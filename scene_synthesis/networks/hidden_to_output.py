from re import A
import torch
import torch.nn as nn
from pytorch3d import transforms

from .bbox_output import AutoregressiveBBoxOutput
from .base import FixedPositionalEncoding, sample_from_dmll


class Hidden2Output(nn.Module):
    def __init__(self, hidden_size, n_classes, with_extra_fc=False):
        super().__init__()
        self.with_extra_fc = with_extra_fc
        self.n_classes = n_classes
        self.hidden_size = hidden_size

        mlp_layers = [
            nn.Linear(hidden_size, 2*hidden_size),
            nn.ReLU(),
            nn.Linear(2*hidden_size, hidden_size),
            nn.ReLU()
        ]
        self.hidden2output = nn.Sequential(*mlp_layers)

    def apply_linear_layers(self, x):
        if self.with_extra_fc:
            x = self.hidden2output(x)

        class_labels = self.class_layer(x)
        translations = (
            self.centroid_layer_x(x),
            self.centroid_layer_y(x),
            self.centroid_layer_z(x)
        )
        sizes = (
            self.size_layer_x(x),
            self.size_layer_y(x),
            self.size_layer_z(x)
        )
        angles = self.angle_layer(x)
        return class_labels, translations, sizes, angles

    def forward(self, x, sample_params=None):
        raise NotImplementedError()


class AutoregressiveDMLL(Hidden2Output):
    def __init__(
        self,
        hidden_size,
        n_classes,
        n_pose_classes,
        n_mixtures,
        bbox_output,
        with_extra_fc=False,
        use_6D=False,
        use_t_coarse=False,
        use_p_coarse=False,
        use_s_coarse=False,
        single_head=False,
        single_head_trans=False,
        sampling=False,
        dropout=0.5,
    ):
        super().__init__(hidden_size, n_classes, with_extra_fc)

        if not isinstance(n_mixtures, list):
            n_mixtures = [n_mixtures]*12

        self.use_6D = use_6D
        self.use_t_coarse = use_t_coarse
        self.use_p_coarse = use_p_coarse
        self.use_s_coarse = use_s_coarse
        self.single_head = single_head
        self.single_head_trans = single_head_trans
        self.sampling = sampling
        self.class_layer = nn.Linear(hidden_size, n_classes)

        self.fc_class_labels = nn.Linear(n_classes, 64)
        # Positional embedding for the target translation
        self.pe_trans_x = FixedPositionalEncoding(proj_dims=64)
        self.pe_trans_y = FixedPositionalEncoding(proj_dims=64)
        self.pe_trans_z = FixedPositionalEncoding(proj_dims=64)
        # Positional embedding for the target angle
        self.pe_pose_0 = FixedPositionalEncoding(proj_dims=64)
        self.pe_pose_1 = FixedPositionalEncoding(proj_dims=64)
        self.pe_pose_2 = FixedPositionalEncoding(proj_dims=64)
        self.pe_pose_3 = FixedPositionalEncoding(proj_dims=64)
        self.pe_pose_4 = FixedPositionalEncoding(proj_dims=64)
        self.pe_pose_5 = FixedPositionalEncoding(proj_dims=64)

        c_hidden_size = hidden_size + 64
        if self.use_t_coarse:
            self.tran_bin_layer = AutoregressiveDMLL._mlp(c_hidden_size, n_pose_classes, dropout)
            c_hidden_size += n_pose_classes
        
        # prediction layers for translations
        if self.single_head or self.single_head_trans:
            self.centroid_layers = AutoregressiveDMLL._mlp(\
                c_hidden_size, sum([n_mixtures[i]*3 for i in range(3)]), dropout)
        else:
            self.centroid_layers = nn.ModuleList([AutoregressiveDMLL._mlp(\
                c_hidden_size, n_mixtures[i]*3, dropout) for i in range(3)])

        c_hidden_size = c_hidden_size + 64*3
        # if use coarse prediction, concate coarse bins to the input for pose prediction
        if self.use_p_coarse:
            self.pose_bin_layer = AutoregressiveDMLL._mlp(c_hidden_size, n_pose_classes, dropout)
            c_hidden_size += n_pose_classes

        # prediction layer for poses
        if self.sampling:
            if self.single_head:
                self.pose_layers = AutoregressiveDMLL._mlp(
                   c_hidden_size, sum([n_mixtures[i]*3 for i in range(6, 10 + int(self.use_6D) * 2)]) , dropout)
            else:
                self.pose_layers = nn.ModuleList([AutoregressiveDMLL._mlp(
                    c_hidden_size, n_mixtures[i]*3, dropout) for i in range(6, 10 + int(self.use_6D) * 2)])
        else:
            self.pose_layers = nn.ModuleList([AutoregressiveDMLL._mlp(
                c_hidden_size, 6, dropout)])
        
        if use_6D:
            c_hidden_size = c_hidden_size + 64 * 6
        else:
            c_hidden_size = c_hidden_size + 64 * 4

        if self.use_s_coarse:
            self.size_bin_layer = AutoregressiveDMLL._mlp(c_hidden_size, n_pose_classes, dropout)
            c_hidden_size = c_hidden_size + n_pose_classes

        # prediction layer for sizes
        if self.single_head:
            self.size_layers = AutoregressiveDMLL._mlp(
                c_hidden_size, sum([n_mixtures[i]*3 for i in range(3, 6)]), dropout)
        else:
            self.size_layers = nn.ModuleList([AutoregressiveDMLL._mlp(
                c_hidden_size, n_mixtures[i]*3, dropout) for i in range(3, 6)])

        self.bbox_output = bbox_output

    @staticmethod
    def _mlp(hidden_size, output_size, dropout):
        mlp_layers = [
            nn.Linear(hidden_size, 2*hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(2*hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, output_size)
        ]
        return nn.Sequential(*mlp_layers)

    @staticmethod
    def _extract_properties_from_target(sample_params):
        class_labels = sample_params["target_class"].float()
        translations = sample_params["target_boxes"][:, :, :3].float()
        sizes = sample_params["target_boxes"][:, :, 3:6].float()
        angles = sample_params["target_boxes"][:, :, 6:].float()
        tran_bins = sample_params["target_tran_bins_oh"]
        angle_bins = sample_params["target_pose_bins_oh"]
        size_bins = sample_params["target_size_bins_oh"]

        return class_labels, translations, sizes, angles, tran_bins, angle_bins, size_bins

    @staticmethod
    def get_dmll_params(pred):
        assert len(pred.shape) == 2

        N = pred.size(0)
        nr_mix = pred.size(1) // 3

        probs = torch.softmax(pred[:, :nr_mix], dim=-1)
        means = pred[:, nr_mix:2 * nr_mix]
        scales = torch.nn.functional.elu(pred[:, 2*nr_mix:3*nr_mix]) + 1.0001

        return probs, means, scales

    def get_translations_dmll_params(self, x, class_labels):
        # Extract the sizes in local variables for convenience
        B, L, _ = class_labels.shape

        c = self.fc_class_labels(class_labels)
        cf = torch.cat([x, c], dim=-1)
        translations_x = self.centroid_layer_x(cf).reshape(B*L, -1)
        translations_y = self.centroid_layer_y(cf).reshape(B*L, -1)
        translations_z = self.centroid_layer_z(cf).reshape(B*L, -1)

        dmll_params = {}
        p = AutoregressiveDMLL.get_dmll_params(translations_x)
        dmll_params["translations_x_probs"] = p[0]
        dmll_params["translations_x_means"] = p[1]
        dmll_params["translations_x_scales"] = p[2]

        p = AutoregressiveDMLL.get_dmll_params(translations_y)
        dmll_params["translations_y_probs"] = p[0]
        dmll_params["translations_y_means"] = p[1]
        dmll_params["translations_y_scales"] = p[2]

        p = AutoregressiveDMLL.get_dmll_params(translations_z)
        dmll_params["translations_z_probs"] = p[0]
        dmll_params["translations_z_means"] = p[1]
        dmll_params["translations_z_scales"] = p[2]

        return dmll_params

    def sample_class_labels(self, x):
        class_labels = self.class_layer(x)

        pred_class = torch.argmax(class_labels, dim=-1)
        C = class_labels.shape[-1]
        pred_class = nn.functional.one_hot(pred_class, C)

        return pred_class.float(), class_labels

    def get_cf_feature(self, x, class_labels):
        c = self.fc_class_labels(class_labels)
        cf = torch.cat([x, c], dim=-1)

        return cf
    
    def get_tf_feature(self, cf, ts):
        tx = self.pe_trans_x(ts[:, :, 0:1])
        ty = self.pe_trans_y(ts[:, :, 1:2])
        tz = self.pe_trans_z(ts[:, :, 2:3])
        tf = torch.cat([cf.repeat(repeats=(1, tx.shape[1], 1)), tx, ty, tz], dim=-1)

        return tf

    def get_sf_feature(self, tf, ans):
        p0 = self.pe_pose_0(ans[:, :, 0:1])
        p1 = self.pe_pose_1(ans[:, :, 1:2])
        p2 = self.pe_pose_2(ans[:, :, 2:3])
        p3 = self.pe_pose_3(ans[:, :, 3:4])
        sf = [tf, p0, p1, p2, p3]
        if self.use_6D:
            sf.extend([self.pe_pose_4(ans[:, :, 4:5]), self.pe_pose_5(ans[:, :, 5:6])])
        
        sf = torch.cat(sf, dim=-1)

        return sf

    def sample_tran_bins(self, cf):
        # predict the bins of the angle first
        bins = self.tran_bin_layer(cf)
        pred_bins = torch.argmax(bins, dim=-1)
        C = bins.shape[-1]
        pred_bins = nn.functional.one_hot(pred_bins, C)
        return pred_bins, bins

    def sample_translations(self, cf):
        # Extract the sizes in local variables for convenience
        B, L, _ = cf.shape
        
        if self.single_head or self.single_head_trans:
            translations = self.centroid_layers(cf)
            split_chunck = translations.shape[-1] // 3
            translations_x, translations_y, translations_z = torch.split(\
                translations, split_chunck, -1)
        else:
            translations_x = self.centroid_layers[0](cf)
            translations_y = self.centroid_layers[1](cf)
            translations_z = self.centroid_layers[2](cf)

        t_x = sample_from_dmll(translations_x.reshape(B*L, -1))
        t_y = sample_from_dmll(translations_y.reshape(B*L, -1))
        t_z = sample_from_dmll(translations_z.reshape(B*L, -1))
        return torch.cat([t_x, t_y, t_z], dim=-1).view(B, L, 3), \
            (translations_x, translations_y, translations_z)

    def sample_pose_bins(self, tf):
        # predict the bins of the angle first
        bins = self.pose_bin_layer(tf)
        pred_bins = torch.argmax(bins, dim=-1)
        C = bins.shape[-1]
        pred_bins = nn.functional.one_hot(pred_bins, C)
        return pred_bins, bins

    def sample_poses(self, tf):
        # Extract the sizes in local variables for convenience
        B, L, _ = tf.shape

        if self.sampling:
            if self.single_head:
                poses = self.pose_layers(tf)
                split_chunck = 6 if self.use_6D else 4 
                split_chunck = poses.shape[-1] // split_chunck
                poses = torch.split(poses, split_chunck, -1)                
            else:
                poses = [pose_layer(tf) for pose_layer in self.pose_layers]
            ps = [sample_from_dmll(pose.reshape(B*L, -1)) for pose in poses]
        else:
            poses = [pose_layer(tf) for pose_layer in self.pose_layers]
            ps = poses

        return torch.cat(ps, dim=-1).view(B, L, -1), poses

    def sample_size_bins(self, sf):
        # predict the bins of the angle first
        bins = self.size_bin_layer(sf)
        pred_bins = torch.argmax(bins, dim=-1)
        C = bins.shape[-1]
        pred_bins = nn.functional.one_hot(pred_bins, C)
        return pred_bins, bins

    def sample_sizes(self, sf):
        # Extract the sizes in local variables for convenience
        B, L, _ = sf.shape
        if self.single_head:
            sizes = self.size_layers(sf)
            split_chunck = sizes.shape[-1] // 3
            sizes_x, sizes_y, sizes_z = torch.split(sizes, split_chunck, -1)
        else:
            sizes_x = self.size_layers[0](sf)
            sizes_y = self.size_layers[1](sf)
            sizes_z = self.size_layers[2](sf)

        s_x = sample_from_dmll(sizes_x.reshape(B*L, -1))
        s_y = sample_from_dmll(sizes_y.reshape(B*L, -1))
        s_z = sample_from_dmll(sizes_z.reshape(B*L, -1))
        return torch.cat([s_x, s_y, s_z], dim=-1).view(B, L, 3), \
            (sizes_x, sizes_y, sizes_z)

    def pred_class_probs(self, x):
        class_labels = self.class_layer(x)

        # Extract the sizes in local variables for convenience
        b, l, _ = class_labels.shape
        c = self.n_classes

        # Sample the class
        class_probs = torch.softmax(class_labels, dim=-1).view(b*l, c)

        return class_probs

    def pred_dmll_params_translation(self, x, class_labels):
        def dmll_params_from_pred(pred):
            assert len(pred.shape) == 2

            N = pred.size(0)
            nr_mix = pred.size(1) // 3

            probs = torch.softmax(pred[:, :nr_mix], dim=-1)
            means = pred[:, nr_mix:2 * nr_mix]
            scales = torch.nn.functional.elu(pred[:, 2*nr_mix:3*nr_mix])
            scales = scales + 1.0001

            return probs, means, scales

        # Extract the sizes in local variables for convenience
        B, L, _ = class_labels.shape

        c = self.fc_class_labels(class_labels)
        cf = torch.cat([x, c], dim=-1)
        t_x = self.centroid_layer_x(cf).reshape(B*L, -1)
        t_y = self.centroid_layer_y(cf).reshape(B*L, -1)
        t_z = self.centroid_layer_z(cf).reshape(B*L, -1)

        return dmll_params_from_pred(t_x), dmll_params_from_pred(t_y),\
            dmll_params_from_pred(t_z)

    def forward(self, x, sample_params, schedule_sampling=False):
        if self.with_extra_fc:
            x = self.hidden2output(x)

        # Extract the target properties from sample_params and embed them into
        # a higher dimensional space.
        target_properties = \
            AutoregressiveDMLL._extract_properties_from_target(
                sample_params
            )

        # predict next part's class label use teacher forcing when applied
        class_labels = self.class_layer(x)
        if not schedule_sampling:
            cls_oh = target_properties[0]
        else:
            cls_oh, class_labels = self.sample_class_labels(x)
        cf = self.get_cf_feature(x, cls_oh)

        tran_bins = None
        if self.use_t_coarse:
            t_bins, tran_bins = self.sample_tran_bins(cf)
            if not schedule_sampling:
                t_bins = target_properties[4]
            cf = torch.cat([cf, t_bins], dim=-1)

        # predict translation of the part use teacher forcing when applied
        if not schedule_sampling:
            ts = target_properties[1]
            if self.single_head or self.single_head_trans:
                translations = self.centroid_layers(cf)
                split_chunck = translations.shape[-1] // 3
                translations = torch.split(translations, split_chunck, -1)
            else:
                translations = [centroid_layer(cf) for centroid_layer in self.centroid_layers]                
        else:
            ts, translations = self.sample_translations(cf)
        tf = self.get_tf_feature(cf, ts)
        
        # coarse prediction of the poses if necessary
        pose_bins = None
        if self.use_p_coarse:
            p_bins, pose_bins = self.sample_pose_bins(tf)
            if not schedule_sampling:
                p_bins = target_properties[5]
            tf = torch.cat([tf, p_bins], dim=-1)

        # predict part's angle use teacher forcing when applied
        if not schedule_sampling:
            ans = target_properties[3]
            if self.single_head:
                poses = self.pose_layers(tf)
                split_chunck = 6 if self.use_6D else 4 
                split_chunck = poses.shape[-1] // split_chunck
                poses = torch.split(poses, split_chunck, -1)
            else:
                poses = [pose_layer(tf) for pose_layer in self.pose_layers]            
            if self.use_6D:
                ans = transforms.quaternion_to_matrix(ans)
                ans = transforms.matrix_to_rotation_6d(ans)
        elif self.sampling: # when not using teacher forcing and using sampling
            ans, poses = self.sample_poses(tf)
        else:
            # direct predict results 
            ans = self.pose_layers[0](tf)
            poses = [ans]
        
        sf = self.get_sf_feature(tf, ans)

        size_bins = None
        if self.use_s_coarse:
            s_bins, size_bins = self.sample_size_bins(sf)
            if not schedule_sampling:
                s_bins = target_properties[6]
            sf = torch.cat([sf, s_bins], dim=-1)

        if self.single_head:
            sizes = self.size_layers(sf)
            split_chunck = sizes.shape[-1] // 3
            sizes = torch.split(sizes, split_chunck, -1)
        else:
            sizes = [size_layer(sf) for size_layer in self.size_layers]

        return self.bbox_output(t_params=translations, p_params=poses, s_params=sizes, \
                                class_labels=class_labels, tran_bins=tran_bins, pose_bins=pose_bins, \
                                size_bins=size_bins, use_6D=self.use_6D, use_t_coarse=self.use_t_coarse, \
                                use_p_coarse=self.use_p_coarse, use_s_coarse=self.use_s_coarse, \
                                use_dmll=self.sampling)
    
    def logits_to_one_hot(self, logits):
        C = logits.shape[-1]
        idx = torch.argmax(logits, dim=-1)
        return torch.eye(C, device=logits.device)[idx]


def get_bbox_output(bbox_type):
    return {
        "autoregressive_mlc": AutoregressiveBBoxOutput
    }[bbox_type]

import torch
import numpy as np
from sklearn.metrics import accuracy_score
from pytorch3d import transforms

from ..losses import cross_entropy_loss, mse_loss, dmll, compute_geodesic_distance_from_two_matrices
from ..stats_logger import StatsLogger


class BBoxOutput(object):
    def __init__(self, sizes, translations, angles, class_labels):
        self.sizes = sizes
        self.translations = translations
        self.angles = angles
        self.class_labels = class_labels

    def __len__(self):
        return len(self.members)

    @property
    def members(self):
        return (self.sizes, self.translations, self.angles, self.class_labels)

    @property
    def n_classes(self):
        return self.class_labels.shape[-1]

    @property
    def device(self):
        return self.class_labels.device

    @staticmethod
    def extract_bbox_params_from_tensor(t, use_6D, use_dmll):
        if isinstance(t, dict):
            class_labels = t["target_label"]
            translations = t["target_boxes"][:, :, :3]
            sizes = t["target_boxes"][:, :, 3:6]
            angles = t["target_boxes"][:, :, 6:]
            rotation_matrix = transforms.quaternion_to_matrix(angles) # convert quat to rotation matrix
            tran_bins = t["target_tran_bins"]
            pose_bins = t["target_pose_bins"]
            size_bins = t["target_size_bins"]
            if use_6D:
                B, L, *_ = rotation_matrix.shape
                angles = transforms.matrix_to_rotation_6d(rotation_matrix) \
                    if use_dmll else rotation_matrix.view(B, L, 3, 3)
        else:
            assert len(t.shape) == 3
            class_labels = t[:, :, :-7]
            translations = t[:, :, -7:-4]
            sizes = t[:, :, -4:-1]
            angles = t[:, :, -1:]

        return class_labels, translations, sizes, angles, rotation_matrix, \
            tran_bins, pose_bins, size_bins

    @property
    def feature_dims(self):
        raise NotImplementedError()

    def get_losses(self, X_target):
        raise NotImplementedError()

    def reconstruction_loss(self, sample_params):
        raise NotImplementedError()


class AutoregressiveBBoxOutput(BBoxOutput):
    def __init__(self, s_params, t_params, p_params, class_labels, \
        poses=None, size_bins=None, tran_bins=None, pose_bins=None, use_6D=False, \
        use_t_coarse=False, use_p_coarse=False, use_s_coarse=False, use_dmll=False):

        self.s_params_x, self.s_params_y, self.s_params_z = s_params
        self.t_params_x, self.t_params_y, self.t_params_z = t_params
        self.class_labels = class_labels
        self.use_6D = use_6D
        self.use_dmll= use_dmll
        self.use_t_coarse = use_t_coarse
        self.use_p_coarse = use_p_coarse
        self.use_s_coarse = use_s_coarse
        
        # convert to rotation matrix if using MSE
        B, L, _ = p_params[0].shape
        self.p_params = p_params if use_dmll else transforms.rotation_6d_to_matrix(p_params[0]).view(B, L, 9)
        self.poses = poses
        self.tran_bins = tran_bins
        self.pose_bins = pose_bins
        self.size_bins = size_bins

    @property
    def members(self):
        return (
            self.s_params_x, self.s_params_y, self.s_params_z,
            self.t_params_x, self.t_params_y, self.t_params_z,
            self.p_params, self.pose_bins, self.class_labels, 
        )

    @property
    def feature_dims(self):
        return self.n_classes + 3 + 3 + 1

    def _targets_from_tensor(self, X_target):
        # Make sure that everything has the correct shape
        # Extract the bbox_params for the target tensor
        target_bbox_params = self.extract_bbox_params_from_tensor(X_target, self.use_6D, self.use_dmll)
        target = {}
        target["labels"] = target_bbox_params[0]
        target["translations_x"] = target_bbox_params[1][:, :, 0:1]
        target["translations_y"] = target_bbox_params[1][:, :, 1:2]
        target["translations_z"] = target_bbox_params[1][:, :, 2:3]
        target["sizes_x"] = target_bbox_params[2][:, :, 0:1]
        target["sizes_y"] = target_bbox_params[2][:, :, 1:2]
        target["sizes_z"] = target_bbox_params[2][:, :, 2:3]
        
        if self.use_dmll:
            pose_length = 6 if self.use_6D else 4
            for i in range(pose_length):
                target["poses_{}".format(i)] = target_bbox_params[3][:, :, i:i+1]
        else:
            # Otherwise directly use the target and calculate MSE loss
            target["poses"] = target_bbox_params[3]
        target["rotation_matrix"] = target_bbox_params[4]
        target["tran_bins"] = target_bbox_params[5]
        target["pose_bins"] = target_bbox_params[6]
        target["size_bins"] = target_bbox_params[7]

        return target

    def get_losses(self, X_target):
        target = self._targets_from_tensor(X_target)
        # assert torch.sum(target["labels"][..., -2]).item() == 0

        # For the class labels compute the cross entropy loss between the
        # target and the predicted labels
        label_loss = cross_entropy_loss(self.class_labels, torch.squeeze(target["labels"], -1))

        label_acc = accuracy_score(torch.squeeze(target["labels"], -1).cpu().numpy(),\
             torch.argmax(self.class_labels, dim=-1).reshape(-1).cpu().numpy())
        # For the t_params, s_params and p_params compute the discretized
        # logistic mixture likelihood as described in 
        # PIXELCNN++: Improving the PixelCNN with Discretized Logistic Mixture Likelihood and
        # Other Modifications, by Salimans et al.

        # only calculate box losses that matter
        box_mask = torch.squeeze(target['labels'], -1).bool()

        translation_loss = dmll(self.t_params_x[box_mask], target["translations_x"][box_mask])
        translation_loss += dmll(self.t_params_y[box_mask], target["translations_y"][box_mask])
        translation_loss += dmll(self.t_params_z[box_mask], target["translations_z"][box_mask])
        size_loss = dmll(self.s_params_x[box_mask], target["sizes_x"][box_mask])
        size_loss += dmll(self.s_params_y[box_mask], target["sizes_y"][box_mask])
        size_loss += dmll(self.s_params_z[box_mask], target["sizes_z"][box_mask])
        pose_loss = torch.zeros_like(size_loss)
        pose_bin_loss = torch.zeros_like(size_loss)
        pose_error = torch.zeros_like(size_loss)
        
        B, L, _ = target["translations_x"][box_mask].shape
        if self.use_dmll:
            pose_length = 6 if self.use_6D else 4
            for i in range(0, pose_length):
                pose_loss += dmll(self.p_params[i][box_mask], target["poses_{}".format(i)][box_mask])
        else:
            pose_pred = self.p_params[box_mask].view(B*L, 3, 3)
            pose_target = target['poses'][box_mask].view(B*L, 3, 3)
            pose_loss += mse_loss(pose_pred, pose_target)

        if self.poses is not None:
            if self.use_6D:
                rotation_matrix_pred = transforms.rotation_6d_to_matrix(self.poses[box_mask]).view(B*L, 3, 3)
            else:
                rotation_matrix_pred = transforms.quaternion_to_matrix(self.poses[box_mask]).view(B*L, 3, 3)
            
            pose_error = compute_geodesic_distance_from_two_matrices(rotation_matrix_pred, \
                target["rotation_matrix"][box_mask].view(B*L, 3, 3))

        # calculate losses for pose_bins
        tran_bin_loss, pose_bin_loss, size_bin_loss = torch.tensor(0.), torch.tensor(0.), torch.tensor(0.)
        if torch.sum(box_mask) != 0:
            if self.use_t_coarse:
                tran_bin_loss = cross_entropy_loss(self.tran_bins[box_mask], torch.squeeze(target["tran_bins"][box_mask], -1))
            if self.use_p_coarse:
                pose_bin_loss = cross_entropy_loss(self.pose_bins[box_mask], torch.squeeze(target["pose_bins"][box_mask], -1))
            if self.use_s_coarse:
                size_bin_loss = cross_entropy_loss(self.size_bins[box_mask], torch.squeeze(target["size_bins"][box_mask], -1))

        # pick the box with the least loss
        if translation_loss.shape[1] > 1:
            losses = (translation_loss + size_loss + pose_loss).mean(dim=0)
            min_ind = torch.argmin(losses)
            translation_loss = translation_loss[:, min_ind:min_ind+1]
            size_loss = size_loss[:, min_ind:min_ind+1]
            pose_loss = pose_loss[:, min_ind:min_ind+1]

        return label_loss, translation_loss, size_loss, pose_loss, \
            tran_bin_loss, pose_bin_loss, size_bin_loss, label_acc, pose_error

    def reconstruction_loss(self, X_target, lengths=None):
        # Compute the losses
        label_loss, translation_loss, size_loss, pose_loss, \
            tran_bin_loss, pose_bin_loss, size_bin_loss, \
            label_acc, pose_error = self.get_losses(X_target)

        label_loss = label_loss.mean()
        translation_loss = translation_loss.mean()
        size_loss = size_loss.mean()
        pose_loss = pose_loss.mean()
        tran_bin_loss = tran_bin_loss.mean()
        pose_bin_loss = pose_bin_loss.mean()
        size_bin_loss = size_bin_loss.mean()
        pose_error = pose_error.mean()

        StatsLogger.instance()["losses.translation"].value = \
            translation_loss.item()
        StatsLogger.instance()["losses.pose"].value = pose_loss.item()
        StatsLogger.instance()["losses.size"].value = size_loss.item()

        StatsLogger.instance()["losses.tran_bin"].value = tran_bin_loss.item()
        StatsLogger.instance()["losses.pose_bin"].value = pose_bin_loss.item()
        StatsLogger.instance()["losses.size_bin"].value = size_bin_loss.item()
        StatsLogger.instance()["losses.label"].value = label_loss.item()
        
        StatsLogger.instance()["error.pose"].value = pose_error.item()
        StatsLogger.instance()["accuracy.label"].value = label_acc.item()

        return label_loss + translation_loss + size_loss + pose_loss + \
            tran_bin_loss + pose_bin_loss + size_bin_loss

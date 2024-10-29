import torch
import torch.nn as nn
from pytorch3d import transforms

from fast_transformers.builders import TransformerEncoderBuilder, TransformerDecoderBuilder
from fast_transformers.masking import LengthMask, FullMask

from sklearn.metrics import accuracy_score, recall_score, precision_score
from .base import FixedPositionalEncoding
from ..stats_logger import StatsLogger
from .bbox_output import AutoregressiveBBoxOutput
from .positional_encoding import PositionalEncoding1D


class BaseAutoregressiveTransformer(nn.Module):
    def __init__(self, input_dims, hidden2output, feature_extractor, config, n_classes):
        super().__init__()
        # Build a transformer encoder
        self.transformer_encoder = TransformerEncoderBuilder.from_kwargs(
            n_layers=config.get("n_layers", 6),
            n_heads=config.get("n_heads", 12),
            query_dimensions=config.get("query_dimensions", 64),
            value_dimensions=config.get("value_dimensions", 64),
            feed_forward_dimensions=config.get(
                "feed_forward_dimensions", 3072
            ),
            dropout=config.get("dropout", 0.5),
            attention_type="full",
            activation="gelu"
        ).get()

        self.register_parameter(
            "start_token_embedding",
            nn.Parameter(torch.randn(1, 640))
        )

        # TODO: Add the projection dimensions for the room features in the
        # config!!!
        self.feature_extractor = feature_extractor if feature_extractor is not None \
                                                 else nn.Sequential(
                                                     nn.Linear(640, 512),
                                                     nn.ReLU(),
                                                 )
        self.fc_root = nn.Linear(512, 704)

        # Positional encoding for each property
        self.pe_pos_x = FixedPositionalEncoding(proj_dims=64)
        self.pe_pos_y = FixedPositionalEncoding(proj_dims=64)
        self.pe_pos_z = FixedPositionalEncoding(proj_dims=64)

        self.pe_size_x = FixedPositionalEncoding(proj_dims=64)
        self.pe_size_y = FixedPositionalEncoding(proj_dims=64)
        self.pe_size_z = FixedPositionalEncoding(proj_dims=64)

        self.pe_angle_z = FixedPositionalEncoding(proj_dims=64)

        # Embedding matix for property class label.
        # Compute the number of classes from the input_dims. Note that we
        # remove 3 to account for the masked bins for the size, position and
        # angle properties
        self.input_dims = input_dims
        self.n_classes = n_classes
        self.fc_class = nn.Linear(self.n_classes, 64, bias=False)

        hidden_dims = config.get("hidden_dims", 768)
        self.fc = nn.Linear(704, hidden_dims)
        self.hidden2output = hidden2output


    def start_symbol(self, device="cpu"):
        start_class = torch.zeros(1, 1, self.n_classes, device=device)
        start_class[0, 0, 1] = 1
        return {
            "input_class": start_class,
            "input_boxes": torch.zeros(1, 1, 10, device=device)
        }


    def end_symbol(self, device="cpu"):
        end_class = torch.zeros(1, 1, self.n_classes, device=device)
        end_class[0, 0, 0] = 1
        return {
            "class_labels": end_class,
            "translations": torch.zeros(1, 1, 3, device=device),
            "sizes": torch.zeros(1, 1, 3, device=device),
            "quats": torch.zeros(1, 1, 4, device=device)
        }

    def start_symbol_features(self, B, root_X):
        # start symbol takes the entire box
        root_X_f = self.fc_root(self.feature_extractor(root_X))
        return root_X_f[:, None, :]

    def forward(self, sample_params):
        raise NotImplementedError()

    def autoregressive_decode(self, boxes, room_mask):
        raise NotImplementedError()

    @torch.no_grad()
    def generate_boxes(self, room_mask, max_boxes=32, device="cpu"):
        raise NotImplementedError()


class AutoregressiveTransformer(BaseAutoregressiveTransformer):
    def __init__(self, input_dims, hidden2output, feature_extractor, config):
        super().__init__(input_dims, hidden2output, feature_extractor, config)
        # Embedding to be used for the empty/mask token
        self.register_parameter(
            "empty_token_embedding", nn.Parameter(torch.randn(1, 576))
        )

    def forward(self, sample_params):
        # Unpack the sample_params
        class_labels = sample_params["class_labels"]
        translations = sample_params["translations"]
        sizes = sample_params["sizes"]
        angles = sample_params["angles"]
        room_layout = sample_params["room_layout"]
        B, _, _ = class_labels.shape

        # Apply the positional embeddings only on bboxes that are not the start
        # token
        class_f = self.fc_class(class_labels)
        # Apply the positional embedding along each dimension of the position
        # property
        pos_f_x = self.pe_pos_x(translations[:, :, 0:1])
        pos_f_y = self.pe_pos_x(translations[:, :, 1:2])
        pos_f_z = self.pe_pos_x(translations[:, :, 2:3])
        pos_f = torch.cat([pos_f_x, pos_f_y, pos_f_z], dim=-1)

        size_f_x = self.pe_size_x(sizes[:, :, 0:1])
        size_f_y = self.pe_size_x(sizes[:, :, 1:2])
        size_f_z = self.pe_size_x(sizes[:, :, 2:3])
        size_f = torch.cat([size_f_x, size_f_y, size_f_z], dim=-1)

        angle_f = self.pe_angle_z(angles)
        X = torch.cat([class_f, pos_f, size_f, angle_f], dim=-1)

        start_symbol_f = self.start_symbol_features(B, room_layout)
        # Concatenate with the mask embedding for the start token
        X = torch.cat([
            start_symbol_f, self.empty_token_embedding.expand(B, -1, -1), X
        ], dim=1)
        X = self.fc(X)

        # Compute the features using causal masking
        lengths = LengthMask(
            sample_params["lengths"]+2,
            max_len=X.shape[1]
        )
        F = self.transformer_encoder(X, length_mask=lengths)
        return self.hidden2output(F[:, 1:2], sample_params)

    def _encode(self, boxes, room_mask):
        class_labels = boxes["class_labels"]
        translations = boxes["translations"]
        sizes = boxes["sizes"]
        angles = boxes["angles"]
        B, _, _ = class_labels.shape

        if class_labels.shape[1] == 1:
            start_symbol_f = self.start_symbol_features(B, room_mask)
            X = torch.cat([
                start_symbol_f, self.empty_token_embedding.expand(B, -1, -1)
            ], dim=1)
        else:
            # Apply the positional embeddings only on bboxes that are not the
            # start token
            class_f = self.fc_class(class_labels[:, 1:])
            # Apply the positional embedding along each dimension of the
            # position property
            pos_f_x = self.pe_pos_x(translations[:, 1:, 0:1])
            pos_f_y = self.pe_pos_x(translations[:, 1:, 1:2])
            pos_f_z = self.pe_pos_x(translations[:, 1:, 2:3])
            pos_f = torch.cat([pos_f_x, pos_f_y, pos_f_z], dim=-1)

            size_f_x = self.pe_size_x(sizes[:, 1:, 0:1])
            size_f_y = self.pe_size_x(sizes[:, 1:, 1:2])
            size_f_z = self.pe_size_x(sizes[:, 1:, 2:3])
            size_f = torch.cat([size_f_x, size_f_y, size_f_z], dim=-1)

            angle_f = self.pe_angle_z(angles[:, 1:])
            X = torch.cat([class_f, pos_f, size_f, angle_f], dim=-1)

            start_symbol_f = self.start_symbol_features(B, room_mask)
            # Concatenate with the mask embedding for the start token
            X = torch.cat([
                start_symbol_f, self.empty_token_embedding.expand(B, -1, -1), X
            ], dim=1)
        X = self.fc(X)
        F = self.transformer_encoder(X, length_mask=None)[:, 1:2]

        return F

    def autoregressive_decode(self, boxes, room_mask):
        class_labels = boxes["class_labels"]

        # Compute the features using the transformer
        F = self._encode(boxes, room_mask)
        # Sample the class label for the next bbbox
        class_labels = self.hidden2output.sample_class_labels(F)
        # Sample the translations
        translations = self.hidden2output.sample_translations(F, class_labels)
        # Sample the angles
        angles = self.hidden2output.sample_angles(
            F, class_labels, translations
        )
        # Sample the sizes
        sizes = self.hidden2output.sample_sizes(
            F, class_labels, translations, angles
        )

        return {
            "class_labels": class_labels,
            "translations": translations,
            "sizes": sizes,
            "angles": angles
        }

    @torch.no_grad()
    def generate_boxes(self, room_mask, max_boxes=32, device="cpu"):
        boxes = self.start_symbol(device)
        for i in range(max_boxes):
            box = self.autoregressive_decode(boxes, room_mask=room_mask)

            for k in box.keys():
                boxes[k] = torch.cat([boxes[k], box[k]], dim=1)

            # Check if we have the end symbol
            if box["class_labels"][0, 0, -1] == 1:
                break

        return {
            "class_labels": boxes["class_labels"].cpu(),
            "translations": boxes["translations"].cpu(),
            "sizes": boxes["sizes"].cpu(),
            "angles": boxes["angles"].cpu()
        }

    def autoregressive_decode_with_class_label(
        self, boxes, room_mask, class_label
    ):
        class_labels = boxes["class_labels"]
        B, _, C = class_labels.shape

        # Make sure that everything has the correct size
        assert len(class_label.shape) == 3
        assert class_label.shape[0] == B
        assert class_label.shape[-1] == C

        # Compute the features using the transformer
        F = self._encode(boxes, room_mask)

        # Sample the translations conditioned on the query_class_label
        translations = self.hidden2output.sample_translations(F, class_label)
        # Sample the angles
        angles = self.hidden2output.sample_angles(
            F, class_label, translations
        )
        # Sample the sizes
        sizes = self.hidden2output.sample_sizes(
            F, class_label, translations, angles
        )

        return {
            "class_labels": class_label,
            "translations": translations,
            "sizes": sizes,
            "angles": angles
        }

    @torch.no_grad()
    def add_object(self, room_mask, class_label, boxes=None, device="cpu"):
        boxes = dict(boxes.items())

        # Make sure that the provided class_label will have the correct format
        if isinstance(class_label, int):
            one_hot = torch.eye(self.n_classes)
            class_label = one_hot[class_label][None, None]
        elif not torch.is_tensor(class_label):
            class_label = torch.from_numpy(class_label)

        # Make sure that the class label the correct size,
        # namely (batch_size, 1, n_classes)
        assert class_label.shape == (1, 1, self.n_classes)

        # Create the initial input to the transformer, namely the start token
        start_box = self.start_symbol(device)
        for k in start_box.keys():
            boxes[k] = torch.cat([start_box[k], boxes[k]], dim=1)

        # Based on the query class label sample the location of the new object
        box = self.autoregressive_decode_with_class_label(
            boxes=boxes,
            room_mask=room_mask,
            class_label=class_label
        )

        for k in box.keys():
            boxes[k] = torch.cat([boxes[k], box[k]], dim=1)

        # Creat a box for the end token and update the boxes dictionary
        end_box = self.end_symbol(device)
        for k in end_box.keys():
            boxes[k] = torch.cat([boxes[k], end_box[k]], dim=1)

        return {
            "class_labels": boxes["class_labels"],
            "translations": boxes["translations"],
            "sizes": boxes["sizes"],
            "angles": boxes["angles"]
        }

    @torch.no_grad()
    def complete_scene(
        self,
        boxes,
        room_mask,
        max_boxes=100,
        device="cpu"
    ):
        boxes = dict(boxes.items())

        # Create the initial input to the transformer, namely the start token
        start_box = self.start_symbol(device)
        # Add the start box token in the beginning
        for k in start_box.keys():
            boxes[k] = torch.cat([start_box[k], boxes[k]], dim=1)

        for i in range(max_boxes):
            box = self.autoregressive_decode(boxes, room_mask=room_mask)

            for k in box.keys():
                boxes[k] = torch.cat([boxes[k], box[k]], dim=1)

            # Check if we have the end symbol
            if box["class_labels"][0, 0, -1] == 1:
                break

        return {
            "class_labels": boxes["class_labels"],
            "translations": boxes["translations"],
            "sizes": boxes["sizes"],
            "angles": boxes["angles"]
        }

    def autoregressive_decode_with_class_label_and_translation(
        self,
        boxes,
        room_mask,
        class_label,
        translation
    ):
        class_labels = boxes["class_labels"]
        B, _, C = class_labels.shape

        # Make sure that everything has the correct size
        assert len(class_label.shape) == 3
        assert class_label.shape[0] == B
        assert class_label.shape[-1] == C

        # Compute the features using the transformer
        F = self._encode(boxes, room_mask)

        # Sample the angles
        angles = self.hidden2output.sample_angles(F, class_label, translation)
        # Sample the sizes
        sizes = self.hidden2output.sample_sizes(
            F, class_label, translation, angles
        )

        return {
            "class_labels": class_label,
            "translations": translation,
            "sizes": sizes,
            "angles": angles
        }

    @torch.no_grad()
    def add_object_with_class_and_translation(
        self,
        boxes,
        room_mask,
        class_label,
        translation,
        device="cpu"
    ):
        boxes = dict(boxes.items())

        # Make sure that the provided class_label will have the correct format
        if isinstance(class_label, int):
            one_hot = torch.eye(self.n_classes)
            class_label = one_hot[class_label][None, None]
        elif not torch.is_tensor(class_label):
            class_label = torch.from_numpy(class_label)

        # Make sure that the class label the correct size,
        # namely (batch_size, 1, n_classes)
        assert class_label.shape == (1, 1, self.n_classes)


        # Create the initial input to the transformer, namely the start token
        start_box = self.start_symbol(device)
        for k in start_box.keys():
            boxes[k] = torch.cat([start_box[k], boxes[k]], dim=1)

        # Based on the query class label sample the location of the new object
        box = self.autoregressive_decode_with_class_label_and_translation(
            boxes=boxes,
            class_label=class_label,
            translation=translation,
            room_mask=room_mask
        )

        for k in box.keys():
            boxes[k] = torch.cat([boxes[k], box[k]], dim=1)

        # Creat a box for the end token and update the boxes dictionary
        end_box = self.end_symbol(device)
        for k in end_box.keys():
            boxes[k] = torch.cat([boxes[k], end_box[k]], dim=1)

        return {
            "class_labels": boxes["class_labels"],
            "translations": boxes["translations"],
            "sizes": boxes["sizes"],
            "angles": boxes["angles"]
        }

    @torch.no_grad()
    def distribution_classes(self, boxes, room_mask, device="cpu"):
        # Shallow copy the input dictionary
        boxes = dict(boxes.items())
        # Create the initial input to the transformer, namely the start token
        start_box = self.start_symbol(device)
        # Add the start box token in the beginning
        for k in start_box.keys():
            boxes[k] = torch.cat([start_box[k], boxes[k]], dim=1)

        # Compute the features using the transformer
        F = self._encode(boxes, room_mask)
        return self.hidden2output.pred_class_probs(F)

    @torch.no_grad()
    def distribution_translations(
        self,
        boxes,
        room_mask, 
        class_label,
        device="cpu"
    ):
        # Shallow copy the input dictionary
        boxes = dict(boxes.items())

        # Make sure that the provided class_label will have the correct format
        if isinstance(class_label, int):
            one_hot = torch.eye(self.n_classes)
            class_label = one_hot[class_label][None, None]
        elif not torch.is_tensor(class_label):
            class_label = torch.from_numpy(class_label)

        # Make sure that the class label the correct size,
        # namely (batch_size, 1, n_classes)
        assert class_label.shape == (1, 1, self.n_classes)

        # Create the initial input to the transformer, namely the start token
        start_box = self.start_symbol(device)
        # Concatenate to the given input (that's why we shallow copy in the
        # beginning of this method
        for k in start_box.keys():
            boxes[k] = torch.cat([start_box[k], boxes[k]], dim=1)

        # Compute the features using the transformer
        F = self._encode(boxes, room_mask)

        # Get the dmll params for the translations
        return self.hidden2output.pred_dmll_params_translation(
            F, class_label
        )


class AutoregressiveTransformerPE(AutoregressiveTransformer):
    def __init__(self, input_dims, hidden2output, feature_extractor, config):
        super().__init__(input_dims, hidden2output, feature_extractor, config)
        # Embedding to be used for the empty/mask token
        self.register_parameter(
            "empty_token_embedding", nn.Parameter(torch.randn(1, 512))
        )

        # Positional embedding for the ordering
        max_seq_length = 32
        self.register_parameter(
            "positional_embedding",
            nn.Parameter(torch.randn(max_seq_length, 32))
        )

        # Positional encoding for each property
        self.pe_pos_x = FixedPositionalEncoding(proj_dims=60)
        self.pe_pos_y = FixedPositionalEncoding(proj_dims=60)
        self.pe_pos_z = FixedPositionalEncoding(proj_dims=60)

        self.pe_size_x = FixedPositionalEncoding(proj_dims=60)
        self.pe_size_y = FixedPositionalEncoding(proj_dims=60)
        self.pe_size_z = FixedPositionalEncoding(proj_dims=64)

        self.pe_angle_z = FixedPositionalEncoding(proj_dims=60)

        # Embedding matix for property class label.
        # Compute the number of classes from the input_dims. Note that we
        # remove 3 to account for the masked bins for the size, position and
        # angle properties
        self.input_dims = input_dims
        self.n_classes = self.input_dims - 3 - 3 - 1
        self.fc_class = nn.Linear(self.n_classes, 60, bias=False)

    def forward(self, sample_params, schedule_sampling):
        # Unpack the sample_params
        class_labels = sample_params["class_labels"]
        translations = sample_params["translations"]
        sizes = sample_params["sizes"]
        angles = sample_params["angles"]
        room_layout = sample_params["room_layout"]
        B, L, _ = class_labels.shape

        # Apply the positional embeddings only on bboxes that are not the start
        # token
        class_f = self.fc_class(class_labels)
        # Apply the positional embedding along each dimension of the position
        # property
        pos_f_x = self.pe_pos_x(translations[:, :, 0:1])
        pos_f_y = self.pe_pos_x(translations[:, :, 1:2])
        pos_f_z = self.pe_pos_x(translations[:, :, 2:3])
        pos_f = torch.cat([pos_f_x, pos_f_y, pos_f_z], dim=-1)

        size_f_x = self.pe_size_x(sizes[:, :, 0:1])
        size_f_y = self.pe_size_x(sizes[:, :, 1:2])
        size_f_z = self.pe_size_x(sizes[:, :, 2:3])
        size_f = torch.cat([size_f_x, size_f_y, size_f_z], dim=-1)

        angle_f = self.pe_angle_z(angles)
        pe = self.positional_embedding[None, :L].expand(B, -1, -1)
        X = torch.cat([class_f, pos_f, size_f, angle_f, pe], dim=-1)

        start_symbol_f = self.start_symbol_features(B, room_layout)
        # Concatenate with the mask embedding for the start token
        X = torch.cat([
            start_symbol_f, self.empty_token_embedding.expand(B, -1, -1), X
        ], dim=1)
        X = self.fc(X)

        # Compute the features using causal masking
        lengths = LengthMask(
            sample_params["lengths"]+2,
            max_len=X.shape[1]
        )
        F = self.transformer_encoder(X, length_mask=lengths)
        return self.hidden2output(F[:, 1:2], sample_params, schedule_sampling)

    def _encode(self, boxes, room_mask):
        class_labels = boxes["class_labels"]
        translations = boxes["translations"]
        sizes = boxes["sizes"]
        angles = boxes["angles"]
        B, L, _ = class_labels.shape

        if class_labels.shape[1] == 1:
            start_symbol_f = self.start_symbol_features(B, room_mask)
            X = torch.cat([
                start_symbol_f, self.empty_token_embedding.expand(B, -1, -1)
            ], dim=1)
        else:
            # Apply the positional embeddings only on bboxes that are not the
            # start token
            class_f = self.fc_class(class_labels[:, 1:])
            # Apply the positional embedding along each dimension of the
            # position property
            pos_f_x = self.pe_pos_x(translations[:, 1:, 0:1])
            pos_f_y = self.pe_pos_x(translations[:, 1:, 1:2])
            pos_f_z = self.pe_pos_x(translations[:, 1:, 2:3])
            pos_f = torch.cat([pos_f_x, pos_f_y, pos_f_z], dim=-1)

            size_f_x = self.pe_size_x(sizes[:, 1:, 0:1])
            size_f_y = self.pe_size_x(sizes[:, 1:, 1:2])
            size_f_z = self.pe_size_x(sizes[:, 1:, 2:3])
            size_f = torch.cat([size_f_x, size_f_y, size_f_z], dim=-1)

            angle_f = self.pe_angle_z(angles[:, 1:])
            pe = self.positional_embedding[None, 1:L].expand(B, -1, -1)
            X = torch.cat([class_f, pos_f, size_f, angle_f, pe], dim=-1)

            start_symbol_f = self.start_symbol_features(B, room_mask)
            # Concatenate with the mask embedding for the start token
            X = torch.cat([
                start_symbol_f, self.empty_token_embedding.expand(B, -1, -1), X
            ], dim=1)
        X = self.fc(X)
        F = self.transformer_encoder(X, length_mask=None)[:, 1:2]

        return F

    
class ObjectGenerationTransformer(BaseAutoregressiveTransformer):
    def __init__(self, input_dims, hidden2output, feature_extractor, config, n_classes):
        super().__init__(input_dims, hidden2output, feature_extractor, config, n_classes)
        self.pe_quad_a = FixedPositionalEncoding(proj_dims=64)
        self.pe_quad_b = FixedPositionalEncoding(proj_dims=64)
        self.pe_quad_c = FixedPositionalEncoding(proj_dims=64)
        self.pe_quad_d = FixedPositionalEncoding(proj_dims=64)
        # Embedding to be used for the empty/mask token
        self.register_parameter(
            "empty_token_embedding", nn.Parameter(torch.randn(1, 704))
        )
        self.use_6D = config['use_6D']
        self.use_t_coarse = config['use_t_coarse']
        self.use_p_coarse = config['use_p_coarse']
        self.use_s_coarse = config['use_s_coarse']
        self.use_clip = config['use_clip']
        self.use_pos_enc = config['pos_enc']
        self.sampling = config['sampling']

        if self.use_clip:
            self.clip_fuser = nn.Linear(512, 704)
        
        if self.use_pos_enc == 'absolute':
            self.pos_enc = PositionalEncoding1D(576)

    def get_pe_box(self, translations, sizes, quads):
        pos_f_x = self.pe_pos_x(translations[:, :, 0:1])
        pos_f_y = self.pe_pos_y(translations[:, :, 1:2])
        pos_f_z = self.pe_pos_z(translations[:, :, 2:3])
        pos_f = torch.cat([pos_f_x, pos_f_y, pos_f_z], dim=-1)

        size_f_x = self.pe_size_x(sizes[:, :, 0:1])
        size_f_y = self.pe_size_y(sizes[:, :, 1:2])
        size_f_z = self.pe_size_z(sizes[:, :, 2:3])
        size_f = torch.cat([size_f_x, size_f_y, size_f_z], dim=-1)

        quad_f_a = self.pe_quad_a(quads[:, :, 0:1])
        quad_f_b = self.pe_quad_b(quads[:, :, 1:2])
        quad_f_c = self.pe_quad_c(quads[:, :, 2:3])
        quad_f_d = self.pe_quad_d(quads[:, :, 3:4])
        quad_f = torch.cat([quad_f_a, quad_f_b, quad_f_c, quad_f_d], dim=-1)
        X = torch.cat([pos_f, size_f, quad_f], dim=-1)
        return X

    def forward(self, sample_params, schedule_sampling=False):
        # Unpack the sample_params
        class_labels = sample_params["input_class"]
        boxes = torch.cat((sample_params['root_box'][:, None], sample_params['input_boxes']), dim=1)
        translations = boxes[:, :, :3]
        sizes = boxes[:, :, 3:6]
        quads = boxes[:, :, 6:]
        B = sample_params['input_boxes'].shape[0]

        # Apply the positional embeddings only on bboxes that are not the start
        # token
        class_f = self.fc_class(class_labels)
        # Apply the positional embedding along each dimension of the position
        # property
        X = self.get_pe_box(translations, sizes, quads)
        root_X, X = X[:, 0], X[:, 1:]
        X = torch.cat([class_f, X], dim=-1)
        start_symbol_f = self.start_symbol_features(B, root_X)

        if self.use_clip:
            # Concatenate with the mask embedding for the start token
            X = torch.cat([
                start_symbol_f, self.empty_token_embedding.expand(B, -1, -1), 
                self.clip_fuser(sample_params['clip_feature']), X
            ], dim=1)
            lengths = LengthMask(
                sample_params["lengths"]+3,
                max_len=X.shape[1]
            )
        else:
            X = torch.cat([
                start_symbol_f, self.empty_token_embedding.expand(B, -1, -1), 
                X], dim=1)

            lengths = LengthMask(
                sample_params["lengths"]+2,
                max_len=X.shape[1]
            )

        X = self.fc(X)

        if torch.any(torch.isnan(X)):
            print("NaN in Model foward() function")
            exit()

        if self.use_pos_enc is not None:
            X += self.pos_enc(X)

        # Compute the features using causal masking
        F = self.transformer_encoder(X, length_mask=lengths)

        return self.hidden2output(F[:, 1:2], sample_params) # predict on cls

    def _encode(self, sample_params):
        class_labels = sample_params["input_class"]
        translations = sample_params['input_boxes'][:, :, :3]
        sizes = sample_params['input_boxes'][:, :, 3:6]
        quads = sample_params['input_boxes'][:, :, 6:]
        
        B, _, _ = class_labels.shape

        lengths=None
        if class_labels.shape[1] == 1:

            if self.use_clip:
                # Concatenate with the mask embedding for the start token
                X = torch.cat([
                    self.empty_token_embedding.expand(B, -1, -1), 
                    self.clip_fuser(sample_params['clip_feature'])], dim=1)
            else:
                X = torch.cat([
                    self.empty_token_embedding.expand(B, -1, -1)], dim=1)            
        else:
            # Apply the positional embeddings only on bboxes that are not the
            # start token
            class_f = self.fc_class(class_labels[:, 1:])
            # Apply the positional embedding along each dimension of the
            # position property
            pos_f_x = self.pe_pos_x(translations[:, 1:, 0:1])
            pos_f_y = self.pe_pos_y(translations[:, 1:, 1:2])
            pos_f_z = self.pe_pos_z(translations[:, 1:, 2:3])
            pos_f = torch.cat([pos_f_x, pos_f_y, pos_f_z], dim=-1)

            size_f_x = self.pe_size_x(sizes[:, 1:, 0:1])
            size_f_y = self.pe_size_y(sizes[:, 1:, 1:2])
            size_f_z = self.pe_size_z(sizes[:, 1:, 2:3])
            size_f = torch.cat([size_f_x, size_f_y, size_f_z], dim=-1)

            quad_f_a = self.pe_quad_a(quads[:, 1:, 0:1])
            quad_f_b = self.pe_quad_b(quads[:, 1:, 1:2])
            quad_f_c = self.pe_quad_c(quads[:, 1:, 2:3])
            quad_f_d = self.pe_quad_d(quads[:, 1:, 3:4])
            quad_f = torch.cat([quad_f_a, quad_f_b, quad_f_c, quad_f_d], dim=-1)
            X = torch.cat([class_f, pos_f, size_f, quad_f], dim=-1)

            # Concatenate with the mask embedding for the start token
            if self.use_clip:
                # Concatenate with the mask embedding for the start token
                X = torch.cat([
                    self.empty_token_embedding.expand(B, -1, -1), 
                    self.clip_fuser(sample_params['clip_feature']), X
                ], dim=1)
            else:
                X = torch.cat([
                    self.empty_token_embedding.expand(B, -1, -1), 
                    X], dim=1)

        if self.use_clip:
            lengths = LengthMask(
                sample_params["lengths"]+2,
                max_len=X.shape[1]
            )
        else:
            lengths = LengthMask(
                sample_params["lengths"]+1,
                max_len=X.shape[1]
            )

        X = self.fc(X)

        if self.use_pos_enc is not None:
            X += self.pos_enc(X)

        F = self.transformer_encoder(X, length_mask=lengths)

        if torch.any(torch.isnan(F)):
            print("NaN in Model _encode() function")
            exit()

        return F

    @torch.no_grad()
    def autoregressive_decode(self, sample_params):

        # Compute the features using the transformer
        F = self._encode(sample_params)[:, 1:2]
        # Sample the class label for the next bbbox
        class_labels, c_logits = self.hidden2output.sample_class_labels(F)
        cf = self.hidden2output.get_cf_feature(F, class_labels)
        
        t_bins_param = None
        if self.use_t_coarse:
            tran_bins, t_bins_param = self.hidden2output.sample_tran_bins(cf)
            cf = torch.cat([cf, tran_bins], dim=-1)
        
        # Sample the translations
        translations, t_params = self.hidden2output.sample_translations(cf)
        tf = self.hidden2output.get_tf_feature(cf, translations)

        p_bins_param = None
        if self.use_p_coarse:
            pose_bins, p_bins_param = self.hidden2output.sample_pose_bins(tf)
            tf = torch.cat([tf, pose_bins], dim=-1)

        # Sample the angles
        poses, p_params = self.hidden2output.sample_poses(tf)
        sf = self.hidden2output.get_sf_feature(tf, poses)
        
        s_bins_param = None
        if self.use_s_coarse:
            size_bins, s_bins_param = self.hidden2output.sample_size_bins(sf)
            sf = torch.cat([sf, size_bins], dim=-1)
        
        # Sample the sizes
        sizes, s_params = self.hidden2output.sample_sizes(sf)


        # convert 6D back to quat
        if self.use_6D:
            poses = transforms.rotation_6d_to_matrix(poses)
            poses = transforms.matrix_to_quaternion(poses)

        return {
            "input_class": class_labels,
            "input_boxes": torch.cat([translations, sizes, poses], dim=-1),
        }, {
            "s_params": s_params,
            "t_params": t_params,
            "p_params": p_params,
            "c_logits": c_logits,
            "t_bins_param": t_bins_param,  
            "p_bins_param": p_bins_param,
            "s_bins_param": s_bins_param 
        }

    @torch.no_grad()
    def generate_boxes(self, root_box, max_boxes=3, device="cpu"):
        boxes = self.start_symbol(device)
        for i in range(max_boxes):
            box = self.autoregressive_decode(boxes, root_box=root_box)
            for k in box.keys():
                boxes[k] = torch.cat([boxes[k], box[k]], dim=1)

            # Check if we have the end symbol
            if box["class_labels"][0, 0, -1] == 0:
                break
        return {
            "class_labels": boxes["class_labels"].cpu(),
            "translations": boxes["translations"].cpu(),
            "sizes": boxes["sizes"].cpu(),
            "angles": boxes["angles"].cpu()
        }

    def autoregressive_decode_with_class_label(
        self, boxes, root_box, target_boxes
    ):

        class_labels = target_boxes['target_class']

        if class_labels.shape[1] == 0:
            B = class_labels.shape[0]
            C_c = class_labels.shape[2]
            C_t = boxes['translations'].shape[2]
            C_s = boxes['sizes'].shape[2]
            C_q = boxes['quats'].shape[2]
            
            return {
                "class_labels": torch.ones(B, 1, C_c),
                "translations": torch.zeros(B, 1, C_t),
                "sizes": torch.zeros(B, 1, C_s),
                "quats": torch.zeros(B, 1, C_q)
            }, 0.0

        # Compute the features using the transformer
        F = self._encode(boxes, root_box)
        # Sample the class label for the next bbbox
        
        cf = self.hidden2output.get_cf_feature(F, class_labels)
        # Sample the translations
        translations, t_params = self.hidden2output.sample_translations(cf)
        tf = self.hidden2output.get_tf_feature(cf, translations)

        p_bins_param = None
        if self.use_p_coarse:
            pose_bins, p_bins_param = self.hidden2output.sample_angle_bins(tf)
            tf = torch.cat([tf, pose_bins], dim=-1)

        # Sample the angles
        poses, p_params = self.hidden2output.sample_angles(tf)
        sf = self.hidden2output.get_sf_feature(tf, poses)
        # Sample the sizes
        sizes, s_params = self.hidden2output.sample_sizes(sf)

        # loss will be None if the model is over generating, will be torch.tensor(nan) if the output is the ending box
        loss = None
        if target_boxes['target_label'].shape[-1] != 0: # we only want to calculate the loss if we are not over generating
            loss = AutoregressiveBBoxOutput(s_params, t_params, p_params, class_labels, poses, p_bins_param, \
                self.use_6D, self.use_p_coarse, self.sampling).reconstruction_loss(target_boxes)

        # convert 6D back to quat
        if self.use_6D:
            poses = transforms.rotation_6d_to_matrix(poses)
            poses = transforms.matrix_to_quaternion(poses) 

        return {
            "class_labels": class_labels,
            "translations": translations,
            "sizes": sizes,
            "quats": poses, 
        }, loss

    @torch.no_grad()
    def add_object(self, root_box, boxes=None, device="cpu"):
        # boxes = dict(boxes.items())

        # # Make sure that the provided class_label will have the correct format
        # if isinstance(class_label, int):
        #     one_hot = torch.eye(self.n_classes)
        #     class_label = one_hot[class_label][None, None]
        # elif not torch.is_tensor(class_label):
        #     class_label = torch.from_numpy(class_label)

        # # Make sure that the class label the correct size,
        # # namely (batch_size, 1, n_classes)
        # assert class_label.shape == (1, 1, self.n_classes)

        # Create the initial input to the transformer, namely the start token
        start_box = self.start_symbol(device)
        for k in start_box.keys():
            boxes[k] = torch.cat([start_box[k], boxes[k]], dim=1)

        # Based on the query class label sample the location of the new object
        box = self.autoregressive_decode(
            boxes=boxes,
            root_box=root_box
        )

        for k in box.keys():
            boxes[k] = torch.cat([boxes[k], box[k]], dim=1)

        # Creat a box for the end token and update the boxes dictionary
        end_box = self.end_symbol(device)
        for k in end_box.keys():
            boxes[k] = torch.cat([boxes[k], end_box[k]], dim=1)

        return {
            "class_labels": boxes["class_labels"],
            "translations": boxes["translations"],
            "sizes": boxes["sizes"],
            "angles": boxes["angles"]
        }

    @torch.no_grad()
    def complete_scene(
        self,
        object_id,
        sample_params,
        max_boxes=100,
        device="cpu"
    ):

        # Create the initial input to the transformer, namely the start token
        start_box = self.start_symbol(device)
        # Add the start box token in the beginning
        for k in start_box.keys():
            sample_params[k] = torch.cat([start_box[k], sample_params[k].to(device)], dim=1)

        for i in range(max_boxes):
            
            current_target_boxes = dict()
            for k in sample_params.keys():
                if 'target' in k:
                    current_target_boxes[k] = sample_params[k][:, i:i+1]
            
            box, box_params = self.autoregressive_decode(sample_params=sample_params)
            # box, loss = self.autoregressive_decode_with_class_label(boxes, root_box=root_box, target_boxes=current_target_boxes)

              # loss will be None if the model is over generating, will be torch.tensor(nan) if the output is the ending box
            loss = None
            if current_target_boxes['target_label'].shape[-1] != 0: # we only want to calculate the loss if we are not over generating
                loss = AutoregressiveBBoxOutput(s_params=box_params['s_params'],
                t_params=box_params['t_params'], p_params=box_params['p_params'],
                class_labels=box_params['c_logits'], poses=box['input_boxes'][..., 6:],
                size_bins=box_params['s_bins_param'], tran_bins=box_params['t_bins_param'],
                pose_bins=box_params['p_bins_param'], use_6D=self.use_6D,
                use_t_coarse=self.use_t_coarse, use_p_coarse=self.use_p_coarse,
                use_s_coarse=self.use_s_coarse, use_dmll=self.sampling).reconstruction_loss(current_target_boxes)

            # Check if we have the end symbol
            if box["input_class"][0, 0, 0] == 1:
                break
            else:
                for k in box.keys():
                    sample_params[k] = torch.cat([sample_params[k], box[k]], dim=1)
                sample_params['lengths'] += 1
                if loss is not None:
                    StatsLogger.instance().print_progress(object_id, i, loss, cur=True)
            # if target_boxes["target_label"][:, i+1:i+2].shape[-1] == 0:
            #     break
        
        return sample_params

    def autoregressive_decode_with_class_label_and_translation(
        self,
        boxes,
        root_box,
        class_label,
        translation
    ):
        class_labels = boxes["class_labels"]
        B, _, C = class_labels.shape

        # Make sure that everything has the correct size
        assert len(class_label.shape) == 3
        assert class_label.shape[0] == B
        assert class_label.shape[-1] == C

        # Compute the features using the transformer
        F = self._encode(boxes, root_box)

        # Sample the angles
        angles = self.hidden2output.sample_angles(F, class_label, translation)
        # Sample the sizes
        sizes = self.hidden2output.sample_sizes(
            F, class_label, translation, angles
        )

        return {
            "class_labels": class_label,
            "translations": translation,
            "sizes": sizes,
            "angles": angles
        }

    @torch.no_grad()
    def add_object_with_class_and_translation(
        self,
        boxes,
        root_box,
        class_label,
        translation,
        device="cpu"
    ):
        boxes = dict(boxes.items())

        # Make sure that the provided class_label will have the correct format
        if isinstance(class_label, int):
            one_hot = torch.eye(self.n_classes)
            class_label = one_hot[class_label][None, None]
        elif not torch.is_tensor(class_label):
            class_label = torch.from_numpy(class_label)

        # Make sure that the class label the correct size,
        # namely (batch_size, 1, n_classes)
        assert class_label.shape == (1, 1, self.n_classes)


        # Create the initial input to the transformer, namely the start token
        start_box = self.start_symbol(device)
        for k in start_box.keys():
            boxes[k] = torch.cat([start_box[k], boxes[k]], dim=1)

        # Based on the query class label sample the location of the new object
        box = self.autoregressive_decode_with_class_label_and_translation(
            boxes=boxes,
            class_label=class_label,
            translation=translation,
            root_box=root_box
        )

        for k in box.keys():
            boxes[k] = torch.cat([boxes[k], box[k]], dim=1)

        # Creat a box for the end token and update the boxes dictionary
        end_box = self.end_symbol(device)
        for k in end_box.keys():
            boxes[k] = torch.cat([boxes[k], end_box[k]], dim=1)

        return {
            "class_labels": boxes["class_labels"],
            "translations": boxes["translations"],
            "sizes": boxes["sizes"],
            "angles": boxes["angles"]
        }

    @torch.no_grad()
    def distribution_classes(self, boxes, root_box, device="cpu"):
        # Shallow copy the input dictionary
        boxes = dict(boxes.items())
        # Create the initial input to the transformer, namely the start token
        start_box = self.start_symbol(device)
        # Add the start box token in the beginning
        for k in start_box.keys():
            boxes[k] = torch.cat([start_box[k], boxes[k]], dim=1)

        # Compute the features using the transformer
        F = self._encode(boxes, root_box)
        return self.hidden2output.pred_class_probs(F)

    @torch.no_grad()
    def distribution_translations(
        self,
        boxes,
        root_box, 
        class_label,
        device="cpu"
    ):
        # Shallow copy the input dictionary
        boxes = dict(boxes.items())

        # Make sure that the provided class_label will have the correct format
        if isinstance(class_label, int):
            one_hot = torch.eye(self.n_classes)
            class_label = one_hot[class_label][None, None]
        elif not torch.is_tensor(class_label):
            class_label = torch.from_numpy(class_label)

        # Make sure that the class label the correct size,
        # namely (batch_size, 1, n_classes)
        assert class_label.shape == (1, 1, self.n_classes)

        # Create the initial input to the transformer, namely the start token
        start_box = self.start_symbol(device)
        # Concatenate to the given input (that's why we shallow copy in the
        # beginning of this method
        for k in start_box.keys():
            boxes[k] = torch.cat([start_box[k], boxes[k]], dim=1)

        # Compute the features using the transformer
        F = self._encode(boxes, root_box)

        # Get the dmll params for the translations
        return self.hidden2output.pred_dmll_params_translation(
            F, class_label
        )

class DiscriminatorTransformer(BaseAutoregressiveTransformer):
    def __init__(self, input_dims, hidden2output, feature_extractor, config, n_classes):
        super().__init__(input_dims, hidden2output, feature_extractor, config, n_classes)
        self.pe_quad_a = FixedPositionalEncoding(proj_dims=64)
        self.pe_quad_b = FixedPositionalEncoding(proj_dims=64)
        self.pe_quad_c = FixedPositionalEncoding(proj_dims=64)
        self.pe_quad_d = FixedPositionalEncoding(proj_dims=64)
        
        self.register_parameter(
            "empty_token_embedding", nn.Parameter(torch.randn(1, 640))
        )
        hidden_dims = config.get("hidden_dims", 768)
        self.fc = nn.Linear(640, hidden_dims)
        self.fc_root = nn.Linear(512, 640)

    def get_pe_box(self, translations, sizes, quads):
        pos_f_x = self.pe_pos_x(translations[:, :, 0:1])
        pos_f_y = self.pe_pos_y(translations[:, :, 1:2])
        pos_f_z = self.pe_pos_z(translations[:, :, 2:3])
        pos_f = torch.cat([pos_f_x, pos_f_y, pos_f_z], dim=-1)

        size_f_x = self.pe_size_x(sizes[:, :, 0:1])
        size_f_y = self.pe_size_y(sizes[:, :, 1:2])
        size_f_z = self.pe_size_z(sizes[:, :, 2:3])
        size_f = torch.cat([size_f_x, size_f_y, size_f_z], dim=-1)

        quad_f_a = self.pe_quad_a(quads[:, :, 0:1])
        quad_f_b = self.pe_quad_b(quads[:, :, 1:2])
        quad_f_c = self.pe_quad_c(quads[:, :, 2:3])
        quad_f_d = self.pe_quad_d(quads[:, :, 3:4])
        quad_f = torch.cat([quad_f_a, quad_f_b, quad_f_c, quad_f_d], dim=-1)
        X = torch.cat([pos_f, size_f, quad_f], dim=-1)
        return X

    def forward(self, sample_params):
        # Unpack the sample_params
        # class_labels = sample_params["input_class"]
        boxes = sample_params['input_boxes']
        translations = boxes[:, :, :3]
        sizes = boxes[:, :, 3:6]
        quads = boxes[:, :, 6:]
        B = sample_params['input_boxes'].shape[0]

        # Apply the positional embeddings only on bboxes that are not the start
        # token
        # class_f = self.fc_class(class_labels)
        # Apply the positional embedding along each dimension of the position
        # property
        X = self.get_pe_box(translations, sizes, quads)
        root_X, X = X[:, 0], X[:, 1:]
        # X = torch.cat([class_f, X], dim=-1)
        start_symbol_f = self.start_symbol_features(B, root_X)
        # Concatenate with the mask embedding for the start token
        X = torch.cat([
            start_symbol_f, self.empty_token_embedding.expand(B, -1, -1), X
        ], dim=1)
        
        X = self.fc(X)

        # Compute the features using causal masking
        lengths = LengthMask(
            sample_params["lengths"]+2,
            max_len=X.shape[1]
        )
        F = self.transformer_encoder(X, length_mask=lengths)

        return torch.sigmoid(torch.squeeze(self.hidden2output(F[:, 1:2]))) # predict on cls

def custom_foward(self, x, memory, x_mask=None, x_length_mask=None,
    memory_mask=None, memory_length_mask=None):
    # Normalize the masks
    N = x.shape[0]
    L = x.shape[1]
    L_prime = memory.shape[1]
    x_mask = x_mask or FullMask(L, device=x.device)
    x_length_mask = x_length_mask  or \
        LengthMask(x.new_full((N,), L, dtype=torch.int64))
    memory_mask = memory_mask or FullMask(L, L_prime, device=x.device)
    memory_length_mask = memory_length_mask or \
        LengthMask(x.new_full((N,), L_prime, dtype=torch.int64))

    if self.use_self_attn:
        # First apply the self attention and add it to the input
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask,
            query_lengths=x_length_mask,
            key_lengths=x_length_mask
        ))
    else:
        self_attn = self.self_attention
        x = x + self.dropout(
            self_attn.out_projection(
                self_attn.value_projection(x)
            )
        )
    x = self.norm1(x)

    # Secondly apply the cross attention and add it to the previous output
    x = x + self.dropout(self.cross_attention(
        x, memory, memory,
        attn_mask=memory_mask,
        query_lengths=x_length_mask,
        key_lengths=memory_length_mask
    ))

    # Finally run the fully connected part of the layer
    y = x = self.norm2(x)
    y = self.dropout(self.activation(self.linear1(y)))
    y = self.dropout(self.linear2(y))

    return self.norm3(x+y)

class PointCloudDecoder(nn.Module):
    def __init__(self, input_dims, hidden2output, config):
        super().__init__()

        self.transformer_decoder = TransformerDecoderBuilder.from_kwargs(
            n_layers=config.get("n_layers", 6),
            n_heads=config.get("n_heads", 12),
            query_dimensions=config.get("query_dimensions", 64),
            value_dimensions=config.get("value_dimensions", 64),
            feed_forward_dimensions=config.get(
                "feed_forward_dimensions", 3072
            ),
            dropout=config.get("dropout", 0.5),
            self_attention_type="full",
            cross_attention_type="full",
            activation="gelu"
        ).get()

 
        for layer in self.transformer_decoder.layers:
            bound_method = custom_foward.__get__(layer, layer.__class__)
            setattr(layer, 'forward', bound_method)
            setattr(layer, 'use_self_attn', False)

        self.proj = nn.Linear(3, config.get("hidden_dims"))
        self.hidden2output = hidden2output


    def forward(self, points, memory, lengths):
        device = points.get_device()
        lengths = LengthMask(
            lengths,
            max_len=memory.shape[1],
            device=device
        )
        points = self.proj(points)
        F = self.transformer_decoder(points, memory, memory_length_mask=lengths)
        y_hat = torch.squeeze(self.hidden2output(F), dim=-1)

        return y_hat


def train_on_batch(model, optimizer, sample_params, config, schedule_sampling=False):
    # Make sure that everything has the correct size
    optimizer.zero_grad()    
    B = sample_params['input_boxes'].shape[0]
    device = sample_params['input_boxes'].get_device()
    # Create the initial input to the transformer, namely the start token
    if schedule_sampling:
        with torch.no_grad():
            start_box = model.start_symbol(device)
            # Add the start box token in the beginning
            for k in start_box.keys():
                sample_params[k] = torch.cat([start_box[k].expand(B, -1, -1), \
                    sample_params[k].to(device)], dim=1)
            ori_length = sample_params['lengths']

            # sampling from (0, 1]
            portion = (0 - 1) * torch.rand(ori_length.shape).to(device) + 1

            # sampling from (0.5, length + 0.5]
            schedule_sampled_idx = sample_params['lengths'] * portion + 0.5
            
            # sampling from (0, length] round changes 0.5 to 0
            schedule_sampled_idx = torch.round(schedule_sampled_idx).long()
            sample_params['lengths'] = torch.maximum(torch.zeros_like(schedule_sampled_idx), schedule_sampled_idx - 1)

            output, _ = model.autoregressive_decode(sample_params)

            # A special handling where the model predicts to end genertaion even though it shouldn't end.
            # We will use ground truth box and ground truth class in this case.
            condition = output['input_class'][:, 0, 0] == 1
            sample_params['input_boxes'][torch.arange(B), schedule_sampled_idx] = \
                torch.where(torch.unsqueeze(condition, 1), \
                sample_params['input_boxes'][torch.arange(B), schedule_sampled_idx], \
                output['input_boxes'][:, 0])

            # sample_params['input_class'][torch.arange(B), schedule_sampled_idx] = \
            #     torch.where(torch.unsqueeze(condition, 1), \
            #     sample_params['input_class'][torch.arange(B), schedule_sampled_idx], \
            #     output['input_class'][:, 0])

            sample_params['lengths'] = ori_length

            for k in start_box.keys():
                sample_params[k] = sample_params[k][:, 1:, :]

    X_pred = model(sample_params)
    # Compute the loss
    loss = X_pred.reconstruction_loss(sample_params, sample_params["lengths"])
    # Do the backpropagation
    loss.backward()
    # Do the update
    optimizer.step()

    return loss.item()


@torch.no_grad()
def validate_on_batch(model, sample_params, config, schedule_sampling=False):
    
    B = sample_params['input_boxes'].shape[0]
    device = sample_params['input_boxes'].get_device()

    if schedule_sampling:
        with torch.no_grad():
            # Create the initial input to the transformer, namely the start token
            start_box = model.start_symbol(device)
            # Add the start box token in the beginning
            for k in start_box.keys():
                sample_params[k] = torch.cat([start_box[k].expand(B, -1, -1), \
                    sample_params[k].to(device)], dim=1)
            ori_length = sample_params['lengths']

            # sampling from (0, 1]
            portion = (0 - 1) * torch.rand(ori_length.shape).to(device) + 1

            # sampling from (0.5, length + 0.5]
            schedule_sampled_idx = sample_params['lengths'] * portion + 0.5

            # sampling from (0, length] round changes 0.5 to 0
            schedule_sampled_idx = torch.round(schedule_sampled_idx).long() #sample_params['lengths']
            sample_params['lengths'] = torch.maximum(torch.zeros_like(schedule_sampled_idx), schedule_sampled_idx - 1)
            output, _ = model.autoregressive_decode(sample_params)

            # A special handling where the model predicts to end genertaion even though it shouldn't end.
            # We will use ground truth box in this case.
            condition = output['input_class'][:, 0, 0] == 1
            sample_params['input_boxes'][torch.arange(B), schedule_sampled_idx] = \
                torch.where(torch.unsqueeze(condition, 1), \
                sample_params['input_boxes'][torch.arange(B), schedule_sampled_idx], \
                output['input_boxes'][:, 0])

            # sample_params['input_class'][torch.arange(B), schedule_sampled_idx] = \
            #     torch.where(torch.unsqueeze(condition, 1), \
            #     sample_params['input_class'][torch.arange(B), schedule_sampled_idx], \
            #     output['input_class'][:, 0])

            sample_params['lengths'] = ori_length

            for k in start_box.keys():
                sample_params[k] = sample_params[k][:, 1:, :]

    X_pred = model(sample_params)
    # Compute the loss
    loss = X_pred.reconstruction_loss(sample_params, sample_params["lengths"])
    # Compute metrics

    # Inference

    return loss.item(), None


@torch.no_grad()
def batch_decode(sample_params, gen_model, device):
    output, _ = gen_model.autoregressive_decode(sample_params)

    sample_params['is_end'] = \
        torch.logical_or(output['input_class'][:, 0, 0] == 1, sample_params['is_end'])
            
    # box padding will be all 0s, class padding will be all 0s     
    B, _, C = sample_params['input_boxes'].shape       
    box_padding = torch.zeros((B, 1, C), device=device)
    sample_params['input_boxes'] = torch.cat([sample_params['input_boxes'], \
        box_padding], axis=1)

    B, _, C = sample_params['input_class'].shape
    class_padding = torch.zeros((B, 1, C), device=device)
    class_padding[:, :, 0] = 1
    sample_params['input_class'] = torch.cat([sample_params['input_class'], \
        class_padding], axis=1)

    batch_idx = torch.arange(B)

    sample_params['input_boxes'][batch_idx, sample_params['lengths'] + 1] = \
        torch.where(sample_params['is_end'].unsqueeze(1), 
        box_padding[:, 0], output['input_boxes'][:, 0])

    sample_params['input_class'][batch_idx, sample_params['lengths'] + 1] = \
        torch.where(sample_params['is_end'].unsqueeze(1), 
        class_padding[:, 0], output['input_class'][:, 0]) 

    # increment length by one if generation has stopped
    sample_params['lengths'] += torch.logical_not(sample_params['is_end'])


def train_on_batch_decoder(models, optimizers, \
        sample_params, config):
    
    encoder = models['encoder']
    decoder = models['decoder']
    generator = models['generator']

    for opt in list(optimizers.values()):
        opt.zero_grad()

    B = sample_params['input_boxes'].shape[0]
    device = sample_params['input_boxes'].get_device()
    
    sample_params['input_boxes'] = sample_params['part_boxes_ori']
    sample_params['root_box'] = sample_params['root_box_ori']    
    
    # Create the initial input to the transformer, namely the start token
    start_box = encoder.start_symbol(device)
    # Add the start box token in the beginning
    for k in start_box.keys():
        sample_params[k] = torch.cat([start_box[k].expand(B, -1, -1), \
            sample_params[k].to(device)], dim=1)

    # only generate if we are not using ground truth boxes
    if config['data']['sample_strategy'] != 'gt_bbox':    
        generator.eval()
        max_box = 25
        sample_params['is_end'] = torch.zeros(B).to(device)
        for _ in range(max_box):
            batch_decode(sample_params, generator, device)
            if torch.sum(sample_params['is_end']) == B:
                break

        max_seq_length = torch.max(sample_params['lengths'])
        # This exclude the end class token and the predicted box since it's useless to us.
        # The start token is still at the front
        sample_params['input_boxes'] = sample_params['input_boxes'][:, :max_seq_length + 1]
        sample_params['input_class'] = sample_params['input_class'][:, :max_seq_length + 1]
        generator.train()

    memory = encoder._encode(sample_params)
    # Additional 2 for lengths due to root and cls tokens
    y_hat = decoder(sample_params['point_clouds'], memory, sample_params['lengths'] + 1)
    y_hat = torch.sigmoid(y_hat)

    loss = torch.nn.functional.binary_cross_entropy(y_hat, \
                sample_params['point_label'].float(), \
                weight=sample_params['point_weights'])
    StatsLogger.instance()["loss.point"].value = loss
    
    y_hat = (y_hat >= 0.5).view(-1).int().cpu().numpy()
    y = sample_params['point_label'].view(-1).cpu().numpy()
    acc = accuracy_score(y, y_hat)
    recall = recall_score(y, y_hat)
    precision = precision_score(y, y_hat, zero_division=0)
    StatsLogger.instance()["accuracy.point"].value = acc
    StatsLogger.instance()["recall.point"].value = recall
    StatsLogger.instance()["precision.point"].value = precision

    loss.backward()

    for opt in list(optimizers.values()):
        opt.step()
    return loss


@torch.no_grad()
def validate_on_batch_decoder(models, sample_params, config):
    encoder = models['encoder']
    decoder = models['decoder']
    generator = models['generator']

    B = sample_params['input_boxes'].shape[0]
    device = sample_params['input_boxes'].get_device()

    sample_params['input_boxes'] = sample_params['part_boxes_ori']
    sample_params['root_box'] = sample_params['root_box_ori']    

    # Create the initial input to the transformer, namely the start token
    start_box = encoder.start_symbol(device)
    # Add the start box token in the beginning
    for k in start_box.keys():
        sample_params[k] = torch.cat([start_box[k].expand(B, -1, -1), \
            sample_params[k].to(device)], dim=1)
        
        # only generate if we are not using ground truth boxes
    if config['data']['sample_strategy'] != 'gt_bbox':    
        max_box = 25
        sample_params['is_end'] = torch.zeros(B).to(device)
        for _ in range(max_box):
            batch_decode(sample_params, generator, device)
            if torch.sum(sample_params['is_end']) == B:
                break
        
        max_seq_length = torch.max(sample_params['lengths'])
        sample_params['input_boxes'] = sample_params['input_boxes'][:, :max_seq_length + 1]
        sample_params['input_class'] = sample_params['input_class'][:, :max_seq_length + 1]

    # Additional 2 for lengths due to root and cls tokens
    memory = encoder._encode(sample_params)
    y_hat = decoder(sample_params['point_clouds'], memory, sample_params['lengths'] + 1)
    y_hat = torch.sigmoid(y_hat)

    loss = torch.nn.functional.binary_cross_entropy(y_hat, \
                sample_params['point_label'].float(), \
                weight=sample_params['point_weights'])
    StatsLogger.instance()["loss.point"].value = loss

    y_hat = (y_hat >= 0.5).view(-1).int().cpu().numpy()
    y = sample_params['point_label'].view(-1).cpu().numpy()
    acc = accuracy_score(y, y_hat)
    recall = recall_score(y, y_hat)
    precision = precision_score(y, y_hat, zero_division=0)
    StatsLogger.instance()["accuracy.point"].value = acc
    StatsLogger.instance()["recall.point"].value = recall
    StatsLogger.instance()["precision.point"].value = precision
    
    return loss.item(), y_hat

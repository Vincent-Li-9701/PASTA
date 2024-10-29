 

import torch
try:
    from radam import RAdam
except ImportError:
    pass

from .autoregressive_transformer import AutoregressiveTransformer, \
    AutoregressiveTransformerPE, ObjectGenerationTransformer, DiscriminatorTransformer, PointCloudDecoder, \
    train_on_batch as train_on_batch_simple_autoregressive, \
    validate_on_batch as validate_on_batch_simple_autoregressive, \
    train_on_batch_decoder as train_on_batch_simple_autoregressive_decode, \
    validate_on_batch_decoder as validate_on_batch_simple_autoregressive_decode

from .hidden_to_output import AutoregressiveDMLL, get_bbox_output
from .feature_extractors import get_feature_extractor
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR, CosineAnnealingWarmRestarts


def hidden2output_layer(config_n, n_classes):
    # config_n = config["network"]
    hidden2output_layer = config_n.get("hidden2output_layer")

    if hidden2output_layer == "autoregressive_mlc":
        return AutoregressiveDMLL(
            config_n.get("hidden_dims", 768),
            n_classes,
            config_n.get("n_pose_classes", 10),
            config_n.get("n_mixtures", 4),
            get_bbox_output(config_n.get("bbox_output", "autoregressive_mlc")),
            config_n.get("with_extra_fc", False),
            config_n.get("use_6D", False),
            config_n.get("use_t_coarse", False),
            config_n.get("use_p_coarse", False),
            config_n.get("use_s_coarse", False),
            config_n.get("single_head", False),
            config_n.get("single_head_trans", False),
            config_n.get("sampling", False),
            config_n.get("dropout", 0.5)
        )
    elif hidden2output_layer == "fc_head":
        return AutoregressiveDMLL._mlp(config_n.get("hidden_dims", 768), n_classes, config_n.get("dropout", 0.1))
    else:
        raise NotImplementedError()


def optimizer_factory(config, parameters):
    """Based on the provided config create the suitable optimizer."""
    optimizer = config.get("optimizer", "Adam")
    lr = config.get("lr", 1e-3)
    momentum = config.get("momentum", 0.9)
    weight_decay = config.get("weight_decay", 0.0)

    if optimizer == "SGD":
        return torch.optim.SGD(
            parameters, lr=lr, momentum=momentum, weight_decay=weight_decay
        )
    elif optimizer == "Adam":
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer == "RAdam":
        return RAdam(parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise NotImplementedError()


def scheduler_factory(config, optimizer):
    scheduler = config.get("scheduler", None)
    if scheduler == "exponential":
        return ExponentialLR(optimizer, gamma=0.9)
    elif scheduler == "cosineAnnealing":
        return CosineAnnealingLR(optimizer, T_max=100)
    elif scheduler == "cosineAnnealingWarmRestarts":
        return CosineAnnealingWarmRestarts(optimizer, T_0=1000)
    elif scheduler == "sineAnnealingWarmRestarts":
        return SineAnnealingWarmRestarts(optimizer, T_0=1000)

def build_network(
    input_dims,
    n_classes,
    config,
    weight_file=None,
    device="cpu"):
    network_type = config['type']

    if network_type == "object_generation_transformer":
        train_on_batch = train_on_batch_simple_autoregressive
        validate_on_batch = validate_on_batch_simple_autoregressive
        network = ObjectGenerationTransformer(
            input_dims,
            hidden2output_layer(config, n_classes),
            None,
            config,
            n_classes
        )
    elif network_type == "discriminator_transformer":
        train_on_batch = train_on_batch_simple_autoregressive
        validate_on_batch = validate_on_batch_simple_autoregressive
        network = DiscriminatorTransformer(
            input_dims,
            hidden2output_layer(config, n_classes),
            None,
            config,
            n_classes
        )
    elif network_type == 'pointcloud_decoder_transformer':
        train_on_batch = train_on_batch_simple_autoregressive_decode
        validate_on_batch = validate_on_batch_simple_autoregressive_decode
        network = PointCloudDecoder(
            input_dims,
            hidden2output_layer(config, n_classes),
            config
        )       
    else:
        raise NotImplementedError()

    # Check whether there is a weight file provided to continue training from
    if weight_file is not None:
        print("Loading weight file from {}".format(weight_file))
        network.load_state_dict(
            torch.load(weight_file, map_location=device)
        )
    network.to(device)
    return network, train_on_batch, validate_on_batch

class SineAnnealingWarmRestarts(CosineAnnealingWarmRestarts):
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1, verbose=False):
        super().__init__(optimizer, T_0, T_mult, eta_min, last_epoch, verbose)
    def get_lr(self):
        return [self.eta_min + (base_lr - self.eta_min) * 
            (1 + math.cos(math.pi * self.T_cur / self.T_i + math.pi)) 
            / 2 for base_lr in self.base_lrs] 
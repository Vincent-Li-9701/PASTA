"""
Usage:
python main.py --model PointMLP --msg demo
"""

from pkg_resources import fixup_namespace_packages
from sklearn import datasets

import torch
import models as models
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from ignite.engine import Engine, create_supervised_evaluator
from ignite.metrics import FID
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union


def load_partial_model(model, pretrain_states):
    own_state = model.state_dict()
    for name, param in pretrain_states.items():
        if name not in own_state:
            continue
        if isinstance(param, torch.nn.parameter.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
            own_state[name].copy_(param)


class DummyDataset(Dataset):
    def __init__(self):
        pass
    def __len__(self):
        return 20

    def __getitem__(self, idx):
        real = torch.rand(3, 1024)
        fake = torch.rand(3, 1024)
        return (real, fake)

def main():

    # load data
    dataset = DummyDataset()
    test_loader = DataLoader(dataset, batch_size=10, shuffle=True,
                            collate_fn=lambda x: tuple(x_.cuda() for x_ in default_collate(x)))

    PATH = './best_checkpoint.pth'

    # load checkpoints
    checkpoint = torch.load(PATH)
    encoder = models.pointMLP()
    load_partial_model(encoder, checkpoint)

    val_metrics = {
        "fid": FID(num_features=1024, feature_extractor=encoder, device='cuda')
    }

    # TODO: Create custom engine and evaluation function to avoid unnecessary forward pass
    evaluator = create_supervised_evaluator(model=encoder, metrics=val_metrics, output_transform=lambda x, y, y_pred: (x, y))

    evaluator.run(test_loader)
    metrics = evaluator.state.metrics

    print(metrics["fid"])

if __name__ == '__main__':
    main()

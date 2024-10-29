"""Script used to evaluate the scene classification accuracy between real and
synthesized scenes.
"""
import argparse
import os
import sys

from PIL import Image

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torchvision import models

from scene_synthesis.networks import ObjectGenerationTransformer
from training_utils import id_generator, save_experiment_params, load_config
from scene_synthesis.datasets import get_raw_dataset, filter_function
from scene_synthesis.networks import build_network

torch.set_num_threads(5)
torch.autograd.set_detect_anomaly(True)

class PartNetSyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        file_names = sorted(os.listdir(data_path), key=lambda x: int(os.path.splitext(x)[0]))
        self.file_paths = [os.path.join(data_path, path) for path in file_names]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        return torch.load(self.file_paths[idx])
        
class SyntheticVRealDataset(torch.utils.data.Dataset):
    def __init__(self, real, synthetic):
        self.N = min(len(real), len(synthetic))
        self.real = real
        self.synthetic = synthetic

    def __len__(self):
        return 2*self.N

    def __getitem__(self, idx):
        if idx < self.N:
            tree, id = self.real[idx]
            part_boxes, _, _, _, _ = tree.graph(leafs_only=True)
            root_box = tree.root.get_box_quat().squeeze()
            part_boxes = torch.stack([root_box] + part_boxes)
            label = 1
        else:
            part_boxes = self.synthetic[idx - self.N]
            label = 0

        return part_boxes, torch.tensor([label], dtype=torch.float)

    def collate_fn(self, samples):

        # getting boxes and labels separately
        x = [sample[0] for sample in samples]
        y = [sample[1] for sample in samples]
        length = torch.tensor([len(sample[0]) for sample in samples])

        return {"input_boxes": pad_sequence(x, True).float(), "lengths": length}, torch.squeeze(torch.stack(y))


class AverageMeter:
    def __init__(self):
        self._value = 0
        self._cnt = 0

    def __iadd__(self, x):
        if torch.is_tensor(x):
            self._value += x.sum().item()
            self._cnt += x.numel()
        else:
            self._value += x
            self._cnt += 1
        return self

    @property
    def value(self):
        return self._value / self._cnt


def main(argv):
    parser = argparse.ArgumentParser(
        description=("Train a classifier to discriminate between real "
                     "and synthetic rooms")
    )
    parser.add_argument(
        "config_file",
        help="Path to the file that contains the experiment configuration"
    )
    parser.add_argument(
        "--weight_file",
        default=None,
        help=("The path to a previously trained model to continue"
              " the training from")
    )    
    parser.add_argument(
        "--path_to_syn_train",
        help="Path to the folder containing the synthesized training data"
    )
    parser.add_argument(
        "--path_to_syn_test",
        help="Path to the folder containing the synthesized testing data"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Set the batch size for training and evaluating (default: 256)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Set the PyTorch data loader workers (default: 0)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Train for that many epochs (default: 10)"
    )
    parser.add_argument(
        "--output_directory",
        default="/tmp/",
        help="Path to the output directory"
    )
    args = parser.parse_args(argv)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print("Running code on", device)

    # Check if output directory exists and if it doesn't create it
    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    config = load_config(args.config_file)
    # use dummy bounds since we don't need to scale anything
    train_real = get_raw_dataset(
        config["data"],
        filter_fn=filter_function(
            config["data"],
            split=config["training"].get("splits", ["train", "val"])
        ),
        split=config["training"].get("splits", ["train", "val"])
    )

    test_real = get_raw_dataset(
        config["data"],
        filter_fn=filter_function(
            config["data"],
            split=config["training"].get("splits", ["train", "val"])
        ),
        split=config["testing"].get("splits", ["test"])
    )

    # Create the synthetic datasets
    train_synthetic = PartNetSyntheticDataset(args.path_to_syn_train)
    test_synthetic = PartNetSyntheticDataset(args.path_to_syn_test)

    # Join them in useable datasets
    train_dataset = SyntheticVRealDataset(train_real, train_synthetic)
    test_dataset = SyntheticVRealDataset(test_real, test_synthetic)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=train_dataset.collate_fn
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=test_dataset.collate_fn
    )


    # Train the model
    scores = []
    for _ in range(10):
        # Create the model
        model, train_on_batch, validate_on_batch = build_network(
            # train_dataset.feature_size, train_dataset.n_classes,
            256, 1,
            config['network'], args.weight_file, device=device
        )
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        for e in range(args.epochs):
            loss_meter = AverageMeter()
            acc_meter = AverageMeter()
            for i, (x, y) in enumerate(train_dataloader):
                model.train()
                for k, v in x.items():
                    x[k] = v.to(device)
                y = y.to(device)
                optimizer.zero_grad()
                y_hat = model(x)
                loss = torch.nn.functional.binary_cross_entropy(y_hat, y)
                acc = (torch.abs(y-y_hat) < 0.5).float().mean()
                loss.backward()
                optimizer.step()

                loss_meter += loss
                acc_meter += acc

                msg = "{: 3d} loss: {:.4f} - acc: {:.4f}".format(
                    i, loss_meter.value, acc_meter.value
                )
                print(msg + "\b"*len(msg), end="", flush=True)
            print()

            if (e + 1) % 5 == 0:
                with torch.no_grad():
                    model.eval()
                    loss_meter = AverageMeter()
                    acc_meter = AverageMeter()
                    for i, (x, y) in enumerate(test_dataloader):
                        for k, v in x.items():
                            x[k] = v.to(device)
                        y = y.to(device)
                        y_hat = model(x)
                        loss = torch.nn.functional.binary_cross_entropy(
                            y_hat, y
                        )
                        acc = (torch.abs(y-y_hat) < 0.5).float().mean()

                        loss_meter += loss
                        acc_meter += acc

                        msg_pre = "{: 3d} val_loss: {:.4f} - val_acc: {:.4f}"

                        msg = msg_pre.format(
                            i, loss_meter.value, acc_meter.value
                        )
                        print(msg + "\b"*len(msg), end="", flush=True)
                    print()
        scores.append(acc_meter.value)
    print(sum(scores) / len(scores))
    print(np.std(scores))


if __name__ == "__main__":
    main(None)

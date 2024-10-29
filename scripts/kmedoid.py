import argparse
import numpy as np
import torch

from training_utils import id_generator, save_experiment_params, load_config
from scene_synthesis.datasets import get_raw_dataset, filter_function
from pytorch3d import transforms

from sklearn_extra.cluster import KMedoids

def quaternion_distance(X, Y=None):
    return 1 - np.dot(X, Y) ** 2

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
        "n_centroids",
        default=10,
        type=int,
        help="Number of centroids we want to produce"
    )
    
    args = parser.parse_args(argv)
    config = load_config(args.config_file)

    print(f"The number of centroids we will produce is {args.n_centroids}")

    # put in dummy values for bounds
    train_dataset = get_raw_dataset(
        config["data"],
        filter_fn=filter_function(
            config["data"],
            split=config["training"].get("splits", ["train", "val"])
        ),
        split=config["training"].get("splits", ["train", "val"]),
    )

    X_train = torch.stack([part_box for tree, _ in train_dataset for part_box in tree.graph(leafs_only=True)[0]])

    # only extract the orientaion
    X_train_translation = X_train[:, :3]
    X_train_size = X_train[:, 3:6]
    X_train_pose = X_train[:, 6:]

    for X, key in zip([X_train_translation, X_train_size, X_train_pose], ['tran', 'size', 'pose']):
        
        save_path = f"./scripts/train_{key}_centroids"
        if "all" in args.config_file:
            save_path += "_all"

        metric = 'euclidean'
        if key == 'pose' and config["network"]["use_6D"]:
            X = transforms.quaternion_to_matrix(X)
            X = transforms.matrix_to_rotation_6d(X)
            metric = quaternion_distance
            save_path += "_6D"

        save_path += f"_{args.n_centroids}.npy"

        kmedoids = KMedoids(n_clusters=args.n_centroids, metric=metric).fit(X.numpy())
        with open(save_path, 'wb') as f:
            np.save(f, kmedoids.cluster_centers_)

if __name__ == "__main__":
    main(None)
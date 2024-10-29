#!/usr/bin/env python
"""Script used to run the evaluation on generated objects of a previously
trained network."""
import argparse
import math
import os
import sys
from typing import Optional
from tempfile import gettempdir

import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

import numpy as np
import torch
import trimesh
from tqdm import tqdm

from pytorch3d.loss import chamfer_distance as pytorch_chamfer_distance

from datasets import build_dataset
from mesh import read_mesh_file

from occupancy_voxelizer import OccupancyGrid
from simple_3dviz import Mesh


def load_config(config_file):
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=Loader)
    return config


def add_dataset_parameters(parser):
    parser.add_argument(
        "--model_tags",
        type=lambda x: x.split(","),
        default=[],
        help="Tags of the models to be used"
    )
    parser.add_argument(
        "--category_tags",
        type=lambda x: x.split(","),
        default=[],
        help="Category tags of the models to be used"
    )
    parser.add_argument(
        "--random_subset",
        type=float,
        default=1.0,
        help="Percentage of dataset to be used"
    )
    parser.add_argument(
        "--val_random_subset",
        type=float,
        default=1.0,
        help="Percentage of dataset to be used for validation"
    )


class PointsFromMeshesDataset:
    def __init__(
        self,
        path_to_dataset,
        n_points_on_mesh=100000,
        normalize_mesh=False,
    ):
        self._n_points_on_mesh = n_points_on_mesh

        self._path_to_dataset = path_to_dataset
        self.path_to_meshes = sorted([
            os.path.join(self._path_to_dataset, oi)
            for oi in os.listdir(self._path_to_dataset)
            if os.path.join(self._path_to_dataset, oi).endswith(".obj")
        ])
        self.normalize_mesh = normalize_mesh
        self.occupancy_grid = OccupancyGrid('64, 64, 64')

    def __len__(self):
        return len(self.path_to_meshes)

    def __getitem__(self, idx):
        path_to_mesh = self.path_to_meshes[idx]

        mesh = read_mesh_file(path_to_mesh, self.normalize_mesh)
        voxel = self.occupancy_grid.voxelize(mesh)[0]
        m = Mesh.from_voxel_grid(voxel)
        v, f = m.to_points_and_faces()
        tr_mesh = Trimesh(v, f, process=False)
        mesh = from_trimesh(tr_mesh, self.normalize_mesh)
        points = mesh.sample_faces(self._n_points_on_mesh)[:, :3]

        # Add a batch dimension
        return torch.tensor(points[None]).float()


def yield_batches(X: torch.Tensor, n: int) -> torch.Tensor:
    """Yield successive 'n'-sized chunks from iterable X.
    """
    for ii in range(0, len(X), n):
        yield X[ii:ii + n]


def bidirectional_chamfer_distance(
    X: torch.Tensor,
    Y: torch.Tensor,
    with_squared_distances: Optional[bool] = True
) -> torch.Tensor:
    """ Chamfer distance betwee two pointclouds

    Arguments:
    ----------
        X: torch.Tensor of shape (B, P1, D) representing
           a batch of point clouds with at most P1 points in each batch element,
           batch size B and feature dimension D.
        Y: torch.Tensor of shape (B, P2, D) representing
           a batch of point clouds with at most P1 points in each batch element,
           batch size B and feature dimension D.
        with_squared_distances: Boolean indicating whether the Euclidean
                                distances between the pointclouds will be squared.
                                By default we consider squared distances in
                                order to be compatible with pytorch3d's API.
    """
    # Make sure that everything has the correct size
    B, P1, D = X.shape
    _, P2, _ = Y.shape
    assert Y.shape[0] == B
    assert Y.shape[2] == D

    # Compute the Euclidean distances between X and Y
    dists = ((X[:, :, None, :] - Y[:, None, :, :])**2).sum(-1)
    if not with_squared_distances:
        dists = torch.sqrt(dists)
    # Ensure that everything has the correct size
    assert dists.shape == (B, P1, P2)

    # For every point in X find its closest point in Y
    completeness, _ = torch.min(dists, dim=-1)
    assert completeness.shape == (B, P1)
    # For every point in Y find its closest point in X
    accuracy, _ = torch.min(dists, dim=1)
    assert accuracy.shape == (B, P2)

    chamfer_dists = completeness.mean(-1) + accuracy.mean(-1)
    return chamfer_dists


def minimum_matching_distance(
    X: torch.Tensor,
    Y: torch.Tensor,
    batch_size: Optional[int] = 250,
    distance: Optional[str] = "chamfer_distance",
    with_squared_distances: Optional[bool] = True
):
    """Compute the Minimum Matching Distance (MMD) between two sets of
    pointclouds. In particular, we match every pointcloud in real_pts to the
    one in fake_pts using a distance such as Chamfer distance, EMD or Light
    Field distance.

    Code adapted from
    https://github.com/Steve-Tod/DeformSyncNet/blob/c4e6628ae4fd80e6e6aa702f4cd5885368047b4f/code/metrics/mmd_cov.py#L14

    Arguments:
    ----------
        X: torch.Tensor of shape (N, P1, D) representing
           N point clouds with P1 points and feature dimension D. X corresponds
           to the pointclouds from the real shapes.
        Y: torch.Tensor of shape (M, P2, D) representing
           M point clouds with P2 points and feature dimension D. Y corresponds
           to the pointclouds from the generated shapes.
        batch_size: Integet indicating the batch size over which we compute the
                    distances between pointclouds   
        distance: String indicating the distance function to be used
        with_squared_distances: Boolean indicating whether the Euclidean
                                distances between the pointclouds will be squared.
                                By default we consider squared distances in
                                order to be compatible with pytorch3d's API.
    """
    # Denote some local variables
    n_ref, n_pc_points, pc_dim = X.shape
    _, n_pc_points_s, pc_dim_s = Y.shape

    if n_pc_points != n_pc_points_s or pc_dim != pc_dim_s:
        raise ValueError('Incompatible size of point-clouds.')
    # For each sample who is the (minimal-distance) gt pc
    matched_ref_items = list()
    # For each sample what is the (minimal-distance) from the match gt pc
    matched_dists = list()

    all_refs_iter = iter(range(n_ref))    
    for i in tqdm(all_refs_iter):
        min_in_all_batches = list()
        loc_in_all_batches = list()
        for sample_chunk in yield_batches(Y, batch_size):            
            n_samples = len(sample_chunk)
            ref_i = X[i].repeat(n_samples, 1, 1)
            if distance == "chamfer_distance":
                # all_dist_in_batch = bidirectional_chamfer_distance(
                #     ref_i, sample_chunk
                # )
                from pytorch3d.loss import chamfer_distance
                all_dist_in_batch = chamfer_distance(
                    ref_i, sample_chunk, batch_reduction=None
                )[0]
            elif distance == "emd":
                raise NotImplementedError()

            location_of_min = all_dist_in_batch.argmin()
            # Best distance, of in-batch samples matched to single ref pc.
            min_in_batch = all_dist_in_batch[location_of_min]
            min_in_all_batches.append(min_in_batch)
            loc_in_all_batches.append(location_of_min)

        min_in_all_batches = torch.stack(min_in_all_batches)       
        min_batch_for_ref_i = torch.argmin(min_in_all_batches)
        min_dist_for_ref_i = min_in_all_batches[min_batch_for_ref_i]

        min_loc_inside_min_batch = torch.stack(loc_in_all_batches)[min_batch_for_ref_i]   
        matched_item_for_ref_i = min_batch_for_ref_i * batch_size + min_loc_inside_min_batch

        matched_dists.append(min_dist_for_ref_i)
        matched_ref_items.append(matched_item_for_ref_i)

    matched_dists = torch.stack(matched_dists)
    matched_ref_items = torch.stack(matched_ref_items)
    mmd = torch.mean(matched_dists).item()
    
    return mmd, matched_dists.cpu().numpy(), matched_ref_items.cpu().numpy()

def coverage_distance(
    X: torch.Tensor,
    Y: torch.Tensor,
    batch_size: Optional[int] = 250,
    distance: Optional[str] = "chamfer_distance",
    with_squared_distances: Optional[bool] = True
):
    """Compute the Coverage distance (COV) between two sets of pointclouds. The
    coverage is measured as the percentage of shapes in X that are covered by a
    shape in Y. To measure this, we find for each shape in Y its closest shape
    in X. A shape in X is considered covered if it is assigned by at least one
    shape in Y.

    Arguments:
    ----------
        X: torch.Tensor of shape (N, P1, D) representing
           N point clouds with P1 points and feature dimension D. X corresponds
           to the pointclouds from the real shapes.
        Y: torch.Tensor of shape (M, P2, D) representing
           M point clouds with P2 points and feature dimension D. Y corresponds
           to the pointclouds from the generated shapes.
        batch_size: Integet indicating the batch size over which we compute the
                    distances between pointclouds   
        distance: String indicating the distance function to be used
        with_squared_distances: Boolean indicating whether the Euclidean
                                distances between the pointclouds will be squared.
                                By default we consider squared distances in
                                order to be compatible with pytorch3d's API.
    """
    matched_elements = minimum_matching_distance(Y, X, batch_size)[-1]
    return len(np.unique(matched_elements)) / len(X), matched_elements


def chamfer_distance(ref_pcs, sample_pcs, batch_size):
    n_sample = 2048
    normalized_scale = 1.0

    all_cd = []
    for i_ref_p in tqdm(range(len(ref_pcs))):
        ref_p = ref_pcs[i_ref_p]
        cd_lst = []
        for sample_b_start in range(0, len(sample_pcs), batch_size):
            sample_b_end = min(len(sample_pcs), sample_b_start + batch_size)
            sample_batch = sample_pcs[sample_b_start:sample_b_end]

            batch_size_sample = sample_batch.size(0)

            chamfer = pytorch_chamfer_distance(
                ref_p.unsqueeze(dim=0).expand(batch_size_sample, -1, -1),
                sample_batch, batch_reduction=None
            )[0]
            # chamfer = kal.metrics.pointcloud.chamfer_distance(
            #     ref_p.unsqueeze(dim=0).expand(batch_size_sample, -1, -1),
            #     sample_batch)
            cd_lst.append(chamfer)
        cd_lst = torch.cat(cd_lst, dim=0)
        all_cd.append(cd_lst.unsqueeze(dim=0))
    all_cd = torch.cat(all_cd, dim=0)
    return all_cd


def compute_all_metrics(sample_pcs, ref_pcs, batch_size, save_name=None):
    M_rs_cd = chamfer_distance(ref_pcs, sample_pcs, batch_size)
    return M_rs_cd
            

def main(argv):
    parser = argparse.ArgumentParser(
        description="Evaluate the generated objects"
    )
    parser.add_argument(
        "path_to_generated_meshes",
        help="Path containing the generated meshes"
    )
    parser.add_argument(
        "config_file",
        help="Path to the file that contains the experiment configuration"
    )
    parser.add_argument(
        "--output_directory",
        default=os.path.join(gettempdir(), "evaluations"),
        help="Save the output files in that directory"
    )
    parser.add_argument(
        "--path_to_target_meshes",
        help="Path containing the target meshes"
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
        help="Run the evaluation on the given split"
    )
    parser.add_argument(
        "--n_points_on_mesh",
        default=2048,
        type=int,
        help="Number of points used to approximate a mesh"
    )
    parser.add_argument(
        "--n_fake_samples",
        default=-1,
        type=int,
        help="Number of fake pointclouds to be used in the evaluation"
    )
    parser.add_argument(
        "--n_real_samples",
        default=-1,
        type=int,
        help="Number of real pointclouds to be used in the evaluation"
    )
    parser.add_argument(
        "--real_fake_pts",
        default=None,
        help=(
            "Path to file containing the real and fake points. "
            "This allows faster computation of metrics."
        )
    )

    add_dataset_parameters(parser)
    args = parser.parse_args(argv)

    # Check if output directory exists and if it doesn't create it
    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    device = torch.device("cuda:0")
    print("Running code on", device)

    # Parse the config file
    config = load_config(args.config_file)

    # Update various fields in config to avoid mistakes in case they are not
    # properly set
    config["data"]["dataset_factory"] = "PointsAutoEncodingDataset"
    config["data"]["n_points_on_mesh"] = args.n_points_on_mesh

    fake_dataset = PointsFromMeshesDataset(
        args.path_to_generated_meshes, args.n_points_on_mesh,
        normalize_mesh=config["data"]["normalize"],
    )
    N = len(fake_dataset) if args.n_fake_samples == -1 else args.n_fake_samples
    print(f"Gathering {N} fake pointclouds...")
    # Gather all the points from the fake pointclouds and move everything to
    # the correct device
    fake_pts = torch.tensor([])
    for i, fi in enumerate(fake_dataset):
        fake_pts = torch.cat([fake_pts, fi], dim=0)
        if i == N:
            break
    fake_pts = fake_pts.to(device)

    # Instantiate a dataloader to generate the samples for evaluation
    real_dataset = build_dataset(
        config,
        args.model_tags,
        args.category_tags,
        ["test"]
        # [args.split],
    )
    N = len(real_dataset) if args.n_real_samples == -1 else args.n_real_samples
    real_pts_subset = np.random.choice( len(real_dataset), N)
    print(f"Gathering {len(real_pts_subset)} real pointclouds...")
    # Gather all the points from the real pointclouds and move everything to
    # the correct device
    real_pts = torch.tensor([])
    for i in real_pts_subset:
        di = real_dataset[i]
        points = di["points_on_mesh"][:, :3]
        # points = points / (points.max(axis=0)[0] - points.min(axis=0)[0])
        points = torch.tensor(points[None]).float()
        real_pts = torch.cat([real_pts, points], dim=0)
    real_pts = real_pts.to(device)

    torch.save(
        [real_pts, fake_pts], os.path.join(args.output_directory, f"points.pt")
    )
    # results = compute_all_metrics(fake_pts, real_pts, batch_size=200)
    # results = results.cpu().numpy()
    # mmd = results.min(axis=1).mean()
    # min_ref = results.argmin(axis=0)
    # unique_idx = np.unique(min_ref)
    # cov = float(len(unique_idx)) / results.shape[0]
    # if mmd < 1:
    #     # Chamfer distance
    #     mmd = mmd * 1000  # for showing results
    # print("MMD: %.2f, COV: %2.4f" % (mmd, cov))

    mmd_cd, _, _ = minimum_matching_distance(real_pts, fake_pts, batch_size=3000)
    mmd_cd = mmd_cd * 1000
    print(f"MMD-cd: {mmd_cd}")
    cov_cd, _ = coverage_distance(real_pts, fake_pts, batch_size=3000)
    print(f"MMD-cd: {mmd_cd}, COV-cd: {cov_cd}")


if __name__ == "__main__":
    main(sys.argv[1:])

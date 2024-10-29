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
from scene_synthesis.datasets import get_raw_dataset, filter_function, Tree
from pytorch3d import transforms

from simple_3dviz import Mesh, Scene
from simple_3dviz.utils import render
from simple_3dviz.behaviours.io import SaveFrames
from simple_3dviz.behaviours.movements import CameraTrajectory
from simple_3dviz.behaviours.trajectory import Circle

torch.set_num_threads(5)
torch.autograd.set_detect_anomaly(True)

class PartNetSyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        file_names = sorted(os.listdir(data_path), key=lambda x: int(os.path.splitext(x)[0]))
        self.file_paths = [os.path.join(data_path, path) for path in file_names]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        part_boxes = torch.load(self.file_paths[idx])
        idx = os.path.split(self.file_paths[idx])[1].split("_")[0]
        
        return int(idx), part_boxes[1:]
        
class PartNetRealDataset(torch.utils.data.Dataset):
    def __init__(self, real):
        self.real = real

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        tree, id = self.real[idx]
        part_boxes, _, _, _, part_sems = tree.graph(leafs_only=True)
        part_boxes = torch.stack(part_boxes)
        R = transforms.quaternion_to_matrix(part_boxes[..., 6:]).numpy()

        return id, part_boxes, part_sems

def mesh_from_boxes(boxes):
    boxes = boxes.cpu()
    # Divide the sizes by half since simple3dviz expect alpha to be
    # half of the side length
    R = transforms.quaternion_to_matrix(boxes[..., 6:]).numpy()
    rot = np.matrix([[1, 0, 0], [0, 0, -1], [0, 1, 0]]).T

    translations = boxes[..., :3].numpy() * rot
    
    R = np.array([R[i].T * rot for i in range(R.shape[0])])
    alphas = boxes[..., 3:6].numpy() / 2 
    epsilons = np.ones((len(alphas), 2)) * 0.1
 
    generator = np.random.default_rng(0)
    colors = generator.random(size=(boxes.shape[0], 3))
    # colors = np.array([[178/255, 190/255, 181/255]])
    # colors = np.repeat(colors, boxes.shape[0], axis=0)

    m = Mesh.from_superquadrics(alphas, epsilons, translations, R, colors)
    return m

def mesh_from_voxel(voxel_grid):
    m = Mesh.from_voxel_grid(voxel_grid)
    
    return m

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
        "--batch_size",
        type=int,
        default=1024,
        help="Set the batch size for training and evaluating (default: 256)"
    )    
    parser.add_argument(
        "--output_directory",
        help="Path to output directory"
    )    
    parser.add_argument(
        "--background",
        type=lambda x: list(map(float, x.split(","))),
        default="1,1,1,1",
        help="Set the background of the scene"
    )
    parser.add_argument(
        "--up_vector",
        type=lambda x: tuple(map(float, x.split(","))),
        default="0,0,1",
        help="Up vector of the scene"
    )
    parser.add_argument(
        "--camera_target",
        type=lambda x: tuple(map(float, x.split(","))),
        default="0,0,0",
        help="Set the target for the camera"
    )
    parser.add_argument(
        "--camera_position",
        type=lambda x: tuple(map(float, x.split(","))),
        default="-2.5,-2.5,2.5",
        help="Camer position in the scene"
    )
    parser.add_argument(
        "--window_size",
        type=lambda x: tuple(map(int, x.split(","))),
        default="512,512",
        help="Define the size of the scene and the window"
    )
    parser.add_argument(
        "--path_to_syn_test",
        help="Path to the folder containing the synthesized testing data"
    )
    parser.add_argument(
        "--render_real",
        default=False,
        action='store_true',
        help="Define whether to render real objs as well"
    )

    args = parser.parse_args(argv)    
    config = load_config(args.config_file)
    
    test_real = get_raw_dataset(
        config["data"],
        filter_fn=filter_function(
            config["data"],
            split=config["training"].get("splits", ["train", "val"])
        ),
        split=config["testing"].get("splits", ["test"])
    )
    test_synthetic = PartNetSyntheticDataset(args.path_to_syn_test)
    test_real = PartNetRealDataset(test_real)    

    real_path = os.path.join(args.output_directory, "real")
    if not os.path.isdir(real_path):
        os.makedirs(real_path)

    syn_path = os.path.join(args.output_directory, "syn")
    if not os.path.isdir(syn_path):
        os.makedirs(syn_path)

    scene = Scene(size=args.window_size)
    scene.light = args.camera_position
    
    if args.render_real:
        for idx, r_sample, _ in test_real:
            path_to_image = "{}/{}_".format(real_path, idx)
            behaviours = [CameraTrajectory(Circle([0, 0, 2.5], args.camera_position, [0, 0, 1]),
                speed=0.1), SaveFrames(path_to_image+"{:03d}.png", 2)]
            mesh = mesh_from_boxes(r_sample)
            render(
                [mesh],
                size=args.window_size,
                camera_position=args.camera_position,
                camera_target=args.camera_target,
                up_vector=args.up_vector,
                background=args.background,
                behaviours=behaviours,
                n_frames=10,
                scene=scene
            )
    
    for idx, s_sample in test_synthetic:
        path_to_image = "{}/{}_".format(syn_path, idx)
        behaviours = [CameraTrajectory(Circle([0, 0, 2.5], args.camera_position, [0, 0, 1]),
            speed=0.1), SaveFrames(path_to_image+"{:03d}.png", 2)]
        mesh = mesh_from_boxes(s_sample)
        render(
            mesh,
            size=args.window_size,
            camera_position=args.camera_position,
            camera_target=args.camera_target,
            up_vector=args.up_vector,
            background=args.background,
            behaviours=behaviours,
            n_frames=10,
            scene=scene
        )

if __name__ == "__main__":
    main(None)

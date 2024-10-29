"""Script used for generating scenes using a previously trained model."""
import argparse
import logging
import os
import sys
from tqdm import tqdm

import numpy as np
import torch
from pytorch3d import transforms

from training_utils import load_config
# from utils import floor_plan_from_scene, export_scene

sys.path.append(os.getcwd())

from scene_synthesis.datasets import filter_function, \
    get_dataset_raw_and_encoded
from scene_synthesis.datasets.partnet import Tree
from scene_synthesis.networks import build_network

from vis_utils import draw_partnet_objects


def main(argv):
    parser = argparse.ArgumentParser(
        description="Generate scenes using a previously trained model"
    )

    parser.add_argument(
        "config_file",
        help="Path to the file that contains the experiment configuration"
    )
    parser.add_argument(
        "output_directory",
        default="/tmp/",
        help="Path to the output directory"
    )
    parser.add_argument(
        "--weight_file",
        default=None,
        help="Path to a pretrained model"
    )
    parser.add_argument(
        "--n_sequences",
        default=10,
        type=int,
        help="The number of sequences to be generated"
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
        default="0,1,0",
        help="Up vector of the scene"
    )
    parser.add_argument(
        "--camera_position",
        type=lambda x: tuple(map(float, x.split(","))),
        default="-0.10923499,1.9325259,-7.19009",
        help="Camer position in the scene"
    )
    parser.add_argument(
        "--camera_target",
        type=lambda x: tuple(map(float, x.split(","))),
        default="0,0,0",
        help="Set the target for the camera"
    )
    parser.add_argument(
        "--window_size",
        type=lambda x: tuple(map(int, x.split(","))),
        default="512,512",
        help="Define the size of the scene and the window"
    )
    parser.add_argument(
        "--with_rotating_camera",
        action="store_true",
        help="Use a camera rotating around the object"
    )
    parser.add_argument(
        "--save_frames",
        help="Path to save the visualization frames to"
    )
    parser.add_argument(
        "--n_frames",
        type=int,
        default=360,
        help="Number of frames to be rendered"
    )
    parser.add_argument(
        "--without_screen",
        action="store_true",
        help="Perform no screen rendering"
    )
    parser.add_argument(
        "--scene_id",
        default=None,
        help="The scene id to be used for conditioning"
    )

    args = parser.parse_args(argv)

    # Disable trimesh's logger
    logging.getLogger("trimesh").setLevel(logging.ERROR)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print("Running code on", device)

    # Check if output directory exists and if it doesn't create it
    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    config = load_config(args.config_file)

    raw_dataset, train_dataset = get_dataset_raw_and_encoded(
        config["data"],
        filter_fn=filter_function(
            config["data"],
            split=config["training"].get("splits", ["train", "val"])
        ),
        split=config["training"].get("splits", ["train", "val"])
    )

    raw_dataset, dataset = get_dataset_raw_and_encoded(
        config["data"],
        filter_fn=filter_function(
            config["data"],
            split=config["validation"].get("splits", ["test"])
        ),
        split=config["validation"].get("splits", ["test"])
    )
    print("Loaded {} objects".format(
        len(dataset)
    ))

    network, _, _ = build_network(
        256, 2,
        config, args.weight_file, device=device
    )
    network.eval()

    for tree, id in tqdm(raw_dataset):
        # plot the root box
        root_only = Tree(Tree.Node(
            is_leaf = True,
            box = tree.root.box,
            label = tree.root.label,
            geo = tree.root.geo,
            full_label = tree.root.full_label
        ))
        draw_partnet_objects(objects=[root_only], object_names=['root box only'], 
                     figsize=(9, 5), leafs_only=True, 
                     sem_colors_filename='scripts/part_colors_Chair.txt',
                     out_fn=os.path.join(args.output_directory, f'{id}_root.png'))
        draw_partnet_objects(objects=[tree], object_names=['GT'], 
                     figsize=(9, 5), leafs_only=True, 
                     sem_colors_filename='scripts/part_colors_Chair.txt',
                     out_fn=os.path.join(args.output_directory, f'{id}_orig_all.png'))

        # generate the object
        box_quat = tree.root.get_box_quat().squeeze(0)
        box_input = torch.cat((box_quat[:6], transforms.quaternion_to_axis_angle(box_quat[6:]))).unsqueeze(0).to(device)
        bbox_params = network.generate_boxes(box_input, device=device)
        translations = bbox_params['translations'].squeeze(0)[1:]
        sizes = bbox_params['sizes'].squeeze(0)[1:]
        angles = bbox_params['angles'].squeeze(0)[1:]
        quats = transforms.axis_angle_to_quaternion(angles)
        box_quats = torch.cat((translations, sizes, quats), dim=-1)
        completed_tree = Tree(Tree.Node(
            is_leaf = False,
            children = [Tree.Node(is_leaf = True) for q in box_quats],
            full_label = 'chair',
        ))
        for child, quat in zip(completed_tree.root.children, box_quats):
            child.set_from_box_quat(quat)
        
        # plot the generated object
        draw_partnet_objects(objects=[completed_tree], object_names=['pred'], 
                     figsize=(9, 5), leafs_only=True, 
                     sem_colors_filename='scripts/part_colors_Chair.txt',
                     out_fn=os.path.join(args.output_directory, f'{id}_pred.png'))


if __name__ == "__main__":
    main(sys.argv[1:])

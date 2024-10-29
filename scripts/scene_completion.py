"""Script to perform scene completion."""
import argparse
import logging
import os
import random
from random import sample
import itertools
import sys
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader


from scene_synthesis.datasets.threed_front_dataset import Scale, dataset_encoding_factory
from scene_synthesis.stats_logger import StatsLogger
from scene_synthesis.datasets import filter_function, Tree, get_raw_dataset
from scene_synthesis.networks import build_network
from render_partnet import mesh_from_boxes
from training_utils import load_config
from vis_utils import draw_partnet_objects, export_simple3dviz_mesh
from utils import save_box_quats
from torch.utils.data import RandomSampler

sys.path.append(os.getcwd())
torch.set_num_threads(5)

def poll_objects(dataset, current_boxes, scene_id):
    """Show the objects in the current_scene and ask which ones to be
    removed."""
    object_types = np.array(dataset.object_types)
    labels = object_types[current_boxes["class_labels"].argmax(-1)].tolist()
    print(
        "The {} scene you selected contains {}".format(
            scene_id, list(enumerate(labels))
        )
    )
    msg = "Enter the indices of objects to be removed, separated with commas\n"
    ois = [int(oi) for oi in input(msg).split(",") if oi != ""]
    idxs_kept = list(set(range(len(labels))) - set(ois))
    print("You are keeping the following indices {}".format(idxs_kept))

    return idxs_kept


def main(argv):
    parser = argparse.ArgumentParser(
        description="Complete a partially complete scene"
    )

    parser.add_argument(
        "config_file",
        help="Path to the file that contains the experiment configuration"
    )
    parser.add_argument(
        "output_directory",
        help="Path to the output directory"
    )
    parser.add_argument(
        "--weight_file",
        default=None,
        help="Path to a pretrained model"
    )
    parser.add_argument(
        "--generation_mode",
        default='generation'
    )
    parser.add_argument(
        "--add_single",
        action="store_true"
    )
    parser.add_argument(
        "--synthesize_data",
        action="store_true"
    )
    parser.add_argument(
        "--inference_train",
        action="store_true"
    )
    parser.add_argument(
        "--render_part",
        action='store_true'
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
    os.makedirs(f"{args.output_directory}/box_quats", exist_ok=True)
    os.makedirs(f"{args.output_directory}/part_obj/gt", exist_ok=True)
    os.makedirs(f"{args.output_directory}/part_obj/pred", exist_ok=True)
    os.makedirs(f"{args.output_directory}/obj", exist_ok=True)

    config = load_config(args.config_file)

    if args.inference_train:
        split = config["training"].get("splits", ["train", "val"])
        filter_fn = filter_function(
            config["data"],
            split=config["training"].get("splits", ["train", "val"])
        )
    else:
        split = config["testing"].get("splits", ["test"])
        filter_fn = filter_function(
            config["data"],
            split=config["testing"].get("splits", ["test"])
        )

    raw_test = get_raw_dataset(
        config["data"],
        filter_fn=filter_fn,
        split=split
    )

    test_dataset = dataset_encoding_factory(
        name=config["data"].get("encoding_type"),
        dataset=raw_test,
        box_ordering=config.get("box_ordering", None),
        sample_strategy=args.generation_mode
    )

    sampler = None
    if args.generation_mode == 'gen_from_scratch':
        sampler = RandomSampler(test_dataset, replacement=False, num_samples=2000)

    test_loader = DataLoader(
        test_dataset,
        sampler=sampler,
        batch_size=1,
        collate_fn=test_dataset.collate_fn,
        shuffle=False
    )

    print("Loaded {} objects".format(
        len(raw_test))
    )

    network, _, _ = build_network(
        256, config["generator"].get("n_class"),
        config["generator"], args.weight_file, device=device
    )
    network.eval()

    if args.add_single:
        print("Adding a single part")
    else:
        print("Completing the object")
    
    test_dataset.test = True

    for i, sample_params in enumerate(itertools.chain(test_loader)): 

        id = sample_params['obj_id'].item()
        
        for k, v in sample_params.items():
            sample_params[k] = v.to(device)

        input_boxes_ori = torch.tensor([])
        target_boxes_ori = torch.tensor([])
        if sample_params['seen_idxs'].shape[-1] > 0:
            input_boxes_ori = sample_params['part_boxes_ori'][[0], sample_params['seen_idxs']].squeeze(0)
        
        if sample_params['target_idxs'].shape[-1] > 0:
            target_boxes_ori = sample_params['part_boxes_ori'][[0], sample_params['target_idxs']].squeeze(0)

        input_length = sample_params['lengths'].squeeze(0).item()

        observed = Tree(Tree.Node(
            is_leaf = False,
            children = [Tree.Node(is_leaf=True) for _ in input_boxes_ori] + \
                [Tree.Node(is_leaf=True) for _ in target_boxes_ori]
        ))

        for child, box_quat in zip(observed.root.children[: input_length], input_boxes_ori):
            child.set_from_box_quat(box_quat)
        
        for child, box_quat in zip(observed.root.children[input_length: ], target_boxes_ori):
            child.set_from_box_quat(box_quat)
            child.full_label = 'chair_0'

        draw_partnet_objects(objects=[observed], object_names=['GT'], 
                     figsize=(9, 5), leafs_only=True, 
                     sem_colors_filename='scripts/part_colors_Chair.txt',
                     out_fn=os.path.join(args.output_directory, f'{id}_observed.png'))

        if args.render_part:
            if args.generation_mode != 'gen_from_scratch':
                start_gt = mesh_from_boxes(input_boxes_ori)
                export_simple3dviz_mesh(start_gt, os.path.join(args.output_directory, "part_obj/gt", f'{id}_{i}_start_gt.obj'))

            start = len(input_boxes_ori)
            for j in range(start, len(sample_params['part_boxes_ori'][0])):
                partial_gt = mesh_from_boxes(sample_params['part_boxes_ori'][0, :j + 1])
                export_simple3dviz_mesh(partial_gt, os.path.join(args.output_directory, "part_obj/gt", f'{id}_{i}_{j}_partial_gt.obj'))

        num_runs = 1 if args.synthesize_data else 3
        objects = []
        object_names = []
        
        all_children = [Tree.Node(
            is_leaf = True,
            full_label = ''
        ) for _ in range(input_length)]

        for run in range(num_runs):    
            if args.add_single:
                bbox_params = network.add_object(
                    root_box=box_input['root_box'],
                    boxes=box_input,
                    device=device
                )
            else:
                bbox_params = network.complete_scene(
                    id,
                    sample_params=sample_params,
                    device=device 
                )

            for k, v in bbox_params.items():
                bbox_params[k] = v.to('cpu')

            # Scale all outputs back to original space, this will introduce a small
            # numerical error around e-9 ~ e-10 on average
            bbox_params = test_dataset.post_process(bbox_params)

            box_quats = bbox_params["input_boxes"][0][1:]
            box_quats = torch.cat((input_boxes_ori.cpu(), box_quats[input_length:]), dim=0)
            all_children.extend([Tree.Node(
                is_leaf = True,
                full_label = 'chair_{}'.format(run)
            ) for _ in range(len(box_quats) - input_length)])

            completed_tree = Tree(Tree.Node(
                is_leaf = False,
                children = all_children
            ))
            for child, quat in zip(completed_tree.root.children, box_quats):
                child.set_from_box_quat(quat)
            objects.append(completed_tree)
            object_names.append(f'pred_{run}')

            if args.render_part:
                for j in range(start, len(box_quats)):
                    partial_pred = mesh_from_boxes(box_quats[:j + 1])
                    export_simple3dviz_mesh(partial_pred, os.path.join(args.output_directory, "part_obj/pred", f'{id}_{i}_{j}_partial.obj'))
            object_mesh = mesh_from_boxes(box_quats)
            export_simple3dviz_mesh(object_mesh, os.path.join(args.output_directory, "obj", f'{id}_{i}.obj'))

            if args.synthesize_data:
                # concate unscaled root_box to the front of the sequence
                box_quats = torch.cat([sample_params["root_box_ori"], box_quats], dim=0)
                save_box_quats(box_quats, f'{id}_{run}', args.output_directory)


        # plot the generated object
        draw_partnet_objects(objects=objects, object_names=object_names,
            figsize=(27, 5), leafs_only=True,
            sem_colors_filename='scripts/part_colors_Chair.txt',
            out_fn=os.path.join(args.output_directory, f'{id}_pred.png'))

    StatsLogger.instance().print_progress(-1, -1, 0, final=True)
if __name__ == "__main__":
    main(sys.argv[1:])

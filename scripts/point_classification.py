from ctypes import pointer
from training_utils import load_config
from scene_synthesis.datasets import get_raw_dataset, dataset_encoding_factory, filter_function
from scene_synthesis.networks import build_network
from scene_synthesis.networks.autoregressive_transformer import batch_decode
from scene_synthesis.stats_logger import StatsLogger
from scene_synthesis.libmesh import check_mesh_contains
from scene_synthesis.evaluator.generation import Generator3D
from utils import save_box_quats
from render_partnet import mesh_from_boxes

import numpy as np
import sys
import os

import torch
from torch.utils.data import DataLoader
import argparse
from sklearn.metrics import accuracy_score, recall_score, precision_score

from simple_3dviz import Mesh, Scene, Spherecloud
from simple_3dviz.utils import render
from simple_3dviz.behaviours.io import SaveFrames
from simple_3dviz.behaviours.movements import CameraTrajectory
from simple_3dviz.behaviours.trajectory import Circle
from torch.utils.data import RandomSampler
import trimesh

def vis_points_cloud(D, D_labels, scene, path, objs):

    D_positive = D[D_labels == True]
    D_negative = D[D_labels == False]

    render_objs = objs
    if len(D_positive) > 0:
        s1 = Spherecloud(D_positive, colors=(0.1, 0.1, 0.9, 0.5), sizes=0.005)
        render_objs.append(s1)

    if len(D_negative) > 0:
        s2 = Spherecloud(D_negative, colors=(0.9, 0.1, 0.1, 0.5), sizes=0.005)
        render_objs.append(s2)

    camera_position = [-2, -2, 2]
    behaviours = [CameraTrajectory(Circle([0, 0, camera_position[2]], camera_position, [0, 0, 1]),
        speed=0.1)]
    scene.light = camera_position

    render(
        render_objs,
        size=[512,512],
        camera_position=camera_position,
        camera_target=[0,0,0],
        up_vector=[0,0,1],
        background=[1,1,1,1],
        behaviours=behaviours + [SaveFrames(path+"_{:03d}.png", 2)],
        n_frames=10,
        scene=scene
    )
    scene.clear()


def vis_bboxes(scene, path, boxes):
    mesh = mesh_from_boxes(boxes)
    render(
        [mesh],
        size=[512,512],
        camera_position=[-2.5, -2.5, 1],
        camera_target=[0,0,0],
        up_vector=[0,0,1],
        background=[1,1,1,1],
        behaviours=[SaveFrames(path+".png", 1)],
        n_frames=1,
        scene=scene
    )

    return mesh


def align_meshes(mesh1, mesh2):

    # Compute the size of each mesh
    mesh1_size = np.linalg.norm(mesh1.vertices.max(axis=0) - mesh1.vertices.min(axis=0))
    mesh2_size = np.linalg.norm(mesh2.vertices.max(axis=0) - mesh2.vertices.min(axis=0))

    # Calculate the scale factor to match the size of the second mesh
    scale_factor = mesh2_size / mesh1_size

    # Calculate the translation vector to move the first mesh to the position of the second mesh
    mesh1_center = mesh1.centroid
    mesh2_center = mesh2.centroid
    translation = mesh2_center - scale_factor * mesh1_center

    # Apply the scaling and translation to the first mesh
    mesh1.apply_scale(scale_factor)
    mesh1.apply_translation(translation)

    return mesh1, mesh2, (scale_factor, translation)


def convert_boxes_meshes(boxes, scale, translate):
    all_meshes = []
    for i in range(len(boxes)):
        m = mesh_from_boxes(boxes[i:i+1])
        v, f = m.to_points_and_faces()
        m = trimesh.Trimesh(v, f, process=False)
        m.apply_scale(scale)
        m.apply_translation(-translate)

        all_meshes.append(m)
    
    return all_meshes


def recolor_faces(mesh, boxes, scale, translate):
    from scipy.spatial.distance import cdist 
    generator = np.random.default_rng(0)
    colors = generator.random(size=(boxes.shape[0], 3))

    box_meshes = convert_boxes_meshes(boxes, scale, translate)
    vs = mesh.vertices
    fs = mesh.faces
    
    output_colors = []
    distances_to_meshes = []
    corners = []

    for m in box_meshes:
        m_c = m.bounding_box_oriented.vertices
        m_c = np.vstack([m_c, m.centroid])
        corners.append(m_c)

    dedup_corners = []
    dedup_idx = []

    for i in range(len(corners)):
        is_contained = False
        for j in range(len(corners)):
            if i != j:
                box1_xmin, box1_ymin, box1_zmin = min(p[0] for p in corners[i]), min(p[1] for p in corners[i]), min(p[2] for p in corners[i])
                box1_xmax, box1_ymax, box1_zmax = max(p[0] for p in corners[i]), max(p[1] for p in corners[i]), max(p[2] for p in corners[i])
                box2_xmin, box2_ymin, box2_zmin = min(p[0] for p in corners[j]), min(p[1] for p in corners[j]), min(p[2] for p in corners[j])
                box2_xmax, box2_ymax, box2_zmax = max(p[0] for p in corners[j]), max(p[1] for p in corners[j]), max(p[2] for p in corners[j])
                if box2_xmin <= box1_xmin <= box2_xmax and box2_ymin <= box1_ymin <= box2_ymax and box2_zmin <= box1_zmin <= box2_zmax and box1_xmax <= box2_xmax and box1_ymax <= box2_ymax and box1_zmax <= box2_zmax:
                    is_contained = True
                    break
        if not is_contained:
            dedup_corners.append(corners[i][-1])
            dedup_idx.append(i)

    
    for m_c in dedup_corners:
        
        distances = cdist([m_c], vs, metric='euclidean')
        distances = np.amin(np.swapaxes(distances, 0, 1), axis=-1) 
        distances_to_meshes.append(distances.reshape(-1, 1))

    distances_to_meshes = np.hstack(distances_to_meshes)
    color_idx = np.argmin(distances_to_meshes, axis=-1)
    output_colors = colors[np.array(dedup_idx)[color_idx]]

    return output_colors, box_meshes, np.vstack(dedup_corners)


def vis_mesh(scene, path, mesh):
    render(
        mesh,
        size=[512,512],
        camera_position=[-1.25, -1.25, .5],
        camera_target=[0,0,0],
        up_vector=[0,0,1],
        background=[1,1,1,1],
        behaviours=[SaveFrames(path+".png", 1)],
        n_frames=1,
        scene=scene
    )


SMOOTH = 1e-6
def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    intersection = (outputs & labels).float().sum(())  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum(())         # Will be zzero if both are 0
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
    
    return iou


def main(argv):
    parser = argparse.ArgumentParser(
        description="Train a generative model on bounding boxes"
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
        "--generator_weight_file",
        default=None,
        help=("The path to a previously trained model to continue"
              " the training from")
    )
    parser.add_argument(
        "--encoder_weight_file",
        default=None,
        help=("The path to a previously trained encoder model to continue"
              " the training from")
    )
    parser.add_argument(
        "--decoder_weight_file",
        default=None,
        help=("The path to a previously trained decoder model to continue"
              " the training from")
    )

    parser.add_argument(
        "--inference_train",
        action="store_true"
    )

    args = parser.parse_args(argv)
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
        split=split,
    )

    config['data']['sample_strategy'] = 'gt_bbox'
    test_dataset = dataset_encoding_factory(
        name=config["data"].get("encoding_type"),
        dataset=raw_test,
        box_ordering=config.get("box_ordering", None),
        sample_strategy=config['data']['sample_strategy'],
        num_points_to_sample=200000,
    )
    test_dataset.test = True 

    sampler = RandomSampler(test_dataset, replacement=False)
    test_loader = DataLoader(
        test_dataset,
        sampler=sampler,
        batch_size=1,
        collate_fn=test_dataset.collate_fn,
        shuffle=False
    )
    print('Sampling Strategy is {}'.format(config['data']['sample_strategy']))

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print("Running code on", device)

    # Check if output directory exists and if it doesn't create it
    os.makedirs(os.path.join(args.output_directory, "obj/vis"), exist_ok=True)
    os.makedirs(os.path.join(args.output_directory, "starting/box_quats/"), exist_ok=True)
    os.makedirs(os.path.join(args.output_directory, "starting/vis/"), exist_ok=True)    
    os.makedirs(os.path.join(args.output_directory, "completed/box_quats/"), exist_ok=True)
    os.makedirs(os.path.join(args.output_directory, "completed/vis/"), exist_ok=True)    

    network, train_on_batch, validate_on_batch = build_network(
        # train_dataset.feature_size, train_dataset.n_classes,
        256, config["generator"].get("n_class"),
        config["generator"], args.generator_weight_file, device=device
    )
    network.eval()

    encoder_network, _, _ = build_network(
        256, config["encoder"].get("n_class"),
        config["encoder"], args.encoder_weight_file, device=device
    )
    encoder_network.eval()

    decoder_network, train_on_batch_decoder, validate_on_batch_decoder = build_network(
        256, config["decoder"].get("n_class"),
        config["decoder"], args.decoder_weight_file, device=device
    )
    decoder_network.eval()

    mesh_gt_path = config['data'].get('dataset_directory').replace('_hier', '_mesh')
    chamfer_L1s = []
    chamfer_L2s = []
    ious = []

    for idx, sample_params in enumerate(test_loader):
        id = sample_params['obj_id'].item()
        B = sample_params['input_boxes'].shape[0]


        for k, v in sample_params.items():
            sample_params[k] = v.to(device)

        start_box = network.start_symbol(device)
        # Add the start box token in the beginning
        for k in start_box.keys():
            sample_params[k] = torch.cat([start_box[k].expand(B, -1, -1), \
                sample_params[k].to(device)], dim=1)
        
        if config['data']['sample_strategy'] != 'gt_bbox':    
            max_box = 25
            sample_params['is_end'] = torch.zeros(B).to(device)
            for _ in range(max_box):
                batch_decode(sample_params, network, device)
                if torch.sum(sample_params['is_end']) == B:
                    break

            max_seq_length = torch.max(sample_params['lengths'])
            # This exclude the end class token and the predicted box since it's useless to us.
            # The start token is still at the front
            sample_params['input_boxes'] = sample_params['input_boxes'][:, :max_seq_length + 1]
            sample_params['input_class'] = sample_params['input_class'][:, :max_seq_length + 1]

        for k, v in sample_params.items():
            sample_params[k] = v.to('cpu')
        
        sample_params = test_dataset.post_process(sample_params)

        for k, v in sample_params.items():
            sample_params[k] = v.to(device)
        
        sample_params['root_box'] = sample_params['root_box_ori']

        # removing the root box
        completed_boxes = sample_params['input_boxes'][0][1:]
        # vis and save the generated boxes        
        scene = Scene(size=[512,512])
        
        # Save the output bboxes and visualize the bbox
        save_box_quats(completed_boxes, f'{id}_{len(completed_boxes)}', os.path.join(args.output_directory, "completed"))
        m = vis_bboxes(scene, os.path.join(args.output_directory, f"completed/vis/{id}_{len(completed_boxes)}"), completed_boxes)

        v, f = m.to_points_and_faces()
        m = trimesh.Trimesh(v, f, process=False)
        m.export(
            os.path.join(args.output_directory, 'completed', f"{id}_{len(completed_boxes)}.obj"),
            file_type="obj"
        )

        if config['data']['sample_strategy'] != 'gen_from_scratch':
            start_boxes = sample_params['part_boxes_ori'][[0], sample_params['seen_idxs']].squeeze(0)
            save_box_quats(start_boxes, f'{id}', os.path.join(args.output_directory, "starting"))
            vis_bboxes(scene, os.path.join(args.output_directory, f"starting/vis/{id}_{len(start_boxes)}"), start_boxes)
            print(f"For object {id}, the starting step is {len(start_boxes)}, number of prediction steps is {len(completed_boxes) - len(start_boxes)}")

        memory = encoder_network._encode(sample_params)
        mesh_generator = Generator3D(decoder_network, \
                                    points_batch_size=80000, \
                                    resolution0=128, \
                                    upsampling_steps=0)

        # Generate a mesh from the predicted occupancies
        mesh = mesh_generator.generate_mesh(memory, sample_params['lengths'] + 1)
        
        m, mesh, (scale, translation) = align_meshes(m, mesh)
        mesh.export(
            os.path.join(args.output_directory, 'obj', f"{id}_{len(completed_boxes)}.obj"),
            file_type="obj"
        )

        colors, box_meshes, corners = recolor_faces(mesh, completed_boxes, scale, translation)   
        meshes = [Mesh.from_faces(mesh.vertices, mesh.faces, colors=colors)]
        # meshes.extend([Mesh.from_faces(m.vertices + np.array((0.5, 0, 0)), m.faces, colors=(0.45, 0.57, 0.70)) for m in box_meshes])
        # meshes.extend([Spherecloud(corners)])

        vis_mesh(scene, os.path.join(args.output_directory, f"obj/vis/{id}_{len(completed_boxes)}"), \
            meshes)

        scene.clear()
        
if __name__ == "__main__":
    main(sys.argv[1:])



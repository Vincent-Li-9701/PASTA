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

import pickle
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
from pytorch3d import transforms
from pyquaternion import Quaternion

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


def convert_boxes_meshes(boxes, scale, translate, dataset_idx):
    all_meshes = []
    B = boxes['class_labels'].shape[0]

    for i in range(B):
        box = {k: boxes[k][i:i+1] for k in boxes.keys()}
        _, m = get_renderable_from_boxes(box, dataset_idx)
        m.apply_scale(scale)
        m.apply_translation(-translate)

        all_meshes.append(m)
    
    return all_meshes


def remap_num2str(class_labels, dataset_idx, n_class, sem2num):
    idx = torch.argmax(class_labels, -1)

    names = dataset_idx[idx]
    idx = [sem2num[name] for name in names]
    remapped_classes = torch.zeros(len(class_labels), n_class)
    remapped_classes[torch.arange(len(class_labels)), idx] = 1
    remapped_classes = remapped_classes.unsqueeze(0)
    return remapped_classes


def get_renderable_from_boxes(boxes, dataset_idx, suffix=""):
    class_labels = boxes[f"class_labels{suffix}"]
    translations = boxes[f"translations{suffix}"].cpu().numpy()
    sizes = boxes[f"sizes{suffix}"].cpu().numpy()
    angles = boxes[f"angles{suffix}"].cpu().numpy()
    N = len(class_labels)
    
    # Get the part labels from the boxes
    part_labels_from_boxes = np.array(
        dataset_idx
    )[np.array(boxes[f"class_labels{suffix}"].argmax(-1))]

    sq_sizes = []
    sq_translations = []
    sq_rotations = []
    sq_colors = []
    for pi in range(N):
        # No need to draw the end box
        if part_labels_from_boxes[pi] == "end":
            print("END TOKEN DETECTED NEED HANDLING")
            continue
        if angles[pi].shape[-1] == 4:
            rotation_matrix = Quaternion(angles[pi]).rotation_matrix
        elif angles[pi].shape[-1] == 6:
            rotation_matrix = transforms.rotation_6d_to_matrix(
                torch.tensor(angles[pi])
            ).numpy()
        else:
            raise NotImplementedError()
        sq_sizes.append(sizes[pi])
        sq_translations.append(translations[pi])
        sq_rotations.append(rotation_matrix)
    
    generator = np.random.default_rng(0)
    colors = generator.random(size=(len(sq_sizes), 3))
    
    mesh = Mesh.from_superquadrics(
        np.array(sq_sizes),
        np.ones((len(np.array(sq_sizes)), 2)) * 0.1,
        np.array(sq_translations),
        np.array(sq_rotations),
        np.array(sq_colors)
    )

    vertices, faces = mesh.to_points_and_faces()
    tr_mesh = trimesh.Trimesh(vertices, faces, process=False)
    # Define the rotation angle in radians (90 degrees in this case)
    rotation_angle = 90.0 * (3.14159 / 180.0)

    # Define the rotation matrix for the desired rotation
    rotation_matrix = trimesh.transformations.rotation_matrix(rotation_angle, [1, 0, 0])

    # Apply the rotation to the mesh vertices
    tr_mesh.apply_transform(rotation_matrix)
    
    sample_params = {}
    poses = transforms.matrix_to_quaternion(torch.tensor(sq_rotations))
    sample_params['input_boxes'] = torch.hstack([torch.tensor(sq_translations), torch.tensor(sq_sizes) * 2, poses]).unsqueeze(0)
    sample_params['input_class'] = torch.tensor(np.array(boxes[f"class_labels{suffix}"].argmax(-1)))
    sample_params['lengths'] = torch.tensor(len(part_labels_from_boxes))

    return sample_params, tr_mesh


def recolor_faces(mesh, boxes, scale, translate, dataset_idx):
    from scipy.spatial.distance import cdist 

    box_meshes = convert_boxes_meshes(boxes, scale, translate, dataset_idx)
    generator = np.random.default_rng(0)
    colors = generator.random(size=(len(box_meshes), 3))

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


def main(argv):
    parser = argparse.ArgumentParser(
        description="Train a generative model on bounding boxes"
    )
    parser.add_argument(
        "config_file",
        help="Path to the file that contains the experiment configuration"
    )
    parser.add_argument(
        "ori_label_dict",
        help="Original label dict that used to train decoder"
    )    
    parser.add_argument(
        "npz_dir",
        help=("The path to the npz directory")
    )
    parser.add_argument(
        "output_directory",
        help="Path to the output directory"
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
        "--recolor",
        default=False,
        action="store_true",
        help="Whether to recolor the mesh using part base"
    )

    args = parser.parse_args(argv)
    config = load_config(args.config_file)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print("Running code on", device)

    # Check if output directory exists and if it doesn't create it
    os.makedirs(os.path.join(args.output_directory, "obj/vis"), exist_ok=True)
    os.makedirs(os.path.join(args.output_directory, "obj_colored/vis"), exist_ok=True)
    os.makedirs(os.path.join(args.output_directory, "starting/box_quats/"), exist_ok=True)
    os.makedirs(os.path.join(args.output_directory, "starting/vis/"), exist_ok=True)    
    os.makedirs(os.path.join(args.output_directory, "completed/box_quats/"), exist_ok=True)
    os.makedirs(os.path.join(args.output_directory, "completed/vis/"), exist_ok=True)    

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

    paths = os.listdir(args.npz_dir)
    npz_files = [path for path in paths if path.endswith('.npz') and 'bounds' not in path]

    sem2num = pickle.load(open(args.ori_label_dict, 'rb'))
    n_class = len(sem2num.keys())

    for id, path in enumerate(npz_files):
        npz_path = os.path.join(args.npz_dir, path)
        dataset_idx = np.load(npz_path, allow_pickle=True)['dataset_classes']
        bbox = np.load(npz_path, allow_pickle=True)['boxes_processed'].item()

        scene = Scene(size=[512,512])
        sample_params, m = get_renderable_from_boxes(bbox, dataset_idx)
        sample_params['input_class'] = remap_num2str(bbox['class_labels'], dataset_idx, n_class, sem2num)

        # Create the initial input to the transformer, namely the start token
        start_box = encoder_network.start_symbol(device)
        # Add the start box token in the beginning
        for k in start_box.keys():
            sample_params[k] = torch.cat([start_box[k].expand(1, -1, -1), \
                sample_params[k].to(device)], dim=1)

        m.export(
            os.path.join(args.output_directory, 'completed', f"{id}_{len(sample_params['input_boxes'][0])}.obj"),
            file_type="obj"
        )

        memory = encoder_network._encode(sample_params)
        mesh_generator = Generator3D(decoder_network, \
                                    points_batch_size=80000, \
                                    resolution0=128, \
                                    upsampling_steps=0)

        # Generate a mesh from the predicted occupancies
        mesh = mesh_generator.generate_mesh(memory, sample_params['lengths'] + 1)
        
        m, mesh, (scale, translation) = align_meshes(m, mesh)
        mesh.export(
            os.path.join(args.output_directory, 'obj', f"{id}_{len(sample_params['input_boxes'][0])}.obj"),
            file_type="obj"
        )

        if args.recolor:
            colors, box_meshes, corners = recolor_faces(mesh, bbox, scale, translation, dataset_idx)   
            meshes = [Mesh.from_faces(mesh.vertices, mesh.faces, colors=colors)]
            # meshes.extend([Mesh.from_faces(m.vertices + np.array((0.5, 0, 0)), m.faces, colors=(0.45, 0.57, 0.70)) for m in box_meshes])
            # meshes.extend([Spherecloud(corners)])

            vis_mesh(scene, os.path.join(args.output_directory, f"obj/vis/{id}_{len(sample_params['input_boxes'][0])}"), \
                meshes)
            mesh.visual.vertex_colors = colors
            mesh.export(
                os.path.join(args.output_directory, 'obj_colored', f"{id}_{len(sample_params['input_boxes'][0])}.ply"),
                file_type="ply",
            )

        scene.clear()
        
if __name__ == "__main__":
    main(sys.argv[1:])



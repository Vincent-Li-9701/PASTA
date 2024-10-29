import numpy as np
import torch
from pykdtree.kdtree import KDTree
import trimesh

from ..libmesh import check_mesh_contains


def distance_p2p(points_src, normals_src, points_tgt, normals_tgt):
    """Computes minimal distances of each point in points_src to points_tgt.
    Arguments:
    ----------
        points_src (numpy array): source points
        normals_src (numpy array): source normals
        points_tgt (numpy array): target points
        normals_tgt (numpy array): target normals
    """
    kdtree = KDTree(points_tgt)
    dist, idx = kdtree.query(points_src)
    

    if normals_src is not None and normals_tgt is not None:
        normals_src = \
            normals_src / np.linalg.norm(normals_src, axis=-1, keepdims=True)
        normals_tgt = \
            normals_tgt / np.linalg.norm(normals_tgt, axis=-1, keepdims=True)

        normals_dot_product = (normals_tgt[idx] * normals_src).sum(axis=-1)
        # Handle normals that point into wrong direction gracefully
        # (mostly due to mehtod not caring about this in generation)
        normals_dot_product = np.abs(normals_dot_product)
    else:
        normals_dot_product = np.array(
            [np.nan] * points_src.shape[0], dtype=np.float32)
    return dist, normals_dot_product


def distance_p2m(points, mesh):
    """Compute minimal distances of each point in points to mesh.
    Arguments:
    ----------
        points (numpy array): points array
        mesh (trimesh): mesh
    """
    _, dist, _ = trimesh.proximity.closest_point(mesh, points)
    return dist


def compute_iou(occ1, occ2, weights=None, average=True):
    """Compute the intersection over union (IoU) for two sets of occupancy
    values.
    Arguments:
    ----------
        occ1: Tensor of size BxN containing the first set of occupancy values
        occ2: Tensor of size BxN containing the first set of occupancy values
    Returns:
    -------
        the IoU
    """
    if not torch.is_tensor(occ1):
        occ1 = torch.tensor(occ1)
        occ2 = torch.tensor(occ2)

    if weights is None:
        weights = occ1.new_ones(occ1.shape)

    assert len(occ1.shape) == 2
    assert occ1.shape == occ2.shape

    # Convert them to boolean
    occ1 = occ1 >= 0.5
    occ2 = occ2 >= 0.5

    # Compute IoU
    area_union = (occ1 | occ2).float()
    area_union = (weights * area_union).sum(dim=-1)
    area_union = torch.max(area_union.new_tensor(1.0), area_union)
    area_intersect = (occ1 & occ2).float()
    area_intersect = (weights * area_intersect).sum(dim=-1)
    iou = (area_intersect / area_union)

    if average:
        return iou.mean().item()
    else:
        return iou

class MeshEvaluator(object):
    """Mesh evaluation class. Code adapted from
    https://github.com/autonomousvision/occupancy_networks/eval.py
    Arguments:
    ----------
        n_points (int): number of points to be used for evaluation
    """

    def __init__(self, n_points=100000):
        self.n_points = n_points

    def eval_mesh(self, mesh, pointcloud_tgt, normals_tgt,
                  points_iou, occ_tgt):
        """Evaluates a mesh.
        Arguments:
        -----------
            mesh (trimesh): mesh which should be evaluated
            pointcloud_tgt (numpy array): target point cloud
            normals_tgt (numpy array): target normals
            points_iou (numpy_array): points tensor for IoU evaluation
            occ_tgt (numpy_array): GT occupancy values for IoU points
        """
        if len(mesh.vertices) != 0 and len(mesh.faces) != 0:
            pointcloud, idx = mesh.sample(self.n_points, return_index=True)
            pointcloud = pointcloud.astype(np.float32)
            normals = mesh.face_normals[idx]
        else:
            pointcloud = np.empty((0, 3))
            normals = np.empty((0, 3))

        out_dict = self.eval_pointcloud(
            pointcloud, pointcloud_tgt, normals, normals_tgt)

        if len(mesh.vertices) != 0 and len(mesh.faces) != 0:
            occ = check_mesh_contains(mesh, points_iou)
            if len(occ.shape) < 2:
                occ = np.expand_dims(occ, 0)
                occ_tgt = np.expand_dims(occ_tgt, 0)

            out_dict['iou'] = compute_iou(occ, occ_tgt)
        else:
            out_dict['iou'] = 0.

        return out_dict

    def eval_pointcloud(self, pointcloud, pointcloud_tgt,
                        normals=None, normals_tgt=None):
        """Evaluates a point cloud wrt to another pointcloud.
        Arguments:
        ----------
            pointcloud (numpy array): predicted point cloud
            pointcloud_tgt (numpy array): target point cloud
            normals (numpy array): predicted normals
            normals_tgt (numpy array): target normals
        """
        pointcloud = np.asarray(pointcloud)
        pointcloud_tgt = np.asarray(pointcloud_tgt)

        # Completeness: how far are the points of the target point cloud
        # from thre predicted point cloud
        completeness, completeness_normals = distance_p2p(
            pointcloud_tgt, normals_tgt, pointcloud, normals
        )
        completeness2 = completeness**2

        completeness = completeness.mean()
        completeness2 = completeness2.mean()
        completeness_normals = completeness_normals.mean()

        # Accuracy: how far are th points of the predicted pointcloud
        # from the target pointcloud
        accuracy, accuracy_normals = distance_p2p(
            pointcloud, normals, pointcloud_tgt, normals_tgt
        )
        accuracy2 = accuracy**2

        accuracy = accuracy.mean()
        accuracy2 = accuracy2.mean()
        accuracy_normals = accuracy_normals.mean()

        # Chamfer distance
        chamferL2 = 0.5 * (completeness2 + accuracy2)
        normals_correctness = (
            0.5 * completeness_normals + 0.5 * accuracy_normals
        )
        chamferL1 = 0.5 * (completeness + accuracy)

        out_dict = {
            'completeness': completeness,
            'accuracy': accuracy,
            'normals_completeness': completeness_normals,
            'normals_accuracy': accuracy_normals,
            'normals': normals_correctness,
            'completeness2': completeness2,
            'accuracy2': accuracy2,
            'chamfer_L2': chamferL2,
            'chamfer_L1': chamferL1,
        }

        return out_dict
import numpy as np

from functools import lru_cache
from scene_synthesis.datasets.partnet import PartNetDataset
from scipy.ndimage import rotate

import torch
from torch.utils.data import Dataset
from .quaternion import Quaternion

from pytorch3d import transforms
import pickle
from sklearn_extra.cluster import KMedoids

class DatasetDecoratorBase(Dataset):
    """A base class that helps us implement decorators for ThreeDFront-like
    datasets."""
    def __init__(self, dataset):
        self._dataset = dataset

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        return self._dataset[idx]

    @property
    def load_points(self):
        return self._dataset.load_points
    
    @property
    def load_clip(self):
        return self._dataset.load_clip

    @property
    def bounds(self):
        return self._dataset.bounds

    @property
    def n_classes(self):
        return self._dataset.n_classes

    @property
    def class_labels(self):
        return self._dataset.class_labels

    @property
    def class_frequencies(self):
        return self._dataset.class_frequencies

    @property
    def n_object_types(self):
        return self._dataset.n_object_types

    @property
    def object_types(self):
        return self._dataset.object_types

    @property
    def feature_size(self):
        return self.bbox_dims + self.n_classes

    @property
    def label_dict(self):
        return self._dataset.label_dict

    @property
    def bin_centers(self):
        return self._dataset.bin_centers

    @property
    def bbox_dims(self):
        raise NotImplementedError()

    def post_process(self, s):
        return self._dataset.post_process(s)


class BoxOrderedDataset(DatasetDecoratorBase):
    def __init__(self, dataset, box_ordering=None):
        super().__init__(dataset)
        self.box_ordering = box_ordering

    @lru_cache(maxsize=16)
    def _get_boxes(self, scene_idx):
        scene = self._dataset[scene_idx]
        if self.box_ordering is None:
            return scene.bboxes
        elif self.box_ordering == "class_frequencies":
            return scene.ordered_bboxes_with_class_frequencies(
                self.class_frequencies
            )
        else:
            raise NotImplementedError()


class DataEncoder(BoxOrderedDataset):
    """DataEncoder is a wrapper for all datasets we have
    """
    @property
    def property_type(self):
        raise NotImplementedError()


class RoomLayoutEncoder(DataEncoder):
    @property
    def property_type(self):
        return "room_layout"

    def __getitem__(self, idx):
        """Implement the encoding for the room layout as images."""
        img = self._dataset[idx].room_mask[:, :, 0:1]
        return np.transpose(img, (2, 0, 1))

    @property
    def bbox_dims(self):
        return 0


class ClassLabelsEncoder(DataEncoder):
    """Implement the encoding for the class labels."""
    @property
    def property_type(self):
        return "class_labels"

    def __getitem__(self, idx):
        # Make a local copy of the class labels
        classes = self.class_labels

        # Get the scene
        boxes = self._get_boxes(idx)
        L = len(boxes)  # sequence length
        C = len(classes)  # number of classes
        class_labels = np.zeros((L, C), dtype=np.float32)
        for i, bs in enumerate(boxes):
            class_labels[i] = bs.one_hot_label(classes)
        return class_labels

    @property
    def bbox_dims(self):
        return 0


class TranslationEncoder(DataEncoder):
    @property
    def property_type(self):
        return "translations"

    def __getitem__(self, idx):
        # Get the scene
        scene = self._dataset[idx]
        boxes = self._get_boxes(idx)
        L = len(boxes)  # sequence length
        translations = np.zeros((L, 3), dtype=np.float32)
        for i, bs in enumerate(boxes):
            translations[i] = bs.centroid(-scene.centroid)
        return translations

    @property
    def bbox_dims(self):
        return 3


class SizeEncoder(DataEncoder):
    @property
    def property_type(self):
        return "sizes"

    def __getitem__(self, idx):
        # Get the scene
        boxes = self._get_boxes(idx)
        L = len(boxes)  # sequence length
        sizes = np.zeros((L, 3), dtype=np.float32)
        for i, bs in enumerate(boxes):
            sizes[i] = bs.size
        return sizes

    @property
    def bbox_dims(self):
        return 3


class AngleEncoder(DataEncoder):
    @property
    def property_type(self):
        return "angles"

    def __getitem__(self, idx):
        # Get the scene
        boxes = self._get_boxes(idx)
        # Get the rotation matrix for the current scene
        L = len(boxes)  # sequence length
        angles = np.zeros((L, 1), dtype=np.float32)
        for i, bs in enumerate(boxes):
            angles[i] = bs.z_angle
        return angles

    @property
    def bbox_dims(self):
        return 1


class DatasetCollection(DatasetDecoratorBase):
    def __init__(self, *datasets):
        super().__init__(datasets[0])
        self._datasets = datasets

    @property
    def bbox_dims(self):
        return sum(d.bbox_dims for d in self._datasets)

    def __getitem__(self, idx):
        sample_params = {}
        for di in self._datasets:
            sample_params[di.property_type] = di[idx]
        return sample_params

    @staticmethod
    def collate_fn(samples):

        # Pad the first dimension.
        sample_params = dict()
        
        for k in samples[0].keys():
            if any([k_p in k for k_p in ['boxes', 'class', 'bins', 'label']]):
                # pad input shapes
                max_length = max(len(sample[k]) for sample in samples)
                pad_shape = torch.tensor(samples[0][k].shape).data.numpy().tolist()[1:]
                sample_params.update({
                    k: torch.stack([
                        torch.cat([
                            sample[k],
                            torch.zeros([max_length - len(sample[k])] + pad_shape, dtype=sample[k].dtype)
                        ], dim=0) for sample in samples
                    ])
                })
            else:
                sample_params.update({
                    k: torch.stack([
                        sample[k] for sample in samples
                    ])
                })
        return sample_params


class CachedDatasetCollection(DatasetCollection):
    def __init__(self, dataset):
        super().__init__(dataset)
        self._dataset = dataset

    def __getitem__(self, idx):
        return self._dataset.get_room_params(idx)

    @property
    def bbox_dims(self):
        return self._dataset.bbox_dims


class CachedDatasetCollectionPartNet(DatasetCollection):
    def __init__(self, dataset):
        super().__init__(dataset)
        self._dataset = dataset

    def __getitem__(self, idx):
        return self._dataset[idx]

    @property
    def bbox_dims(self):
        return self._dataset.bbox_dims


class RotationAugmentation(DatasetDecoratorBase):
    def __init__(self, dataset, min_rad=0, max_rad=6.28319):
        super().__init__(dataset)
        self._min_rad = min_rad
        self._max_rad = max_rad

    def uniform_random_rotation(self):
        return Quaternion.random().rotation_matrix

    def __getitem__(self, idx):
        sample_params = self._dataset[idx]
        random_rotation = self.uniform_random_rotation()

        for v in ['part_boxes', 'root_box']:
            part_poses = sample_params[v][..., 6:]
            part_rotation_matrix = transforms.quaternion_to_matrix(part_poses)
            part_rotation_matrix @= random_rotation

            sample_params[v][..., 6:] = \
                transforms.matrix_to_quaternion(part_rotation_matrix)
       
        # Rotate points will result in points being out of bound [-0.5, 0.5]
        # How do we solve this issue? Scale down both the chair and the point cloud?
        # sample_params['point_clouds'] @= random_rotation.T
       
        return sample_params


class Jitter(DatasetDecoratorBase):
    def __init__(self, dataset):
        super().__init__(dataset)

    def __getitem__(self, idx):
        sample_params = self._dataset[idx]

        x = np.random.normal(-0.01, 0.01)
        y = np.random.normal(-0.01, 0.01)
        z = np.random.normal(-0.01, 0.01)
        # All parts need to move in the same direction
        sample_params['part_boxes'][..., 0] += x
        sample_params['part_boxes'][..., 1] += y
        sample_params['part_boxes'][..., 2] += z

        # All parts need to move in the same direction
        sample_params['root_box'][..., 0] += x
        sample_params['root_box'][..., 1] += y
        sample_params['root_box'][..., 2] += z

        return sample_params


class Scale(DatasetDecoratorBase):
    def __init__(self, dataset):
        super().__init__(dataset)    

    @staticmethod
    def scale(x, minimum, maximum):
        # X = x.astype(np.float32)
        X = x
        X = np.clip(X, minimum, maximum)
        X = ((X - minimum) / (maximum - minimum))
        X = 2 * X - 1
        return X.float()

    @staticmethod
    def descale(x, minimum, maximum):
        x = (x + 1) / 2
        x = x * (maximum - minimum) + minimum
        return x

    def __getitem__(self, idx):
        bounds = self.bounds
        sample_params = self._dataset[idx]

        canvas = {}
        for k, v in sample_params.items():
            if k in bounds:
                # we don't scale quat
                quat = sample_params[k][..., 6:]
                canvas[k + "_ori"] = v  
                sample_params[k] = Scale.scale(
                    v[..., :6], bounds[k][0], bounds[k][1]
                )
                sample_params[k] = torch.cat((sample_params[k], quat), dim=-1)
        sample_params.update(canvas)          
        return sample_params

    def post_process(self, sample_params):
        bounds = self.bounds
        for k, v in sample_params.items():
            if k in ['input_boxes']:
                quat = sample_params[k][..., 6:]
                sample_params[k] = Scale.descale(
                    v[..., :6], bounds['part_boxes'][0], bounds['part_boxes'][1]
                )
                sample_params[k] = torch.cat((sample_params[k], quat), dim=-1)
        return super().post_process(sample_params)

    def collate_fn(self, samples):
        return DatasetCollection.collate_fn(samples)

    @property
    def bbox_dims(self):
        return 3 + 3 + 1

    @property
    def test(self):
        try:
            return self.testing
        except AttributeError:
            return False

    @test.setter
    def test(self, value):
        self.testing = value

    @property
    def sample_strategy(self):
        return self._dataset.sample_strategy

    @sample_strategy.setter
    def sample_strategy(self, value):
        self._dataset.sample_strategy = value


class Permutation(DatasetDecoratorBase):
    def __init__(self, dataset, permutation_keys, permutation_axis=0):
        super().__init__(dataset)
        self._permutation_keys = permutation_keys
        self._permutation_axis = permutation_axis

    def __getitem__(self, idx):
        sample_params = self._dataset[idx]

        shapes = sample_params["class_labels"].shape
        ordering = np.random.permutation(shapes[self._permutation_axis])

        for k in self._permutation_keys:
            sample_params[k] = sample_params[k][ordering]
        return sample_params


class OrderedDataset(DatasetDecoratorBase):
    def __init__(self, dataset, ordered_keys, box_ordering=None):
        super().__init__(dataset)
        self._ordered_keys = ordered_keys
        self._box_ordering = box_ordering

    def __getitem__(self, idx):
        if self._box_ordering is None:
            return self._dataset[idx]

        if self._box_ordering != "class_frequencies":
            raise NotImplementedError()

        sample = self._dataset[idx]
        order = self._get_class_frequency_order(sample)
        for k in self._ordered_keys:
            sample[k] = sample[k][order]
        return sample

    def _get_class_frequency_order(self, sample):
        t = sample["translations"]
        c = sample["class_labels"].argmax(-1)
        class_frequencies = self.class_frequencies
        class_labels = self.class_labels
        f = np.array([
            [class_frequencies[class_labels[ci]]]
            for ci in c
        ])

        return np.lexsort(np.hstack([t, f]).T)[::-1]


class Autoregressive(DatasetDecoratorBase):
    def __getitem__(self, idx):
        sample_params = self._dataset[idx]

        sample_params_target = {}
        # Compute the target from the input
        for k, v in sample_params.items():
            if k == "room_layout" or k == "length":
                pass
            elif k == "class_labels":
                class_labels = np.copy(v)
                L, C = class_labels.shape
                # Add the end label the end of each sequence
                end_label = np.eye(C)[-1]
                sample_params_target[k+"_tr"] = np.vstack([
                    class_labels, end_label
                ])
            else:
                p = np.copy(v)
                # Set the attributes to for the end symbol
                _, C = p.shape
                sample_params_target[k+"_tr"] = np.vstack([p, np.zeros(C)])

        sample_params.update(sample_params_target)

        # Add the number of bounding boxes in the scene
        sample_params["length"] = sample_params["class_labels"].shape[0]

        return sample_params

    def collate_fn(self, samples):
        return DatasetCollection.collate_fn(samples)

    @property
    def bbox_dims(self):
        return 7


class AutoregressiveWOCM(Autoregressive):
    def __getitem__(self, idx):
        sample_params = super().__getitem__(idx)

        # Split the boxes and generate input sequences and target boxes
        L, C = sample_params["class_labels"].shape
        n_boxes = np.random.randint(0, L+1)

        for k, v in sample_params.items():
            if k == "room_layout" or k == "length":
                pass
            else:
                if "_tr" in k:
                    sample_params[k] = v[n_boxes]
                else:
                    sample_params[k] = v[:n_boxes]
        sample_params["length"] = n_boxes

        return sample_params


class PartNetAutoregressive(DatasetDecoratorBase):
    def __init__(self, dataset):
        super().__init__(dataset)
        self.sem2num = pickle.load(open(self.label_dict, 'rb'))
        self.n_class = len(self.sem2num.keys())        
        all_kmedoids = []
        for param in ['tran', 'size', 'pose_6D']:
            kmedoids = KMedoids()
            kmedoids.cluster_centers_ = np.load(self.bin_centers.format(param))
            all_kmedoids.append(kmedoids)
        
        self.kmedoids_tran, self.kmedoids_size, self.kmedoids_pose = all_kmedoids
        self.n_bin_centers = all_kmedoids[0].cluster_centers_.shape[0]


    def get_tran_bins(self, part_tran):
        bins = []
        bins_one_hot = torch.zeros(part_tran.shape[0], self.n_bin_centers)
        bins = self.kmedoids_tran.predict(part_tran.numpy())
        bins_one_hot[torch.arange(len(bins)), bins] = 1
        return torch.tensor(bins, dtype=torch.int64), bins_one_hot


    def get_size_bins(self, part_size):
        bins = []
        bins_one_hot = torch.zeros(part_size.shape[0], self.n_bin_centers)
        bins = self.kmedoids_size.predict(part_size.numpy())
        bins_one_hot[torch.arange(len(bins)), bins] = 1
        return torch.tensor(bins, dtype=torch.int64), bins_one_hot


    def get_pose_bins(self, part_poses):

        if True: #TODO: add 6D
            part_poses = transforms.quaternion_to_matrix(part_poses)
            part_poses = transforms.matrix_to_rotation_6d(part_poses)
        bins = []
        bins_one_hot = torch.zeros(part_poses.shape[0], self.n_bin_centers)
        bins = self.kmedoids_pose.predict(part_poses.numpy())
        bins_one_hot[torch.arange(len(bins)), bins] = 1
        return torch.tensor(bins, dtype=torch.int64), bins_one_hot


    def __getitem__(self, idx):
        tree, id = self._dataset[idx]
        part_boxes, _, _, _, part_sems = tree.graph(leafs_only=True)
        part_boxes = torch.stack(part_boxes)

        part_tran_bins, part_size_bins, part_pose_bins = [], [], []
        part_tran_bins_onehot, part_size_bins_onehot, part_pose_bins_onehot = [], [], []
        
        part_tran_bins, part_tran_bins_onehot = self.get_tran_bins(part_boxes[..., :3])
        part_size_bins, part_size_bins_onehot = self.get_size_bins(part_boxes[..., 3:6])
        part_pose_bins, part_pose_bins_onehot = self.get_pose_bins(part_boxes[..., 6:])


        root_box = tree.root.get_box_quat().squeeze()

        clip_feature = torch.tensor(tree.clip_feature) if self.load_clip else torch.tensor([])

        # sample only part of the point cloud for training the decoder
        point_clouds = torch.tensor(tree.point_clouds) if self.load_points else torch.tensor([])
        point_label = torch.tensor(tree.point_label.astype("int")) if self.load_points else torch.tensor([])

        # process raw semantics to numercial labels
        part_class = torch.zeros(len(part_sems), self.n_class)
        part_label = []

        # One-hot encode part label 
        for idx, part_sem in enumerate(part_sems):
            part_class[idx][self.sem2num[part_sem]] = 1
            part_label.append(self.sem2num[part_sem])
        
        part_label = torch.tensor(part_label)

        return dict(
            obj_id=torch.tensor(int(id)),
            part_boxes=part_boxes,
            part_tran_bins=part_tran_bins,
            part_tran_bins_onehot=part_tran_bins_onehot,
            part_size_bins=part_size_bins,
            part_size_bins_onehot=part_size_bins_onehot,
            part_pose_bins=part_pose_bins,
            part_pose_bins_onehot=part_pose_bins_onehot,
            root_box=root_box,
            clip_feature=clip_feature,
            part_class=part_class,
            part_label=part_label,
            point_clouds=point_clouds,
            point_label=point_label,
        )

    def post_process(self, s):
        return s


class Sampler(DatasetDecoratorBase):
    def __init__(self, dataset, sample_strategy, box_ordering, num_points_to_sample):
        super().__init__(dataset)
        self.sample_strategy = sample_strategy
        self.num_points_to_sample = num_points_to_sample
        self.box_ordering = box_ordering
        self.test = False


    def get_random_idxs(self, sequence_idx, num_seen=-1, sample_strategy='one_target'):
        # idxs = np.random.permutation(length)
        length = len(sequence_idx)

        generator = np.random.default_rng()
        num_seen = generator.integers(low=0, high=length + 1) if num_seen < 0 else num_seen
        if sample_strategy == 'gen_from_scratch':
            return [], sequence_idx
        elif sample_strategy == 'gt_bbox' or num_seen == length:
            return sequence_idx, []
        elif sample_strategy == 'one_target':
            return sequence_idx[:num_seen], sequence_idx[num_seen:num_seen+1]
        elif sample_strategy == 'mask_learning':
            mask = np.ones(sequence_idx.shape, dtype=bool)
            mask[num_seen] = 0
            return sequence_idx[mask], sequence_idx[num_seen:num_seen + 1]
        else:
            return sequence_idx[:num_seen], sequence_idx[num_seen:]


    def get_point_weights(self, point_labels, N):
        n_positive = point_labels.sum()
        n_negative = len(point_labels) - n_positive
        p = n_negative * point_labels + n_positive * (1-point_labels)
        p = p.double()
        p /= p.sum()
        idxs = np.random.choice(len(point_labels), N, p=p)

        weights = 1 - point_labels * 3/4 

        return idxs, weights[idxs] #1/len(point_labels) / p[idxs]


    def get_point_idx(self, length, num_points_to_sample, point_labels, rebalance=True):
        if rebalance:
            positive_idx = np.arange(length)[point_labels == 1]
            negative_idx = np.arange(length)[point_labels == 0]
            sampled_positive_idx = np.random.default_rng().choice(positive_idx, num_points_to_sample // 2)
            sampled_negative_idx = np.random.default_rng().choice(negative_idx, num_points_to_sample // 2)

            return np.concatenate([sampled_positive_idx, sampled_negative_idx])
        else:
            return np.random.default_rng().choice(np.arange(length), num_points_to_sample)


    def __gettest__(self, idx):
        sample_params = self._dataset[idx]
        part_boxes = sample_params['part_boxes']
        part_class = sample_params['part_class']
        part_label = sample_params['part_label']
        point_clouds = sample_params['point_clouds']
        point_label = sample_params['point_label']
        part_tran_bins = sample_params['part_tran_bins']
        part_tran_bins_onehot = sample_params['part_tran_bins_onehot']
        part_size_bins = sample_params['part_size_bins']
        part_size_bins_onehot = sample_params['part_size_bins_onehot']
        part_pose_bins = sample_params['part_pose_bins']
        part_pose_bins_onehot = sample_params['part_pose_bins_onehot']

        if self.box_ordering == 'class_level_random':
            sequence_idx = []
            intermediate_idx = [0]
            prev_label = part_label[0] 
            for idx, label in zip(torch.arange(1, len(part_label)), part_label[1:]):
                if label != prev_label:
                    sequence_idx.extend(np.random.permutation(intermediate_idx))
                    intermediate_idx = [idx]
                else:
                    intermediate_idx.append(idx)
                prev_label = label
            sequence_idx.extend(np.random.permutation(intermediate_idx))
            sequence_idx = torch.tensor(sequence_idx).long()
        else:
            sequence_idx = torch.arange(len(part_label)).long()

        num_seen = -1
        # generate random permutation
        seen_idxs, target_idxs = self.get_random_idxs(sequence_idx, 
            sample_strategy=self.sample_strategy, num_seen=num_seen)
        # seen_idxs, target_idxs = self.get_idxs_by_part(len(part_boxes), 3, part_label)

        part_boxes_input = part_boxes[seen_idxs]
        part_boxes_target = part_boxes[target_idxs]
        target_label = part_label[target_idxs]
        target_class = part_class[target_idxs]
        part_tran_bins_target = part_tran_bins[target_idxs]
        part_tran_bins_onehot_target = part_tran_bins_onehot[target_idxs]
        part_size_bins_target = part_size_bins[target_idxs]
        part_size_bins_onehot_target = part_size_bins_onehot[target_idxs]
        part_pose_bins_target = part_pose_bins[target_idxs]
        part_pose_bins_onehot_target = part_pose_bins_onehot[target_idxs]

        # Create Dummy values to avoid NaN during forward pass
        if target_idxs == []:
            target_label = torch.tensor([0])
            target_class = torch.zeros(1, part_class.shape[-1])
            target_class[0, 0] = 1
            part_boxes_target = torch.zeros(1, part_boxes.shape[-1])
            part_boxes_target[0, -1] = 1 # unit quaternion
            part_tran_bins_target = torch.tensor([0])
            part_tran_bins_onehot_target = torch.zeros(1, part_pose_bins_onehot.shape[-1])
            part_size_bins_target = torch.tensor([0])
            part_size_bins_onehot_target = torch.zeros(1, part_pose_bins_onehot.shape[-1])
            part_pose_bins_target = torch.tensor([0])
            part_pose_bins_onehot_target = torch.zeros(1, part_pose_bins_onehot.shape[-1])
            target_idxs = torch.tensor([])

        point_weights = torch.tensor([])
        if self.load_points:
            point_idx, point_weights = self.get_point_weights(point_label, self.num_points_to_sample)
            point_clouds = point_clouds[point_idx]
            point_label = point_label[point_idx]

        # the difference between target_class and target_label is one is one-hot and the other one is a single number
        sample = dict(
            obj_id = sample_params['obj_id'],
            part_boxes_ori = sample_params['part_boxes_ori'],
            input_boxes = part_boxes_input.float(),
            target_boxes = part_boxes_target.float(),
            input_class = part_class[seen_idxs],
            target_class = target_class,
            target_label = target_label,
            root_box_ori = sample_params['root_box_ori'],            
            root_box = sample_params['root_box'],
            target_tran_bins = part_tran_bins_target,
            target_tran_bins_oh = part_tran_bins_onehot_target.float(),
            target_size_bins = part_size_bins_target,
            target_size_bins_oh = part_size_bins_onehot_target.float(),                        
            target_pose_bins = part_pose_bins_target,
            target_pose_bins_oh = part_pose_bins_onehot_target.float(),
            point_clouds = point_clouds, 
            point_label = point_label,
            point_weights = point_weights,
            seen_idxs = torch.tensor(seen_idxs),
            target_idxs = torch.tensor(target_idxs),
            lengths = torch.tensor(len(seen_idxs)),
            unseen_lengths = torch.tensor(len(target_idxs)),
            clip_feature = sample_params['clip_feature'],
        )

        return sample


    def __getitem__(self, idx):
        sample_param = self.__gettest__(idx)
        if not self.test:
            sample_param.pop("seen_idxs", None)
            sample_param.pop("target_idxs", None)
        return sample_param        


    def post_process(self, s):
        return self._dataset.post_process(s)


    def collate_fn(self, samples):
        return DatasetCollection.collate_fn(samples)


def dataset_encoding_factory(
    name,
    dataset,
    augmentations=None,
    box_ordering=None,
    sample_strategy="mask_learning",
    num_points_to_sample=2400,
):
    # NOTE: The ordering might change after augmentations so really it should
    #       be done after the augmentations. For class frequencies it is fine
    #       though.

    dataset_collection = PartNetAutoregressive(dataset)

    if isinstance(augmentations, list):
        for aug_type in augmentations:
            if aug_type == "rotations":
                print("Applying rotation augmentations")
                dataset_collection = RotationAugmentation(dataset_collection)
            elif aug_type == "jitter":
                pass
                print("Applying jittering augmentations")
                dataset_collection = Jitter(dataset_collection)


    # Scale the input
    dataset_collection = Scale(dataset_collection)

    # Final formatter
    dataset_collection = Sampler(dataset_collection, sample_strategy, box_ordering, num_points_to_sample)
    
    return dataset_collection

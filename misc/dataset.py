import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import Compose
from torchvision.transforms import Normalize as TorchNormalize


class BaseDataset(Dataset):
    """Dataset is a wrapper for all datasets we have
    """
    def __init__(self, dataset_object, transform=None):
        """
        Arguments:
        ---------
            dataset_object: a dataset object that can be an object of type
                            model_collections.ModelCollection
            transform: Callable that applies a transform to a sample
        """
        self._dataset_object = dataset_object
        self._transform = transform

    def __len__(self):
        return len(self._dataset_object)

    def __getitem__(self, idx):
        datapoint = self._get_item_inner(idx)
        if self._transform:
            for key in datapoint.keys():
                datapoint[key] = self._transform(datapoint[key])

        return datapoint

    def _get_item_inner(self, idx):
        raise NotImplementedError()

    @property
    def internal_dataset_object(self):
        return self._dataset_object


class VoxelInput(BaseDataset):
    """Get a voxel-based representation of a mesh."""
    def _get_item_inner(self, idx):
        return {"voxels": self._dataset_object[idx].voxel_grid}


class ImageInput(BaseDataset):
    """Get a random image of a mesh."""
    def _get_item_inner(self, idx):
        try:
            return {"images": self._dataset_object[idx].random_image}
        except:
            return {"images": self._dataset_object[idx][0]}


class PointsOnMesh(BaseDataset):
    """Get random points on the surface of a mesh."""
    def _get_item_inner(self, idx):
        return {"points_on_mesh": self._dataset_object[idx].sample_faces()}


class PointsAndLabels(BaseDataset):
    """Get random points in the bbox and label them inside or outside."""
    def _get_item_inner(self, idx):
        points, labels, weights = self._dataset_object[idx].sample_points()
        return {"points": points, "labels": labels, "weights": weights}


class PointsAndSignedDistances(BaseDataset):
    """Get random points in the bbox and their signed distances."""
    def _get_item_inner(self, idx):
        points, sdfs, weights = \
            self._dataset_object[idx].sample_points_with_signed_distances()

        return {"points": points, "sdfs": sdfs, "weights": weights}


class DatasetCollection(BaseDataset):
    """Implement a pytorch Dataset with a list of BaseDataset objects."""
    def __init__(self, *datasets):
        super(DatasetCollection, self).__init__(
            datasets[0]._dataset_object,
            None
        )
        self._datasets = datasets

    def _get_item_inner(self, idx):
        sample = {}
        for dataset in self._datasets:
            ddd = dataset[idx]
            sample.update({k: ddd[k] for k in ddd.keys()})
        return sample


class ToTensor:
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, x):
        return torch.tensor(x).float()


def dataset_factory(dataset_name, dataset_object):
    image_to_tensor = transforms.ToTensor()
    to_tensor = ToTensor()
    on_surface = PointsOnMesh(dataset_object, transform=to_tensor)
    in_bbox = PointsAndLabels(dataset_object, transform=to_tensor)
    signed_distanes = PointsAndSignedDistances(
        dataset_object, transform=to_tensor
    )
    image_input = ImageInput(
        dataset_object,
        transform=Compose([
            image_to_tensor,
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    )
    cifar_image_input = ImageInput(
        dataset_object,
        transform=transforms.Compose([
            image_to_tensor,
            transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
        ])
    )
    voxelized = VoxelInput(dataset_object, to_tensor)

    return {
        "ImageAutoEncodingDataset": DatasetCollection(cifar_image_input),
        "ImageDataset": DatasetCollection(image_input, on_surface),
        "ImageDatasetWithOccupancyLabels": DatasetCollection(
            image_input, in_bbox
        ),
        "ImageDatasetForChamferAndIOU": DatasetCollection(
            image_input, in_bbox, on_surface
        ),
        "ImageDatasetWithSignedDistances": DatasetCollection(
            image_input, signed_distanes
        ),
        "ImageDatasetForChamferAndSignedDistances": DatasetCollection(
            image_input, signed_distanes, on_surface
        ),
        "VoxelAutoEncodingDataset": DatasetCollection(voxelized),
        "VoxelDatasetWithOccupancyLabels": DatasetCollection(
            voxelized, in_bbox
        ),
        "VoxelDatasetForChamferAndIOU": DatasetCollection(
            voxelized, in_bbox, on_surface
        ),
        "PointsAutoEncodingDataset": DatasetCollection(on_surface),
        "ChamferAndIOU": DatasetCollection(on_surface, in_bbox)
    }[dataset_name]

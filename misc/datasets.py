import torchvision.datasets as datasets
from torch.utils.data import DataLoader

from dataset import dataset_factory
from model_collections import ModelCollectionBuilder
from splits_builder import ShapeNetSplitsBuilder, DynamicFaustSplitsBuilder, \
    CSVSplitsBuilder, MultiFilesSplitsBuilder


def splits_factory(dataset_type):
    return {
        "dynamic_faust": DynamicFaustSplitsBuilder,
        "shapenet_v1": ShapeNetSplitsBuilder,
        "freihand": CSVSplitsBuilder,
        "turbosquid_animal": CSVSplitsBuilder,
        "partnet": MultiFilesSplitsBuilder
    }[dataset_type]


def build_dataset(
    config,
    model_tags,
    category_tags,
    keep_splits,
    random_subset=1.0,
    cache_size=0
):
    # Create a dataset instance to generate the samples for training
    dataset_type = config["data"]["dataset_type"]
    dataset_directory = config["data"]["dataset_directory"]
    train_test_splits_file = config["data"]["splits_file"]
    return dataset_factory(
        config["data"]["dataset_factory"],
        (ModelCollectionBuilder(config)
            .with_dataset(dataset_type)
            .filter_train_test(
                splits_factory(dataset_type)(train_test_splits_file),
                keep_splits
                )
            .filter_category_tags(category_tags)
            .filter_tags(model_tags)
            .random_subset(random_subset)
            .lru_cache(cache_size)
            .build(dataset_directory))
    )


def build_dataloader(
    config,
    model_tags,
    category_tags,
    split,
    batch_size,
    n_processes=0.0,
    random_subset=1.0,
    cache_size=0,
    shuffle=True
):
    # Create a dataset instance to generate the samples for training
    dataset = build_dataset(
        config,
        model_tags,
        category_tags,
        split,
        random_subset=random_subset,
        cache_size=cache_size,
    )
    print("Dataset has {} elements".format(len(dataset)))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=n_processes,
        shuffle=shuffle
    )


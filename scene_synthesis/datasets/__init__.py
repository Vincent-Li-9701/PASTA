from .threed_front_dataset import dataset_encoding_factory
from .partnet import PartNetDataset

import pickle


def get_raw_dataset(
    config,
    filter_fn=lambda s: s,
    split=["train", "val"]
):
    if config['box_bounds'] is not None:
        with open(config['box_bounds'], 'rb') as f:
            bounds = pickle.load(f)
    else:
        bounds = None

    # Make PartNet dataset
    assert len(split) == 1

    data_features = ['object', 'name'] 
    split_file = split[0]
    if config["load_points"]:
        split_file += '_points_tsdf'
    elif config["load_clip"]:
        split_file += '_clip_truncated'
    split_file += '.txt'

    dataset = PartNetDataset(
        config["dataset_directory"],
        config["label_dict"],
        config["bin_centers"],
        split_file,
        data_features,
        load_geo=config["load_geo"],
        load_points=config["load_points"],
        load_clip=config["load_clip"],
        bounds=bounds
    )
    return dataset


def get_dataset_raw_and_encoded(
    config,
    filter_fn=lambda s: s,
    augmentations=None,
    split=["train", "val"]
):
    dataset = get_raw_dataset(config, filter_fn, split=split)
    encoding = dataset_encoding_factory(
        config.get("encoding_type"),
        dataset,
        augmentations,
        config.get("box_ordering", None),
        config['sample_strategy'],
        config['num_points_to_sample']
    )

    return dataset, encoding


def get_encoded_dataset(
    config,
    filter_fn=lambda s: s,
    augmentations=None,
    split=["train", "val"]
):
    _, encoding = get_dataset_raw_and_encoded(
        config, filter_fn, augmentations, split
    )
    return encoding


def filter_function(config, split=["train", "val"], without_lamps=False):
    print("Applying {} filtering".format(config["filter_fn"]))

    if config["filter_fn"] == "no_filtering":
        return lambda s: s

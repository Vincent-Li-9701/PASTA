"""Script used to train a ATISS."""
import argparse
import logging
import os
import sys

import numpy as np

import torch
from torch.utils.data import DataLoader
from training_utils import id_generator, save_experiment_params, load_config

import sys

sys.path.append(os.getcwd())

from scene_synthesis.datasets import get_encoded_dataset, filter_function
from scene_synthesis.networks import build_network, optimizer_factory, scheduler_factory
from scene_synthesis.stats_logger import StatsLogger, WandB

torch.set_num_threads(10)
torch.autograd.set_detect_anomaly(True)

def yield_forever(iterator):
    while True:
        for x in iterator:
            yield x


def load_checkpoints(model, optimizer, experiment_directory, args, device, ckpt_prefix=['model', 'opt']):
    model_files = [
        f for f in os.listdir(experiment_directory)
        if f.startswith(f"{ckpt_prefix[0]}_")
    ]
    if len(model_files) == 0:
        return
    ids = [int(f[f.rfind('_')+1:]) for f in model_files]
    max_id = max(ids)
    model_path = os.path.join(
        experiment_directory, "{}_{:05d}"
    ).format(ckpt_prefix[0], max_id)
    opt_path = os.path.join(
        experiment_directory, "{}_{:05d}"
    ).format(ckpt_prefix[1], max_id)
    if not (os.path.exists(model_path) and os.path.exists(opt_path)):
        return

    print("Loading model checkpoint from {}".format(model_path))
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("Loading optimizer checkpoint from {}".format(opt_path))
    optimizer.load_state_dict(
        torch.load(opt_path, map_location=device)
    )
    args.continue_from_epoch = max_id+1


def save_checkpoints(epoch, model, optimizer, experiment_directory, ckpt_prefix=['model', 'opt']):
    torch.save(
        model.state_dict(),
        os.path.join(experiment_directory, "{}_{:05d}").format(ckpt_prefix[0], epoch)
    )
    torch.save(
        optimizer.state_dict(),
        os.path.join(experiment_directory, "{}_{:05d}").format(ckpt_prefix[1], epoch)
    )

def schedule_training_strategy(epoch, train_loader):
    train_loader.dataset.sample_strategy

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
        "--weight_file",
        default=None,
        help=("The path to a previously trained model to continue"
              " the training from")
    )
    parser.add_argument(
        "--decoder_weight_file",
        default=None,
        help=("The path to a previously trained decoder model to continue"
              " the training from")
    )
    parser.add_argument(
        "--continue_from_epoch",
        default=0,
        type=int,
        help="Continue training from epoch (default=0)"
    )
    parser.add_argument(
        "--n_processes",
        type=int,
        default=0,
        help="The number of processed spawned by the batch provider"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=27,
        help="Seed for the PRNG"
    )
    parser.add_argument(
        "--experiment_tag",
        default=None,
        help="Tag that refers to the current experiment"
    )
    parser.add_argument(
        "--with_wandb_logger",
        action="store_true",
        help="Use wandB for logging the training progress"
    )
    parser.add_argument(
        "--train_box_generator",
        action="store_true",
        help="Train point decoder"
    )
    parser.add_argument(
        "--train_point_decoder",
        action="store_true",
        help="Train point decoder"
    )


    args = parser.parse_args(argv)

    # Disable trimesh's logger
    logging.getLogger("trimesh").setLevel(logging.ERROR)

    # Set the random seed
    np.random.seed(args.seed)
    torch.manual_seed(np.random.randint(np.iinfo(np.int32).max))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(np.random.randint(np.iinfo(np.int32).max))

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print("Running code on", device)

    # Check if output directory exists and if it doesn't create it
    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    # Create an experiment directory using the experiment_tag
    if args.experiment_tag is None:
        experiment_tag = id_generator(9)
    else:
        experiment_tag = args.experiment_tag

    experiment_directory = os.path.join(
        args.output_directory,
        experiment_tag
    )
    if not os.path.exists(experiment_directory):
        os.makedirs(experiment_directory)

    # Save the parameters of this run to a file
    save_experiment_params(args, experiment_tag, experiment_directory)
    print("Save experiment statistics in {}".format(experiment_directory))

    # Parse the config file
    config = load_config(args.config_file)

    train_dataset = get_encoded_dataset(
        config["data"],
        filter_function(
            config["data"],
            split=config["training"].get("splits", ["train", "val"])
        ),
        augmentations=config["data"].get("augmentations", None),
        split=config["training"].get("splits", ["train", "val"])
    )

    print("Training data size", len(train_dataset))
    # Compute the bounds for this experiment, save them to a file in the
    # experiment directory and pass them to the validation dataset

    validation_dataset = get_encoded_dataset(
        config["data"],
        filter_function(
            config["data"],
            split=config["validation"].get("splits", ["test"])
        ),
        augmentations=None,
        split=config["validation"].get("splits", ["test"])
    )
    print("Validation data size", len(validation_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"].get("batch_size", 128),
        num_workers=args.n_processes,
        collate_fn=train_dataset.collate_fn,
        pin_memory=True,
        shuffle=True
    )

    val_loader = DataLoader(
        validation_dataset,
        batch_size=config["validation"].get("batch_size", 1),
        num_workers=args.n_processes,
        collate_fn=validation_dataset.collate_fn,
        pin_memory=True,        
        shuffle=False
    )

    # Build the network architecture to be used for training
    network, train_on_batch, validate_on_batch = build_network(
        256, config["generator"].get("n_class"),
        config["generator"], args.weight_file, device=device
    )

    # Build an optimizer object to compute the gradients of the parameters
    optimizer = optimizer_factory(config["training"], network.parameters())
    # Load the checkpoints if they exist in the experiment directory
    load_checkpoints(network, optimizer, experiment_directory, args, device)

    scheduler = None
    if config['training'].get("scheduler", None):
        print("Learning Rate scheduler is ", config['training'].get("scheduler", None))
        scheduler = scheduler_factory(config["training"], optimizer)

    if args.train_point_decoder:
        # load encoder network weights
        encoder_network, _, _ = build_network(
            256, config["encoder"].get("n_class"),
            config["encoder"], args.weight_file, device=device
        )
        encoder_optimizer = optimizer_factory(config["training"], encoder_network.parameters())
        load_checkpoints(encoder_network, encoder_optimizer, experiment_directory, \
            args, device)

        encoder_scheduler = None
        if config['training'].get("scheduler", None):
            encoder_scheduler = scheduler_factory(config["training"], encoder_optimizer)

        # load decoder network weights
        decoder_network, train_on_batch_decoder, validate_on_batch_decoder = build_network(
            256, config["decoder"].get("n_class"),
            config["decoder"], args.decoder_weight_file, device=device
        )
        decoder_optimizer = optimizer_factory(config["training"], decoder_network.parameters())
        load_checkpoints(decoder_network, decoder_optimizer, experiment_directory, \
            args, device, ['decoder', 'decoder_opt'])
        
        decoder_scheduler = None
        if config['training'].get("scheduler", None):
            decoder_scheduler = scheduler_factory(config["training"], decoder_optimizer)
        
        models = {'encoder': encoder_network, 'decoder': decoder_network, 'generator': network}
        optimizers = {'encoder': encoder_optimizer, 'decoder': decoder_optimizer}


    # Initialize the logger
    if args.with_wandb_logger:
        WandB.instance().init(
            config,
            model=network,
            project=config["logger"].get(
                "project", "part-gen"
            ),
            name=experiment_tag,
            watch=False,
            log_frequency=10,
            dir="./wandb"
        )

    # Log the stats to a file
    StatsLogger.instance().add_output_file(open(
        os.path.join(experiment_directory, "stats.txt"),
        "w"
    ))

    epochs = config["training"].get("epochs", 150)
    steps_per_epoch = config["training"].get("steps_per_epoch", 500)
    save_every = config["training"].get("save_frequency", 10)
    val_every = config["validation"].get("frequency", 100)
    use_ss = config['training'].get('use_scheduled_sampling', False)

    # Do the training
    for i in range(args.continue_from_epoch, epochs):
        # if we are not training a box generator, turn generator to eval
        if args.train_box_generator:
            network.train()
        else:
            network.eval()
    
        if args.train_point_decoder:
            decoder_network.train()
            encoder_network.train()
    
        for b, sample in zip(range(steps_per_epoch), yield_forever(train_loader)):
            # Move everything to device
            for k, v in sample.items():
                sample[k] = v.to(device)
            loss = 0.0
            if args.train_point_decoder:
                point_loss = train_on_batch_decoder(models, optimizers, sample, config)
                loss += point_loss
                if encoder_scheduler:
                    encoder_scheduler.step(i + b / steps_per_epoch)
                if decoder_scheduler:
                    decoder_scheduler.step(i + b / steps_per_epoch)
            if args.train_box_generator:
                ss = np.random.uniform() < (i / epochs) and use_ss
                batch_loss = train_on_batch(network, optimizer, sample, config, ss)
                loss += batch_loss
                if scheduler:
                    scheduler.step(i + b / steps_per_epoch)
            
            StatsLogger.instance().print_progress(i+1, b+1, loss)
            

        if (i % save_every) == 0:
            if args.train_box_generator:
                save_checkpoints(
                    i,
                    network,
                    optimizer,
                    experiment_directory,
                )
            if args.train_point_decoder:
                save_checkpoints(
                    i,
                    decoder_network,
                    decoder_optimizer,
                    experiment_directory,
                    ['decoder', 'decoder_opt']
                )
                save_checkpoints(
                    i,
                    encoder_network,
                    encoder_optimizer,
                    experiment_directory,
                    ['encoder', 'encoder_opt']
                )

        StatsLogger.instance().clear()

        if i % val_every == 0 and i > 0:
            print("====> Validation Epoch ====>")
            network.eval()
            if args.train_point_decoder:
                decoder_network.eval()
                encoder_network.eval()
    
            for b, sample in enumerate(val_loader):
                # Move everything to device
                for k, v in sample.items():
                    sample[k] = v.to(device)
                loss = 0.0
                if args.train_point_decoder:
                    point_loss, _ = validate_on_batch_decoder(models, sample, \
                        config)
                    loss += point_loss
                if args.train_box_generator:
                    ss = np.random.uniform() < (i / epochs) and use_ss
                    batch_loss, _ = validate_on_batch(network, sample, \
                        config, ss)
                    loss += batch_loss
                
                StatsLogger.instance().print_progress(-1, b+1, loss)
            StatsLogger.instance().clear()
            print("====> Validation Epoch ====>")


if __name__ == "__main__":
    main(sys.argv[1:])

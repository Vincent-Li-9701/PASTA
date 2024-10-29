CUDA_VISIBLE_DEVICES=0 python -W ignore scripts/train_network.py \
    config/partnet_config.yaml \
    ./output \
    --n_processes 5 \
    --train_box_generator \
    --with_wandb_logger #--experiment_tag 67OT9C38V
CUDA_VISIBLE_DEVICES=5 python -W ignore scripts/train_network.py \
    config/partnet_config.table.yaml \
    ./output \
    --n_processes 5 \
    --train_box_generator \
    --with_wandb_logger \
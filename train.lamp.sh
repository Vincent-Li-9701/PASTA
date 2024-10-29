CUDA_VISIBLE_DEVICES=2 python -W ignore scripts/train_network.py \
    config/partnet_config.lamp.yaml \
    ./output \
    --n_processes 5 \
    --train_box_generator \
    --with_wandb_logger #--experiment_tag VGT7F7KJE
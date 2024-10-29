CUDA_VISIBLE_DEVICES=1 python -W ignore scripts/train_network.py config/partnet_config.baseline.yaml ./output --n_processes 5 \
    --train_box_generator --with_wandb_logger --experiment_tag 1ADPM7A3J
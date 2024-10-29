CUDA_VISIBLE_DEVICES=1 python -W ignore scripts/train_network.py \
   config/partnet_config_points.all.yaml \
   ./output \
   --n_processes 5 \
   --with_wandb_logger \
   --train_box_generator \

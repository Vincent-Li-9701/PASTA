CUDA_VISIBLE_DEVICES=1 python scripts/train_network.py config/partnet_config_points.yaml ./output --n_processes 10 \
    --experiment_tag 6R5O050TQ \
    --train_point_decoder \
    --with_wandb_logger \
    --weight_file ./output/WFHPO6I4N/model_00630 
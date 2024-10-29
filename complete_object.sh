CUDA_VISIBLE_DEVICES=5 python -W ignore scripts/scene_completion.py config/partnet_config.yaml \
    record/one_tgt_6D_all_bins_single_head_abs_ps_random_ss_cosslw/all_gen_990_pred \
    --weight_file ./output/N28OAYM8M/model_00990 \
    --synthesize_data \
    --generation_mode gen_from_scratch \
    # --inference_train \
    # --render_part \
    
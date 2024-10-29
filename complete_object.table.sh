CUDA_VISIBLE_DEVICES=5 python -W ignore scripts/scene_completion.py config/partnet_config.table.yaml record/one_tgt_6D_all_bins_single_head_random_ss_cosslw.table/test_gen_990 \
    --weight_file ./output/VP5EX74E7/model_00990 \
    --synthesize_data \
    --generation_mode gen_from_scratch
    # --render_part \
    # --inference_train \
        
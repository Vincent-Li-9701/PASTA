CUDA_VISIBLE_DEVICES=5 python -W ignore scripts/scene_completion.py config/partnet_config.lamp.yaml record/one_tgt_6D_all_bins_single_head_random_ss_cosslw.lamp/test_gen_810 \
    --weight_file ./output/HXS4CLRZ2/model_00810 \
    --synthesize_data \
    --generation_mode gen_from_scratch \
    # --render_part \
    #--inference_train   
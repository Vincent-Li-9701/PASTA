CUDA_VISIBLE_DEVICES=2 python scripts/synthetic_vs_real_classifier.py config/fid_config.yaml \
    --path_to_syn_train record/one_tgt_6D_all_bins_single_head_random_ss_cosslw_class_random/train_630/box_quats \
    --path_to_syn_test record/one_tgt_6D_all_bins_single_head_random_ss_cosslw_class_random/test_630/box_quats
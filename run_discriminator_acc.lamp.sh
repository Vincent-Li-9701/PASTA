CUDA_VISIBLE_DEVICES=7 python scripts/synthetic_vs_real_classifier.py config/fid_config.lamp.yaml \
    --path_to_syn_train record/one_tgt_quat_baseline.lamp/train_630/box_quats \
    --path_to_syn_test record/one_tgt_quat_baseline.lamp/test_630/box_quats
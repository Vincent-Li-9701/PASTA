CUDA_VISIBLE_DEVICES=2 python scripts/synthetic_vs_real_classifier.py config/fid_config.table.yaml \
    --path_to_syn_train record/one_tgt_quat_baseline.table/train_480/box_quats \
    --path_to_syn_test record/one_tgt_quat_baseline.table/test_480/box_quats
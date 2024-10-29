CUDA_VISIBLE_DEVICES=3 python scripts/point_classification.py config/partnet_config.yaml ./record/point_decoder_both_train_reuse_encoder_2 \
    --weight_file  ./output/A6Y094N8P/model_01200 \
    --decoder_weight_file  ./output/A6Y094N8P/decoder_01200
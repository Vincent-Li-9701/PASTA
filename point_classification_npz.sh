CUDA_VISIBLE_DEVICES=7 python -u -W ignore scripts/point_classification_npz.py \
    config/partnet_config_points.yaml \
    config/label_dict.pkl \
    sample/chairs \
    out/ \
    --encoder_weight_file ./output/FISCF8OAG/encoder_00080 \
    --decoder_weight_file ./output/FISCF8OAG/decoder_00080 \
    #--recolor

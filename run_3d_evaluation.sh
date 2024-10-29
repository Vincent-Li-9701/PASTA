CUDA_VISIBLE_DEVICES=9 python misc/evaluate_partnet_generation.py \
    record/one_tgt_6D_all_bins_single_head_random_ss_cosslw/all_gen_810_pred/objs \
    misc/partnet_chairs.yaml \
    --output_directory /afs/cs.stanford.edu/u/hansonlu/remote/ATISS/tmp_all/ \
    --n_fake_samples 2000 --n_real_samples 2000
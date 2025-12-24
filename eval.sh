export CUDA_VISIBLE_DEVICES=7
python eval_rebuild.py \
    --world_size=1 \
    --cfg="outputs/test_models/Vessel/config.yaml" \
    --resume="outputs/test_models/Vessel/best_checkpoint.pth" \
    --vis_dir="outputs/test_models/Vessel/vis_nsp" 

    # --eval_pad="padding_center" \
    # --prob_map_lc='None' \
    # --eval_robust='None' \
    # eval_pad in [padding, padding_center, resize]
    # not finished, do not use: eval_robust in [None, direction, scale]
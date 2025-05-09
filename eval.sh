export CUDA_VISIBLE_DEVICES=7
python eval_rebuild.py \
    --world_size=1 \
    --cfg='outputs/SOY/s_noencoder/config.yaml' \
    --resume="outputs/SOY/s_noencoder/best_checkpoint.pth" \
    --eval_pad="resize" \
    --eval_robust=None \
    --vis_dir="outputs/SOY/s_noencoder/vis_resize" 

    # eval_pad in [padding, padding_center, resize]
    # eval_robust in None, [direction]
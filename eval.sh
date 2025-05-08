export CUDA_VISIBLE_DEVICES=7
python eval_rebuild.py \
    --world_size=1 \
    --cfg='outputs/Car/s_noencoder_split/config.yaml' \
    --resume="outputs/Car/s_noencoder_split/best_checkpoint.pth" \
    --eval_pad="resize" \
    --eval_robust=None \
    --vis_dir="outputs/Car/s_noencoder_split/vis_resize" 

    # eval_pad in [padding, padding_center, resize]
    # eval_robust in None, [direction]
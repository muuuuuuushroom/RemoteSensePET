export CUDA_VISIBLE_DEVICES=3
python eval_rebuild.py \
    --world_size=1 \
    --cfg='outputs/Ship/t_noencoder_hxn/config.yaml' \
    --resume="outputs/Ship/t_noencoder_hxn/best_checkpoint.pth" \
    --eval_pad="padding_center" \
    --eval_robust=None \
    --vis_dir="outputs/Ship/t_noencoder_hxn/vis" 

    # eval_pad in [padding, padding_center, resize]
    # not finished, do not use: eval_robust in [None, direction]
export CUDA_VISIBLE_DEVICES=1
python eval_rebuild.py \
    --world_size=1 \
    --cfg='outputs/Ship/t_noencoder_opre_probonf4x/config.yaml' \
    --resume="outputs/Ship/t_noencoder_opre_probonf4x/best_checkpoint.pth" \
    --eval_pad="padding_center" \
    --eval_robust='None' \
    --vis_dir="outputs/Ship/t_noencoder_opre_probonf4x/vis" 

    # eval_pad in [padding, padding_center, resize]
    # not finished, do not use: eval_robust in [None, direction, scale]
export CUDA_VISIBLE_DEVICES=1
python eval_rebuild.py \
    --world_size=1 \
    --cfg='/data/slcao_data/zlt/rscount/outputs/Ship/tiny_noencoder_attnsplit/config.yaml' \
    --resume="/data/slcao_data/zlt/rscount/outputs/Ship/tiny_noencoder_attnsplit/best_checkpoint.pth" \
    --vis_dir="/data/slcao_data/zlt/rscount/outputs/Ship/tiny_noencoder_attnsplit/vis_pad"
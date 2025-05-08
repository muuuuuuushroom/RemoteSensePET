export CUDA_VISIBLE_DEVICES=2
python eval_rebuild.py \
    --world_size=1 \
    --cfg='/data/slcao_data/zlt/RemoteSensePET/outputs/Car/small_noencoder/tiny_noencoder/config.yaml' \
    --resume="/data/slcao_data/zlt/RemoteSensePET/outputs/Car/small_noencoder/tiny_noencoder/best_checkpoint.pth" \
    --vis_dir="/data/slcao_data/zlt/RemoteSensePET/outputs/Car/small_noencoder/test_vis"

    # /data/slcao_data/zlt/RemoteSensePET/outputs/test_vis
    # /data/slcao_data/zlt/RemoteSensePET/outputs/Car/small_noencoder/tiny_noencoder/vis_pad
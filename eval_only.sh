CUDA_VISIBLE_DEVICES='7' \
python eval_rebuil_ev2.py \
    --world_size=1 \
    --cfg="/data/zlt/RemoteSensePET/outputs/soy_newran/t_base/config.yaml" \
    --resume="/data/zlt/RemoteSensePET/outputs/soy_newran/t_base/best_checkpoint.pth" \
    --eval_pad="padding_center" \
    --prob_map_lc='None' \
    --eval_robust='None' \
    --vis_dir=None
    
    #"/root/PET/eval_data/vis/adapter_s"
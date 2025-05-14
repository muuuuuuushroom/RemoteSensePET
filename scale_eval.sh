#!/bin/bash

LOGFILE="output_nohup/robust_eval.log"

mkdir -p output_nohup

for robust_para in $(awk 'BEGIN{for(i=2.4;i<=3.0;i+=0.1) printf "%.1f ", i}'); do
    echo "Running eval with robust_para=${robust_para}" | tee -a "$LOGFILE"

    CUDA_VISIBLE_DEVICES=3 python eval_rebuild.py \
        --world_size=1 \
        --cfg='outputs/Ship/t_noencoder_attn_opre/config.yaml' \
        --resume="outputs/Ship/t_noencoder_attn_opre/best_checkpoint.pth" \
        --eval_pad="padding_center" \
        --eval_robust="scale" \
        --robust_para="${robust_para}" \
        2>&1 | tee -a "$LOGFILE"

    echo "Finished eval with robust_para=${robust_para}" | tee -a "$LOGFILE"
    echo "----------------------------------------" | tee -a "$LOGFILE"
done


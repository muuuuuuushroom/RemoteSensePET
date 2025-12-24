#!/bin/bash

LOGFILE="output_test/s0_car_scale.log"

mkdir -p output_nohup

for robust_para in $(awk 'BEGIN{for(i=0.2;i<=2.0;i+=0.2) printf "%.1f ", i}'); do
    echo "Running eval with robust_para=${robust_para}" | tee -a "$LOGFILE"

    CUDA_VISIBLE_DEVICES=2 python eval_rebuild.py \
        --world_size=1 \
        --cfg='outputs/RS/Car/base_s0/config.yaml' \
        --resume="outputs/RS/Car/base_s0/best_checkpoint.pth" \
        --eval_pad="padding_center" \
        --eval_robust="scale" \
        --robust_para="${robust_para}" \
        2>&1 | tee -a "$LOGFILE"

    echo "Finished eval with robust_para=${robust_para}" | tee -a "$LOGFILE"
    echo "----------------------------------------" | tee -a "$LOGFILE"
done


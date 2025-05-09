#!/bin/bash
export CUDA_VISIBLE_DEVICES=7
DEFAULT_NPROC_PER_NODE=1
DEFAULT_MASTER_PORT=10007
DEFAULT_CONFIG_FILE_PATH="configs_con/soybeam.yaml"

NPROC_PER_NODE=${2:-$DEFAULT_NPROC_PER_NODE}
MASTER_PORT=${3:-$DEFAULT_MASTER_PORT}
CONFIG_FILE_PATH=${4:-$DEFAULT_CONFIG_FILE_PATH}

echo "nproc_per_node: $NPROC_PER_NODE"
echo "master_port: $MASTER_PORT"
echo "config path: $CONFIG_FILE_PATH"

python -m torch.distributed.launch \
    --nproc_per_node=$NPROC_PER_NODE \
    --master_port=$MASTER_PORT \
    --use_env main_rebuild.py \
    --cfg="$CONFIG_FILE_PATH" 
    # --resume="/data/zlt/PET/RTC/outputs/Car/swin_t_encoder/best_checkpoint.pth"

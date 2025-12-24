# bash shell
#!/bin/bash
trap "trap - SIGINT && kill -- -$$" SIGINT
LOG_DIR="/data/zlt/datasets/Crowd2Fleet_val/eval_logs"
mkdir -p $LOG_DIR
echo "Logs will be saved in $LOG_DIR/"

TASKS=(
    "People:5"
    "Vehicle:6"
    "Vessel:7"
)
BASE_MODEL_PATH="/data/zlt/RemoteSensePET/outputs/test_models"
BASE_VIS_PATH="/data/zlt/datasets/Crowd2Fleet_val/visualiztions_predict"

for task in "${TASKS[@]}"; do

    IFS=':' read -r CATEGORY GPU <<< "$task"
    
    LOG_NAME=${CATEGORY,,}
    LOG_FILE="$LOG_DIR/${LOG_NAME}.log"
    CFG_PATH="${BASE_MODEL_PATH}/${CATEGORY}/config.yaml"
    RESUME_PATH="${BASE_MODEL_PATH}/${CATEGORY}/best_checkpoint.pth"
    VIS_DIR="${BASE_VIS_PATH}/${CATEGORY}"

    echo "Starting $CATEGORY on GPU $GPU... Log: $LOG_FILE"
    nohup env CUDA_VISIBLE_DEVICES=$GPU python eval_rebuild.py \
        --world_size=1 \
        --cfg="$CFG_PATH" \
        --resume="$RESUME_PATH" \
        --vis_dir="$VIS_DIR" \
        > "$LOG_FILE" 2>&1 &
done

echo "All jobs started in background. Waiting for completion..."
wait
echo "All evaluations have finished. Check logs in $LOG_DIR/"
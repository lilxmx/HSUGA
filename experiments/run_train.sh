#!/bin/bash
# Universal training launcher for HSUGA experiments.
# - Activates the hsuga conda environment
# - Forces GPU usage
# - Runs training in background via nohup
# - All outputs (log, tensorboard, checkpoints) go to runs/{run_id}/
#
# Usage:
#   bash experiments/run_train.sh --model_name hsuga_llmesr_gru4rec --dataset beauty [other args...]

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

# Initialize conda
eval "$('/root/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
conda activate hsuga || true

# Parse --model_name and --dataset from args
MODEL_NAME=""
DATASET=""
prev=""
for i in "$@"; do
    case $prev in
        --model_name) MODEL_NAME="$i" ;;
        --dataset) DATASET="$i" ;;
    esac
    prev="$i"
done

if [ -z "$MODEL_NAME" ] || [ -z "$DATASET" ]; then
    echo "[ERROR] --model_name and --dataset are required."
    exit 1
fi

# Ensure --gpu_id is present (default to 0)
HAS_GPU_ID=false
for arg in "$@"; do
    if [ "$arg" = "--gpu_id" ]; then
        HAS_GPU_ID=true
        break
    fi
done

GPU_ARGS=""
if [ "$HAS_GPU_ID" = false ]; then
    GPU_ARGS="--gpu_id 0"
fi

# Generate run_id and pass to python for directory consistency
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_ID="${MODEL_NAME}_${DATASET}_${TIMESTAMP}"
RUN_DIR="./runs/${RUN_ID}"
mkdir -p "$RUN_DIR"

echo "================================================================"
echo "[INFO] Run ID:      $RUN_ID"
echo "[INFO] Run Dir:     $(realpath $RUN_DIR)"
echo "[INFO] Config:      $RUN_DIR/config.yaml"
echo "[INFO] Train log:   $RUN_DIR/train.log"
echo "[INFO] Stdout:      $RUN_DIR/stdout.log"
echo "[INFO] Checkpoints: $RUN_DIR/checkpoints/"
echo "[INFO] TensorBoard: $RUN_DIR/tensorboard/"
echo "================================================================"

nohup python main_legacy.py "$@" $GPU_ARGS --run_id "$RUN_ID" > "${RUN_DIR}/stdout.log" 2>&1 &
PID=$!

echo "[INFO] PID:         $PID"
echo "[INFO] Training started in background."
echo ""
echo "[INFO] Monitor with:"
echo "       tail -f ${RUN_DIR}/stdout.log"
echo "================================================================"

echo "$PID" > "${RUN_DIR}/pid.txt"

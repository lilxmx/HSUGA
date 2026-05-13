#!/bin/bash
# HSUGA + SASRec on Fashion dataset
SEEDS="${SEEDS:-42 43 44}"
GPU_ID="${GPU_ID:-0}"

for seed in ${SEEDS}; do
    echo "Running HSUGA+SASRec on Fashion, seed=${seed}"
    python main.py --config configs/model/hsuga_sasrec.yaml \
        --override \
        dataset.name=fashion \
        training.seed=${seed} \
        training.gpu_id=${GPU_ID} \
        alignment.alpha=0.1 \
        alignment.sim_user_num=6 \
        alignment.sim_long_user_num=6
done

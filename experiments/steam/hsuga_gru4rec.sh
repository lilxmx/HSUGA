#!/bin/bash
# HSUGA + GRU4Rec on Steam dataset
SEEDS="${SEEDS:-42 43 44}"
GPU_ID="${GPU_ID:-0}"

for seed in ${SEEDS}; do
    echo "Running HSUGA+GRU4Rec on Steam, seed=${seed}"
    python main.py --config configs/model/hsuga_gru4rec.yaml \
        --override \
        dataset.name=steam \
        training.seed=${seed} \
        training.gpu_id=${GPU_ID} \
        alignment.alpha=0.1 \
        alignment.sim_user_num=6 \
        alignment.sim_long_user_num=6
done

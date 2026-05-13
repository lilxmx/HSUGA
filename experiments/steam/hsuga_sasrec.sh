#!/bin/bash
# HSUGA + SASRec on Steam dataset (new config-driven format)
#
# Equivalent to the legacy steam_mean_sas_best.bash
#
# Old command:
#   python main.py --dataset steam --model_name llmesr_mean_sasrec --hidden_mode _qwen_mean_llm ...
#
# New command:
#   python main.py --config configs/model/hsuga_sasrec.yaml --override dataset.name=steam training.seed=42

SEEDS="${SEEDS:-42 43 44}"
GPU_ID="${GPU_ID:-0}"

for seed in ${SEEDS}; do
    echo "Running HSUGA+SASRec on Steam, seed=${seed}"
    python main.py --config configs/model/hsuga_sasrec.yaml \
        --override \
        dataset.name=steam \
        training.seed=${seed} \
        training.gpu_id=${GPU_ID} \
        alignment.alpha=0.1 \
        alignment.sim_user_num=6 \
        alignment.sim_long_user_num=6
done

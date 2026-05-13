#!/bin/bash
# HSUGA Experiment Runner
# Usage: bash experiments/run_experiment.sh <config_path> [override1=val1] [override2=val2] ...
#
# Examples:
#   bash experiments/run_experiment.sh configs/model/hsuga_sasrec.yaml dataset.name=steam
#   bash experiments/run_experiment.sh configs/model/hsuga_sasrec.yaml dataset.name=beauty training.seed=42

CONFIG=$1
shift

# Default seeds for reproducibility
SEEDS="${SEEDS:-42 43 44}"

for seed in ${SEEDS}; do
    echo "=========================================="
    echo "Running with seed=${seed}"
    echo "=========================================="
    python main.py --config ${CONFIG} --override training.seed=${seed} "$@"
done

echo "All experiments completed."

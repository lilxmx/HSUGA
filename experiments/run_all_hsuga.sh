#!/bin/bash
# ==============================================================================
# HSUGA Full Reproduction: 3 backbones x 3 datasets x 1 seed = 9 experiments
#
# Components enabled:
#   - Dual-view modeling (ID + LLM embeddings)
#   - GAA (Group-Aware Alignment via user_sim_func=kd)
#   - HSU (Hierarchical Semantic Understanding via use_hsu_fusion)
#
# Hardware: 8x H20 GPU (97GB each)
# Strategy: 9 unique (model, dataset) combos mapped to 8 GPUs.
#           Each combo runs 1 seed on its assigned GPU.
#           All 8 GPU slots launch in parallel.
#
# GPU Mapping:
#   GPU 0: Beauty   + GRU4Rec
#   GPU 1: Beauty   + SASRec
#   GPU 2: Beauty   + BERT4Rec
#   GPU 3: Steam    + GRU4Rec
#   GPU 4: Steam    + SASRec
#   GPU 5: Steam    + BERT4Rec
#   GPU 6: Fashion  + GRU4Rec
#   GPU 7: Fashion  + SASRec  -> then Fashion + BERT4Rec
#
# Usage:
#   nohup bash experiments/run_all_hsuga.sh > experiments/run_all.log 2>&1 &
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

echo "================================================================"
echo " HSUGA Full Reproduction"
echo " Working dir: $(pwd)"
echo " Start time:  $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"

# conda activate is incompatible with set -e, so we activate first
eval "$('/root/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
conda activate hsuga || true
echo " Python: $(which python)"
echo " Env:    $CONDA_DEFAULT_ENV"
echo "================================================================"

set -e

SEED=44

# Run one (model, dataset) combo on a single GPU.
run_combo() {
    local GPU=$1
    shift

    local MODEL_NAME="" DATASET=""
    local prev=""
    for arg in "$@"; do
        case $prev in
            --model_name) MODEL_NAME="$arg" ;;
            --dataset)    DATASET="$arg"    ;;
        esac
        prev="$arg"
    done

    local TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    local RUN_ID="${MODEL_NAME}_${DATASET}_s${SEED}_${TIMESTAMP}"
    local RUN_DIR="./runs/${RUN_ID}"
    mkdir -p "$RUN_DIR"

    echo "[LAUNCH] GPU=$GPU  seed=$SEED  run_id=$RUN_ID"

    python main_legacy.py "$@" \
        --gpu_id $GPU \
        --seed $SEED \
        --run_id "$RUN_ID" \
        > "${RUN_DIR}/stdout.log" 2>&1

    echo "[DONE]   GPU=$GPU  seed=$SEED  run_id=$RUN_ID  exit=$?"
}

# ==============================================================================
# GPU 0: Beauty + GRU4Rec
# Ref: beauty_mean_gru_best.bash
# ==============================================================================
run_combo 0 \
  --model_name hsuga_llmesr_gru4rec \
  --dataset beauty \
  --hidden_mode _qwen_mean \
  --hidden_size 64 \
  --train_batch_size 128 \
  --max_len 200 \
  --num_workers 4 \
  --num_train_epochs 200 \
  --patience 20 \
  --ts_user 9 \
  --ts_item 4 \
  --freeze \
  --log \
  --user_sim_func kd \
  --alpha 0.05 \
  --use_cross_att \
  --lr 0.001 \
  --dropout_rate 0.5 \
  --num_layers 1 \
  --sim_user_num 14 \
  --use_hsu_fusion \
  --hsu_bank_path data/beauty/handled/usr_emb_np_qwen_mean.pkl \
  --hsu_fusion_type scalar \
  --hsu_gate_init -10.0 &

# ==============================================================================
# GPU 1: Beauty + SASRec
# Ref: beauty_sas_best.bash
# ==============================================================================
run_combo 1 \
  --model_name hsuga_llmesr_sasrec \
  --dataset beauty \
  --hidden_mode _qwen_mean \
  --hidden_size 64 \
  --train_batch_size 128 \
  --max_len 200 \
  --num_workers 4 \
  --num_train_epochs 200 \
  --patience 20 \
  --ts_user 9 \
  --ts_item 4 \
  --freeze \
  --log \
  --user_sim_func kd \
  --alpha 0.05 \
  --use_cross_att \
  --lr 0.0005 \
  --dropout_rate 0.6 \
  --trm_num 2 \
  --num_heads 1 \
  --use_hsu_fusion \
  --hsu_bank_path data/beauty/handled/usr_emb_np_qwen_mean.pkl \
  --hsu_fusion_type scalar \
  --hsu_gate_init -10.0 &

# ==============================================================================
# GPU 2: Beauty + BERT4Rec
# Ref: beauty_mean_bert_best.bash
# ==============================================================================
run_combo 2 \
  --model_name hsuga_llmesr_bert4rec \
  --dataset beauty \
  --hidden_mode _qwen_mean \
  --hidden_size 64 \
  --train_batch_size 128 \
  --max_len 200 \
  --num_workers 4 \
  --num_train_epochs 200 \
  --patience 20 \
  --ts_user 9 \
  --ts_item 4 \
  --freeze \
  --log \
  --user_sim_func kd \
  --alpha 0.01 \
  --use_cross_att \
  --lr 0.0005 \
  --dropout_rate 0.4 \
  --sim_user_num 10 \
  --mask_prob 0.7 \
  --use_hsu_fusion \
  --hsu_bank_path data/beauty/handled/usr_emb_np_qwen_mean.pkl \
  --hsu_fusion_type scalar \
  --hsu_gate_init -10.0 &

# ==============================================================================
# GPU 3: Steam + GRU4Rec
# Ref: steam_mean_gru_best.bash
# ==============================================================================
run_combo 3 \
  --model_name hsuga_llmesr_gru4rec \
  --dataset steam \
  --hidden_mode _qwen_mean_llm \
  --hidden_size 64 \
  --train_batch_size 128 \
  --max_len 200 \
  --num_workers 4 \
  --num_train_epochs 200 \
  --patience 20 \
  --ts_user 16 \
  --ts_item 69 \
  --freeze \
  --log \
  --user_sim_func kd \
  --alpha 0.01 \
  --use_cross_att \
  --lr 0.00005 \
  --dropout_rate 0.1 \
  --num_layers 1 \
  --sim_user_num 2 \
  --sim_long_user_num 14 \
  --similar_gate -1 \
  --filter_similar_metric pearson \
  --use_hsu_fusion \
  --hsu_bank_path data/steam/handled/usr_emb_np_qwen_mean.pkl \
  --hsu_fusion_type scalar \
  --hsu_gate_init -10.0 &

# ==============================================================================
# GPU 4: Steam + SASRec
# Ref: steam_mean_sas_best.bash
# ==============================================================================
run_combo 4 \
  --model_name hsuga_llmesr_sasrec \
  --dataset steam \
  --hidden_mode _qwen_mean_llm \
  --hidden_size 64 \
  --train_batch_size 128 \
  --max_len 200 \
  --num_workers 4 \
  --num_train_epochs 200 \
  --patience 20 \
  --ts_user 16 \
  --ts_item 69 \
  --freeze \
  --log \
  --user_sim_func kd \
  --alpha 0.1 \
  --use_cross_att \
  --lr 0.001 \
  --dropout_rate 0.4 \
  --sim_user_num 6 \
  --sim_long_user_num 6 \
  --similar_gate -1 \
  --filter_similar_metric pearson \
  --use_hsu_fusion \
  --hsu_bank_path data/steam/handled/usr_emb_np_qwen_mean.pkl \
  --hsu_fusion_type scalar \
  --hsu_gate_init -10.0 &

# ==============================================================================
# GPU 5: Steam + BERT4Rec
# Ref: steam_mean_bert_best.bash
# ==============================================================================
run_combo 5 \
  --model_name hsuga_llmesr_bert4rec \
  --dataset steam \
  --hidden_mode _qwen_mean_llm \
  --hidden_size 64 \
  --train_batch_size 128 \
  --max_len 200 \
  --num_workers 4 \
  --num_train_epochs 200 \
  --patience 20 \
  --ts_user 16 \
  --ts_item 69 \
  --freeze \
  --log \
  --user_sim_func kd \
  --alpha 0.05 \
  --use_cross_att \
  --lr 0.0005 \
  --dropout_rate 0.3 \
  --sim_user_num 10 \
  --mask_prob 0.7 \
  --similar_gate 0.0 \
  --filter_similar_metric pearson \
  --use_hsu_fusion \
  --hsu_bank_path data/steam/handled/usr_emb_np_qwen_mean.pkl \
  --hsu_fusion_type scalar \
  --hsu_gate_init -10.0 &

# ==============================================================================
# GPU 6: Fashion + GRU4Rec
# Ref: mean_gru_best_group_recall_wo_unchanged.bash
# Note: hidden_mode uses _qwen_mean_llm (matching available sim_user files)
# ==============================================================================
run_combo 6 \
  --model_name hsuga_llmesr_gru4rec \
  --dataset fashion \
  --hidden_mode _qwen_mean_llm \
  --hidden_size 64 \
  --train_batch_size 128 \
  --max_len 200 \
  --num_workers 4 \
  --num_train_epochs 200 \
  --patience 20 \
  --ts_user 3 \
  --ts_item 4 \
  --freeze \
  --log \
  --user_sim_func kd \
  --alpha 0.1 \
  --use_cross_att \
  --lr 0.0005 \
  --dropout_rate 0.3 \
  --sim_user_num 10 \
  --sim_long_user_num 18 \
  --similar_gate -0.000734 \
  --filter_similar_metric pearson \
  --use_hsu_fusion \
  --hsu_bank_path data/fashion/handled/usr_emb_np_qwen_mean.pkl \
  --hsu_fusion_type scalar \
  --hsu_gate_init -10.0 &

# ==============================================================================
# GPU 7: Fashion + SASRec -> then Fashion + BERT4Rec (share one GPU)
# Ref: mean_sas_best_group_recall_wo_unchanged.bash
#      mean_bert_best_group_recall_wo_unchanged.bash
# ==============================================================================
(
run_combo 7 \
  --model_name hsuga_llmesr_sasrec \
  --dataset fashion \
  --hidden_mode _qwen_mean_llm \
  --hidden_size 64 \
  --train_batch_size 128 \
  --max_len 200 \
  --num_workers 4 \
  --num_train_epochs 200 \
  --patience 20 \
  --ts_user 3 \
  --ts_item 4 \
  --freeze \
  --log \
  --user_sim_func kd \
  --alpha 0.1 \
  --use_cross_att \
  --lr 0.0005 \
  --dropout_rate 0.3 \
  --sim_user_num 2 \
  --sim_long_user_num 14 \
  --similar_gate -1 \
  --filter_similar_metric pearson \
  --use_hsu_fusion \
  --hsu_bank_path data/fashion/handled/usr_emb_np_qwen_mean.pkl \
  --hsu_fusion_type scalar \
  --hsu_gate_init -10.0

run_combo 7 \
  --model_name hsuga_llmesr_bert4rec \
  --dataset fashion \
  --hidden_mode _qwen_mean_llm \
  --hidden_size 64 \
  --train_batch_size 128 \
  --max_len 200 \
  --num_workers 4 \
  --num_train_epochs 200 \
  --patience 20 \
  --ts_user 3 \
  --ts_item 4 \
  --freeze \
  --log \
  --user_sim_func kd \
  --alpha 0.1 \
  --use_cross_att \
  --lr 0.0005 \
  --dropout_rate 0.3 \
  --sim_user_num 14 \
  --sim_long_user_num 14 \
  --similar_gate -0.000734 \
  --filter_similar_metric pearson \
  --mask_prob 0.7 \
  --use_hsu_fusion \
  --hsu_bank_path data/fashion/handled/usr_emb_np_qwen_mean.pkl \
  --hsu_fusion_type scalar \
  --hsu_gate_init -10.0
) &

echo ""
echo "================================================================"
echo " All 8 GPU slots launched (9 runs total, seed=$SEED)."
echo ""
echo " GPU 0-6: each runs 1 experiment"
echo " GPU 7:   runs 2 experiments sequentially"
echo ""
echo " Monitor progress:"
echo "   ls -lt runs/ | head -20"
echo "   tail -f runs/<run_id>/stdout.log"
echo ""
echo " Waiting for all jobs to complete..."
echo "================================================================"

wait
echo ""
echo "================================================================"
echo " ALL 9 EXPERIMENTS COMPLETED!"
echo " End time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"

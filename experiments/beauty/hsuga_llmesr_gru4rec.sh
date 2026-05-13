#!/bin/bash
# HSUGA + LLMESR + GRU4Rec on Beauty dataset
# Full HSUGA: Dual-view modeling + GAA (Group-Aware Alignment) + HSU (Hierarchical Semantic Understanding)

cd "$(dirname "${BASH_SOURCE[0]}")/.."

bash experiments/run_train.sh \
  --model_name hsuga_llmesr_gru4rec \
  --dataset beauty \
  --hidden_mode _qwen_mean \
  --hidden_size 64 \
  --train_batch_size 64 \
  --max_len 200 \
  --gpu_id 0 \
  --num_workers 4 \
  --num_train_epochs 200 \
  --seed 44 \
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
  --hsu_gate_init -10.0

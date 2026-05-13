#!/bin/bash
# ============================================================================
# Steam Mean BERT Best Group Recall 实验脚本
# ============================================================================

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# 日志配置
LOG_DIR="${PROJECT_DIR}/experiments/steam/logs"
mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/steam_mean_bert_best_group_recall_${TIMESTAMP}.log"

# 是否后台运行（可以通过环境变量控制）
BACKGROUND=${BACKGROUND:-false}

# 实验参数
gpu_id=7
dataset="steam"
seeds=(44)
# lr_list=(0.001)
lr_list=(0.0005)
dropout_list=(0.3) # 越低越好
mask_prob_list=(0.7) # 越高越好， 很明显
alpha_list=(0.05) # 0.1稍微好，但不明显。论文的方法是0.1更好
sim_user_num=(6)
sim_long_user_num=(10)
ts_user=16
ts_item=69
model_name="llmesr_mean_bert4rec"
mode="_qwen_mean_llm"
similar_gates=(-0.0028 -0.0022 -0.0019 -0.0017 -0.0014)
filter_similar_metric="pearson"
# 自定义相似用户文件和pearson分数文件路径
sim_user_file="data/steam/handled/usr_emb_cot_qwen2.5-7b-instruct_doubao_pca128_sim_user_top18.pkl"
sim_user_score_file="data/steam/handled/usr_emb_cot_qwen2.5-7b-instruct_doubao_pca128_pearson_score.pkl"
# Bank 缓存参数（注意：llmesr_mean_bert4rec 模型目前不支持 bank，这些参数会被忽略）
# 如果使用支持 bank 的模型（如 llmemb_gaa_bert4rec），可以启用这些参数
use_user_bank=false  # 是否启用 user bank 缓存
bank_momentum=0.9     # EMA 更新系数 (0.9~0.99)
bank_dtype="fp32"     # 存储精度: fp32 或 fp16
bank_on_cpu=true       # 是否存储在 CPU 上（推荐 true，节省 GPU 内存）
bank_warmup_epochs=1   # Bank warmup 轮数
bank_min_fill_ratio=0.1  # Bank 最小填充率
ckpt_save_name="group_recall_best"
filter_value=0  # 设置过滤值
counter=0        # 初始化计数器

# ============================================================================
# 日志函数
# ============================================================================
log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_FILE}"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $*" | tee -a "${LOG_FILE}" >&2
}

# ============================================================================
# 主实验函数
# ============================================================================
run_experiments() {
    # 切换到项目目录
    cd "${PROJECT_DIR}"
    
    log_info "============================================================"
    log_info "开始实验: Steam Mean BERT Best Group Recall"
    log_info "============================================================"
    log_info "项目目录: ${PROJECT_DIR}"
    log_info "日志文件: ${LOG_FILE}"
    log_info "GPU ID: ${gpu_id}"
    log_info "数据集: ${dataset}"
    log_info "模型: ${model_name}"
    log_info "模式: ${mode}"
    log_info "============================================================"
    log_info ""
    
    for lr in ${lr_list[@]}; do
        for dropout_rate in ${dropout_list[@]}; do
                for mask_prob in ${mask_prob_list[@]}; do
                        for alpha in ${alpha_list[@]}; do
                                for sim_users in ${sim_user_num[@]}; do
                                        for similar_gate in ${similar_gates[@]}; do
                                                for sim_long_user in ${sim_long_user_num[@]}; do
                                                        for seed in ${seeds[@]}; do
                                                                ((counter++))  # 每次循环计数器加1
                                                                if [ $counter -lt $filter_value ]; then
                                                                        log_info "跳过组合 $counter (小于过滤值 $filter_value)"
                                                                        continue  # 跳过当前循环
                                                                fi
                                                                
                                                                # 构建 bank 参数（如果启用）
                                                                bank_args=""
                                                                if [ "$use_user_bank" = true ]; then
                                                                        bank_args="--gaa_use_user_bank --gaa_bank_momentum ${bank_momentum} --gaa_bank_dtype ${bank_dtype} --gaa_warmup_epochs ${bank_warmup_epochs} --gaa_bank_min_fill_ratio ${bank_min_fill_ratio}"
                                                                        if [ "$bank_on_cpu" = true ]; then
                                                                                bank_args="${bank_args} --gaa_bank_on_cpu"
                                                                        fi
                                                                fi
                                                                
                                                                log_info "============================================================"
                                                                log_info "开始实验组合 #${counter}"
                                                                log_info "参数: lr=${lr}, dropout=${dropout_rate}, mask_prob=${mask_prob}"
                                                                log_info "      alpha=${alpha}, sim_users=${sim_users}, sim_long_user=${sim_long_user}"
                                                                log_info "      similar_gate=${similar_gate}, seed=${seed}"
                                                                log_info "============================================================"
                                                                
                                                                python main.py --dataset ${dataset} \
                                                                        --model_name ${model_name} \
                                                                        --hidden_mode ${mode} \
                                                                        --hidden_size 64 \
                                                                        --train_batch_size 128 \
                                                                        --max_len 200 \
                                                                        --gpu_id ${gpu_id} \
                                                                        --num_workers 4 \
                                                                        --num_train_epochs 200 \
                                                                        --seed ${seed} \
                                                                        --check_path ${ckpt_save_name} \
                                                                        --patience 20 \
                                                                        --ts_user ${ts_user} \
                                                                        --ts_item ${ts_item} \
                                                                        --freeze \
                                                                        --log \
                                                                        --user_sim_func kd \
                                                                        --alpha ${alpha} \
                                                                        --use_cross_att \
                                                                        --lr ${lr} \
                                                                        --dropout_rate ${dropout_rate} \
                                                                        --sim_user_num ${sim_users} \
                                                                        --mask_prob ${mask_prob} \
                                                                        --similar_gate ${similar_gate} \
                                                                        --filter_similar_metric ${filter_similar_metric} \
                                                                        --sim_long_user_num ${sim_long_user} \
                                                                        --sim_user_file_path ${sim_user_file} \
                                                                        --sim_user_score_file_path ${sim_user_score_file} \
                                                                        ${bank_args} 2>&1 | tee -a "${LOG_FILE}"
                                                                
                                                                if [ ${PIPESTATUS[0]} -eq 0 ]; then
                                                                        log_info "实验组合 #${counter} 完成"
                                                                else
                                                                        log_error "实验组合 #${counter} 失败 (退出码: ${PIPESTATUS[0]})"
                                                                fi
                                                                log_info ""
                                                        done
                                                done
                                        done
                                done
                        done
                done
        done
    done
    
    log_info "============================================================"
    log_info "所有实验完成!"
    log_info "结束时间: $(date)"
    log_info "============================================================"
}

# ============================================================================
# 执行主函数
# ============================================================================
if [ "$BACKGROUND" = true ]; then
    echo "以后台模式启动实验..."
    echo "主日志文件: ${LOG_FILE}"
    echo "项目目录: ${PROJECT_DIR}"
    echo ""
    
    # 使用 nohup 在后台运行主函数
    (
        cd "${PROJECT_DIR}"
        run_experiments
    ) > "${LOG_FILE}" 2>&1 &
    MAIN_PID=$!
    
    # 保存 PID
    echo "${MAIN_PID}" > "${LOG_DIR}/steam_mean_bert_best_group_recall_${TIMESTAMP}.pid"
    
    echo "主进程 PID: ${MAIN_PID}"
    echo "PID 文件: ${LOG_DIR}/steam_mean_bert_best_group_recall_${TIMESTAMP}.pid"
    echo ""
    echo "监控进度:"
    echo "  tail -f ${LOG_FILE}"
    echo ""
    echo "停止实验:"
    echo "  kill ${MAIN_PID}"
    echo "  或使用: pkill -f 'python main.py.*--model_name ${model_name}'"
    echo ""
    
    # 使用 disown 使进程在终端关闭后继续运行
    disown ${MAIN_PID}
else
    # 前台模式：直接运行并同时输出到终端和日志文件
    run_experiments 2>&1 | tee "${LOG_FILE}"
fi

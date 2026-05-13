gpu_id=5
dataset="fashion"
seeds=(42 43 44)
lr_list=(0.0005)
# dropout_list=(0.2 0.3 0.4 0.5)
# alpha_list=(0.01 0.05 0.1)
# lr_list=(0.001)
dropout_list=(0.3)
# alpha_list=(0.05)
alpha_list=(0.1)
# sim_user_num=(2 6 10 14 18)
sim_user_num=(10)
sim_long_user_num=(10)
ts_user=3
ts_item=4
model_name="llmesr_mean_gru4rec"
mode="_qwen_mean_rnn_llm"
# similar_gates=(0)
# 分别对应Top18相似用户过滤后为 18 13 8 4
similar_gates=(-1) #-1 -0.000734 -0.000636 0.935315
filter_similar_metric="pearson"
ckpt_save_name="group_recall_gru_rnn"
filter_value=0  # 设置过滤值 比要过来的组合多1
counter=0        # 初始化计数器
for seed in ${seeds[@]}; do
        for lr in ${lr_list[@]}; do
                for dropout_rate in ${dropout_list[@]}; do
                        for alpha in ${alpha_list[@]}; do
                                for sim_users in ${sim_user_num[@]}; do
                                        for long_sim_users in ${sim_long_user_num[@]}; do
                                                for similar_gate in ${similar_gates[@]}; do
                                                        ((counter++))  # 每次循环计数器加1
                                                        if [ $counter -lt $filter_value ]; then
                                                                echo "跳过组合 $counter (小于过滤值 $filter_value)"
                                                                continue  # 跳过当前循环
                                                        fi
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
                                                                --similar_gate ${similar_gate} \
                                                                --filter_similar_metric ${filter_similar_metric} \
                                                                --sim_long_user_num ${long_sim_users}
                                                done
                                        done
                                done
                        done
                done
        done
done

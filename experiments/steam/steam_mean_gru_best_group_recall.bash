gpu_id=6
dataset="steam"
seeds=(44 43 42)
# lr_list=(0.00005 0.0001 0.0005)
# dropout_list=(0.3 0.4 0.5 0.2 0.1)
# alpha_list=(0.01 0.05 0.1)
lr_list=(0.00005)
dropout_list=(0.1)
alpha_list=(0.01)
sim_user_num=(2)
sim_long_user_num=(14)
gru_layers=(1)
ts_user=16
ts_item=69
similar_gates=(-0.0014309) #-0.0014309 -0.001209
filter_similar_metric="pearson"
model_name="llmesr_mean_gru4rec"
mode="_qwen_mean_llm"
ckpt_save_name="group_recall_best"
filter_value=0  # 设置过滤值 比要过来的组合多1
counter=0        # 初始化计数器
for lr in ${lr_list[@]}; do
        for dropout_rate in ${dropout_list[@]}; do
                for gru_layer in ${gru_layers[@]}; do
                        for alpha in ${alpha_list[@]}; do
                                for sim_users in ${sim_user_num[@]}; do
                                        for sim_long_user in ${sim_long_user_num[@]}; do
                                                for similar_gate in ${similar_gates[@]}; do
                                                        for seed in ${seeds[@]}; do
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
                                                                        --num_workers 8 \
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
                                                                        --num_layers ${gru_layer} \
                                                                        --sim_user_num ${sim_users} \
                                                                        --similar_gate ${similar_gate} \
                                                                        --filter_similar_metric ${filter_similar_metric} \
                                                                        --sim_long_user_num ${sim_long_user}
                                                        done
                                                done
                                        done
                                done
                        done
                done
        done
done

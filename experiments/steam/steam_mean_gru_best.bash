gpu_id=4
dataset="steam"
seed_list=(44 43 42)
lr_list=(0.00005)
dropout_list=(0.1)
alpha_list=(0.01)
sim_user_num=(2)
sim_long_user_num=(14)
gru_layers=(1)
ts_user=16
ts_item=69
model_name="llmesr_mean_gru4rec"
mode="_qwen_mean_llm"
similar_gate=-1
filter_similar_metric="pearson"
for lr in ${lr_list[@]}; do
        for dropout_rate in ${dropout_list[@]}; do
                for gru_layer in ${gru_layers[@]}; do
                        for alpha in ${alpha_list[@]}; do
                                for sim_users in ${sim_user_num[@]}; do
                                        for seed in ${seed_list[@]}; do
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
                                                        --check_path "long_user" \
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
                                                        --sim_long_user_num ${sim_long_user_num}
                                        done
                                done
                        done
                done
        done
done

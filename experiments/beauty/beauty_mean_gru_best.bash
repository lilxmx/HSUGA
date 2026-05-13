gpu_id=6
dataset="beauty"
seed_lists=(44)
lr_list=(0.001)
dropout_list=(0.5)
alpha_list=(0.05)
gru_layers=(1)
sim_user_num=(14)
ts_user=9
ts_item=4
model_name="llmesr_mean_gru4rec"
mode="_qwen_mean"
for lr in ${lr_list[@]}; do
        for dropout_rate in ${dropout_list[@]}; do
                for gru_layer in ${gru_layers[@]}; do
                        for alpha in ${alpha_list[@]}; do
                                for sim_users in ${sim_user_num[@]}; do
                                        for seed in ${seed_lists[@]}; do
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
                                                        --check_path "" \
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
                                                        --sim_user_num ${sim_users}
                                        done
                                done
                        done
                done
        done
done

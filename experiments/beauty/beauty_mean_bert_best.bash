gpu_id=0
dataset="beauty"
seed_lists=(42 43 44)
lr_list=(0.0005)
dropout_list=(0.4)
mask_prob_list=(0.7)
alpha_list=(0.01)
sim_user_num=(10)
ts_user=9
ts_item=4
model_name="llmesr_mean_bert4rec"
mode="_qwen_mean"
for lr in ${lr_list[@]}; do
        for dropout_rate in ${dropout_list[@]}; do
                for mask_prob in ${mask_prob_list[@]}; do
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
                                                        --sim_user_num ${sim_users} \
                                                        --mask_prob ${mask_prob}
                                        done
                                done
                        done
                done
        done
done

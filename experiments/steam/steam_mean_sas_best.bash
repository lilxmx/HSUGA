gpu_id=2
dataset="steam"
seed_lists=(42 43 44)
lr_list=(0.001)
dropout_list=(0.4)
alpha_list=(0.1)
# alpha_list=(0.01 0.05 0.1 0.5)
# sim_user_num=(2 6 10 14 18)
sim_user_num=(6)
sim_long_user_num=(6)
ts_user=16
ts_item=69
model_name="llmesr_mean_sasrec"
# mode="_qwen_mean_collab"
mode="_qwen_mean_llm"
similar_gate=-1
filter_similar_metric="pearson"
filter_value=0  # Set filter value (one more than the combination to filter)
counter=0       # Initialize counter
for lr in ${lr_list[@]}; do
        for dropout_rate in ${dropout_list[@]}; do
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
                                                --num_workers 4 \
                                                --num_train_epochs 200 \
                                                --seed ${seed} \
                                                --check_path "only_filter" \
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
                                                --sim_long_user_num ${sim_long_user_num}
                                done
                        done
                done
        done
done

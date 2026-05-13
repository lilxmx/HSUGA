## LLM-ESR -- SASRec, Bert4Rec, GRU4Rec
gpu_id=6
dataset="beauty"
seed_list=(42 43 44)
ts_user=9
ts_item=4
mode="_qwen_mean"
model_name="llmesr_sasrec"

for seed in ${seed_list[@]}
do
        python main.py --dataset ${dataset} \
                --model_name ${model_name} \
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
                --alpha 0.05 \
                --use_cross_att \
                --hidden_mode ${mode} \
                --lr 0.0005 \
                --dropout_rate 0.6 \
                --trm_num 2 \
                --num_heads 1
done
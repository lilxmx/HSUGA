# HSUGA: LLM-Enhanced Recommendation with Hierarchical Semantic Understanding and Group-Aware Alignment

This is the implementation of the paper "HSUGA: LLM-Enhanced Recommendation with Hierarchical Semantic Understanding and Group-Aware Alignment".

## Configure the environment

To ease the configuration of the environment, I list versions of my hardware and software equipments:

- Hardware:
  - GPU: Tesla V100 32GB
  - Cuda: 11.4
- Software:
  - Python: 3.8.20
  - torch: 2.3.0

You can pip install the `requirements.txt` to configure the environment.

## Preprocess the dataset

You can preprocess the dataset and get the LLMs embedding according to the following steps:

1. The raw dataset downloaded from website should be put into `/data/<steam/fashion/beauty>/raw/`. The steam dataset can be obtained from [). The fashion and beauty datasets can be obtained from [https://cseweb.ucsd.edu/~jmcauley/datasets.html\#amazon_reviews](https://cseweb.ucsd.edu/~jmcauley/datasets.html\#amazon_reviews)(2014).
2. Conduct the preprocessing code `data/data_process.py` to filter cold-start users and items. After the procedure, you will get the id file  `/data/<steam/fashion/beauty>/hdanled/id_map.json` and the interaction file  `/data/<steam/fashion/beauty>/handled/inter_seq.txt`.
3. Convert the interaction file to the format used in this repo by running `data/convert_inter.ipynb`.
4. To get the LLMs item embedding for each dataset, please run the jupyter notebooks  `/data/<steam/fashion/beauty>/get_item_embedding.ipynb`. After the running, you will get the LLMs item embedding file `/data/<steam/fashion/beauty>/handled/itm_emb_np.pkl`.
5. To get the LLMs User Embedding for each dataset, please use llm to inference for user behavior. The inference log will be saved in the folder 'inference_log'. The inference result(text & hidden embedding) will be saved in the folder 'prompt'. We have provided relevant merging, analysis, and processing scripts in the folder. Please use them according to the sequence number. Finally, we get `usr_emb_np_qwen_mean.pkl`
```
bash inference/batch_inference_router.py
```
6. For dual-view modeling module, we need to run the jupyter notebook `data/pca.ipynb` to get the dimension-reduced LLMs item embedding for initialization, i.e., `/data/<steam/fashion/beauty>/handled/pca64_itm_emb_np.pkl`.
7. For retrieval augmented self-distillation, we need to run the jupyter notebook `data/retrieval_users.ipynb` to get the similar user set for each user. The output file in this step is `sim_user_100.pkl`
8. For active user filter, we need to calculate the relevance between users. run `bash data/cal_similar.bash` to get the `sim_user_pearson_score_100_qwen_mean_llm.pkl`

In conclusion, the prerequisite files to run the code are as follows: `inter.txt`, `itm_emb_np.pkl`, `usr_emb_np_qwen_mean.pkl`, `pca64_itm_emb_np.pkl`, `sim_user_100.pkl` and `sim_user_pearson_score_100_qwen_mean_llm.pkl`

⭐️ To ease the reproducibility of our paper, we also upload all preprocessed files to this [link]().

## Run and test

2. You can reproduce all HSUGA experiments by running the bash as follows:

```
bash experiments/steam/steam_mean_sas_best.bash
bash experiments/mean_gru_grid_rnn.bash
bash experiments/beauty_sas_best.bash
```

3. The log and results will be saved in the folder `log/`. The checkpoint will be saved in the folder `saved/`.


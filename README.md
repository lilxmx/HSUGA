# HSUGA: LLM-Enhanced Recommendation with Hierarchical Semantic Understanding and Group-Aware Alignment

[![paper](https://img.shields.io/badge/Paper-ACL'26_Findings-blue)](TODO_PAPER_LINK)
[![data](https://img.shields.io/badge/Data-Google_Drive-yellow)](TODO_DATA_LINK)

This is the official implementation of the paper: **"HSUGA: LLM-Enhanced Recommendation with Hierarchical Semantic Understanding and Group-Aware Alignment"** (ACL 2026 Findings).

> **Note:** This repository is a refactored and reorganized version of the original research codebase. While the core algorithms and model logic remain identical, the project structure, configuration management, and experiment workflow have been restructured for better readability, reproducibility, and maintainability. As a result, file paths and module organization may differ from the raw experimental codebase used during development.

## Configure the environment

### Hardware & Software Requirements

- GPU: NVIDIA GPU with at least 16GB VRAM (Tesla T4 / V100 / A100 / H20)
- NVIDIA Driver: >= 525 (supporting CUDA 12.x)
- OS: Linux

### Step 1: Create conda environment

```bash
conda create -n hsuga python=3.10 -y
conda activate hsuga
```

### Step 2: Install PyTorch (match your CUDA driver version)

```bash
# For CUDA 12.1 (recommended for driver >= 525)
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8 (older drivers)
# pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu118
```

### Step 3: Install other dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify installation

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
```

## Project Structure

```
HSUGA/
├── main_legacy.py              # Main entry point (training & evaluation)
├── main.py                     # YAML-config based entry (alternative)
├── requirements.txt            # Python dependencies
├── setup.py                    # Package installation
│
├── models/                     # Model implementations
│   ├── LLMESR.py               # HSUGA core: LLMESR + HSU fusion
│   ├── HSUGatedFusion.py       # HSU (Hierarchical Semantic Understanding)
│   ├── ICSRec.py               # ICSRec baseline model
│   ├── RCL.py                  # RCL baseline model
│   └── modules.py              # Shared backbone modules (SASRec, GRU4Rec, BERT4Rec)
│
├── trainers/                   # Training logic
│   ├── trainer.py              # Base trainer (HSUGA / LLMESR)
│   ├── icsrec_trainer.py       # ICSRec trainer
│   └── rcl_trainer.py          # RCL trainer
│
├── generators/                 # Data loading & sampling
│   ├── data.py                 # Core dataset classes
│   └── icsrec_generator.py     # ICSRec-specific data generator
│
├── baselines/                  # Baseline method implementations
│   ├── icsrec/                 # ICSRec (clustering, losses, model, trainer)
│   ├── rcl/                    # RCL
│   ├── llm2rec/                # LLM2Rec
│   └── llmemb/                 # LLMEmb
│
├── utils/                      # Utility functions
│   ├── utils.py                # Metrics, logging helpers
│   └── logger.py               # Logger with TensorBoard support
│
├── experiments/                # Experiment scripts
│   ├── run_all_hsuga.sh        # One-click: all 9 experiments (3 backbones x 3 datasets)
│   ├── run_train.sh            # Single experiment launcher (nohup + auto run_id)
│   ├── beauty/                 # Per-dataset experiment configs
│   ├── steam/
│   └── fashion/
│
├── configs/                    # YAML configuration files
│   ├── default.yaml
│   ├── dataset/
│   └── model/
│
├── data/                       # Data directory
│   ├── data_process.py         # Filter cold-start users/items
│   ├── convert_inter.ipynb     # Convert interaction format
│   ├── pca.py                  # PCA dimension reduction
│   ├── retrieval_users.py      # Retrieve similar users
│   ├── retrieval_users.ipynb   # Interactive version of the above
│   └── <dataset>/
│       ├── raw/                # Raw dataset (download required)
│       └── handled/            # Preprocessed data files
│
├── inference/                  # LLM inference for user embeddings
│   ├── batch_inference_router.py
│   └── qwen2.5-7B/Modelfile
│
└── scripts/                    # Auxiliary scripts
    ├── inference/
    └── preprocess/
```

## Preprocess the dataset

You can preprocess the dataset and get the LLM embeddings according to the following steps:

1. The raw dataset downloaded from website should be put into `data/<steam/fashion/beauty>/raw/`. The Steam dataset can be obtained from [https://cseweb.ucsd.edu/~jmcauley/datasets.html#steam_data](https://cseweb.ucsd.edu/~jmcauley/datasets.html#steam_data). The Fashion and Beauty datasets can be obtained from [https://cseweb.ucsd.edu/~jmcauley/datasets.html#amazon_reviews](https://cseweb.ucsd.edu/~jmcauley/datasets.html#amazon_reviews) (2014 version).

2. Conduct the preprocessing code `data/data_process.py` to filter cold-start users and items. After the procedure, you will get the id file `data/<dataset>/handled/id_map.json` and the interaction file `data/<dataset>/handled/inter_seq.txt`.

3. Convert the interaction file to the format used in this repo by running `data/convert_inter.ipynb`.

4. To get the LLM item embeddings for each dataset, refer to the notebook `data/steam/get_item_embedding.ipynb` as an example (adapt paths for other datasets). After running, you will get the LLM item embedding file `data/<dataset>/handled/itm_emb_np.pkl`.

5. To get the LLM User Embeddings for each dataset, please use LLM to inference for user behavior. We use [Ollama](https://ollama.com/) to deploy local LLM service (see `inference/qwen2.5-7B/Modelfile` for the model config). Then run:
```bash
python inference/batch_inference_router.py
```
The inference results (text & hidden embeddings) will be saved in the folder `prompt/`. We have provided relevant merging, analysis, and processing scripts in the folder. Finally, we get `usr_emb_np_qwen_mean.pkl`.

6. For the dual-view modeling module, run `data/pca.py` to get the dimension-reduced LLM item embedding for initialization: `data/<dataset>/handled/pca64_itm_emb_np.pkl`.

7. For retrieval augmented self-distillation, run `python data/retrieval_users.py` to get the similar user set for each user. The output file is `sim_user_100_qwen_mean_llm.pkl`.

8. For the active user filter, calculate the relevance between users by running `python data/retrieval_users.py --sim_metric pearson` to get `sim_user_pearson_score_100_qwen_mean_llm.pkl`.

In conclusion, the prerequisite files to run the code are as follows:
- `inter.txt` — interaction sequences
- `itm_emb_np.pkl` — LLM item embeddings
- `usr_emb_np_qwen_mean.pkl` — LLM user embeddings
- `pca64_itm_emb_np.pkl` — PCA-reduced item embeddings
- `sim_user_100_qwen_mean_llm.pkl` — similar user set (cosine)
- `sim_user_pearson_score_100_qwen_mean_llm.pkl` — similar user set (Pearson)

To ease the reproducibility of our paper, we upload all preprocessed files here: **[Google Drive](TODO_DATA_LINK)** | **[Hugging Face](TODO_HF_LINK)**

## Run and test

### Quick start

```bash
conda activate hsuga

# Run a single experiment (launches in background via nohup)
bash experiments/run_train.sh \
  --model_name hsuga_llmesr_gru4rec \
  --dataset beauty \
  --lr 0.001 \
  --dropout_rate 0.5 \
  --use_hsu_fusion \
  --hsu_bank_path data/beauty/handled/usr_emb_np_qwen_mean.pkl

# Run all 9 experiments (3 backbones x 3 datasets) in parallel
nohup bash experiments/run_all_hsuga.sh > experiments/run_all.log 2>&1 &
```

### Output structure

All outputs are organized under `runs/{run_id}/`:
```
runs/hsuga_llmesr_gru4rec_beauty_s44_20260513_120000/
├── config.yaml        # Full hyperparameter snapshot (for reproducibility)
├── train.log          # Structured training log
├── stdout.log         # Raw stdout/stderr
├── tensorboard/       # TensorBoard events
└── checkpoints/       # Best model weights (pytorch_model.bin)
```

### Monitor training

```bash
# Check all running experiments
ls -lt runs/ | head

# Follow a specific experiment's log
tail -f runs/<run_id>/stdout.log

# Launch TensorBoard
tensorboard --logdir runs/
```
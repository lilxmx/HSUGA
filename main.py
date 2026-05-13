"""
HSUGA: LLM-Enhanced Recommendation with Hierarchical Semantic Understanding and Group-Aware Alignment

New config-driven entry point.

Usage:
    python main.py --config configs/model/hsuga_sasrec.yaml --override dataset.name=steam training.seed=42

For legacy argparse-based usage, see main_legacy.py.
"""
import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hsuga.utils.misc import set_seed, load_config
from hsuga.utils.logger import Logger
from hsuga.data import build_dataloader
from hsuga.trainers import build_trainer


def main():
    parser = argparse.ArgumentParser(description="HSUGA Training")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--override", nargs="*", default=[], help="Override config values: key1.key2=value")
    cli_args = parser.parse_args()

    cfg = load_config(cli_args.config, cli_args.override)

    # Extract commonly used values with backward-compatible attribute access
    seed = getattr(cfg.training, 'seed', 42) if hasattr(cfg, 'training') else 42
    set_seed(seed)

    # Flatten key attributes to top level for backward compat with existing code
    cfg.model_name = cfg.model.name
    cfg.dataset = cfg.dataset.name if hasattr(cfg.dataset, 'name') else 'steam'
    cfg.hidden_size = cfg.model.hidden_size
    cfg.trm_num = getattr(cfg.model, 'trm_num', 2)
    cfg.num_heads = getattr(cfg.model, 'num_heads', 1)
    cfg.num_layers = getattr(cfg.model, 'num_layers', 1)
    cfg.dropout_rate = cfg.model.dropout_rate
    cfg.max_len = cfg.model.max_len
    cfg.mask_prob = getattr(cfg.model, 'mask_prob', 0.4)
    cfg.use_cross_att = getattr(cfg.model, 'use_cross_att', True)
    cfg.freeze_llm_emb = getattr(cfg.model, 'freeze_llm_emb', True)
    cfg.freeze = getattr(cfg.model, 'freeze_llm_emb', True)

    # Training params
    cfg.train_batch_size = getattr(cfg.training, 'batch_size', 128)
    cfg.lr = cfg.training.lr
    cfg.num_train_epochs = getattr(cfg.training, 'epochs', 200)
    cfg.patience = getattr(cfg.training, 'patience', 20)
    cfg.watch_metric = getattr(cfg.training, 'watch_metric', 'NDCG@10')
    cfg.seed = seed
    cfg.gpu_id = getattr(cfg.training, 'gpu_id', 0)
    cfg.num_workers = getattr(cfg.training, 'num_workers', 4)
    cfg.log = getattr(cfg.training, 'log', True)
    cfg.l2 = getattr(cfg.training, 'l2', 0)
    cfg.lr_dc_step = getattr(cfg.training, 'lr_dc_step', 1000)
    cfg.lr_dc = getattr(cfg.training, 'lr_dc', 0)
    cfg.no_cuda = False

    # Alignment params
    cfg.alpha = getattr(cfg.alignment, 'alpha', 0.1) if hasattr(cfg, 'alignment') else 0.1
    cfg.user_sim_func = getattr(cfg.alignment, 'sim_func', 'kd') if hasattr(cfg, 'alignment') else 'kd'
    cfg.sim_user_num = getattr(cfg.alignment, 'sim_user_num', 6) if hasattr(cfg, 'alignment') else 6
    cfg.sim_long_user_num = getattr(cfg.alignment, 'sim_long_user_num', 6) if hasattr(cfg, 'alignment') else 6
    cfg.filter_similar_metric = getattr(cfg.alignment, 'filter_metric', 'pearson') if hasattr(cfg, 'alignment') else 'pearson'
    cfg.similar_gate = getattr(cfg.alignment, 'similar_gate', -1.0) if hasattr(cfg, 'alignment') else -1.0
    cfg.sim_filter_percentile = getattr(cfg.alignment, 'sim_filter_percentile', 0.0) if hasattr(cfg, 'alignment') else 0.0
    cfg.filter_similar_user = cfg.similar_gate > -1.0 or cfg.sim_filter_percentile > 0.0

    # Dataset params
    cfg.inter_file = getattr(cfg.dataset if hasattr(cfg, 'dataset') and not isinstance(cfg.dataset, str) else cfg, 'inter_file', 'inter')
    if isinstance(cfg.dataset, str):
        ds_name = cfg.dataset
    else:
        ds_name = cfg.dataset
    cfg.ts_user = getattr(cfg, 'ts_user', None)
    cfg.ts_item = getattr(cfg, 'ts_item', None)

    # Auto-set dataset defaults
    DATASET_DEFAULTS = {
        "beauty": {"ts_user": 9, "ts_item": 4},
        "fashion": {"ts_user": 3, "ts_item": 4},
        "steam": {"ts_user": 16, "ts_item": 69},
    }
    if cfg.ts_user is None:
        cfg.ts_user = DATASET_DEFAULTS.get(cfg.dataset, {}).get("ts_user", 16)
    if cfg.ts_item is None:
        cfg.ts_item = DATASET_DEFAULTS.get(cfg.dataset, {}).get("ts_item", 69)

    # Output params
    cfg.output_dir = getattr(cfg.output, 'save_dir', './saved') if hasattr(cfg, 'output') else './saved'
    cfg.check_path = getattr(cfg.output, 'check_path', '') if hasattr(cfg, 'output') else ''
    cfg.output_dir = os.path.join(cfg.output_dir, cfg.dataset, cfg.model_name, cfg.check_path)

    # Backward compat flags
    cfg.demo = False
    cfg.do_test = False
    cfg.do_emb = False
    cfg.do_item_emb = False
    cfg.do_group = False
    cfg.do_analysis = False
    cfg.keepon = False
    cfg.keepon_path = 'normal'
    cfg.aug = False
    cfg.aug_seq = False
    cfg.aug_seq_len = 0
    cfg.aug_file = 'inter'
    cfg.train_neg = getattr(cfg, 'train_neg', 1)
    cfg.test_neg = getattr(cfg, 'test_neg', 100)
    cfg.item_from_llm = False
    cfg.item_reg = False
    cfg.beta = 0.1
    cfg.use_llm2rec = False
    cfg.filter_item = False
    cfg.weight_sum = False
    cfg.hidden_mode = getattr(cfg, 'hidden_mode', '_qwen_mean_llm')
    cfg.use_dynamic_k = False

    # Device setup
    device = torch.device(f"cuda:{cfg.gpu_id}" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.output_dir, exist_ok=True)

    # Initialize logger
    log_manager = Logger(cfg)
    logger, writer = log_manager.get_logger()
    cfg.now_str = log_manager.get_now_str()

    # Build dataloader and trainer
    dataloader = build_dataloader(cfg, logger, device)
    trainer = build_trainer(cfg, logger, writer, device, dataloader)
    trainer.train()

    log_manager.end_log()


if __name__ == "__main__":
    main()

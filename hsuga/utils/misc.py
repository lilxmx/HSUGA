import os
import random
import numpy as np
import torch
import yaml
from types import SimpleNamespace


def set_seed(seed):
    """Fix all random seeds for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def _deep_merge(base, override):
    """Recursively merge override dict into base dict."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _flatten_dict(d, parent_key='', sep='.'):
    """Flatten nested dict to dot-notation keys."""
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(_flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def _dict_to_namespace(d):
    """Convert nested dict to nested SimpleNamespace for attribute access."""
    ns = SimpleNamespace()
    for key, value in d.items():
        if isinstance(value, dict):
            setattr(ns, key, _dict_to_namespace(value))
        else:
            setattr(ns, key, value)
    return ns


def load_config(config_path, overrides=None):
    """
    Load YAML config with optional CLI overrides.
    
    Supports merging multiple config files (base + model + dataset).
    Override format: ["key1.key2=value", ...]
    """
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # Load and merge default config if exists
    config_dir = os.path.dirname(config_path)
    default_path = os.path.join(config_dir, '..', 'default.yaml')
    if not os.path.isabs(default_path):
        default_path = os.path.normpath(default_path)
    
    if os.path.exists(default_path) and os.path.abspath(default_path) != os.path.abspath(config_path):
        with open(default_path, 'r') as f:
            default_cfg = yaml.safe_load(f)
        cfg = _deep_merge(default_cfg, cfg)

    # Apply CLI overrides
    if overrides:
        for override in overrides:
            if '=' not in override:
                continue
            key, value = override.split('=', 1)
            keys = key.split('.')
            
            # Auto-cast value
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    if value.lower() == 'true':
                        value = True
                    elif value.lower() == 'false':
                        value = False

            # Set nested value
            d = cfg
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            d[keys[-1]] = value

    return _dict_to_namespace(cfg)


def get_n_params(model, logger=None):
    """Get the number of parameters of model."""
    pp = 0
    for p in list(model.named_parameters()):
        nn = 1
        for s in list(p[1].size()):
            nn = nn * s
        pp += nn
    return pp


def masked_mean(sim_log_feats, valid_mask, eps=1e-8):
    """
    Calculate mean of valid similar users for each batch sample.

    Args:
        sim_log_feats: [batch, sim_num, hidden_size]
        valid_mask: [batch, sim_num] (0/1)

    Returns:
        dynamic_mean: [batch, hidden_size]
        valid_sample_idx: batch indices of valid samples
    """
    mask = valid_mask.unsqueeze(-1).float()
    masked_sum = (sim_log_feats * mask).sum(dim=1)
    valid_counts = mask.sum(dim=1)
    valid_sample_idx = (valid_counts.squeeze(-1) > 0).nonzero(as_tuple=True)[0]
    valid_counts = valid_counts + eps
    dynamic_mean = masked_sum / valid_counts
    return dynamic_mean, valid_sample_idx

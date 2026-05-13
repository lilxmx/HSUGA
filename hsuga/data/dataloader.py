"""
DataLoader factory: creates the appropriate data generator based on config.
Wraps the existing generators for backward compatibility.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from generators.generator import Seq2SeqGeneratorAllUser, GeneratorAllUser
from generators.bert_generator import BertGeneratorAllUser


def build_dataloader(cfg, logger, device):
    """
    Build dataloader based on config. Returns the appropriate Generator instance.
    
    The generator encapsulates train/valid/test dataloaders and dataset metadata.
    """
    model_name = cfg.model_name if hasattr(cfg, 'model_name') else cfg.model.name

    # Map new names to backbone types
    backbone_type = _get_backbone_type(model_name)

    if backbone_type == "gru4rec":
        return GeneratorAllUser(cfg, logger, device)
    elif backbone_type == "bert4rec":
        return BertGeneratorAllUser(cfg, logger, device)
    elif backbone_type == "sasrec":
        return Seq2SeqGeneratorAllUser(cfg, logger, device)
    else:
        raise ValueError(f"Cannot determine backbone type for model: {model_name}")


def _get_backbone_type(model_name):
    """Determine backbone type from model name."""
    name = model_name.lower()
    if "gru4rec" in name or "gru" in name:
        return "gru4rec"
    elif "bert4rec" in name or "bert" in name:
        return "bert4rec"
    elif "sasrec" in name or "sas" in name:
        return "sasrec"
    return "sasrec"

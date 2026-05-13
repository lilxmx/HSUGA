"""Trainer module for HSUGA."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from hsuga.trainers.seq_trainer import SeqTrainer
from hsuga.trainers.base_trainer import Trainer


def build_trainer(cfg, logger, writer, device, generator):
    """
    Build the appropriate trainer based on config.
    """
    model_name = cfg.model_name if hasattr(cfg, 'model_name') else cfg.model.name
    name = model_name.lower()

    if "icsrec" in name:
        from trainers.icsrec_trainer import ICSRecTrainer
        return ICSRecTrainer(cfg, logger, writer, device, generator)
    elif "rcl" in name:
        from trainers.rcl_trainer import RCLTrainer
        return RCLTrainer(cfg, logger, writer, device, generator)
    else:
        return SeqTrainer(cfg, logger, writer, device, generator)

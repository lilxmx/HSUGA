"""
Base Trainer: handles model creation, optimizer setup, early stopping, and training loop.
This is a thin wrapper around the existing trainers/trainer.py for the new package structure.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from trainers.trainer import Trainer

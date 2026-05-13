"""
Sequential Trainer: extends Trainer with specific train/eval logic for sequential recommendation.
Re-exports from the existing trainers/ for backward compatibility.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from trainers.sequence_trainer import SeqTrainer

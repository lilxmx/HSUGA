"""Data loading module for HSUGA."""
import sys
import os

# Add project root to path so legacy generators/ can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from hsuga.data.dataloader import build_dataloader

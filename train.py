"""
Compatibility shim for root-level imports.

Allows legacy scripts to use:
    from train import train_model, create_data_loaders

All training logic is in src/training/train.py.
"""
from src.training.train import *

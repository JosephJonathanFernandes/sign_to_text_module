"""
Compatibility shim for root-level imports.

Allows legacy scripts to use:
    from model import SignLanguageGRU

All model definitions are in src/training/model.py.
"""
from src.training.model import *

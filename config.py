"""
Compatibility shim for root-level imports.

This file allows legacy scripts to use:
    from config import get_config
    from config import CONFIG

All actual configuration is defined in src/core/config.py.
"""
from src.core.config import *

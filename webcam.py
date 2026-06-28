"""
Compatibility shim for root-level imports.

Allows legacy scripts to use:
    from webcam import run_webcam

All live inference logic is in src/core/webcam.py.
"""
from src.core.webcam import *

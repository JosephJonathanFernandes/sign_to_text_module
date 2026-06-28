"""
ISL Sign-to-Text — CLI Entry Point.

This file is the top-level entry point.
All pipeline logic is implemented in src/core/main.py.

Usage:
    python main.py --webcam           # Live webcam recognition
    python main.py --train            # Train single model
    python main.py --kfold            # K-fold ensemble training
    python main.py --preprocess       # Extract landmarks from videos
    python main.py --predict VIDEO    # Predict from a video file
    python main.py --collect          # Collect new training samples
    python main.py -h                 # Full help
"""
import sys  # noqa: F401
from src.core.main import main

if __name__ == '__main__':
    main()

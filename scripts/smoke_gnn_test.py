"""
Smoke test: load model (GNN enabled), run synthetic forward pass, print shapes and timing.
Run from repo root: python -u scripts/smoke_gnn_test.py
"""
import os
import sys
import time
import torch

# Ensure repo root is on sys.path so local modules import correctly
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from config import get_config
cfg = get_config()

from model import SignLanguageGRU

# Synthetic inputs
B = 2
T = cfg.preprocessing.num_frames
INPUT_SIZE_LOCAL = cfg.frame_features.input_sequence_dim
x = torch.randn(B, T, INPUT_SIZE_LOCAL)

# Instantiate model
num_classes = 10
model = SignLanguageGRU(num_classes=num_classes)
model.eval()

# Try loading checkpoint (non-strict to allow missing GNN keys)
ckpt_path = os.path.join(repo_root, 'model.pth')
if os.path.exists(ckpt_path):
    try:
        state = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(state, strict=False)
        print(f"Loaded checkpoint: {ckpt_path} (strict=False)")
    except Exception as e:
        print(f"Warning: failed to load checkpoint: {e}")
else:
    print(f"No checkpoint found at {ckpt_path}; proceeding with random init")

# Forward pass
with torch.no_grad():
    t0 = time.time()
    out = model(x)
    t1 = time.time()

print(f"Input: {x.shape}")
if isinstance(out, tuple):
    logits = out[0]
else:
    logits = out
print(f"Logits shape: {logits.shape}")
print(f"Forward time (ms): {(t1-t0)*1000:.2f}")

# Optional: run with return_attention True if supported
try:
    with torch.no_grad():
        t0 = time.time()
        out2 = model(x, return_attention=True)
        t1 = time.time()
    print(f"Forward(with attention) time (ms): {(t1-t0)*1000:.2f}")
    print(f"Out2 types: {type(out2)}")
except Exception as e:
    print(f"Return-attention forward not supported: {e}")

import os
import sys
import numpy as np

# Ensure project root is on path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from onnx_inference import ONNXModelWrapper
from config import PreprocessingConfig

MODEL = os.path.join(ROOT, "model_fp32.onnx")
print("Model path:", MODEL)

w = ONNXModelWrapper(MODEL)
print("ONNX available:", w.onnx_available)
if not w.onnx_available or w.session is None:
    print("ONNX session unavailable; exiting")
    sys.exit(0)

print("Session inputs:")
for inp in w.session.get_inputs():
    print(" -", inp.name, inp.shape, inp.type)

# Heuristic values from logs
num_frames = PreprocessingConfig().num_frames

# Simulate the 'live' input shape observed in logs (feat_dim=253)
live_feat = 253
seq2d = np.random.randn(num_frames, live_feat).astype(np.float32)
prox2d = np.zeros((num_frames, 1), dtype=np.float32)

print('\n--- Test: 2D seq (seq,feat) ---')
print('seq2d.shape before:', seq2d.shape)
print('prox2d.shape before:', prox2d.shape)
try:
    out = w.infer_onnx(seq2d, prox2d)
    print('infer_onnx success, out.shape =', np.array(out).shape)
except Exception as e:
    print('infer_onnx ERROR:', e)

# Test 3D input (batch, seq, feat)
seq3d = seq2d[np.newaxis, ...]
prox3d = np.zeros((1, num_frames, 1), dtype=np.float32)
print('\n--- Test: 3D seq (batch,seq,feat) ---')
print('seq3d.shape before:', seq3d.shape)
print('prox3d.shape before:', prox3d.shape)
try:
    out = w.infer_onnx(seq3d, prox3d)
    print('infer_onnx success, out.shape =', np.array(out).shape)
except Exception as e:
    print('infer_onnx ERROR:', e)

# Test with expected feature dim if available
sess_inp = w.session.get_inputs()[0]
expected_feat = None
if len(sess_inp.shape) >= 1 and sess_inp.shape[-1] is not None:
    expected_feat = int(sess_inp.shape[-1])

if expected_feat:
    print('\nSession expected feat dim:', expected_feat)
    seq_expected = np.random.randn(num_frames, expected_feat).astype(np.float32)
    prox_expected = np.zeros((num_frames, 1), dtype=np.float32)
    print('seq_expected.shape before:', seq_expected.shape)
    try:
        out = w.infer_onnx(seq_expected, prox_expected)
        print('infer_onnx success (expected dim), out.shape =', np.array(out).shape)
    except Exception as e:
        print('infer_onnx ERROR (expected dim):', e)
else:
    print('\nNo explicit expected feature dim reported by session input shape')

print('\nDone')

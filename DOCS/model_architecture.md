# Model Architecture — SignLanguageGRU

## Overview

The `SignLanguageGRU` model is a multi-branch deep learning classifier combining a Spatial GNN, Conv1D temporal frontend, Bidirectional GRU, and proximity-aware attention. All 10 architectural improvements (Phases 1–10) are enabled by default.

## Input

| Property | Value |
|---|---|
| Shape | `(batch, 20, 506)` |
| Sequence length | 20 frames (~667 ms at 30 FPS) |
| Feature dim | 506 = 253 base features × 2 (with velocity) |

## Data Flow

### Branch A — Spatial GNN (`src/training/spatial_gnn.py`)

**Input:** First 126 dims of each frame (raw hand landmark coordinates).

- Hand landmarks are reshaped to `(batch × 20, 2, 21, 3)` — 2 hands, 21 nodes, 3 coords
- A `LightweightSpatialGNN` applies 2-layer Graph Convolution over the anatomical hand skeleton (21 nodes per hand, edges = known metacarpal → proximal → medial → distal finger joint connections)
- GCN layer 1: Linear(3 → 16) + adjacency-weighted neighbor aggregation + ReLU
- GCN layer 2: Linear(16 → 8) + adjacency aggregation + ReLU
- Global max-pool over 21 nodes per hand → 8 dims per hand
- Both hands concatenated → **16 dims per frame**
- Optional: shared GNN weights between left/right hands (halves parameters)

### Branch B — Conv1D Frontend (Phase 1)

**Input:** All 506 dims of each frame.

- Pointwise Conv1d(506 → 128, kernel=1): feature mixing, dimension reduction
- Depthwise temporal Conv1d(128 → 128, kernel=3, groups=128, padding=1): local temporal pattern extraction
- Residual connection from pointwise output
- GroupNorm(8 groups) → ReLU → Dropout(0.1)
- **Output:** `(batch, 20, 128)`

### Fusion

- Concatenate GNN output (16) + Conv1D output (128) = **144 dims per frame**

### Learnable Frame Weighting (Phase 2)

- MLP: `Linear(144→32) → ReLU → Linear(32→1) → Sigmoid`
- Produces a scalar weight per frame: `weights.shape = (batch, 20, 1)`
- Applied as element-wise multiplication to the fused features
- Allows the model to soft-suppress uninformative transition frames

### Input Projection

- `Linear(144 → 64)` → `LayerNorm(64)` → `ReLU`
- Projects to GRU input space

### Bidirectional GRU (Phase 4)

| Property | Value |
|---|---|
| Layers | 3 stacked |
| Hidden dim | 64 per direction |
| Bidirectional | Yes (forward + backward) |
| Output dim | 128 (concatenated) |
| Inter-layer dropout | 0.30 |
| Post-GRU norm | LayerNorm(128) |

Output shape: `(batch, 20, 128)`

### HybridAttention

4 attention heads operating on GRU output:

| Head type | Count | Description |
|---|---|---|
| Standard temporal | 2 | Learn which frames carry the most information |
| Proximity-aware | 2 | Attention scores additively biased by `log N(prox; 0, σ²)` where σ=0.15 is learnable |

Each head has an independent **learnable temperature** clamped to [0.1, 10.0].
Each head output: 32 dims. All 4 concatenated → 128-dim context vector.

**Residual skips (Phases 5 & 9):**
- Phase 9: `context += gru_out.mean(dim=1)` — temporal mean residual
- Phase 5: `context += input_proj.mean(dim=1)` — input projection residual (if dims align)

### FC Classification Head

```
Dropout(0.25)
→ Linear(128 → 96)
→ ReLU
→ Dropout(0.25)
→ Linear(96 → num_classes)
→ logits (89 classes)
```

## Parameter Count

*(Parameter count computed from current implementation: `sum(p.numel() for p in model.parameters())`)*

| Component | Approximate Parameters |
|---|---|
| Spatial GNN | ~2K |
| Conv1D Frontend | ~70K |
| Frame Weighting MLP | ~5K |
| Input Projection | ~9K |
| BiGRU × 3 layers | ~225K |
| HybridAttention | ~20K |
| FC Head | ~13K |
| **Total** | **343,976 (~344K)** |

## ONNX Export

The model is exported to ONNX format (opset 18) using `scripts/export_onnx.py`:
- Dynamic batch size
- Fixed sequence length (20)
- Automatic `num_classes` inference from checkpoint
- Writes `*_metadata.json` alongside the ONNX file

Dynamic INT8 quantization is applied via `scripts/quantize_onnx.py`:
- ~75% size reduction (4.2 MB FP32 → ~1.05 MB INT8)
- 2–3× faster CPU inference vs PyTorch FP32

## Ablation Flags

All architectural phases can be individually toggled in `ArchitectureImprovementsConfig`:

| Flag | Phase | Default |
|---|---|---|
| `use_conv_frontend` | Phase 1 | `True` |
| `use_frame_weighting` | Phase 2 | `True` |
| `use_depthwise_temporal` | Phase 4 | `True` |
| `use_residual_gru_skip` | Phase 5 | `True` |
| `use_groupnorm` | Phase 6 | `True` |
| `use_residual_attention_skip` | Phase 9 | `True` |
| `use_gnn` | Phase 10 | `True` |

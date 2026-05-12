"""
PHASE 6: Debug utilities for model analysis and optimization.

Provides tools for:
- Tensor shape tracking through forward pass
- Parameter counting (total, per-layer, trainable vs frozen)
- FLOPs estimation
- Inference latency benchmarking
- Attention weight statistics
- Frame weight distribution visualization
- Comparison with baseline models
"""

import torch
import torch.nn as nn
import time
from typing import List, Dict, Any, Tuple, Optional
from collections import OrderedDict
import numpy as np

from config import get_config
from model import SignLanguageGRU
from dataset import ISLDataset

cfg = get_config()


class ShapeTracker(nn.Module):
    """Hooks into model layers to track tensor shapes during forward pass."""
    
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.shapes = OrderedDict()
        self.activations = OrderedDict()
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks on all named modules."""
        for name, module in self.model.named_modules():
            if name:  # Skip root module
                module.register_forward_hook(self._create_hook(name))
    
    def _create_hook(self, name: str):
        """Create a hook function for a specific module."""
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                self.shapes[name] = tuple(output.shape)
                # Store activation statistics for certain layers
                if 'attention' not in name:  # Skip attention for space
                    self.activations[name] = {
                        'mean': output.detach().mean().item(),
                        'std': output.detach().std().item(),
                        'min': output.detach().min().item(),
                        'max': output.detach().max().item(),
                    }
        return hook
    
    def forward(self, x: torch.Tensor, *args, **kwargs):
        """Forward pass with shape tracking."""
        self.shapes.clear()
        self.activations.clear()
        return self.model(x, *args, **kwargs)


def print_model_shapes(
    model: SignLanguageGRU,
    input_shape: Tuple[int, int, int] = (2, 20, 504),
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Track and print tensor shapes through model forward pass.
    
    Args:
        model: SignLanguageGRU instance
        input_shape: (batch_size, seq_len, input_dim)
        device: 'cpu' or 'cuda'
    
    Returns:
        Dictionary with shape and activation statistics
    """
    print("\n" + "="*80)
    print("PHASE 6: MODEL SHAPE TRACKING")
    print("="*80)
    
    model.to(device)
    model.eval()
    
    # Create tracker
    tracker = ShapeTracker(model)
    tracker.to(device)
    
    # Forward pass
    with torch.no_grad():
        x = torch.randn(*input_shape, device=device)
        _ = tracker(x)
    
    # Print results
    print(f"\nInput shape: {input_shape}")
    print("\nTensor shapes through model:")
    print("─" * 80)
    for layer_name, shape in tracker.shapes.items():
        print(f"  {layer_name:<40} → {str(shape):<30}")
    
    print("\nLayer activations (mean, std, min, max):")
    print("─" * 80)
    for layer_name, stats in tracker.activations.items():
        if 'bn' not in layer_name.lower() and 'norm' not in layer_name.lower():
            print(f"  {layer_name:<40} μ={stats['mean']:7.4f} σ={stats['std']:7.4f} " +
                  f"[{stats['min']:7.4f}, {stats['max']:7.4f}]")
    
    return {
        'shapes': tracker.shapes,
        'activations': tracker.activations,
    }


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count total parameters, trainable vs frozen, per layer.
    
    Args:
        model: Neural network model
    
    Returns:
        Dictionary with parameter counts
    """
    print("\n" + "="*80)
    print("PHASE 6: PARAMETER COUNT")
    print("="*80)
    
    total_params = 0
    trainable_params = 0
    frozen_params = 0
    
    per_layer = OrderedDict()
    
    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        
        if param.requires_grad:
            trainable_params += num_params
        else:
            frozen_params += num_params
        
        # Group by layer
        layer_name = name.split('.')[0]
        if layer_name not in per_layer:
            per_layer[layer_name] = {'total': 0, 'trainable': 0, 'frozen': 0}
        
        per_layer[layer_name]['total'] += num_params
        if param.requires_grad:
            per_layer[layer_name]['trainable'] += num_params
        else:
            per_layer[layer_name]['frozen'] += num_params
    
    # Print results
    print(f"\nTotal parameters: {total_params:,}")
    print(f"  ├─ Trainable: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
    print(f"  └─ Frozen: {frozen_params:,} ({100*frozen_params/total_params:.1f}%)")
    
    print("\nParameters per layer:")
    print("─" * 80)
    for layer_name, counts in per_layer.items():
        print(f"  {layer_name:<40} {counts['total']:>10,} " +
              f"(trainable: {counts['trainable']:>8,}, frozen: {counts['frozen']:>8,})")
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': frozen_params,
        'per_layer': per_layer,
    }


def estimate_flops(
    model: SignLanguageGRU,
    input_shape: Tuple[int, int, int] = (1, 20, 504),
) -> Dict[str, Any]:
    """
    Estimate FLOPs for model forward pass (approximate).
    
    Args:
        model: SignLanguageGRU instance
        input_shape: (batch_size, seq_len, input_dim)
    
    Returns:
        Dictionary with FLOP estimates
    """
    print("\n" + "="*80)
    print("PHASE 6: FLOPS ESTIMATION (APPROXIMATE)")
    print("="*80)
    
    batch, seq, in_dim = input_shape
    
    flops = {}
    total_flops = 0
    
    # Conv1D: 2 × (in_ch × out_ch × kernel × seq_len)
    if hasattr(model, 'conv1d') and model.conv1d is not None:
        conv1d_flops = 2 * in_dim * 256 * 3 * seq
        flops['Conv1D'] = conv1d_flops
        total_flops += conv1d_flops
    
    # Input projection: Linear + LayerNorm
    in_to_proj = 256 if flops.get('Conv1D') else in_dim
    input_proj_flops = 2 * in_to_proj * 128 * seq  # Linear
    input_proj_flops += 4 * seq * 128  # LayerNorm
    flops['Input Projection'] = input_proj_flops
    total_flops += input_proj_flops
    
    # GRU: 3 × (4 × (hidden × (hidden + input)) × seq)
    gru_flops = 3 * 4 * 128 * (128 + 128) * seq  # 3 layers
    flops['GRU'] = gru_flops
    total_flops += gru_flops
    
    # Layer Norm: 4 × hidden × seq
    layer_norm_flops = 4 * 256 * seq
    flops['Layer Norm'] = layer_norm_flops
    total_flops += layer_norm_flops
    
    # Attention: ~2 × (hidden × hidden × seq)
    attention_flops = 2 * 256 * 256 * seq
    flops['Attention'] = attention_flops
    total_flops += attention_flops
    
    # Spatial Attention: linear projection + softmax
    spatial_flops = 2 * 256 * 3 * seq + 3 * seq
    flops['Spatial Attention'] = spatial_flops
    total_flops += spatial_flops
    
    # FC Head: 2 × (256 × 96 + 96 × num_classes)
    fc_flops = 2 * 256 * 96 + 2 * 96 * 100  # Assuming 100 classes
    flops['FC Head'] = fc_flops
    total_flops += fc_flops
    
    # Print results
    print(f"\nEstimated FLOPs (batch_size={batch}, seq_len={seq}):")
    print("─" * 80)
    for component, comp_flops in flops.items():
        print(f"  {component:<40} {comp_flops/1e6:>10.2f}M FLOPs " +
              f"({100*comp_flops/total_flops:>5.1f}%)")
    
    print("─" * 80)
    print(f"  {'Total':<40} {total_flops/1e6:>10.2f}M FLOPs")
    
    return {
        'total_flops': total_flops,
        'per_component': flops,
    }


def benchmark_inference(
    model: SignLanguageGRU,
    input_shape: Tuple[int, int, int] = (1, 20, 504),
    num_runs: int = 100,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Benchmark model inference latency.
    
    Args:
        model: SignLanguageGRU instance
        input_shape: (batch_size, seq_len, input_dim)
        num_runs: Number of forward passes to average
        device: 'cpu' or 'cuda'
    
    Returns:
        Dictionary with timing statistics
    """
    print("\n" + "="*80)
    print("PHASE 6: INFERENCE LATENCY BENCHMARK")
    print("="*80)
    
    model.to(device)
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            x = torch.randn(*input_shape, device=device)
            _ = model(x)
    
    # Timing
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            x = torch.randn(*input_shape, device=device)
            
            if device == 'cuda':
                torch.cuda.synchronize()
            start = time.time()
            
            _ = model(x)
            
            if device == 'cuda':
                torch.cuda.synchronize()
            end = time.time()
            
            times.append((end - start) * 1000)  # ms
    
    times = np.array(times)
    
    # Print results
    print(f"\nBenchmark: {num_runs} runs on {device.upper()}")
    print(f"Input shape: {input_shape}")
    print("─" * 80)
    print(f"  Mean latency:    {times.mean():.3f}ms")
    print(f"  Std deviation:   {times.std():.3f}ms")
    print(f"  Min latency:     {times.min():.3f}ms")
    print(f"  Max latency:     {times.max():.3f}ms")
    print(f"  Median latency:  {np.median(times):.3f}ms")
    print(f"  Throughput (batch): {input_shape[0] / (times.mean() / 1000):.1f} samples/sec")
    print(f"  FPS (1 forward):    {1000 / times.mean():.1f} FPS")
    
    return {
        'mean': times.mean(),
        'std': times.std(),
        'min': times.min(),
        'max': times.max(),
        'median': np.median(times),
        'all_times': times,
    }


def analyze_frame_weights(
    model: SignLanguageGRU,
    dataset: ISLDataset,
    num_samples: int = 10,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Analyze learned frame weights across samples.
    
    Only meaningful if Phase 2 (frame weighting) is enabled.
    
    Args:
        model: SignLanguageGRU with frame weighting
        dataset: ISLDataset instance
        num_samples: Number of samples to analyze
        device: 'cpu' or 'cuda'
    
    Returns:
        Dictionary with frame weight statistics
    """
    if not hasattr(model, 'frame_weight_scorer') or model.frame_weight_scorer is None:
        print("\n[PHASE 2] Frame weighting not enabled. Skipping analysis.")
        return {}
    
    print("\n" + "="*80)
    print("PHASE 2: FRAME WEIGHT ANALYSIS")
    print("="*80)
    
    model.to(device)
    model.eval()
    
    all_weights = []
    
    with torch.no_grad():
        for i in range(min(num_samples, len(dataset))):
            x, _ = dataset[i]
            x = x.unsqueeze(0).to(device)  # (1, seq_len, input_dim)
            
            if hasattr(model, 'conv1d') and model.conv1d is not None:
                x_transposed = x.transpose(1, 2)
                x_conv = model.conv1d(x_transposed).transpose(1, 2)
            else:
                x_conv = x
            
            weights = model.frame_weight_scorer(x_conv)  # (1, seq_len, 1)
            all_weights.append(weights.squeeze().cpu().numpy())
    
    all_weights = np.array(all_weights)
    
    # Print results
    print(f"\nFrame weight statistics across {num_samples} samples:")
    print("─" * 80)
    print(f"  Mean weight:     {all_weights.mean():.4f}")
    print(f"  Std deviation:   {all_weights.std():.4f}")
    print(f"  Min weight:      {all_weights.min():.4f}")
    print(f"  Max weight:      {all_weights.max():.4f}")
    
    # Per-frame statistics
    per_frame = all_weights.mean(axis=0)
    print(f"\nPer-frame average weights:")
    print("─" * 80)
    for frame_idx, weight in enumerate(per_frame):
        bar = "█" * int(weight * 50)
        print(f"  Frame {frame_idx:2d}: {weight:.4f} {bar}")
    
    return {
        'all_weights': all_weights,
        'per_frame_mean': per_frame,
        'global_mean': all_weights.mean(),
        'global_std': all_weights.std(),
    }


def print_architecture_summary(model: SignLanguageGRU):
    """Print summary of enabled architectural improvements (PHASE 1–7)."""
    print("\n" + "="*80)
    print("ARCHITECTURE IMPROVEMENTS SUMMARY (PHASE 1–7)")
    print("="*80)
    
    cfg = get_config()
    arch = cfg.arch_improvements
    
    print(f"\n✓ PHASE 1: Conv frontend")
    print(f"    Enabled: {arch.use_conv_frontend}")
    if arch.use_conv_frontend:
        print(f"    Output channels: {arch.conv_frontend_out_channels}")
        print(f"    Pointwise kernel: {arch.conv_frontend_pointwise_kernel}")
        print(f"    Dropout: {arch.conv_frontend_dropout}")
    
    print(f"\n✓ PHASE 2: Learnable Frame Weighting")
    print(f"    Enabled: {arch.use_frame_weighting}")
    if arch.use_frame_weighting:
        print(f"    Init strategy: {arch.frame_weight_init}")
    
    print(f"\n✓ PHASE 4: Reduced Dropout (0.35 → 0.25)")
    print(f"    GRU dropout: {arch.gru_dropout}")
    print(f"    FC dropout: {arch.fc_dropout}")
    
    print(f"\n✓ PHASE 5: Residual GRU Skip Connection")
    print(f"    Enabled: {arch.use_residual_gru_skip}")
    
    print(f"\n✓ PHASE 3: Live Inference Optimization")
    print(f"    Use TTA: {cfg.live_inference.use_tta}")
    print(f"    Ensemble size: {cfg.live_inference.ensemble_size}")
    print(f"    Print latency stats: {cfg.live_inference.print_latency_stats}")
    
    print(f"\n✓ PHASE 6: Debug Mode")
    print(f"    Print shapes: {arch.debug_print_shapes}")
    print(f"    Layer stats: {arch.debug_layer_stats}")
    print(f"\n✓ PHASE 8: Conv Ablations")
    print(f"    Depthwise temporal: {arch.use_depthwise_temporal}")
    print(f"    Residual in conv frontend: {arch.use_residual_conv}")
    print(f"    Use GroupNorm: {arch.use_groupnorm}")


if __name__ == "__main__":
    """
    Example usage:
    
    python debug_model.py
    
    This will:
    1. Load the model
    2. Print architecture summary
    3. Track tensor shapes
    4. Count parameters
    5. Estimate FLOPs
    6. Benchmark inference
    7. Analyze frame weights (if enabled)
    """
    
    # Configuration
    cfg.validate()
    
    # Load model
    print("Loading SignLanguageGRU...")
    model = SignLanguageGRU(num_classes=100)
    
    # Phase 6: Debug analysis
    print_architecture_summary(model)
    print_model_shapes(model, input_shape=(2, 20, cfg.frame_features.input_sequence_dim))
    count_parameters(model)
    estimate_flops(model, input_shape=(1, 20, cfg.frame_features.input_sequence_dim))
    benchmark_inference(model, input_shape=(1, 20, cfg.frame_features.input_sequence_dim), num_runs=50)
    
    # Frame weight analysis (if enabled)
    # load_dataset = ISLDataset(split='train', num_frames=20)
    # analyze_frame_weights(model, load_dataset, num_samples=10)
    
    print("\n" + "="*80)
    print("Debug analysis complete!")
    print("="*80 + "\n")

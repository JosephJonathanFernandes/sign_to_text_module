"""
Comprehensive training configuration tuning script.
Tests different combinations of:
  - USE_FOCAL_LOSS (True/False)
  - MIXUP_PROB (0.3, 0.5, 0.7)
  - FOCAL_GAMMA (1.0, 2.0, 3.0)

Usage:
    # Test all combinations (18 runs × 5 folds = ~lots of time!)
    python tune_training_config.py --full
    
    # Test recommended promising configs (4 runs × 5 folds)
    python tune_training_config.py --quick
    
    # Custom: test specific settings
    python tune_training_config.py --use-focal --mixup-probs 0.3 0.5 --focal-gammas 2.0
    
    # Just test focal loss (2 runs)
    python tune_training_config.py --test-focal-only
    
    # Just test mixup probability (3 runs)
    python tune_training_config.py --test-mixup-only

Example expected runtime:
  - Quick (4 configs):  ~2-3 hours
  - Full (18 configs):  ~10-15 hours
  - Focal only (2):     ~1 hour
  - Mixup only (3):     ~1.5 hours

Results saved to: tune_results_config.txt
"""

import os
import time
import argparse
import numpy as np
import torch
from itertools import product
from config import get_config

cfg = get_config()

# Convenience references
NUM_FOLDS = cfg.paths.num_folds
RANDOM_SEED = cfg.training.random_seed


def format_config_name(use_focal, mixup_prob, focal_gamma):
    """Format config for display."""
    focal_str = "FL" if use_focal else "CE"
    return f"{focal_str}|Mxp{mixup_prob:.1f}|FG{focal_gamma:.1f}"


def run_single_config(use_focal, mixup_prob, focal_gamma, config_idx, total_configs):
    """
    Run k-fold training with specific config.
    
    Args:
        use_focal: Whether to use focal loss
        mixup_prob: Probability of applying mixup (0.0-1.0)
        focal_gamma: Focal loss gamma parameter
        config_idx: Current config number (for display)
        total_configs: Total number of configs to test
    
    Returns:
        dict with results or error
    """
    config.USE_FOCAL_LOSS = use_focal
    config.MIXUP_PROB = mixup_prob
    config.FOCAL_GAMMA = focal_gamma
    
    config_name = format_config_name(use_focal, mixup_prob, focal_gamma)
    
    print(f"\n{'='*70}")
    print(f"  Config {config_idx}/{total_configs}: {config_name}")
    print(f"  USE_FOCAL_LOSS={use_focal}, MIXUP_PROB={mixup_prob}, FOCAL_GAMMA={focal_gamma}")
    print(f"{'='*70}\n")
    
    fold_start = time.time()
    
    try:
        fold_accs = train_kfold()
        fold_time = time.time() - fold_start
        
        mean_acc = np.mean(fold_accs)
        std_acc = np.std(fold_accs)
        
        result = {
            'config_name': config_name,
            'use_focal': use_focal,
            'mixup_prob': mixup_prob,
            'focal_gamma': focal_gamma,
            'fold_accuracies': fold_accs,
            'mean_acc': mean_acc,
            'std_acc': std_acc,
            'min_acc': np.min(fold_accs),
            'max_acc': np.max(fold_accs),
            'training_time': fold_time,
            'error': None,
        }
        
        print(f"\n✓ Results for {config_name}:")
        print(f"  Fold accuracies: {[f'{a:.1f}%' for a in fold_accs]}")
        print(f"  Mean: {mean_acc:.2f}% ± {std_acc:.2f}%")
        print(f"  Min-Max: {np.min(fold_accs):.2f}% - {np.max(fold_accs):.2f}%")
        print(f"  Training time: {fold_time:.1f}s ({fold_time/60:.1f}m)")
        
        return result
        
    except Exception as e:
        print(f"\n✗ ERROR for {config_name}: {e}")
        import traceback
        traceback.print_exc()
        return {
            'config_name': config_name,
            'use_focal': use_focal,
            'mixup_prob': mixup_prob,
            'focal_gamma': focal_gamma,
            'error': str(e),
        }


def run_tuning(configs):
    """
    Run k-fold training for multiple configurations.
    
    Args:
        configs: List of (use_focal, mixup_prob, focal_gamma) tuples
    
    Returns:
        List of result dicts
    """
    print("\n" + "="*70)
    print(f"  Training Configuration Tuning")
    print(f"  Testing {len(configs)} configurations")
    print(f"  Using {NUM_FOLDS}-Fold Cross-Validation")
    print("="*70 + "\n")
    
    results = []
    start_time = time.time()
    
    for idx, (use_focal, mixup_prob, focal_gamma) in enumerate(configs, 1):
        result = run_single_config(use_focal, mixup_prob, focal_gamma, idx, len(configs))
        results.append(result)
    
    total_time = time.time() - start_time
    
    # Summary
    print(f"\n\n{'='*70}")
    print("  SUMMARY & COMPARISON")
    print("="*70 + "\n")
    
    # Sort by mean accuracy (successful runs only)
    successful = [r for r in results if r.get('error') is None]
    if not successful:
        print("  No successful runs!")
        return results
    
    successful_sorted = sorted(successful, key=lambda x: x['mean_acc'], reverse=True)
    
    print(f"  {'Rank':<5} {'Config':<18} {'Mean Acc':<10} {'Std Dev':<10} {'Min-Max':<15} {'Time':<8}")
    print(f"  {'-'*70}")
    
    for rank, res in enumerate(successful_sorted, 1):
        config_name = res['config_name']
        mean = res['mean_acc']
        std = res['std_acc']
        min_acc = res['min_acc']
        max_acc = res['max_acc']
        time_str = f"{res['training_time']/60:.1f}m"
        
        print(
            f"  {rank:<5} {config_name:<18} {mean:6.2f}%  "
            f"±{std:5.2f}%  {min_acc:5.1f}%-{max_acc:5.1f}%  {time_str:>8}"
        )
    
    # Failed runs
    failed = [r for r in results if r.get('error') is not None]
    if failed:
        print(f"\n  {'─'*70}")
        print(f"  Failed configurations ({len(failed)}):")
        for res in failed:
            print(f"    {res['config_name']}: {res['error']}")
    
    # Top recommendation
    print(f"\n  {'─'*70}")
    best = successful_sorted[0]
    print(f"  BEST CONFIG:")
    print(f"    {best['config_name']}")
    print(f"    USE_FOCAL_LOSS={best['use_focal']}")
    print(f"    MIXUP_PROB={best['mixup_prob']}")
    print(f"    FOCAL_GAMMA={best['focal_gamma']}")
    print(f"    Mean Accuracy: {best['mean_acc']:.2f}% ± {best['std_acc']:.2f}%")
    print(f"  {'─'*70}\n")
    
    # Save to file
    results_file = "tune_results_config.txt"
    with open(results_file, 'w') as f:
        f.write("Training Configuration Tuning Results\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total time: {total_time:.1f}s ({total_time/60:.1f}m)\n")
        f.write(f"Configurations tested: {len(configs)}\n")
        f.write(f"Successful: {len(successful)}, Failed: {len(failed)}\n")
        f.write("="*70 + "\n\n")
        
        # Sorted results
        f.write("Ranked Results (by mean accuracy):\n\n")
        f.write(f"  {'Rank':<5} {'Config':<18} {'Mean Acc':<10} {'Std Dev':<10} {'Min-Max':<15}\n")
        f.write(f"  {'-'*70}\n")
        
        for rank, res in enumerate(successful_sorted, 1):
            config_name = res['config_name']
            mean = res['mean_acc']
            std = res['std_acc']
            min_acc = res['min_acc']
            max_acc = res['max_acc']
            
            f.write(
                f"  {rank:<5} {config_name:<18} {mean:6.2f}%  "
                f"±{std:5.2f}%  {min_acc:5.1f}%-{max_acc:5.1f}%\n"
            )
        
        # Detailed results
        f.write("\n\nDetailed Results:\n")
        f.write("="*70 + "\n\n")
        
        for res in successful_sorted:
            f.write(f"Configuration: {res['config_name']}\n")
            f.write(f"  Settings:\n")
            f.write(f"    USE_FOCAL_LOSS: {res['use_focal']}\n")
            f.write(f"    MIXUP_PROB:     {res['mixup_prob']}\n")
            f.write(f"    FOCAL_GAMMA:    {res['focal_gamma']}\n")
            f.write(f"  Results:\n")
            f.write(f"    Fold accuracies: {[f'{a:.1f}%' for a in res['fold_accuracies']]}\n")
            f.write(f"    Mean accuracy:   {res['mean_acc']:.2f}%\n")
            f.write(f"    Std dev:         {res['std_acc']:.2f}%\n")
            f.write(f"    Min/Max:         {res['min_acc']:.2f}% / {res['max_acc']:.2f}%\n")
            f.write(f"    Training time:   {res['training_time']:.1f}s\n\n")
        
        # Recommendation
        f.write("\nRECOMMENDED CONFIGURATION:\n")
        f.write("="*70 + "\n")
        f.write(f"Config: {best['config_name']}\n")
        f.write(f"USE_FOCAL_LOSS = {best['use_focal']}\n")
        f.write(f"MIXUP_PROB = {best['mixup_prob']}\n")
        f.write(f"FOCAL_GAMMA = {best['focal_gamma']}\n")
        f.write(f"Mean Accuracy: {best['mean_acc']:.2f}% ± {best['std_acc']:.2f}%\n")
    
    print(f"✓ Results saved to: {results_file}\n")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tune training configuration parameters"
    )
    parser.add_argument(
        '--full', action='store_true',
        help='Test all combinations (18 configs, ~10-15 hours)'
    )
    parser.add_argument(
        '--quick', action='store_true', default=True,
        help='Test recommended promising configs (4 configs, ~2-3 hours) [DEFAULT]'
    )
    parser.add_argument(
        '--test-focal-only', action='store_true',
        help='Test only USE_FOCAL_LOSS (2 configs, ~1 hour)'
    )
    parser.add_argument(
        '--test-mixup-only', action='store_true',
        help='Test only MIXUP_PROB (3 configs, ~1.5 hours)'
    )
    parser.add_argument(
        '--test-gamma-only', action='store_true',
        help='Test only FOCAL_GAMMA (3 configs, ~1.5 hours)'
    )
    parser.add_argument(
        '--use-focal', action='store_true', dest='force_use_focal',
        help='Only test with USE_FOCAL_LOSS=True'
    )
    parser.add_argument(
        '--mixup-probs', type=float, nargs='+', dest='mixup_probs',
        help='Custom MIXUP_PROB values to test'
    )
    parser.add_argument(
        '--focal-gammas', type=float, nargs='+', dest='focal_gammas',
        help='Custom FOCAL_GAMMA values to test'
    )
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    
    # Determine which configs to test
    if args.full:
        # All combinations: 2 × 3 × 3 = 18
        use_focal_values = [False, True]
        mixup_probs = [0.3, 0.5, 0.7]
        focal_gammas = [1.0, 2.0, 3.0]
        configs = list(product(use_focal_values, mixup_probs, focal_gammas))
        print(f"\nFull tuning: {len(configs)} configurations")
        
    elif args.test_focal_only:
        # Only focal loss: 2 configs
        configs = [
            (False, 0.5, 2.0),  # CE baseline
            (True, 0.5, 2.0),   # Focal loss
        ]
        print(f"\nTesting focal loss only: {len(configs)} configurations")
        
    elif args.test_mixup_only:
        # Only mixup prob: 3 configs
        configs = [
            (True, 0.3, 2.0),
            (True, 0.5, 2.0),
            (True, 0.7, 2.0),
        ]
        print(f"\nTesting mixup probability only: {len(configs)} configurations")
        
    elif args.test_gamma_only:
        # Only focal gamma: 3 configs
        configs = [
            (True, 0.5, 1.0),
            (True, 0.5, 2.0),
            (True, 0.5, 3.0),
        ]
        print(f"\nTesting focal gamma only: {len(configs)} configurations")
        
    elif args.mixup_probs or args.focal_gammas or args.force_use_focal:
        # Custom combination
        use_focal_values = [True] if args.force_use_focal else [False, True]
        mixup_probs = args.mixup_probs or [0.3, 0.5, 0.7]
        focal_gammas = args.focal_gammas or [1.0, 2.0, 3.0]
        configs = list(product(use_focal_values, mixup_probs, focal_gammas))
        print(f"\nCustom tuning: {len(configs)} configurations")
        
    else:
        # Quick: 4 promising configurations
        configs = [
            (False, 0.5, 2.0),  # Baseline (CE + moderate mixup)
            (True, 0.5, 2.0),   # Focal + moderate mixup
            (True, 0.3, 2.0),   # Focal + low mixup
            (True, 0.7, 2.0),   # Focal + high mixup
        ]
        print(f"\nQuick tuning (DEFAULT): {len(configs)} configurations")
    
    print(f"Expected time: ~{len(configs) * 30 / 60:.0f}m ({len(configs) * NUM_FOLDS} fold runs)\n")
    
    results = run_tuning(configs)
    print("✓ Tuning completed!")

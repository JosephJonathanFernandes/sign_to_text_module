"""
Hyperparameter tuning script for CLASS_WEIGHT_POWER.
Tests multiple values (0.5, 0.7, 1.0) and reports comparative results.

Usage:
    python tune_class_weight_power.py
    
    Or specify power values:
    python tune_class_weight_power.py --powers 0.3 0.5 0.7 0.9 1.0

Results are saved to tune_results.txt
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
from config import NUM_FOLDS, ENSEMBLE_DIR, RANDOM_SEED
from dataset import ISLDataset
from train import train_kfold
import config

def run_tuning(power_values=None):
    """
    Run k-fold training with different CLASS_WEIGHT_POWER values.
    
    Args:
        power_values: List of power values to test. Default: [0.5, 0.7, 1.0]
    """
    if power_values is None:
        power_values = [0.5, 0.7, 1.0]
    
    print("\n" + "="*70)
    print("  CLASS_WEIGHT_POWER Tuning")
    print(f"  Testing values: {power_values}")
    print(f"  Using {NUM_FOLDS}-Fold Cross-Validation")
    print("="*70 + "\n")
    
    results = {}
    start_time = time.time()
    
    for power in power_values:
        print(f"\n{'─'*70}")
        print(f"  Testing CLASS_WEIGHT_POWER = {power}")
        print(f"{'─'*70}\n")
        
        # Update config
        config.CLASS_WEIGHT_POWER = power
        
        fold_start = time.time()
        
        # Run k-fold training
        try:
            fold_accs = train_kfold()
            fold_time = time.time() - fold_start
            
            mean_acc = np.mean(fold_accs)
            # Calculate overall accuracy manually
            overall_acc = mean_acc  # Approximation for comparison
            
            results[power] = {
                'fold_accuracies': fold_accs,
                'mean_acc': mean_acc,
                'overall_acc': overall_acc,
                'training_time': fold_time,
            }
            
            print(f"\nPower={power} Results:")
            print(f"  Fold accuracies: {[f'{a:.1f}%' for a in fold_accs]}")
            print(f"  Mean accuracy:   {mean_acc:.2f}%")
            print(f"  Overall accuracy: {overall_acc:.2f}%")
            print(f"  Training time:    {fold_time:.1f}s ({fold_time/60:.1f}m)")
            
        except Exception as e:
            print(f"ERROR with power={power}: {e}")
            import traceback
            traceback.print_exc()
            results[power] = {'error': str(e)}
    
    # Summary and comparison
    total_time = time.time() - start_time
    
    print(f"\n\n{'='*70}")
    print("  Summary & Comparison")
    print("="*70 + "\n")
    
    best_power = None
    best_acc = 0.0
    
    for power in sorted(results.keys()):
        if 'error' in results[power]:
            print(f"  Power={power}: ERROR - {results[power]['error']}")
        else:
            res = results[power]
            mean = res['mean_acc']
            overall = res['overall_acc']
            print(
                f"  Power={power:3.1f}  | "
                f"Mean: {mean:6.2f}%  | "
                f"Overall: {overall:6.2f}%  | "
                f"Time: {res['training_time']:6.1f}s"
            )
            
            if mean > best_acc:
                best_acc = mean
                best_power = power
    
    # Recommendation
    print(f"\n  {'─'*70}")
    if best_power is not None:
        print(f"  RECOMMENDED: CLASS_WEIGHT_POWER = {best_power}")
        print(f"             Achieved {best_acc:.2f}% mean accuracy")
    print(f"  Total tuning time: {total_time:.1f}s ({total_time/60:.1f}m)")
    print(f"  {'─'*70}\n")
    
    # Save results to file
    results_file = "tune_results.txt"
    with open(results_file, 'w') as f:
        f.write("CLASS_WEIGHT_POWER Tuning Results\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total time: {total_time:.1f}s\n")
        f.write("="*70 + "\n\n")
        
        for power in sorted(results.keys()):
            if 'error' in results[power]:
                f.write(f"Power={power}: ERROR\n")
            else:
                res = results[power]
                f.write(f"Power={power}:\n")
                f.write(f"  Fold accuracies: {[f'{a:.1f}%' for a in res['fold_accuracies']]}\n")
                f.write(f"  Mean accuracy:   {res['mean_acc']:.2f}%\n")
                f.write(f"  Overall accuracy: {res['overall_acc']:.2f}%\n")
                f.write(f"  Training time:    {res['training_time']:.1f}s\n\n")
        
        if best_power:
            f.write(f"RECOMMENDATION: Use CLASS_WEIGHT_POWER = {best_power}\n")
            f.write(f"Achieved:       {best_acc:.2f}% mean accuracy\n")
    
    print(f"Results saved to: {results_file}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune CLASS_WEIGHT_POWER parameter")
    parser.add_argument(
        '--powers', type=float, nargs='+', default=[0.5, 0.7, 1.0],
        help='Power values to test (default: 0.5 0.7 1.0)'
    )
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    
    results = run_tuning(args.powers)
    
    print("Tuning completed!")

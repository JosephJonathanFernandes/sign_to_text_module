import os
import sys
import time
import json
import numpy as np
import torch
import psutil
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from sklearn.metrics import confusion_matrix, classification_report

sys.path.insert(0, os.getcwd())
from src.core.config import get_config
from src.preprocessing.dataset import ISLDataset
from src.inference.ensemble import load_ensemble, ensemble_predict, check_ood

cfg = get_config()
DEVICE = cfg.hardware.torch_device

def compute_ece(confidences, accuracies, num_bins=10):
    """Calculate Expected Calibration Error"""
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    indices = np.digitize(confidences, bins, right=True) - 1
    
    ece = 0.0
    for b in range(num_bins):
        mask = indices == b
        if np.any(mask):
            bin_acc = np.mean(accuracies[mask])
            bin_conf = np.mean(confidences[mask])
            bin_weight = np.sum(mask) / len(confidences)
            ece += bin_weight * np.abs(bin_acc - bin_conf)
            
    return ece * 100.0

# --- Corruptions ---

def apply_gaussian_noise(sequence, sigma):
    """Add Gaussian noise to the coordinates."""
    if sigma <= 0: return sequence
    noise = np.random.normal(0, sigma, sequence.shape)
    return sequence + noise

def apply_landmark_dropout(sequence, drop_prob):
    """Randomly zero out landmarks across all frames."""
    if drop_prob <= 0: return sequence
    mask = np.random.rand(sequence.shape[1]) > drop_prob
    corrupted = sequence.copy()
    corrupted[:, ~mask] = 0.0
    return corrupted

def apply_frame_blackout(sequence, drop_prob):
    """Randomly zero out entire frames."""
    if drop_prob <= 0: return sequence
    mask = np.random.rand(sequence.shape[0]) > drop_prob
    corrupted = sequence.copy()
    corrupted[~mask, :] = 0.0
    return corrupted

def run_degradation_experiment(models, ds, test_indices, corruption_fn, param_list, name):
    results = {}
    print(f"\n--- EXPERIMENT: {name} Degradation Curve ---")
    for param in param_list:
        correct = 0
        total = len(test_indices)
        for i in test_indices:
            seq_t, _, _, _, _ = ds[i]
            sequence = seq_t.numpy()
            cls_idx = ds.samples[i][1]
            
            # Apply corruption
            corrupted_seq = corruption_fn(sequence, param)
            
            pred_idx, _, _ = ensemble_predict(models, corrupted_seq)
            if pred_idx == cls_idx:
                correct += 1
                
        acc = (correct / total) * 100
        print(f"{name} {param}: {acc:.2f}%")
        results[param] = acc
    return results

def generate_synthetic_ood(sequence):
    """Generate a synthetic OOD sequence by randomizing temporal order and adding high noise."""
    corrupted = sequence.copy()
    np.random.shuffle(corrupted)
    noise = np.random.normal(0, 0.1, corrupted.shape)
    return corrupted + noise

def evaluate_ood(models, ds, test_indices):
    print("\n--- EXPERIMENT 7: Synthetic OOD Detection Evaluation ---")
    
    # 1. False Rejection Rate (In-Distribution incorrectly rejected)
    in_dist_rejected = 0
    in_dist_total = len(test_indices)
    
    for i in test_indices:
        seq_t, _, _, _, _ = ds[i]
        sequence = seq_t.numpy()
        
        _, _, all_probs = ensemble_predict(models, sequence)
        is_ood, _ = check_ood(all_probs)
        if is_ood:
            in_dist_rejected += 1
            
    # 2. True Positive Rate (Synthetic OOD correctly rejected)
    ood_rejected = 0
    ood_total = len(test_indices)
    
    for i in test_indices:
        seq_t, _, _, _, _ = ds[i]
        sequence = seq_t.numpy()
        
        # Make it OOD
        ood_seq = generate_synthetic_ood(sequence)
        
        _, _, all_probs = ensemble_predict(models, ood_seq)
        is_ood, _ = check_ood(all_probs)
        if is_ood:
            ood_rejected += 1
            
    tpr = (ood_rejected / ood_total) * 100
    frr = (in_dist_rejected / in_dist_total) * 100
    far = ((ood_total - ood_rejected) / ood_total) * 100
    
    precision = ood_rejected / (ood_rejected + in_dist_rejected + 1e-9)
    recall = ood_rejected / ood_total
    
    print(f"True Positive Rate (Correctly rejected OOD): {tpr:.2f}%")
    print(f"False Rejection Rate (Incorrectly rejected valid): {frr:.2f}%")
    print(f"False Acceptance Rate (Accepted OOD as valid): {far:.2f}%")
    print(f"OOD Precision: {precision:.4f}")
    print(f"OOD Recall: {recall:.4f}")
    
class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()

def evaluate_baseline():
    os.makedirs("benchmarks", exist_ok=True)
    os.makedirs("diagrams", exist_ok=True)
    sys.stdout = Logger(os.path.join("benchmarks", "robustness_report.txt"))

    print("Loading models...")
    models, classes, num_classes = load_ensemble()
    if not models:
        print("No models found!")
        return

    print("Loading dataset...")
    # Load raw dataset without augmentation to ensure controlled evaluation
    ds = ISLDataset(augment=False, oversample=False)
    
    print("\n--- EXPERIMENT 1 & 2: Clean Baseline & Domain-wise Accuracy ---")
    
    domain_to_idx = ds.domain_to_idx
    idx_to_domain = {v: k for k, v in domain_to_idx.items()}
    
    domain_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    confidences = []
    accuracies = []
    
    # Identify real domains for degradation evaluation (webcam + MVI)
    real_domains = [domain_to_idx.get("webcam", -1), domain_to_idx.get("MVI", -1)]
    test_indices = [i for i, (_, _, _, d) in enumerate(ds.samples) if d in real_domains]
    
    # Subset to 1000 samples for speed
    if len(test_indices) > 1000:
        np.random.seed(42)
        test_indices = np.random.choice(test_indices, 1000, replace=False).tolist()
    
    # Track latency
    latencies = []
    start_mem = psutil.Process().memory_info().rss / 1024 / 1024
    
    # We only evaluate real-domain samples for the clean baseline to save massive compute time
    for i in test_indices:
        seq_t, _, _, _, _ = ds[i]
        sequence = seq_t.numpy()
        
        cls_idx = ds.samples[i][1]
        d_idx = ds.samples[i][3]
            
        t0 = time.perf_counter()
        pred_idx, conf, all_probs = ensemble_predict(models, sequence)
        t1 = time.perf_counter()
        
        latencies.append((t1 - t0) * 1000) # ms
        
        is_correct = (pred_idx == cls_idx)
        domain_name = idx_to_domain.get(d_idx, "unknown")
        
        domain_stats[domain_name]["correct"] += is_correct
        domain_stats[domain_name]["total"] += 1
        
        domain_stats["ALL"]["correct"] += is_correct
        domain_stats["ALL"]["total"] += 1
        
        confidences.append(float(conf))
        accuracies.append(int(is_correct))
        
        if i % 500 == 0:
            print(f"Processed {i} test samples...")
            
    end_mem = psutil.Process().memory_info().rss / 1024 / 1024
            
    print("\n[Domain-wise Accuracy]")
    for dom, stats in domain_stats.items():
        if stats["total"] > 0:
            acc = (stats["correct"] / stats["total"]) * 100
            print(f"{dom.ljust(15)}: {acc:.2f}% ({stats['correct']}/{stats['total']})")
            
    print("\n[Performance Metrics]")
    print(f"Avg Inference Latency: {np.mean(latencies):.2f} ms")
    print(f"P95 Inference Latency: {np.percentile(latencies, 95):.2f} ms")
    print(f"Estimated FPS:         {1000.0 / np.mean(latencies):.0f} fps")
    print(f"Memory Usage delta:    {end_mem - start_mem:.2f} MB")
    
    print("\n--- EXPERIMENT 3 & 9: Per-class Accuracy & Confusion Matrix ---")
    y_true = []
    y_pred = []
    for i in test_indices:
        seq_t, _, _, _, _ = ds[i]
        sequence = seq_t.numpy()
        cls_idx = ds.samples[i][1]
        
        pred_idx, _, _ = ensemble_predict(models, sequence)
        y_true.append(cls_idx)
        y_pred.append(pred_idx)
        
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    
    # Per-class accuracy
    print("\n[Worst 10 Classes by Accuracy]")
    class_accs = []
    for c in range(num_classes):
        total_c = np.sum(cm[c, :])
        if total_c > 0:
            acc = cm[c, c] / total_c
            class_accs.append((acc, classes[c], cm[c, c], total_c))
    class_accs.sort()
    for acc, name, corr, tot in class_accs[:10]:
        print(f"  {name.ljust(20)}: {acc*100:.1f}% ({corr}/{tot})")
        
    # Most confused pairs
    print("\n[Top 10 Most Confused Pairs]")
    np.fill_diagonal(cm, 0)
    confused_indices = np.dstack(np.unravel_index(np.argsort(cm.ravel()), cm.shape))[0]
    confused_indices = confused_indices[::-1] # descending
    for r, c in confused_indices[:10]:
        if cm[r, c] > 0:
            print(f"  True: {classes[r].ljust(15)} -> Pred: {classes[c].ljust(15)} ({cm[r, c]} times)")
            
    # Per-class Precision/Recall/F1
    print("\n[Per-Class Metrics]")
    print(classification_report(y_true, y_pred, labels=np.arange(len(classes)), target_names=classes, zero_division=0))
    
    # Save Confusion Matrix Visualization
    print("\nSaving Confusion Matrix Visualization to 'diagrams/confusion_matrix.png'...")
    plt.figure(figsize=(20, 20))
    sns.heatmap(cm, xticklabels=classes, yticklabels=classes, cmap="Blues", annot=False, cbar=True)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join("diagrams", "confusion_matrix.png"), dpi=150)
    plt.close()
    
    print("\n--- EXPERIMENT 8: Reliability Diagram + ECE ---")
    ece = compute_ece(np.array(confidences), np.array(accuracies))
    print(f"Expected Calibration Error (ECE): {ece:.2f}%")
    
    if test_indices:
        print(f"\nRunning degradation curves on {len(test_indices)} real-domain samples...")
        run_degradation_experiment(models, ds, test_indices, apply_gaussian_noise, [0.0, 0.005, 0.01, 0.02, 0.03], "Gaussian Noise")
        run_degradation_experiment(models, ds, test_indices, apply_landmark_dropout, [0.0, 0.05, 0.10, 0.15, 0.20], "Landmark Dropout")
        run_degradation_experiment(models, ds, test_indices, apply_frame_blackout, [0.0, 0.05, 0.10, 0.15, 0.20], "Frame Blackout")
        
        # Evaluate OOD
        evaluate_ood(models, ds, test_indices)

if __name__ == "__main__":
    evaluate_baseline()

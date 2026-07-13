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
    if sigma <= 0: return sequence
    noise = np.random.normal(0, sigma, sequence.shape)
    return sequence + noise

def apply_landmark_dropout(sequence, drop_prob):
    if drop_prob <= 0: return sequence
    mask = np.random.rand(sequence.shape[1]) > drop_prob
    corrupted = sequence.copy()
    corrupted[:, ~mask] = 0.0
    return corrupted

def apply_frame_blackout(sequence, drop_prob):
    if drop_prob <= 0: return sequence
    mask = np.random.rand(sequence.shape[0]) > drop_prob
    corrupted = sequence.copy()
    corrupted[~mask, :] = 0.0
    return corrupted

def generate_synthetic_ood(sequence):
    corrupted = sequence.copy()
    np.random.shuffle(corrupted)
    noise = np.random.normal(0, 0.1, corrupted.shape)
    return corrupted + noise

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

def get_reject_category(fpath):
    fpath = fpath.replace("\\", "/")
    if "processed_negatives" in fpath:
        parts = fpath.split("/")
        try:
            idx = parts.index("processed_negatives")
            if idx + 1 < len(parts):
                # If there's no subfolder and it's just files, it might be the file itself
                if parts[idx + 1].endswith(".npy"):
                    return "root"
                return parts[idx + 1]
        except ValueError:
            pass
    return "unknown"

def run_degradation_experiment(models, ds, test_indices, corruption_fn, param_list, name):
    print(f"\n--- EXPERIMENT: {name} Degradation Curve ---")
    for param in param_list:
        correct = 0
        total = len(test_indices)
        for i in test_indices:
            seq_t, _, _, _, _ = ds[i]
            sequence = seq_t.numpy()
            cls_idx = ds.samples[i][1]
            corrupted_seq = corruption_fn(sequence, param)
            pred_idx, _, _ = ensemble_predict(models, corrupted_seq)
            if pred_idx == cls_idx:
                correct += 1
        acc = (correct / total) * 100
        print(f"{name} {param}: {acc:.2f}%")

def evaluate_baseline():
    os.makedirs("benchmarks", exist_ok=True)
    os.makedirs("diagrams", exist_ok=True)
    sys.stdout = Logger(os.path.join("benchmarks", "robustness_report.txt"))

    print("Loading models...")
    models, classes, num_classes = load_ensemble()
    if not models:
        return

    print("Loading dataset...")
    neg_root_p = os.path.join(os.path.dirname(cfg.paths.processed_dir), "processed_negatives")
    ds = ISLDataset(augment=False, oversample=False, neg_root=neg_root_p)
    
    reject_cls_idx = ds.class_to_idx.get("__reject__", -1)
    domain_to_idx = ds.domain_to_idx
    idx_to_domain = {v: k for k, v in domain_to_idx.items()}
    
    print("\n--- 1. Pipeline Verification ---")
    print(f"Loaded {len(ds.samples)} total dataset samples.")
    print(f"__reject__ class index: {reject_cls_idx}")
    
    real_domains = [domain_to_idx.get("webcam", -1), domain_to_idx.get("MVI", -1)]
    
    test_indices = []
    valid_sign_indices = []
    reject_indices = []
    
    for i, (fpath, c_idx, _, d_idx) in enumerate(ds.samples):
        if c_idx != reject_cls_idx:
            if d_idx in real_domains:
                test_indices.append(i)
                valid_sign_indices.append(i)
        else:
            test_indices.append(i)
            reject_indices.append(i)
            
    print(f"Test set (natural distribution) includes {len(test_indices)} samples.")
    print(f"Of these, {len(valid_sign_indices)} are valid signs and {len(reject_indices)} are __reject__ samples.")
    
    y_true = []
    y_pred = []
    y_conf = []
    fpaths = []
    domains = []
    accuracies = []
    
    print("\nEvaluating test set...")
    for i, idx in enumerate(test_indices):
        seq_t, _, _, _, _ = ds[idx]
        sequence = seq_t.numpy()
        cls_idx = ds.samples[idx][1]
        fpath = ds.samples[idx][0]
        d_idx = ds.samples[idx][3]
        
        pred_idx, conf, _ = ensemble_predict(models, sequence)
        
        y_true.append(cls_idx)
        y_pred.append(pred_idx)
        y_conf.append(float(conf))
        fpaths.append(fpath)
        
        d_str = idx_to_domain.get(d_idx, "unknown")
        # If it's a negative sample, let's categorize its domain by its original filename prefix if possible
        if cls_idx == reject_cls_idx:
            fname = os.path.basename(fpath)
            if fname.startswith("webcam"):
                d_str = "webcam"
            elif fname.startswith("MVI"):
                d_str = "MVI"
        domains.append(d_str)
        accuracies.append(int(pred_idx == cls_idx))
        
        if (i+1) % 500 == 0:
            print(f"Processed {i+1} samples...")
            
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_conf = np.array(y_conf)
    accuracies = np.array(accuracies)
    fpaths = np.array(fpaths)
    domains = np.array(domains)
    
    print("\n--- 2. Overall Classification Metrics ---")
    rep = classification_report(y_true, y_pred, labels=np.arange(len(classes)), target_names=classes, zero_division=0, output_dict=True)
    print(f"Overall Accuracy: {rep['accuracy']:.4f}")
    print(f"Macro Precision: {rep['macro avg']['precision']:.4f}")
    print(f"Macro Recall: {rep['macro avg']['recall']:.4f}")
    print(f"Macro F1: {rep['macro avg']['f1-score']:.4f}")
    print(f"Weighted F1: {rep['weighted avg']['f1-score']:.4f}")
    
    if reject_cls_idx != -1:
        print("\n--- 3. __reject__ Metrics ---")
        rej_stats = rep[classes[reject_cls_idx]]
        print(f"Precision: {rej_stats['precision']:.4f}")
        print(f"Recall:    {rej_stats['recall']:.4f}")
        print(f"F1-score:  {rej_stats['f1-score']:.4f}")
        print(f"Support:   {rej_stats['support']}")
        
        # Calculate TP, FP, FN
        tp = np.sum((y_true == reject_cls_idx) & (y_pred == reject_cls_idx))
        fp = np.sum((y_true != reject_cls_idx) & (y_pred == reject_cls_idx))
        fn = np.sum((y_true == reject_cls_idx) & (y_pred != reject_cls_idx))
        print(f"True Positives (TP): {tp}")
        print(f"False Positives (FP): {fp} (Valid signs incorrectly rejected)")
        print(f"False Negatives (FN): {fn} (Reject samples incorrectly accepted as valid signs)")
        
        print("\n--- 4. Per-category Reject Analysis ---")
        reject_mask = y_true == reject_cls_idx
        reject_fpaths = fpaths[reject_mask]
        reject_preds = y_pred[reject_mask]
        
        cat_stats = defaultdict(lambda: {"correct": 0, "total": 0})
        for rp, rpred in zip(reject_fpaths, reject_preds):
            cat = get_reject_category(rp)
            cat_stats[cat]["total"] += 1
            if rpred == reject_cls_idx:
                cat_stats[cat]["correct"] += 1
                
        for cat, stats in cat_stats.items():
            acc = (stats["correct"] / stats["total"]) * 100
            print(f"{cat.ljust(20)}: Recall {acc:.2f}% ({stats['correct']}/{stats['total']})")
            
    print("\n--- 5. Per-domain Analysis ---")
    domain_stats = defaultdict(lambda: {"signs_correct": 0, "signs_total": 0, "reject_correct": 0, "reject_total": 0})
    for yt, yp, d in zip(y_true, y_pred, domains):
        if yt == reject_cls_idx:
            domain_stats[d]["reject_total"] += 1
            if yp == yt:
                domain_stats[d]["reject_correct"] += 1
        else:
            domain_stats[d]["signs_total"] += 1
            if yp == yt:
                domain_stats[d]["signs_correct"] += 1
                
    for d, stats in domain_stats.items():
        print(f"[{d}]")
        st = stats["signs_total"]
        rt = stats["reject_total"]
        if st > 0:
            print(f"  Signs Acc:  {(stats['signs_correct']/st)*100:.2f}% ({stats['signs_correct']}/{st})")
        if rt > 0:
            print(f"  Reject Acc: {(stats['reject_correct']/rt)*100:.2f}% ({stats['reject_correct']}/{rt})")

    print("\n--- 6. Confusion Matrix & Error Analysis ---")
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(classes)))
    
    if reject_cls_idx != -1:
        print("\n[Top valid signs incorrectly classified as __reject__ (FRR contributors)]")
        fn_indices = np.argsort(cm[:, reject_cls_idx])[::-1]
        for idx in fn_indices[:5]:
            if idx != reject_cls_idx and cm[idx, reject_cls_idx] > 0:
                print(f"  {classes[idx].ljust(15)}: {cm[idx, reject_cls_idx]} times")
                
        print("\n[Top valid signs that __reject__ is incorrectly classified as (FAR contributors)]")
        fp_indices = np.argsort(cm[reject_cls_idx, :])[::-1]
        for idx in fp_indices[:5]:
            if idx != reject_cls_idx and cm[reject_cls_idx, idx] > 0:
                print(f"  {classes[idx].ljust(15)}: {cm[reject_cls_idx, idx]} times")
                
    # Save Confusion Matrix Visualization
    plt.figure(figsize=(24, 24))
    sns.heatmap(cm, xticklabels=classes, yticklabels=classes, cmap="Blues", annot=False, cbar=True)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join("diagrams", "confusion_matrix.png"), dpi=150)
    plt.close()
    
    if reject_cls_idx != -1:
        print("\n--- 7. Top Misclassifications Involving __reject__ ---")
        print("Saving to diagrams/misclassified_rejects.txt")
        with open("diagrams/misclassified_rejects.txt", "w") as f:
            f.write("Filename | True Label | Predicted Label | Confidence\n")
            f.write("-" * 80 + "\n")
            
            # Find cases where reject -> good, or good -> reject
            mis_mask = ((y_true == reject_cls_idx) & (y_pred != reject_cls_idx)) | ((y_true != reject_cls_idx) & (y_pred == reject_cls_idx))
            mis_indices = np.where(mis_mask)[0]
            
            # Sort by confidence (highest confidence errors first)
            mis_conf = y_conf[mis_indices]
            sorted_mis = np.argsort(mis_conf)[::-1]
            
            for rank, sm_idx in enumerate(sorted_mis[:100]):
                orig_idx = mis_indices[sm_idx]
                fname = os.path.basename(fpaths[orig_idx])
                true_lbl = classes[y_true[orig_idx]]
                pred_lbl = classes[y_pred[orig_idx]]
                conf_val = y_conf[orig_idx]
                f.write(f"{fname.ljust(40)} | {true_lbl.ljust(15)} | {pred_lbl.ljust(15)} | {conf_val:.4f}\n")
                
    print("\n--- 8. Confidence Analysis ---")
    valid_mask = (y_true != reject_cls_idx)
    reject_mask = (y_true == reject_cls_idx)
    
    if np.any(valid_mask):
        print("Valid Signs Confidence:")
        print(f"  Average: {np.mean(y_conf[valid_mask]):.4f}")
        print(f"  Median:  {np.median(y_conf[valid_mask]):.4f}")
        
    if np.any(reject_mask):
        print("__reject__ Confidence:")
        print(f"  Average: {np.mean(y_conf[reject_mask]):.4f}")
        print(f"  Median:  {np.median(y_conf[reject_mask]):.4f}")
        
    # Generate Confidence Distribution Plot
    if np.any(valid_mask) and np.any(reject_mask):
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        sns.histplot(y_conf[valid_mask], color='blue', alpha=0.5, label='Valid Signs', bins=20, stat='density')
        sns.histplot(y_conf[reject_mask], color='red', alpha=0.5, label='__reject__', bins=20, stat='density')
        plt.title("Confidence Histogram")
        plt.xlabel("Confidence")
        plt.ylabel("Density")
        plt.legend()
        
        plt.subplot(1, 2, 2)
        data = [y_conf[valid_mask], y_conf[reject_mask]]
        plt.boxplot(data, tick_labels=['Valid Signs', '__reject__'])
        plt.title("Confidence Boxplot")
        plt.ylabel("Confidence")
        
        plt.tight_layout()
        plt.savefig(os.path.join("diagrams", "confidence_distribution.png"), dpi=150)
        plt.close()
        print("Saved Confidence Distribution plot to 'diagrams/confidence_distribution.png'")
        
    ece = compute_ece(y_conf, accuracies)
    print(f"\nExpected Calibration Error (ECE): {ece:.2f}%")

    if reject_cls_idx != -1:
        print("\n--- 9. Threshold Sweep (ROC Analysis for Reject) ---")
        print("Sweeping probability threshold for accepting a sign (rejecting if below threshold or predicted as reject)")
        print(f"{'Threshold':<10} | {'Precision':<10} | {'Recall':<10} | {'FAR':<10} | {'FRR':<10}")
        print("-" * 60)
        
        # We need raw probabilities for threshold sweep, but ensemble_predict only returns max conf.
        # Let's do a fast re-eval for sweeping. Wait, we don't have all_probs saved.
        # Since the user specifically wants ROC on confidence, let's just use y_conf.
        # If predicted class is __reject__, it's rejected.
        # If predicted class is NOT __reject__ but confidence < threshold, it's rejected.
        for thresh in np.arange(0.2, 1.0, 0.1):
            # A sample is rejected if pred == reject OR conf < thresh
            is_rejected = (y_pred == reject_cls_idx) | (y_conf < thresh)
            
            # Ground truth is reject
            is_true_reject = (y_true == reject_cls_idx)
            
            tp_rej = np.sum(is_rejected & is_true_reject)
            fp_rej = np.sum(is_rejected & ~is_true_reject)
            tn_rej = np.sum(~is_rejected & ~is_true_reject)
            fn_rej = np.sum(~is_rejected & is_true_reject)
            
            prec = tp_rej / (tp_rej + fp_rej + 1e-9)
            rec = tp_rej / (tp_rej + fn_rej + 1e-9)
            far = fn_rej / (fn_rej + tn_rej + 1e-9) # OOD accepted as valid
            frr = fp_rej / (fp_rej + tn_rej + 1e-9) # Valid rejected
            
            print(f"{thresh:<10.1f} | {prec:<10.4f} | {rec:<10.4f} | {far:<10.4%} | {frr:<10.4%}")
            
    print("\n--- 10. Synthetic Stress Test ---")
    print("WARNING: This tests purely synthetic corruptions (Gaussian noise, temporal shuffling).")
    print("These are OUTSIDE the training distribution and evaluate algorithmic stability, not real-world deployment performance.")
    
    # We only run synthetic stress tests on valid signs (to see if they get misclassified or rejected)
    if len(valid_sign_indices) > 0:
        # Subset for speed
        if len(valid_sign_indices) > 500:
            np.random.seed(42)
            stress_indices = np.random.choice(valid_sign_indices, 500, replace=False).tolist()
        else:
            stress_indices = valid_sign_indices
            
        run_degradation_experiment(models, ds, stress_indices, apply_gaussian_noise, [0.01, 0.03], "Gaussian Noise")
        run_degradation_experiment(models, ds, stress_indices, apply_landmark_dropout, [0.10, 0.20], "Landmark Dropout")
        
        print("\nSynthetic OOD (Pure Noise/Shuffling):")
        ood_rejected = 0
        total_ood = len(stress_indices)
        for i in stress_indices:
            seq_t, _, _, _, _ = ds[i]
            ood_seq = generate_synthetic_ood(seq_t.numpy())
            pred_idx, conf, all_probs = ensemble_predict(models, ood_seq)
            
            # Rejected if it predicts __reject__ OR fails threshold
            is_ood, _ = check_ood(all_probs)
            if pred_idx == reject_cls_idx or is_ood:
                ood_rejected += 1
                
        print(f"Synthetic OOD Rejection Rate: {(ood_rejected / total_ood) * 100:.2f}%")
        print(f"Synthetic FAR (Accepted OOD as valid): {((total_ood - ood_rejected) / total_ood) * 100:.2f}%")

if __name__ == "__main__":
    evaluate_baseline()

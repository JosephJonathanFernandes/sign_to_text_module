"""
Phase-1 audit steps 2-5 (READ-ONLY w.r.t. repo; writes only into repro_audit/).

2. Recompute train.py's split  (_disjoint_stratified_split, val_split=0.30, seed=42)
3. Overlap of each script's evaluated set with train_idx / val_idx
4. Held-out metrics of the existing checkpoint on val_idx only
5. Near-duplicate distances val -> nearest same-class train sample
"""
import json, os, sys, time
import numpy as np, torch, h5py
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix)

sys.path.insert(0, os.getcwd())
from src.core.config import get_config
from src.preprocessing.dataset import ISLDataset
from src.training.model import SignLanguageGRU
from src.training.train import _disjoint_stratified_split

cfg = get_config()
OUT = "repro_audit"
LOG = open(os.path.join(OUT, "01_report.txt"), "w", encoding="utf-8")
def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()

VAL_SPLIT, SEED = cfg.training.val_split, cfg.training.random_seed
P(f"val_split={VAL_SPLIT} seed={SEED} batch={cfg.training.batch_size} epochs={cfg.training.num_epochs}")

# ---------- datasets exactly as the two scripts / train.py build them ----------
neg_root = os.path.join(os.path.dirname(cfg.paths.processed_dir), "processed_negatives")
# train.py:  ISLDataset(augment=False, min_samples=2, oversample=False, neg_root=..., archived_root=None)
train_ds = ISLDataset(augment=False, min_samples=2, oversample=False, neg_root=neg_root)
# evaluate_robustness.py: ISLDataset(augment=False, oversample=False, neg_root=neg_root)
rob_ds = ISLDataset(augment=False, oversample=False, neg_root=neg_root)
assert len(train_ds.samples) == len(rob_ds.samples)
labels = np.array([s[1] for s in train_ds.samples])
assert (labels == np.array([s[1] for s in rob_ds.samples])).all(), "label order differs between train and eval dataset"
N = len(labels)
reject_idx = train_ds.class_to_idx["__reject__"]
P(f"N={N} classes={len(train_ds.classes)} reject_idx={reject_idx} reject_count={(labels==reject_idx).sum()}")

# ---------- step 2: split ----------
train_idx, val_idx = _disjoint_stratified_split(train_ds.samples, labels, VAL_SPLIT, SEED)
assert len(set(train_idx) & set(val_idx)) == 0
P(f"\n[2] split: train={len(train_idx)} val={len(val_idx)} (sum={len(train_idx)+len(val_idx)})")
np.savez(os.path.join(OUT, "recomputed_split_seed42.npz"), train_idx=train_idx, val_idx=val_idx)
is_train = np.zeros(N, bool); is_train[train_idx] = True
is_val = np.zeros(N, bool); is_val[val_idx] = True

# ---------- step 3: sets evaluated by each script ----------
dom = np.array([s[3] for s in rob_ds.samples])
d2i = rob_ds.domain_to_idx
real_domains = [d2i.get("webcam", -1), d2i.get("MVI", -1)]
# evaluate_robustness.py lines 134-141
rob_eval = np.array([i for i in range(N)
                     if (labels[i] != reject_idx and dom[i] in real_domains) or labels[i] == reject_idx])
# generate_report_metrics.py lines 20-40 (dataset WITHOUT neg_root => first 89883 samples only)
wc = d2i["webcam"]; unk = d2i.get("unknown", -1)
n_h5 = rob_ds.num_hdf5_samples
grm_test = np.array([i for i in range(n_h5) if dom[i] == wc])
grm_val  = np.array([i for i in range(n_h5) if dom[i] != wc and dom[i] != unk])
overlap = {}
for name, idx in [("evaluate_robustness.py test set", rob_eval),
                  ("generate_report_metrics.py 'unseen/test' (webcam domain)", grm_test),
                  ("generate_report_metrics.py 'validation' (non-webcam domains)", grm_val)]:
    a = int(is_train[idx].sum()); b = int(is_val[idx].sum())
    overlap[name] = dict(evaluated=int(len(idx)), in_train_idx=a, in_val_idx=b,
                         pct_in_train=round(100*a/len(idx), 2))
    P(f"[3] {name}: evaluated={len(idx)} in_train_idx={a} ({100*a/len(idx):.2f}%) in_val_idx={b} ({100*b/len(idx):.2f}%)")
json.dump(overlap, open(os.path.join(OUT, "overlap.json"), "w"), indent=2)

# ---------- forward pass of existing checkpoint on ALL samples ----------
ck = torch.load("models/model.pth", map_location="cpu", weights_only=False)
model = SignLanguageGRU(num_classes=ck["num_classes"], num_domains=0)
model.load_state_dict(ck["model_state_dict"]); model.eval()
classes = ck["classes"]
assert classes == train_ds.classes or sorted(classes) == sorted(train_ds.classes), "class list mismatch"
P(f"\ncheckpoint: epoch={ck['epoch']} val_acc stored={ck['val_acc']:.3f} num_classes={ck['num_classes']}")
if classes != train_ds.classes:
    P("WARNING: checkpoint class ORDER differs from dataset class order -> remapping not attempted")

preds_path = os.path.join(OUT, "preds_all_single_model.npy")
if os.path.exists(preds_path):
    preds = np.load(preds_path)
else:
    preds = np.zeros(N, dtype=np.int64)
    loader = DataLoader(rob_ds, batch_size=256, shuffle=False, num_workers=0)
    k = 0; t0 = time.time()
    with torch.no_grad():
        for seq, prox, lab, w, d in loader:
            out = model(seq, proximity=prox)["sign_logits"]
            preds[k:k+len(seq)] = out.argmax(1).numpy(); k += len(seq)
            if (k // 256) % 40 == 0: P(f"  forward {k}/{N} ({time.time()-t0:.0f}s)")
    np.save(preds_path, preds)

# sanity: accuracy by subset
P("\n[3b] single-model accuracy by subset (same forward pass):")
for name, idx in [("evaluate_robustness set", rob_eval), ("  of which in train_idx", rob_eval[is_train[rob_eval]]),
                  ("  of which in val_idx", rob_eval[is_val[rob_eval]]),
                  ("webcam-domain (generate_report_metrics 'unseen')", grm_test),
                  ("  of which in train_idx", grm_test[is_train[grm_test]]),
                  ("  of which in val_idx", grm_test[is_val[grm_test]])]:
    P(f"   {name}: n={len(idx)} acc={100*accuracy_score(labels[idx], preds[idx]):.2f}%")

# ---------- step 4: held-out metrics on val_idx ----------
def macro(yt, yp, lab_set):
    p, r, f, _ = precision_recall_fscore_support(yt, yp, labels=lab_set, average="macro", zero_division=0)
    return 100*p, 100*r, 100*f
yt, yp = labels[val_idx], preds[val_idx]
all_labels = list(range(len(classes)))
sign_labels = [i for i in all_labels if i != reject_idx]
P(f"\n[4] HELD-OUT (val_idx only, n={len(val_idx)}), existing checkpoint, single-model forward")
P(f"   accuracy (all 301 incl. reject samples): {100*accuracy_score(yt, yp):.2f}%")
P("   macro P/R/F1, all 301 labels      : %.2f / %.2f / %.2f" % macro(yt, yp, all_labels))
P("   macro P/R/F1, 300 sign labels only: %.2f / %.2f / %.2f  (reject samples still counted as errors)" % macro(yt, yp, sign_labels))
m = yt != reject_idx
P(f"   accuracy on true-sign val samples only (n={m.sum()}): {100*accuracy_score(yt[m], yp[m]):.2f}%")
P("   macro P/R/F1, sign samples only, 300 labels: %.2f / %.2f / %.2f" % macro(yt[m], yp[m], sign_labels))
tp = ((yt == reject_idx) & (yp == reject_idx)).sum(); fp = ((yt != reject_idx) & (yp == reject_idx)).sum()
fn = ((yt == reject_idx) & (yp != reject_idx)).sum()
P(f"   reject: support={int((yt==reject_idx).sum())} TP={tp} FP={fp} FN={fn} precision={100*tp/max(tp+fp,1):.2f}% recall={100*tp/max(tp+fn,1):.2f}%")
p, r, f, s = precision_recall_fscore_support(yt, yp, labels=all_labels, zero_division=0)
with open(os.path.join(OUT, "heldout_per_class.csv"), "w", encoding="utf-8") as fh:
    fh.write("class,precision,recall,f1,support\n")
    for i in all_labels:
        fh.write(f"{classes[i]},{p[i]:.4f},{r[i]:.4f},{f[i]:.4f},{s[i]}\n")
np.save(os.path.join(OUT, "heldout_confusion_matrix.npy"), confusion_matrix(yt, yp, labels=all_labels))
P("   per-class table -> repro_audit/heldout_per_class.csv ; confusion matrix -> heldout_confusion_matrix.npy")

# ---------- step 5: near-duplicates ----------
P("\n[5] nearest same-class TRAIN neighbour for every val sample (flattened 20x506 features after _prepare_sequence, augment=False)")
h5 = h5py.File(rob_ds.h5_path, "r")
def feats(idx):
    idx = np.sort(np.asarray(idx)); out = []
    h = idx[idx < n_h5]; n = idx[idx >= n_h5]
    if len(h):
        raw = h5["features"][h.tolist()] if len(h) < 2 else h5["features"][h.tolist()]
        for s in raw: out.append(ISLDataset._prepare_sequence(s.copy(), augment=False)[0].reshape(-1))
    for i in n:
        s = np.load(rob_ds.samples[i][0]).astype(np.float32)
        out.append(ISLDataset._prepare_sequence(s, augment=False)[0].reshape(-1))
    return idx, np.stack(out).astype(np.float32)

nn_d = np.full(N, np.nan); nn_rel = np.full(N, np.nan); nn_loo_train = np.full(N, np.nan)
for c in np.unique(labels):
    tr = train_idx[labels[train_idx] == c]; va = val_idx[labels[val_idx] == c]
    if len(tr) == 0 or len(va) == 0: continue
    tr_s, Xt = feats(tr); va_s, Xv = feats(va)
    Xt_t, Xv_t = torch.from_numpy(Xt), torch.from_numpy(Xv)
    D = torch.cdist(Xv_t, Xt_t)
    nn_d[va_s] = D.min(1).values.numpy()
    nn_rel[va_s] = nn_d[va_s] / np.maximum(np.linalg.norm(Xv, axis=1), 1e-9)
    if len(tr) > 1:   # train->train leave-one-out reference
        Dt = torch.cdist(Xt_t, Xt_t); Dt.fill_diagonal_(float("inf"))
        nn_loo_train[tr_s] = Dt.min(1).values.numpy()
v = nn_d[val_idx]; vr = nn_rel[val_idx]; ok = ~np.isnan(v)
P(f"   val samples with a same-class train neighbour: {ok.sum()}/{len(val_idx)}")
P("   abs L2 distance percentiles (1,5,25,50,75,95,99): " + ", ".join(f"{np.percentile(v[ok],q):.3f}" for q in (1,5,25,50,75,95,99)))
P("   relative (d/||x||) percentiles (1,5,25,50,75,95,99): " + ", ".join(f"{np.percentile(vr[ok],q):.4f}" for q in (1,5,25,50,75,95,99)))
loo = nn_loo_train[train_idx]; loo = loo[~np.isnan(loo)]
P(f"   reference: train->train leave-one-out NN abs dist median={np.median(loo):.3f}")
P(f"   exact duplicates (d<=1e-6): {(v[ok]<=1e-6).sum()} ({100*(v[ok]<=1e-6).mean():.2f}%)")
for t in (0.01, 0.02, 0.05, 0.10):
    P(f"   near-duplicate if d/||x|| <= {t:.2f}: {(vr[ok]<=t).sum()} ({100*(vr[ok]<=t).mean():.2f}%)")
P("   THRESHOLD RATIONALE: no ground-truth duplicate labels exist (h5 stores no filenames/parent ids). 1e-6 = bit-identical copy;"
  " 2% relative L2 ~ jitter smaller than the training augmentation noise; 5% is a lenient bound. The 5% figure is a convention, not a measured boundary.")
w = np.array([s[2] for s in rob_ds.samples]); dnames = rob_ds.domains
P("\n   composition of val_idx by domain: " + str({dnames[d]: int((dom[val_idx]==d).sum()) for d in np.unique(dom[val_idx]) if d < 4}))
P("   unique sample weights in h5 (flag for derived data?): " + str(np.unique(w, return_counts=True)))
np.savez(os.path.join(OUT, "nn_distances_val.npz"), idx=val_idx, d=nn_d[val_idx], rel=nn_rel[val_idx])
P("\nDONE")

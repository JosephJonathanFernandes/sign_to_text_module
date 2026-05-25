"""Visualize quality discriminator scores and sample embeddings.

Outputs:
- score histogram for real vs synthetic
- PCA or t-SNE scatter of embeddings
- optional accepted/rejected previews
"""

from __future__ import annotations

import argparse
import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from quality_discriminator import DiscriminatorConfig, QualityDiscriminator, SequenceQualityDataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize quality discriminator outputs")
    p.add_argument("--checkpoint", default="models/quality_discriminator.pt")
    p.add_argument("--real-root", default="processed")
    p.add_argument("--synthetic-root", default="generated")
    p.add_argument("--output-dir", default="logs/quality_discriminator_vis")
    p.add_argument("--seq-len", type=int, default=20)
    p.add_argument("--feature-dim", type=int, default=506)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--threshold", type=float, default=0.8)
    p.add_argument("--mode", choices=["pca", "tsne"], default="pca")
    p.add_argument("--max-samples", type=int, default=2500)
    return p.parse_args()


def load_model(checkpoint_path: str, device: torch.device) -> tuple[QualityDiscriminator, dict[str, Any]]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    model_cfg = DiscriminatorConfig(**ckpt["model_config"])
    model = QualityDiscriminator(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def collect_scores(model: QualityDiscriminator, loader: DataLoader, device: torch.device):
    scores = []
    labels = []
    emb = []
    meta = []
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        logits = model(x, y if model.use_class_conditioning else None)
        prob = torch.sigmoid(logits).detach().cpu().numpy()
        scores.append(prob)
        labels.append(batch["target"].cpu().numpy())
        emb.append(batch["x"].mean(dim=1).cpu().numpy())
        meta.extend([
            {"class_name": c, "path": p, "is_real": bool(r), "score": float(s)}
            for c, p, r, s in zip(batch["class_name"], batch["path"], batch["is_real"], prob)
        ])
    return np.concatenate(scores), np.concatenate(labels), np.concatenate(emb), meta


def pca_2d(x: np.ndarray) -> np.ndarray:
    x = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    return x @ vt[:2].T


def tsne_2d(x: np.ndarray) -> np.ndarray:
    try:
        from sklearn.manifold import TSNE  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("t-SNE requested but scikit-learn is not installed") from exc
    return TSNE(n_components=2, init="pca", learning_rate="auto", random_state=42).fit_transform(x)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_model(args.checkpoint, device)

    dataset = SequenceQualityDataset(
        real_root=args.real_root,
        synthetic_root=args.synthetic_root,
        seq_len=args.seq_len,
        feature_dim=args.feature_dim,
        class_aware=model.use_class_conditioning,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    scores, labels, emb, meta = collect_scores(model, loader, device)
    if len(scores) > args.max_samples:
        scores = scores[: args.max_samples]
        labels = labels[: args.max_samples]
        emb = emb[: args.max_samples]
        meta = meta[: args.max_samples]

    hist_path = os.path.join(args.output_dir, "score_histogram.png")
    plt.figure(figsize=(10, 6))
    plt.hist(scores[labels >= 0.5], bins=40, alpha=0.7, label="real", density=True)
    plt.hist(scores[labels < 0.5], bins=40, alpha=0.7, label="synthetic", density=True)
    plt.axvline(args.threshold, color="red", linestyle="--", label=f"threshold={args.threshold}")
    plt.xlabel("P(sample_is_real)")
    plt.ylabel("Density")
    plt.title("Quality Discriminator Score Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(hist_path, dpi=180)
    plt.close()

    if args.mode == "tsne":
        emb_2d = tsne_2d(emb)
    else:
        emb_2d = pca_2d(emb)

    scatter_path = os.path.join(args.output_dir, f"embedding_{args.mode}.png")
    plt.figure(figsize=(10, 8))
    plt.scatter(emb_2d[labels >= 0.5, 0], emb_2d[labels >= 0.5, 1], s=8, alpha=0.55, label="real")
    plt.scatter(emb_2d[labels < 0.5, 0], emb_2d[labels < 0.5, 1], s=8, alpha=0.55, label="synthetic")
    plt.title(f"Sequence Embeddings ({args.mode.upper()})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(scatter_path, dpi=180)
    plt.close()

    accepted = [m for m in meta if m["score"] >= args.threshold and not m["is_real"]]
    rejected = [m for m in meta if m["score"] < args.threshold and not m["is_real"]]

    summary = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "histogram_path": hist_path,
        "scatter_path": scatter_path,
        "threshold": args.threshold,
        "num_samples": int(len(scores)),
        "accepted_synthetic": len(accepted),
        "rejected_synthetic": len(rejected),
    }
    summary_path = os.path.join(args.output_dir, "quality_visualization_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        import json

        json.dump(summary, f, indent=2)

    print(f"Saved: {hist_path}")
    print(f"Saved: {scatter_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()

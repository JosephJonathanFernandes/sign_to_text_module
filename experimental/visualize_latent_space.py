"""Visualize latent space embeddings for a trained CVAE model.

- Computes class-conditioned latent means from real sequences.
- Projects to 2D via PCA (always available) or t-SNE (optional sklearn).
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from cvae_landmarks import CVAEModelConfig, LandmarkCVAE, ProcessedLandmarkDataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize CVAE latent embeddings")
    p.add_argument("--checkpoint", default="models/cvae_landmarks.pt")
    p.add_argument("--processed-root", default="processed")
    p.add_argument("--output", default="logs/cvae_landmarks/latent_pca.png")
    p.add_argument("--mode", choices=["pca", "tsne"], default="pca")
    p.add_argument("--max-samples", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def pca_2d(z: np.ndarray) -> np.ndarray:
    z_center = z - z.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(z_center, full_matrices=False)
    comp = vt[:2]
    return z_center @ comp.T


def tsne_2d(z: np.ndarray, seed: int) -> np.ndarray:
    try:
        from sklearn.manifold import TSNE  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("t-SNE requested but scikit-learn is not installed") from exc

    model = TSNE(n_components=2, random_state=seed, init="pca", learning_rate="auto")
    return model.fit_transform(z)


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)

    model_cfg = CVAEModelConfig(**ckpt["model_config"])
    model = LandmarkCVAE(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dataset = ProcessedLandmarkDataset(
        args.processed_root,
        seq_len=ckpt.get("seq_len", model.cfg.seq_len),
        feature_dim=ckpt.get("feature_dim", model.cfg.feature_dim),
        exclude_prefixes=("cvae_",),
    )

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    z_all = []
    y_all = []
    n_seen = 0
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            mu, _ = model.encode(x, y)
            z_all.append(mu.detach().cpu().numpy())
            y_all.append(y.detach().cpu().numpy())
            n_seen += x.size(0)
            if n_seen >= args.max_samples:
                break

    z = np.concatenate(z_all, axis=0)[: args.max_samples]
    y = np.concatenate(y_all, axis=0)[: args.max_samples]

    if args.mode == "tsne":
        z2 = tsne_2d(z, args.seed)
    else:
        z2 = pca_2d(z)

    plt.figure(figsize=(10, 8))
    classes = dataset.label_encoder.classes
    for class_id in sorted(np.unique(y).tolist()):
        mask = y == class_id
        label = classes[int(class_id)] if int(class_id) < len(classes) else str(class_id)
        plt.scatter(z2[mask, 0], z2[mask, 1], s=7, alpha=0.55, label=label)

    plt.title(f"CVAE Latent Space ({args.mode.upper()})")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    # Keep legend compact for many classes.
    if len(np.unique(y)) <= 25:
        plt.legend(markerscale=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(args.output, dpi=180)
    plt.close()

    print(f"Saved latent visualization: {args.output}")


if __name__ == "__main__":
    main()

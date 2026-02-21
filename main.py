"""
Main entry point for the ISL Word Recognition Pipeline.

Usage:
    python main.py                  # Run full pipeline (preprocess + train)
    python main.py --preprocess     # Only preprocess videos
    python main.py --train          # Only train (requires processed/ to exist)
    python main.py --predict VIDEO  # Predict class for a single video
"""

import argparse
import os
import sys
import torch

from config import DEVICE, MODEL_SAVE_PATH, PROCESSED_DIR


def run_preprocess():
    """Run the preprocessing pipeline."""
    from preprocess import preprocess_dataset
    print("=" * 60)
    print("  STEP 1: Preprocessing Videos -> Landmarks (.npy)")
    print("=" * 60)
    stats = preprocess_dataset()
    total = sum(stats.values())
    print(f"\nTotal processed: {total} videos across {len(stats)} classes\n")
    return stats


def run_train():
    """Run the training pipeline."""
    from train import create_data_loaders, train

    print("=" * 60)
    print("  STEP 2: Training Bidirectional GRU + Attention")
    print("=" * 60)

    if not os.path.exists(PROCESSED_DIR):
        print(f"[ERROR] Processed directory not found: {PROCESSED_DIR}")
        print("        Run with --preprocess first.")
        sys.exit(1)

    train_loader, val_loader, num_classes = create_data_loaders()
    model = train(train_loader, val_loader, num_classes)
    print("\nTraining complete!\n")
    return model


def run_predict(video_path: str):
    """Predict the class of a single video."""
    from preprocess import extract_hand_landmarks
    from model import SignLanguageGRU

    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"[ERROR] No saved model found at: {MODEL_SAVE_PATH}")
        print("        Train the model first.")
        sys.exit(1)

    if not os.path.exists(video_path):
        print(f"[ERROR] Video not found: {video_path}")
        sys.exit(1)

    # Load checkpoint
    checkpoint = torch.load(MODEL_SAVE_PATH, map_location=DEVICE, weights_only=False)
    num_classes = checkpoint["num_classes"]

    # Reconstruct class list from processed/
    classes = sorted([
        d for d in os.listdir(PROCESSED_DIR)
        if os.path.isdir(os.path.join(PROCESSED_DIR, d))
    ])

    # Build model and load weights
    model = SignLanguageGRU(num_classes=num_classes).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Extract landmarks
    print(f"Processing: {video_path}")
    sequence = extract_hand_landmarks(video_path)
    tensor = torch.from_numpy(sequence).unsqueeze(0).to(DEVICE)  # (1, 20, 63)

    # Predict
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        pred_idx = logits.argmax(dim=1).item()
        confidence = probs[0, pred_idx].item()

    pred_class = classes[pred_idx] if pred_idx < len(classes) else f"class_{pred_idx}"
    print(f"\nPrediction: {pred_class}")
    print(f"Confidence: {confidence:.2%}")
    print("\nAll probabilities:")
    for i, cls in enumerate(classes):
        print(f"  {cls:>12}: {probs[0, i].item():.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="ISL Word Recognition Pipeline (INCLUDE Dataset)"
    )
    parser.add_argument(
        "--preprocess", action="store_true",
        help="Only run preprocessing (video -> .npy)",
    )
    parser.add_argument(
        "--train", action="store_true",
        help="Only run training (requires processed/ to exist)",
    )
    parser.add_argument(
        "--predict", type=str, default=None,
        help="Predict class for a single video file",
    )
    args = parser.parse_args()

    # If --predict, run prediction and exit
    if args.predict:
        run_predict(args.predict)
        return

    # If neither flag specified, run full pipeline
    if not args.preprocess and not args.train:
        run_preprocess()
        run_train()
        return

    if args.preprocess:
        run_preprocess()

    if args.train:
        run_train()


if __name__ == "__main__":
    main()

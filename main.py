"""
Main entry point for ISL Sign Language Recognition Pipeline.

Two pipelines:
  --mode word   (default) Video-based word recognition (BiGRU)
  --mode letter           Image-based letter/number recognition (CNN)

Usage:
    python main.py                          # preprocess + train word model
    python main.py --preprocess             # preprocess videos only
    python main.py --train                  # train word model
    python main.py --kfold                  # K-fold word ensemble
    python main.py --predict VIDEO          # predict word from video
    python main.py --webcam                 # live webcam (auto-detect)

    python main.py --mode letter --train    # train letter model
    python main.py --mode letter --kfold    # K-fold letter ensemble
    python main.py --mode letter --predict IMG  # predict letter from image
"""

import argparse
import os
import sys


# ── Word-mode functions (video pipeline) ─────────────────────────


def run_preprocess():
    """Run the video preprocessing pipeline."""
    from preprocess import preprocess_dataset
    print("=" * 60)
    print("  Preprocessing Videos -> Landmarks (.npy)")
    print("=" * 60)
    stats = preprocess_dataset()
    total = sum(stats.values())
    print(
        f"\nTotal processed: {total} videos "
        f"across {len(stats)} classes\n"
    )
    return stats


def run_train_word():
    """Train single word model."""
    from config import PROCESSED_DIR
    from train import create_data_loaders, train

    print("=" * 60)
    print("  Training Word Model (BiGRU + Attention)")
    print("=" * 60)

    if not os.path.exists(PROCESSED_DIR):
        print("[ERROR] Run --preprocess first.")
        sys.exit(1)

    tl, vl, nc, cw = create_data_loaders()
    train(tl, vl, nc, cw)
    print("\nWord training complete!\n")


def run_kfold_word():
    """K-fold word ensemble training."""
    from config import PROCESSED_DIR
    from train import train_kfold

    print("=" * 60)
    print("  K-Fold Word Ensemble Training")
    print("=" * 60)

    if not os.path.exists(PROCESSED_DIR):
        print("[ERROR] Run --preprocess first.")
        sys.exit(1)

    train_kfold()
    print("\nK-fold word training complete!\n")


def run_predict_word(video_path: str):
    """Predict word from a video file."""
    from preprocess import extract_hand_landmarks
    from ensemble import load_ensemble, ensemble_predict

    if not os.path.exists(video_path):
        print(f"[ERROR] Video not found: {video_path}")
        sys.exit(1)

    models, classes, _ = load_ensemble()
    print(f"Processing: {video_path}")
    sequence = extract_hand_landmarks(video_path)
    pred_idx, conf, probs = ensemble_predict(
        models, sequence, use_tta=True,
    )

    pred_class = (
        classes[pred_idx]
        if pred_idx < len(classes) else "?"
    )
    print(f"\nPrediction: {pred_class}")
    print(f"Confidence: {conf:.2%}")
    print("\nAll probabilities:")
    for i, cls in enumerate(classes):
        print(f"  {cls:>12}: {probs[i]:.4f}")


# ── Letter-mode functions (image pipeline) ───────────────────────


def run_train_letter():
    """Train single letter/number image model."""
    from train_image import train_image_model

    print("=" * 60)
    print("  Training Letter/Number Model (CNN)")
    print("=" * 60)
    train_image_model()
    print("\nLetter training complete!\n")


def run_kfold_letter():
    """K-fold letter ensemble training."""
    from train_image import train_image_kfold

    print("=" * 60)
    print("  K-Fold Letter Ensemble Training")
    print("=" * 60)
    train_image_kfold()
    print("\nK-fold letter training complete!\n")


def run_predict_letter(image_path: str):
    """Predict letter/number from an image file."""
    import cv2
    from ensemble_image import (
        load_image_ensemble, preprocess_image,
        image_ensemble_predict,
    )

    if not os.path.exists(image_path):
        print(f"[ERROR] Image not found: {image_path}")
        sys.exit(1)

    models, classes, _ = load_image_ensemble()
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Cannot read image: {image_path}")
        sys.exit(1)

    img_chw = preprocess_image(img)
    pred_idx, conf, probs = image_ensemble_predict(
        models, img_chw,
    )

    pred_class = (
        classes[pred_idx]
        if pred_idx < len(classes) else "?"
    )
    print(f"\nPrediction: {pred_class}")
    print(f"Confidence: {conf:.2%}")
    print("\nTop-5 probabilities:")
    top5 = sorted(
        enumerate(probs), key=lambda x: -x[1]
    )[:5]
    for idx, p in top5:
        print(f"  {classes[idx]:>3}: {p:.4f}")


# ── Main ─────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="ISL Sign Language Recognition Pipeline"
    )
    parser.add_argument(
        "--mode", choices=["word", "letter"],
        default="word",
        help="word = video-based word recognition, "
             "letter = image-based letter/number",
    )
    parser.add_argument(
        "--preprocess", action="store_true",
        help="Preprocess videos (word mode only)",
    )
    parser.add_argument(
        "--train", action="store_true",
        help="Train a single model",
    )
    parser.add_argument(
        "--kfold", action="store_true",
        help="K-fold CV ensemble training",
    )
    parser.add_argument(
        "--predict", type=str, default=None,
        help="Predict from a video (word) or image (letter)",
    )
    parser.add_argument(
        "--webcam", action="store_true",
        help="Live webcam recognition (auto-detects letters & words)",
    )
    args = parser.parse_args()

    # ── Webcam (always dual mode) ──
    if args.webcam:
        from webcam import run_webcam
        run_webcam()
        return

    # ── Letter mode ──
    if args.mode == "letter":
        if args.predict:
            run_predict_letter(args.predict)
        elif args.kfold:
            run_kfold_letter()
        elif args.train:
            run_train_letter()
        else:
            # Default: train letter model
            run_train_letter()
        return

    # ── Word mode (default) ──
    if args.predict:
        run_predict_word(args.predict)
        return

    if args.kfold:
        run_kfold_word()
        return

    # Default: preprocess + train
    if not args.preprocess and not args.train:
        run_preprocess()
        run_train_word()
        return

    if args.preprocess:
        run_preprocess()

    if args.train:
        run_train_word()


if __name__ == "__main__":
    main()

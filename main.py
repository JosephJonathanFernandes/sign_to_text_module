"""
Main entry point for ISL Sign Language Recognition Pipeline.

Video-based word recognition using BiGRU + hand landmarks.

Usage:
    python main.py                          # preprocess + train word model
    python main.py --preprocess             # preprocess videos only
    python main.py --train                  # train word model
    python main.py --kfold                  # K-fold word ensemble
    python main.py --predict VIDEO          # predict word from video
    python main.py --webcam                 # live webcam recognition
"""

import argparse
import os
import sys


# ── Word-mode functions (video pipeline) ─────────────────────────


def run_preprocess(input_dir: str | None = None):
    """Run the video preprocessing pipeline."""
    from preprocess import DATASET_DIR, preprocess_dataset

    source_dir = input_dir or DATASET_DIR
    print("=" * 60)
    print("  Preprocessing Videos -> Landmarks (.npy)")
    print("=" * 60)
    print(f"  Source: {source_dir}")
    stats = preprocess_dataset(source_dir)
    total = sum(stats.values())
    print(
        f"\nTotal processed: {total} videos "
        f"across {len(stats)} classes\n"
    )
    return stats


def run_augment_videos(
    input_dir: str,
    output_dir: str,
    max_videos_per_class: int,
    max_augments_per_video: int,
    target_width: int,
    target_height: int,
    clear_output: bool,
):
    """Build the controlled augmented raw-video dataset."""
    from preprocess import augment_video_dataset

    print("=" * 60)
    print(f"  Augmenting Raw Videos -> {output_dir}")
    print("=" * 60)
    stats = augment_video_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        max_videos_per_class=max_videos_per_class,
        max_augments_per_video=max_augments_per_video,
        target_width=target_width,
        target_height=target_height,
        clear_output=clear_output,
    )
    total = sum(item["total_output"] for item in stats.values())
    print(
        f"\nTotal output videos: {total} across {len(stats)} classes\n"
    )
    return stats


def run_train_word():
    """Train single word model."""
    from config import get_config
    from train import create_data_loaders, train

    cfg = get_config()
    PROCESSED_DIR = cfg.paths.processed_dir

    print("=" * 60)
    print("  Training Word Model (BiGRU + Attention)")
    print("=" * 60)

    if not os.path.exists(PROCESSED_DIR):
        print("[ERROR] Run --preprocess first.")
        sys.exit(1)

    tl, vl, nc, cw, ds = create_data_loaders()
    train(tl, vl, nc, cw, classes_list=ds.classes)
    print("\nWord training complete!\n")


def run_kfold_word():
    """K-fold word ensemble training."""
    from config import get_config
    from train import train_kfold

    cfg = get_config()
    PROCESSED_DIR = cfg.paths.processed_dir

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


# ── Main ─────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="ISL Sign Language Recognition Pipeline"
    )
    parser.add_argument(
        "--preprocess", action="store_true",
        help="Preprocess videos",
    )
    parser.add_argument(
        "--preprocess-dir", type=str, default=None,
        help="Source directory for preprocessing (e.g. augmented_dataset)",
    )
    parser.add_argument(
        "--train", action="store_true",
        help="Train a single model",
    )
    parser.add_argument(
        "--augment-videos", action="store_true",
        help="Create a controlled augmented raw-video dataset",
    )
    parser.add_argument(
        "--augment-input-dir", type=str, default=None,
        help="Source directory for raw video augmentation",
    )
    parser.add_argument(
        "--augment-output-dir", type=str, default=None,
        help="Output directory for augmented videos",
    )
    parser.add_argument(
        "--augment-max-per-class", type=int, default=100,
        help="Maximum total videos to keep per class in augmented output",
    )
    parser.add_argument(
        "--augment-max-per-video", type=int, default=8,
        help="Maximum augmented variants to generate per source video",
    )
    parser.add_argument(
        "--augment-width", type=int, default=224,
        help="Output video width for augmented samples",
    )
    parser.add_argument(
        "--augment-height", type=int, default=224,
        help="Output video height for augmented samples",
    )
    parser.add_argument(
        "--no-clear", action="store_true",
        help="Do not remove existing augmented output; append instead",
    )
    parser.add_argument(
        "--kfold", action="store_true",
        help="K-fold CV ensemble training",
    )
    parser.add_argument(
        "--predict", type=str, default=None,
        help="Predict from a video file",
    )
    parser.add_argument(
        "--webcam", action="store_true",
        help="Live webcam recognition",
    )
    parser.add_argument(
        "--collect", action="store_true",
        help="Record new training samples via webcam",
    )
    parser.add_argument(
        "--cls", type=str, default=None,
        help="Class name for --collect (e.g. 'happy')",
    )
    parser.add_argument(
        "--n", type=int, default=None,
        help="Number of samples for --collect",
    )
    args = parser.parse_args()

    # ── Data collection ──
    if args.collect:
        from collect_data import collect_interactive, record_samples
        if args.cls:
            record_samples(args.cls.lower().strip(), num_samples=args.n)
        else:
            collect_interactive()
        return

    if args.augment_videos:
        from preprocess import DATASET_DIR, AUGMENTED_DATASET_DIR
        input_dir = args.augment_input_dir or DATASET_DIR
        output_dir = args.augment_output_dir or AUGMENTED_DATASET_DIR
        run_augment_videos(
            input_dir=input_dir,
            output_dir=output_dir,
            max_videos_per_class=args.augment_max_per_class,
            max_augments_per_video=args.augment_max_per_video,
            target_width=args.augment_width,
            target_height=args.augment_height,
            clear_output=(not args.no_clear),
        )
        return

    # ── Webcam (always dual mode) ──
    if args.webcam:
        from webcam import run_webcam
        run_webcam()
        return

    # ── Word mode ──
    if args.predict:
        run_predict_word(args.predict)
        return

    if args.kfold:
        run_kfold_word()
        return

    # Default: preprocess + train
    if not args.preprocess and not args.train:
        run_preprocess(args.preprocess_dir)
        run_train_word()
        return

    if args.preprocess:
        run_preprocess(args.preprocess_dir)

    if args.train:
        run_train_word()


if __name__ == "__main__":
    main()

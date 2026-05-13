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

from pipeline_logger import setup_pipeline_logger


# ── Word-mode functions (video pipeline) ─────────────────────────


def run_preprocess(input_dir: str | None = None):
    """Run the video preprocessing pipeline."""
    from preprocess import DATASET_DIR, preprocess_dataset

    source_dir = input_dir or DATASET_DIR
    pipeline_log = setup_pipeline_logger("preprocess")
    
    with pipeline_log.capture_stdio():
        pipeline_log.event("preprocess_pipeline_start", source_dir=source_dir)
        print("=" * 60)
        print("  Preprocessing Videos -> Landmarks (.npy)")
        print("=" * 60)
        print(f"  Source: {source_dir}")
        stats = preprocess_dataset(source_dir, pipeline_log=pipeline_log)
        total = sum(stats.values())
        print(
            f"\nTotal processed: {total} videos "
            f"across {len(stats)} classes\n"
        )
        pipeline_log.event("preprocess_pipeline_end", total_videos=total, total_classes=len(stats), stats=stats)
    return stats


def run_augment_videos(
    input_dir: str,
    output_dir: str,
    max_videos_per_class: int,
    max_augments_per_video: int,
    target_width: int,
    target_height: int,
    clear_output: bool,
    class_only: str | None = None,
):
    """Build the controlled augmented raw-video dataset."""
    from preprocess import augment_video_dataset

    pipeline_log = setup_pipeline_logger("augment")
    
    with pipeline_log.capture_stdio():
        pipeline_log.event("augment_pipeline_start", input_dir=input_dir, output_dir=output_dir, class_only=class_only)
        print("=" * 60)
        print(f"  Augmenting Raw Videos -> {output_dir}")
        if class_only:
            print(f"  (Class filter: {class_only})")
        print("=" * 60)
        stats = augment_video_dataset(
            input_dir=input_dir,
            output_dir=output_dir,
            max_videos_per_class=max_videos_per_class,
            max_augments_per_video=max_augments_per_video,
            target_width=target_width,
            target_height=target_height,
            clear_output=clear_output,
            pipeline_log=pipeline_log,
            class_only=class_only,
        )
        total = sum(item["total_output"] for item in stats.values())
        print(
            f"\nTotal output videos: {total} across {len(stats)} classes\n"
        )
        pipeline_log.event("augment_pipeline_end", total_videos=total, total_classes=len(stats), stats=stats)
    return stats


def run_augment_landmarks(
    input_dir: str = "processed",
    output_dir: str | None = None,
    augment_per_sample: int = 3,
    class_only: str | None = None,
):
    """Augment processed landmark sequences (.npy files)."""
    from augmentations import augment_dataset

    if output_dir is None:
        output_dir = input_dir

    print("=" * 60)
    print("  Augmenting Landmark Sequences")
    if class_only:
        print(f"  (Class filter: {class_only})")
    print("=" * 60)
    augment_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        augment_per_sample=augment_per_sample,
        class_only=class_only,
    )
    print()


def run_train_word():
    """Train single word model."""
    from config import get_config

    cfg = get_config()
    PROCESSED_DIR = cfg.paths.processed_dir
    pipeline_log = setup_pipeline_logger("train")

    with pipeline_log.capture_stdio():
        pipeline_log.event(
            "train_start",
            processed_dir=PROCESSED_DIR,
        )
        print("=" * 60)
        print("  Training Word Model (BiGRU + Attention)")
        print("=" * 60)

        if not os.path.exists(PROCESSED_DIR):
            print("[ERROR] Run --preprocess first.")
            sys.exit(1)

        from train import create_data_loaders, train

        tl, vl, nc, cw, ds = create_data_loaders()
        train(tl, vl, nc, cw, classes_list=ds.classes, pipeline_log=pipeline_log)
        pipeline_log.event(
            "train_end",
            classes=len(ds.classes),
            num_classes=nc,
        )
    print("\nWord training complete!\n")


def run_merge_augmentations(
    input_dir: str = "processed",
    output_dir: str | None = None,
    per_sample: int = 1,
    mode: str = "hybrid",
    class_only: str | None = None,
):
    """Merge augmented samples using frame splicing."""
    from merge_augmentations import merge_dataset

    print("=" * 60)
    print("  Merging Augmented Samples")
    if class_only:
        print(f"  (Class filter: {class_only})")
    print("=" * 60)
    merge_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        per_sample=per_sample,
        mode=mode,
        class_only=class_only,
    )
    print("\nMerge augmentation complete!\n")


def run_cleanup_dataset(
    root_dir: str = "processed",
    max_aug: int = 50,
    max_merge: int = 40,
    dry_run: bool = False,
    class_only: str | None = None,
):
    """Clean up dataset by removing near-duplicates."""
    from cleanup_dataset_npy import clean_dataset

    print("=" * 60)
    print("  Dataset Cleanup")
    if class_only:
        print(f"  (Class filter: {class_only})")
    print("=" * 60)
    clean_dataset(
        root_dir=root_dir,
        max_aug=max_aug,
        max_merge=max_merge,
        dry_run=dry_run,
        class_only=class_only,
    )
    print()


def run_kfold_word():
    """K-fold word ensemble training."""
    from config import get_config

    cfg = get_config()
    PROCESSED_DIR = cfg.paths.processed_dir
    pipeline_log = setup_pipeline_logger("kfold")

    with pipeline_log.capture_stdio():
        pipeline_log.event(
            "kfold_start",
            processed_dir=PROCESSED_DIR,
        )
        print("=" * 60)
        print("  K-Fold Word Ensemble Training")
        print("=" * 60)

        if not os.path.exists(PROCESSED_DIR):
            print("[ERROR] Run --preprocess first.")
            sys.exit(1)

        from train import train_kfold

        fold_accs = train_kfold(pipeline_log=pipeline_log)
        pipeline_log.event(
            "kfold_end",
            fold_accuracies=fold_accs,
            folds=len(fold_accs),
        )
    print("\nK-fold word training complete!\n")


def run_predict_word(video_path: str):
    """Predict word from a video file."""
    from preprocess import extract_hand_landmarks
    from ensemble import load_ensemble, ensemble_predict
    from pipeline_logger import setup_pipeline_logger

    if not os.path.exists(video_path):
        print(f"[ERROR] Video not found: {video_path}")
        sys.exit(1)

    pipeline_log = setup_pipeline_logger("predict")
    with pipeline_log.capture_stdio():
        pipeline_log.event("predict_start", video_path=video_path)
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
        pipeline_log.event(
            "predict_end",
            predicted=pred_class,
            confidence=round(float(conf), 4),
            classes=len(classes),
        )


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
        "--augment-cls", type=str, default=None,
        help="Only augment this specific class",
    )
    parser.add_argument(
        "--augment-landmarks", action="store_true",
        help="Augment processed landmark sequences (.npy files)",
    )
    parser.add_argument(
        "--augment-landmarks-dir", type=str, default="processed",
        help="Directory for landmark augmentation",
    )
    parser.add_argument(
        "--augment-landmarks-n", type=int, default=3,
        help="Number of augmentations per landmark sample (fixed ordered variants; default: 14)",
    )
    parser.add_argument(
        "--augment-landmarks-cls", type=str, default=None,
        help="Only augment this specific class (landmark augmentation)",
    )
    parser.add_argument(
        "--merge", action="store_true",
        help="Create merged augmentation samples",
    )
    parser.add_argument(
        "--merge-dir", type=str, default="processed",
        help="Directory for merge augmentation",
    )
    parser.add_argument(
        "--merge-n", type=int, default=1,
        help="Number of merged samples per original",
    )
    parser.add_argument(
        "--merge-mode",
        choices=[
            "splice",
            "crossfade_splice",
            "multi_splice",
            "tempo_aligned_splice",
            "blend",
            "blend_then_noise",
            "hand_swap",
            "proximity_only_swap",
            "left_right_cross_swap",
            "hybrid",
        ],
        default="hybrid",
        help="Merge mode to use",
    )
    parser.add_argument(
        "--merge-cls", type=str, default=None,
        help="Only merge this specific class",
    )
    parser.add_argument(
        "--cleanup", action="store_true",
        help="Clean up dataset by removing near-duplicates",
    )
    parser.add_argument(
        "--cleanup-dir", type=str, default="processed",
        help="Directory to clean up",
    )
    parser.add_argument(
        "--cleanup-max-aug", type=int, default=50,
        help="Max augmented files to keep per class",
    )
    parser.add_argument(
        "--cleanup-max-merge", type=int, default=40,
        help="Max merged files to keep per class",
    )
    parser.add_argument(
        "--cleanup-dry-run", action="store_true",
        help="Preview cleanup without deleting",
    )
    parser.add_argument(
        "--cleanup-cls", type=str, default=None,
        help="Only clean this specific class",
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

        pipeline_log = setup_pipeline_logger("collect")
        with pipeline_log.capture_stdio():
            if args.cls:
                record_samples(args.cls.lower().strip(), num_samples=args.n, pipeline_log=pipeline_log)
            else:
                collect_interactive(pipeline_log=pipeline_log)
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
            class_only=args.augment_cls,
        )
        return

    if args.augment_landmarks:
        run_augment_landmarks(
            input_dir=args.augment_landmarks_dir,
            output_dir=args.augment_landmarks_dir,
            augment_per_sample=args.augment_landmarks_n,
            class_only=args.augment_landmarks_cls,
        )
        return

    if args.merge:
        run_merge_augmentations(
            input_dir=args.merge_dir,
            per_sample=args.merge_n,
            mode=args.merge_mode,
            class_only=args.merge_cls,
        )
        return

    if args.cleanup:
        run_cleanup_dataset(
            root_dir=args.cleanup_dir,
            max_aug=args.cleanup_max_aug,
            max_merge=args.cleanup_max_merge,
            dry_run=args.cleanup_dry_run,
            class_only=args.cleanup_cls,
        )
        return

    # ── Webcam (always dual mode) ──
    if args.webcam:
        pipeline_log = setup_pipeline_logger("inference")
        with pipeline_log.capture_stdio():
            from webcam import run_webcam

            run_webcam(pipeline_log=pipeline_log)
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

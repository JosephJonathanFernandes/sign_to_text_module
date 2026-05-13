#!/usr/bin/env python3
"""
Video Augmentation Pipeline Orchestrator.

Augments video dataset (original videos and synthetic variants) from Dataset folder.

Covers ALL combinations systematically (33 total):
- Variants 1-3: Spatial only (center, left, right crops)
- Variants 4-33: All 3 crops × 10 effects
  * 4-13: center + [brightness, contrast, hue, fog, rotation, scale, color_jitter, noise, pixel_dropout, coarse_dropout]
  * 14-23: left + [same 10 effects]
  * 24-33: right + [same 10 effects]

Usage:
    python augment_video_pipeline.py                    # All classes, 33 augments each
    python augment_video_pipeline.py --class old        # Specific class
    python augment_video_pipeline.py --augments 33      # All 33 combinations
    python augment_video_pipeline.py --augments 13      # Just center crop combos
    python augment_video_pipeline.py --max-videos 200   # Cap per class
"""

import argparse
import sys
import os
from config import get_config
from preprocess import augment_video_dataset
from pipeline_logger import PipelineLogger

cfg = get_config()


def main():
    parser = argparse.ArgumentParser(
        description="Run video augmentation pipeline for Dataset folder"
    )
    parser.add_argument(
        "--class",
        dest="class_name",
        required=False,
        default=None,
        help="Class name to augment (e.g., 'old', 'young'). If not provided, runs for ALL classes."
    )
    parser.add_argument(
        "--augments",
        type=int,
        default=33,
        help="Augmentations per video (default: 33, range: 1-33). Covers: 3 spatial crops + 10 visual effects"
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=900,
        help="Max videos per class in output (default: 900)"
    )
    parser.add_argument(
        "--clear-output",
        action="store_true",
        default=True,
        help="Clear output directory before augmentation (default: True)"
    )
    parser.add_argument(
        "--no-clear-output",
        action="store_false",
        dest="clear_output",
        help="Do NOT clear output directory"
    )
    
    args = parser.parse_args()
    
    # Validate augments
    if args.augments < 1 or args.augments > 13:
        print("Error: --augments must be between 1 and 13 (3 spatial + 10 effects)")
        sys.exit(1)
    
    # Get class list
    if args.class_name:
        classes = [args.class_name.strip()]
    else:
        # Get all classes from Dataset directory
        dataset_dir = cfg.paths.dataset_dir
        classes = sorted([
            d for d in os.listdir(dataset_dir)
            if os.path.isdir(os.path.join(dataset_dir, d))
        ])
        if not classes:
            print(f"Error: No classes found in {dataset_dir}")
            sys.exit(1)
    
    print(f"\n{'*' * 70}")
    print(f"Video Augmentation Pipeline")
    print(f"{'*' * 70}")
    print(f"Classes to process: {len(classes)} - {', '.join(classes[:5])}{'...' if len(classes) > 5 else ''}")
    print(f"Augments per video: {args.augments}")
    print(f"Max videos per class: {args.max_videos}")
    print(f"Clear output: {args.clear_output}")
    print(f"{'*' * 70}\n")
    
    pipeline_log = PipelineLogger()
    
    failed_classes = []
    
    # Process each class
    for class_idx, class_name in enumerate(classes, 1):
        print(f"\n{'#' * 70}")
        print(f"Processing Class {class_idx}/{len(classes)}: {class_name}")
        print(f"{'#' * 70}\n")
        
        try:
            stats = augment_video_dataset(
                input_dir=cfg.paths.dataset_dir,
                output_dir=cfg.paths.augmented_dataset_dir,
                max_videos_per_class=args.max_videos,
                max_augments_per_video=args.augments,
                target_width=224,
                target_height=224,
                clear_output=(class_idx == 1 and args.clear_output),  # Only clear for first class
                pipeline_log=pipeline_log,
                class_only=class_name,
            )
            print(f"✓ Class '{class_name}' augmented successfully")
        except Exception as e:
            print(f"✗ Class '{class_name}' failed: {e}")
            failed_classes.append(class_name)
    
    # Summary
    print(f"\n{'=' * 70}")
    print("VIDEO AUGMENTATION SUMMARY")
    print(f"{'=' * 70}")
    
    if failed_classes:
        print(f"✗ Pipeline completed with {len(failed_classes)} failure(s):")
        for cls in failed_classes:
            print(f"  - {cls}")
        sys.exit(1)
    else:
        print(f"✓ Video augmentation completed successfully for {len(classes)} class(es)!")
        print(f"Output directory: {cfg.paths.augmented_dataset_dir}")
        sys.exit(0)


if __name__ == "__main__":
    main()

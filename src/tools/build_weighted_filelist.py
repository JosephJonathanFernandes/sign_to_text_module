#!/usr/bin/env python3
"""Build a combined filelist from `processed` and `processed_del`.

Output format (tab-separated):
  <path>\t<label>\t<weight>\n
Weights can be used by a training DataLoader / sampler.
"""
from pathlib import Path
import argparse


def collect(root: Path, ext: str):
    for p in sorted(root.rglob(f"**/*.{ext}")):
        if p.is_file():
            label = p.parent.name
            yield p, label


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--processed-root", default="processed", help="curated data root")
    p.add_argument("--archived-root", default="processed_del", help="archived/removed samples root")
    p.add_argument("--out-file", default="train_combined.txt", help="output filelist")
    p.add_argument("--archived-weight", type=float, default=0.25, help="weight for archived samples (0-1)")
    p.add_argument("--ext", default="npy", help="file extension to include")
    p.add_argument("--staged", action="store_true", help="emit two filelists: stage1 (processed) and stage2 (archived)")
    args = p.parse_args()

    processed = Path(args.processed_root)
    archived = Path(args.archived_root)

    if args.staged:
        stage1 = Path(args.out_file).with_name("stage1_" + Path(args.out_file).name)
        stage2 = Path(args.out_file).with_name("stage2_" + Path(args.out_file).name)
        with stage1.open("w", encoding="utf-8") as s1:
            for fp, label in collect(processed, args.ext):
                s1.write(f"{fp.as_posix()}\t{label}\t1.0\n")
        with stage2.open("w", encoding="utf-8") as s2:
            for fp, label in collect(archived, args.ext):
                s2.write(f"{fp.as_posix()}\t{label}\t{args.archived_weight}\n")
        print(f"Wrote staged lists: {stage1} and {stage2}")
        return

    out = Path(args.out_file)
    with out.open("w", encoding="utf-8") as fh:
        for fp, label in collect(processed, args.ext):
            fh.write(f"{fp.as_posix()}\t{label}\t1.0\n")
        for fp, label in collect(archived, args.ext):
            fh.write(f"{fp.as_posix()}\t{label}\t{args.archived_weight}\n")

    print(f"Wrote combined filelist to {out}")


if __name__ == "__main__":
    main()

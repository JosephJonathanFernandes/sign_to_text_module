"""Generate or move negative samples from per-class manifests.

Features added:
- `--tag-all --class <class> --tag <tag>`: add the tag to every .npy
    in the given class folder (writes/updates `manifest.json`).
- `--move`: move files instead of copying (deletes originals).
- Default behavior without `--move` is to copy into `--out-root`.

Usage examples:

1) Dry-run: collect negatives declared in manifests and show actions

```bash
python tools/generate_negative_root.py --processed-root processed \
        --out-root processed_negatives --dry-run
```

2) Tag all samples in a class as negative and dry-run the collection

```bash
python tools/generate_negative_root.py --processed-root processed \
        --out-root processed_negatives --class hello --tag negative --tag-all --dry-run
```

3) Actually move tagged negatives out of `processed/` into `processed_negatives/`:

```bash
python tools/generate_negative_root.py --processed-root processed \
        --out-root processed_negatives --class hello --tag negative --tag-all --move
```

Manifest format (per-class): place a JSON file named `manifest.json` inside
each class folder under `processed/`. The file should be a mapping of
filename -> tag. Example:

{
    "webcam_20260501_0001.npy": "negative",
    "webcam_20260501_0002.npy": "outdated"
}

If `--tag-all` and `--class` are supplied the script will create/overwrite
the class `manifest.json` entries for all `.npy` files in that class.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from typing import Dict, List


def load_manifest(manifest_path: str) -> Dict[str, str]:
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
            # Support list-of-objects format too
            if isinstance(data, list):
                out = {}
                for e in data:
                    if isinstance(e, dict) and "file" in e and "tag" in e:
                        out[e["file"]] = e["tag"]
                return out
    except Exception:
        return {}
    return {}


def write_manifest(manifest_path: str, manifest: Dict[str, str]) -> None:
    try:
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[WARN] Could not write manifest {manifest_path}: {e}")


def collect_negatives(processed_root: str, tags: List[str]) -> Dict[str, List[str]]:
    """Return mapping class -> list of negative file paths (absolute)."""
    result = {}
    if not os.path.isdir(processed_root):
        print(f"[WARN] processed root not found: {processed_root}")
        return result

    for cls in sorted(os.listdir(processed_root)):
        cls_dir = os.path.join(processed_root, cls)
        if not os.path.isdir(cls_dir):
            continue
        manifest_path = os.path.join(cls_dir, "manifest.json")
        manifest = {}
        if os.path.isfile(manifest_path):
            manifest = load_manifest(manifest_path)
        negs = []
        for fname, tag in manifest.items():
            if tag in tags:
                src = os.path.join(cls_dir, fname)
                if os.path.isfile(src):
                    negs.append(src)
        if negs:
            result[cls] = negs
    return result


def tag_all_in_class(processed_root: str, class_name: str, tag: str) -> int:
    """Tag all .npy files in a class folder with `tag` by writing manifest.json.

    Returns number of files tagged.
    """
    cls_dir = os.path.join(processed_root, class_name)
    if not os.path.isdir(cls_dir):
        print(f"[ERROR] Class not found: {cls_dir}")
        return 0
    files = [f for f in sorted(os.listdir(cls_dir)) if f.endswith(".npy")]
    if not files:
        print(f"[WARN] No .npy files in {cls_dir}")
        return 0
    manifest_path = os.path.join(cls_dir, "manifest.json")
    manifest = load_manifest(manifest_path)
    for fn in files:
        manifest[fn] = tag
    write_manifest(manifest_path, manifest)
    print(f"Wrote manifest for {class_name} with {len(files)} entries (tag={tag})")
    return len(files)


def make_out_dirs(out_root: str, classes: List[str]):
    for cls in classes:
        d = os.path.join(out_root, cls)
        os.makedirs(d, exist_ok=True)


def copy_negatives(mapping: Dict[str, List[str]], out_root: str, dry_run: bool = True):
    make_out_dirs(out_root, list(mapping.keys()))
    for cls, files in mapping.items():
        dst_dir = os.path.join(out_root, cls)
        for src in files:
            dst = os.path.join(dst_dir, os.path.basename(src))
            if dry_run:
                print(f"DRY: {src} -> {dst}")
            else:
                shutil.copy2(src, dst)
                print(f"Copied {src} -> {dst}")


def move_negatives(mapping: Dict[str, List[str]], out_root: str, dry_run: bool = True):
    make_out_dirs(out_root, list(mapping.keys()))
    for cls, files in mapping.items():
        dst_dir = os.path.join(out_root, cls)
        for src in files:
            dst = os.path.join(dst_dir, os.path.basename(src))
            if dry_run:
                print(f"DRY-MOVE: {src} -> {dst}")
            else:
                try:
                    shutil.move(src, dst)
                    print(f"Moved {src} -> {dst}")
                except Exception as e:
                    print(f"[ERROR] Could not move {src} -> {dst}: {e}")


def main():
    p = argparse.ArgumentParser(description="Collect negative samples from per-class manifests")
    p.add_argument("--processed-root", default="processed", help="Processed dataset root")
    p.add_argument("--out-root", default="processed_negatives", help="Output root for negative samples")
    p.add_argument("--tags", default="negative,hard_negative", help="Comma-separated tags to treat as negatives")
    p.add_argument("--dry-run", action="store_true", default=False, help="Print actions without copying/moving")
    p.add_argument("--class", dest="class_name", help="Class folder to tag all files inside")
    p.add_argument("--tag", dest="tag", help="Tag to apply when using --tag-all (default 'negative')")
    p.add_argument("--tag-all", action="store_true", default=False, help="Tag all files in --class with --tag by writing manifest.json")
    p.add_argument("--move", action="store_true", default=False, help="Move files instead of copying (deletes originals)")
    args = p.parse_args()

    tags = [t.strip() for t in args.tags.split(",") if t.strip()]

    # Optionally tag all files in a class by writing manifest.json
    if args.tag_all:
        if not args.class_name:
            print("[ERROR] --tag-all requires --class <class_name>")
            return
        tag_to_use = args.tag or "negative"
        n = tag_all_in_class(args.processed_root, args.class_name, tag_to_use)
        if n == 0:
            print("[WARN] No files tagged; aborting collection.")
            return

    mapping = collect_negatives(args.processed_root, tags)
    if not mapping:
        print("No negative samples found (check manifests and tags).")
        return

    print(f"Found negatives in {len(mapping)} classes.")
    if args.move:
        move_negatives(mapping, args.out_root, dry_run=args.dry_run)
    else:
        copy_negatives(mapping, args.out_root, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Repository integrity verification script.

Verifies:
  1. All core modules import without errors
  2. Config computes correct feature dimensions (253/506 contract)
  3. Feature extractor produces correct output shape
  4. API app can be imported
  5. No circular imports in core packages

Usage:
    python scripts/verify_repo.py
"""

import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
SKIP = "\033[93m~\033[0m"

results = []


def check(name: str, fn):
    try:
        fn()
        results.append((True, name))
        print(f"  {PASS}  {name}")
    except Exception as e:
        results.append((False, name, str(e)))
        print(f"  {FAIL}  {name}")
        print(f"       → {e}")


print("\n══════════════════════════════════════════════════════════")
print("  ISL Sign-to-Text — Repository Verification")
print("══════════════════════════════════════════════════════════\n")

# ─── 1. Core imports ────────────────────────────────────────────
print("Phase 1: Core Module Imports")


def check_config():
    from src.core.config import get_config
    cfg = get_config()
    assert cfg is not None


def check_feature_extractor():
    from src.shared.feature_extractor import build_single_frame_features
    import numpy as np
    result = build_single_frame_features(None, None, None)
    assert result.shape == (253,), f"Expected (253,), got {result.shape}"
    assert result.dtype.name == "float32"


def check_ensemble_import():
    from src.inference.ensemble import ensemble_predict, load_ensemble


def check_api_app():
    from api.app import app
    assert app is not None


check("src.core.config importable", check_config)
check("src.shared.feature_extractor importable", check_feature_extractor)
check("src.inference.ensemble importable", check_ensemble_import)
check("api.app importable", check_api_app)

# ─── 2. Dimension contract ──────────────────────────────────────
print("\nPhase 2: Feature Dimension Contract (253/506)")


def check_spatial_dim():
    from src.core.config import get_config
    cfg = get_config()
    dim = cfg.frame_features.frame_features_dim
    assert dim == 253, f"Expected 253, got {dim}"


def check_sequence_dim():
    from src.core.config import get_config
    cfg = get_config()
    dim = cfg.frame_features.input_sequence_dim
    assert dim == 506, f"Expected 506, got {dim}"


def check_num_frames():
    from src.core.config import get_config
    cfg = get_config()
    assert cfg.preprocessing.num_frames == 20, f"Expected 20 frames"


def check_feature_extractor_output():
    import numpy as np
    from src.shared.feature_extractor import build_single_frame_features
    rng = np.random.default_rng(42)
    left = rng.uniform(0.1, 0.9, 63).astype("float32")
    right = rng.uniform(0.1, 0.9, 63).astype("float32")
    face = rng.uniform(0.1, 0.9, 264 * 3).astype("float32")
    result = build_single_frame_features(left, right, face)
    assert result.shape == (253,)
    assert not any(v != v for v in result), "NaN detected"  # NaN check


check("frame_features_dim == 253", check_spatial_dim)
check("input_sequence_dim == 506", check_sequence_dim)
check("num_frames == 20", check_num_frames)
check("build_single_frame_features output shape (253,)", check_feature_extractor_output)

# ─── 3. Config validation ────────────────────────────────────────
print("\nPhase 3: Config Self-Validation")


def check_config_validate():
    from src.core.config import get_config
    cfg = get_config()
    cfg.validate()


check("cfg.validate() passes", check_config_validate)

# ─── 4. Summary ─────────────────────────────────────────────────
passed = sum(1 for r in results if r[0])
failed = sum(1 for r in results if not r[0])
total = len(results)

print(f"\n══════════════════════════════════════════════════════════")
print(f"  Results: {passed}/{total} passed", end="")
if failed:
    print(f"  |  {failed} FAILED ← fix before committing")
else:
    print("  ✓  All checks passed")
print("══════════════════════════════════════════════════════════\n")

sys.exit(0 if failed == 0 else 1)

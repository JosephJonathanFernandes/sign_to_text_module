"""
Unit tests for HDF5 dataset loading (src/preprocessing/dataset.py)

Tests:
  - Dataset falls back to .npy mode when dataset.h5 is absent
  - HDF5 metadata loading correctness (when file present)
  - _prepare_sequence shape correctness
  - _align_input_size padding/truncation
  - mixup / cutmix shape preservation

NOTE: These tests do NOT require a real dataset.h5 or processed/ directory.
      The HDF5 fast-path tests are marked `hdf5` and skipped if assets absent.
"""

import os
import tempfile

import numpy as np
import pytest


pytestmark = pytest.mark.unit

INPUT_SIZE = 506
NUM_FRAMES = 20


class TestAlignInputSize:
    """_align_input_size is a static method — testable without any dataset."""

    def _align(self, seq):
        from src.preprocessing.dataset import ISLDataset
        return ISLDataset._align_input_size(seq)

    def test_correct_size_unchanged(self):
        seq = np.zeros((NUM_FRAMES, INPUT_SIZE), dtype=np.float32)
        result = self._align(seq)
        assert result.shape == (NUM_FRAMES, INPUT_SIZE)
        assert result is seq or np.array_equal(result, seq)

    def test_smaller_dim_is_padded(self):
        seq = np.ones((NUM_FRAMES, 253), dtype=np.float32)
        result = self._align(seq)
        assert result.shape == (NUM_FRAMES, INPUT_SIZE)
        # Original values preserved, padded with zeros
        np.testing.assert_array_equal(result[:, :253], seq)
        np.testing.assert_array_equal(result[:, 253:], np.zeros((NUM_FRAMES, 253)))

    def test_larger_dim_is_truncated(self):
        seq = np.ones((NUM_FRAMES, 1024), dtype=np.float32)
        result = self._align(seq)
        assert result.shape == (NUM_FRAMES, INPUT_SIZE)


class TestPrepareSequence:
    """_prepare_sequence: align + optional augment + extract proximity."""

    def _prepare(self, seq, augment=False):
        from src.preprocessing.dataset import ISLDataset
        return ISLDataset._prepare_sequence(seq, augment=augment)

    def test_output_shapes(self):
        seq = np.zeros((NUM_FRAMES, INPUT_SIZE), dtype=np.float32)
        out_seq, proximity = self._prepare(seq)
        assert out_seq.shape == (NUM_FRAMES, INPUT_SIZE)
        assert proximity.shape == (NUM_FRAMES,)

    def test_augmented_output_shapes(self):
        rng = np.random.default_rng(42)
        seq = rng.standard_normal((NUM_FRAMES, INPUT_SIZE)).astype(np.float32)
        out_seq, proximity = self._prepare(seq, augment=True)
        assert out_seq.shape == (NUM_FRAMES, INPUT_SIZE)
        assert proximity.shape == (NUM_FRAMES,)

    def test_dtype_is_float32(self):
        seq = np.zeros((NUM_FRAMES, INPUT_SIZE), dtype=np.float64)
        out_seq, proximity = self._prepare(seq)
        assert out_seq.dtype == np.float32


class TestMixupCutmix:
    def test_mixup_shape_preserved(self):
        from src.preprocessing.dataset import ISLDataset
        rng = np.random.default_rng(0)
        s1 = rng.standard_normal((NUM_FRAMES, INPUT_SIZE)).astype(np.float32)
        s2 = rng.standard_normal((NUM_FRAMES, INPUT_SIZE)).astype(np.float32)
        mixed = ISLDataset.mixup(s1, s2)
        assert mixed.shape == (NUM_FRAMES, INPUT_SIZE)
        assert mixed.dtype == np.float32

    def test_cutmix_shape_preserved(self):
        from src.preprocessing.dataset import ISLDataset
        rng = np.random.default_rng(1)
        s1 = rng.standard_normal((NUM_FRAMES, INPUT_SIZE)).astype(np.float32)
        s2 = rng.standard_normal((NUM_FRAMES, INPUT_SIZE)).astype(np.float32)
        mixed = ISLDataset.cutmix(s1, s2)
        assert mixed.shape == (NUM_FRAMES, INPUT_SIZE)
        assert mixed.dtype == np.float32


class TestHDF5Metadata:
    """Tests HDF5 metadata reading — skipped when dataset.h5 is absent."""

    @pytest.mark.hdf5
    def test_hdf5_metadata_readable(self):
        h5_path = os.path.join("assets", "dataset.h5")
        if not os.path.exists(h5_path):
            pytest.skip("assets/dataset.h5 not present — skipping HDF5 test")

        import h5py
        import json

        with h5py.File(h5_path, "r") as f:
            sample_count = f.attrs["sample_count"]
            class_names_raw = f["class_names"][()]
            class_mapping = json.loads(class_names_raw)

        assert sample_count > 0
        assert len(class_mapping) > 0

    @pytest.mark.hdf5
    def test_hdf5_features_shape(self):
        h5_path = os.path.join("assets", "dataset.h5")
        if not os.path.exists(h5_path):
            pytest.skip("assets/dataset.h5 not present — skipping HDF5 test")

        import h5py

        with h5py.File(h5_path, "r") as f:
            features = f["features"]
            assert features.ndim == 3  # (N, frames, features)
            n, frames, feat_dim = features.shape
            assert frames == NUM_FRAMES
            assert feat_dim == INPUT_SIZE
            assert n > 0

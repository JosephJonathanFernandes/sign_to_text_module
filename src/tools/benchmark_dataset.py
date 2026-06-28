import os
import sys
import time
import psutil
from unittest.mock import patch
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.preprocessing.dataset import ISLDataset

def get_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def benchmark():
    print("=========================================")
    print("   Dataset Storage Benchmark (NPY vs HDF5)")
    print("=========================================\n")
    
    # 1. Benchmark NPY
    print("--- 1. NPY Benchmark ---")
    start_mem = get_memory_mb()
    t0 = time.time()
    
    # Mock os.path.exists to always return False for dataset.h5 to force NPY load
    original_exists = os.path.exists
    def mock_exists(path):
        if path.endswith("dataset.h5"):
            return False
        return original_exists(path)
        
    with patch('os.path.exists', side_effect=mock_exists):
        npy_ds = ISLDataset()
        
    t_init_npy = time.time() - t0
    npy_mem = get_memory_mb() - start_mem
    print(f"NPY initialized in {t_init_npy:.2f}s with {len(npy_ds)} samples. Memory overhead: {npy_mem:.1f} MB")
    
    dl_npy = DataLoader(
        npy_ds, 
        batch_size=64, 
        shuffle=True, 
        num_workers=4, 
        persistent_workers=True,
        multiprocessing_context='spawn' if os.name == 'nt' else None
    )
    
    t0 = time.time()
    for batch_idx, batch in enumerate(dl_npy):
        pass
    t_load_npy = time.time() - t0
    print(f"NPY full epoch load time: {t_load_npy:.2f}s")
    
    
    # 2. Benchmark HDF5
    print("\n--- 2. HDF5 Benchmark ---")
    start_mem = get_memory_mb()
    t0 = time.time()
    
    h5_ds = ISLDataset()
    t_init_h5 = time.time() - t0
    h5_mem = get_memory_mb() - start_mem
    print(f"HDF5 initialized in {t_init_h5:.2f}s with {len(h5_ds)} samples. Memory overhead: {h5_mem:.1f} MB")
    
    dl_h5 = DataLoader(
        h5_ds, 
        batch_size=64, 
        shuffle=True, 
        num_workers=4, 
        persistent_workers=True,
        multiprocessing_context='spawn' if os.name == 'nt' else None
    )
    
    t0 = time.time()
    for batch_idx, batch in enumerate(dl_h5):
        pass
    t_load_h5 = time.time() - t0
    print(f"HDF5 full epoch load time: {t_load_h5:.2f}s")
    
    print("\n=========================================")
    print("             FINAL REPORT")
    print("=========================================\n")
    print(f"| Metric | NPY | HDF5 | Improvement |")
    print(f"|---|---|---|---|")
    print(f"| Init Time | {t_init_npy:.3f}s | {t_init_h5:.3f}s | {t_init_npy/max(0.001, t_init_h5):.1f}x faster |")
    print(f"| Batch Load (1 epoch) | {t_load_npy:.2f}s | {t_load_h5:.2f}s | {((t_load_npy - t_load_h5) / max(0.001, t_load_npy) * 100):.1f}% faster |")
    
    # Calculate full epoch time (Init + Load)
    total_npy = t_init_npy + t_load_npy
    total_h5 = t_init_h5 + t_load_h5
    print(f"| Full Epoch 1 Time | {total_npy:.2f}s | {total_h5:.2f}s | {total_npy/max(0.001, total_h5):.1f}x faster |")
    print(f"| Memory diff| {npy_mem:.1f} MB | {h5_mem:.1f} MB | {npy_mem - h5_mem:.1f} MB |")

if __name__ == "__main__":
    benchmark()

import numpy as np

# 1. Test preprocess.py's _add_velocity logic
from src.preprocessing.preprocess import _add_velocity

def test_add_velocity():
    print("--- Testing src.preprocessing.preprocess._add_velocity ---")
    
    # 3 frames, 3 spatial dimensions
    # Frame 1: [1, 2, 3]
    # Frame 2: [2, 4, 6]
    # Frame 3: [5, 7, 9]
    spatial_seq = np.array([
        [1, 2, 3],
        [2, 4, 6],
        [5, 7, 9]
    ], dtype=np.float32)
    
    # _add_velocity computes current - previous.
    result = _add_velocity(spatial_seq)
    
    # Result should be [spatial, velocity] concatenated
    # Frame 1: [1, 2, 3, 0, 0, 0]
    # Frame 2: [2, 4, 6, 1, 2, 3]  (2-1, 4-2, 6-3)
    # Frame 3: [5, 7, 9, 3, 3, 3]  (5-2, 7-4, 9-6)
    
    expected = np.array([
        [1, 2, 3, 0, 0, 0],
        [2, 4, 6, 1, 2, 3],
        [5, 7, 9, 3, 3, 3]
    ], dtype=np.float32)
    
    assert np.all(result == expected), "Preprocess velocity logic failed!"
    print("Preprocess velocity logic passed!\n")
    

# 2. Test simulate_frontend.py temporal logic
def test_frontend_simulator_logic():
    print("--- Testing simulate_frontend.py velocity loop ---")
    spatial_frames = [
        np.array([1, 2, 3], dtype=np.float32),
        np.array([2, 4, 6], dtype=np.float32),
        np.array([5, 7, 9], dtype=np.float32)
    ]
    
    expected_velocities = [
        np.array([0, 0, 0], dtype=np.float32),
        np.array([1, 2, 3], dtype=np.float32),
        np.array([3, 3, 3], dtype=np.float32)
    ]
    
    # Simulating the exact loop structure from simulate_frontend.py
    prev_spatial = np.zeros(3, dtype=np.float32)
    
    for i, spatial in enumerate(spatial_frames):
        if i == 0:
            velocity = np.zeros_like(spatial)
        else:
            velocity = spatial - prev_spatial
            
        prev_spatial = spatial.copy()
        
        # Verify
        assert np.all(velocity == expected_velocities[i]), f"Frontend velocity logic failed on frame {i}! Got {velocity}, Expected {expected_velocities[i]}"
        print(f"Frame {i}: spatial={spatial}, velocity={velocity} [OK]")
        
    print("Frontend simulator temporal logic passed!")

if __name__ == "__main__":
    test_add_velocity()
    test_frontend_simulator_logic()

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.animation import FuncAnimation
import sys

def get_npy_files(dataset_dir="processed"):
    """Get all .npy files from Dataset folder."""
    dataset_path = Path(dataset_dir)
    npy_files = sorted(list(dataset_path.rglob("*.npy")))
    return npy_files

def view_file_info(npy_file):
    """Display info about a .npy file."""
    data = np.load(npy_file)
    print(f"\n{'='*60}")
    print(f"File: {npy_file.relative_to('processed')}")
    print(f"Shape: {data.shape}")
    print(f"Data type: {data.dtype}")
    print(f"Min value: {np.min(data):.4f}")
    print(f"Max value: {np.max(data):.4f}")
    print(f"Mean value: {np.mean(data):.4f}")
    print(f"Non-zero elements: {np.count_nonzero(data)}")
    print(f"{'='*60}\n")
    return data

def plot_landmarks_frame(landmarks, frame_idx=0, figsize=(12, 8)):
    """
    Plot landmarks for a single frame.
    Assumes landmarks shape is (frames, num_landmarks, 3) for (x, y, z) coordinates.
    """
    if len(landmarks.shape) == 3:
        # (frames, landmarks, 3)
        frame_data = landmarks[frame_idx]
    elif len(landmarks.shape) == 2:
        # Already in (landmarks, 3) format
        frame_data = landmarks
    else:
        print(f"Unexpected shape: {landmarks.shape}")
        return
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot landmark points
    if frame_data.shape[1] >= 2:
        x = frame_data[:, 0]
        y = frame_data[:, 1]
        
        ax.scatter(x, y, c='red', s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        # Add landmark indices
        for i, (xi, yi) in enumerate(zip(x, y)):
            ax.annotate(str(i), (xi, yi), fontsize=7, ha='center', va='center')
        
        # Connect some basic skeleton connections if it looks like it
        if len(frame_data) > 10:
            # Try to connect nearby points (simple heuristic)
            for i in range(len(frame_data) - 1):
                x_line = [frame_data[i, 0], frame_data[i+1, 0]]
                y_line = [frame_data[i, 1], frame_data[i+1, 1]]
                ax.plot(x_line, y_line, 'b-', alpha=0.3, linewidth=0.5)
    
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_xlabel('X (normalized)')
    ax.set_ylabel('Y (normalized)')
    ax.set_title(f'Landmarks - Frame {frame_idx}')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def interactive_viewer():
    """Interactive viewer for .npy files."""
    npy_files = get_npy_files()
    
    print(f"\nFound {len(npy_files)} .npy files\n")
    
    for i, file in enumerate(npy_files[:50]):  # Show first 10
        print(f"[{i}] {file.relative_to('processed')}")
    
    if len(npy_files) > 10:
        print(f"... and {len(npy_files) - 10} more files")
    
    print("\n" + "="*60)
    print("Usage examples:")
    print("  python view_npy_webcam.py view <index>")
    print("  python view_npy_webcam.py view 0")
    print("  python view_npy_webcam.py list")
    print("  python view_npy_webcam.py info 0")
    print("="*60)
    
    return npy_files

def main():
    npy_files = get_npy_files()
    
    if len(sys.argv) < 2:
        interactive_viewer()
        return
    
    command = sys.argv[1]
    
    if command == "list":
        print(f"\nAll {len(npy_files)} .npy files:\n")
        for i, file in enumerate(npy_files):
            data = np.load(file)
            print(f"[{i:3d}] {str(file.relative_to('Dataset')):50s} shape: {str(data.shape):30s}")
    
    elif command == "info":
        if len(sys.argv) < 3:
            print("Usage: python view_npy_webcam.py info <index>")
            return
        idx = int(sys.argv[2])
        if 0 <= idx < len(npy_files):
            data = view_file_info(npy_files[idx])
        else:
            print(f"Index {idx} out of range (0-{len(npy_files)-1})")
    
    elif command == "view":
        if len(sys.argv) < 3:
            print("Usage: python view_npy_webcam.py view <index>")
            return
        idx = int(sys.argv[2])
        if 0 <= idx < len(npy_files):
            data = view_file_info(npy_files[idx])
            
            # Show first and middle frames
            if len(data.shape) == 3:
                num_frames = data.shape[0]
                frame_idx = num_frames // 2
                print(f"Displaying frame {frame_idx} of {num_frames}")
                fig = plot_landmarks_frame(data, frame_idx=frame_idx)
            else:
                fig = plot_landmarks_frame(data)
            
            plt.show()
        else:
            print(f"Index {idx} out of range (0-{len(npy_files)-1})")
    
    else:
        print(f"Unknown command: {command}")
        interactive_viewer()

if __name__ == "__main__":
    main()

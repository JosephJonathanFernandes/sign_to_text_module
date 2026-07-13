import argparse
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description="Run the post-processing pipeline for a newly collected sign language class.")
    parser.add_argument("--cls", type=str, required=True, help="Class name to process")
    args = parser.parse_args()

    class_name = args.cls

    commands = [
        [sys.executable, "-m", "src.preprocessing.augment_pipeline", "--class", class_name],
        [sys.executable, "-m", "src.preprocessing.quality_filter_hybrid", "--class", class_name],
        [sys.executable, "-m", "src.preprocessing.balance_processed_dataset", "--class", class_name],
        [sys.executable, "src/tools/generate_dataset_heuristics.py", "--class", class_name],
    ]

    for cmd in commands:
        print("\n" + "="*80)
        print(f"🚀 Running: {' '.join(cmd)}")
        print("="*80 + "\n")
        
        result = subprocess.run(cmd)
        
        if result.returncode != 0:
            print(f"\n❌ Error: Command failed with exit code {result.returncode}")
            print(f"Failed Command: {' '.join(cmd)}")
            sys.exit(result.returncode)
            
    print("\n" + "="*80)
    print(f"✅ Successfully processed class '{class_name}'!")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()

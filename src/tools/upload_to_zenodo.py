import requests
import os
import sys
import hashlib
import time

# ==========================================
# SETUP: FILL IN THESE TWO VALUES
# ==========================================
TOKEN = "Sx3Xla4785XPGDRjmgQomTkHo9VJzMuRaKcSiUuVZKGBTIpX57EKIDc0kwXt"
DEPOSITION_ID = "23004815"

# Path to the large dataset file
FILE_PATH = r"C:\Users\Joseph\Desktop\projects\sign_to_text\assets\dataset_consolidated.h5"
# ==========================================

class ProgressFileReader:
    """A wrapper to read a file in chunks and print upload progress."""
    def __init__(self, filename):
        self.filename = filename
        self.size = os.stat(filename).st_size
        self.read_so_far = 0
        self.f = open(filename, 'rb')
        self.start_time = time.time()
        
    def read(self, chunk_size):
        chunk = self.f.read(chunk_size)
        self.read_so_far += len(chunk)
        
        # Calculate progress and speed
        percent = (self.read_so_far / self.size) * 100
        elapsed = time.time() - self.start_time
        speed_mb = (self.read_so_far / (1024 * 1024)) / elapsed if elapsed > 0 else 0
        
        sys.stdout.write(f"\rUploading: {percent:.1f}% ({self.read_so_far / 1e9:.2f} / {self.size / 1e9:.2f} GB) - Speed: {speed_mb:.1f} MB/s")
        sys.stdout.flush()
        return chunk
        
    def __len__(self):
        return self.size
        
    def reset(self):
        self.f.seek(0)
        self.read_so_far = 0
        self.start_time = time.time()
        print("\n[Upload Progress Reset]")
        
    def close(self):
        self.f.close()
        print() # New line after progress bar finishes

def get_local_md5(filepath):
    print("Computing local MD5 checksum...")
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def main():
    if TOKEN == "paste_your_token_here" or DEPOSITION_ID == "paste_your_deposition_id_here":
        print("ERROR: Please edit the script to add your Zenodo TOKEN and DEPOSITION_ID.")
        return

    if not os.path.exists(FILE_PATH):
        print(f"ERROR: File not found at {FILE_PATH}")
        return

    session = requests.Session()
    
    print(f"Connecting to Zenodo for Deposition ID: {DEPOSITION_ID}...")
    r = session.get(
        f"https://zenodo.org/api/deposit/depositions/{DEPOSITION_ID}",
        params={"access_token": TOKEN},
    )
    r.raise_for_status()
    bucket_url = r.json()["links"]["bucket"]
    print(f"Bucket URL established: {bucket_url}")

    file_name = os.path.basename(FILE_PATH)
    
    # Pre-calculate local MD5 so we can compare immediately after upload
    local_md5 = get_local_md5(FILE_PATH)
    print(f"Local MD5: {local_md5}")

    print(f"\nStarting upload of {file_name}...")
    reader = ProgressFileReader(FILE_PATH)
    
    MAX_RETRIES = 5
    success = False
    
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            reader.reset()
            r = session.put(
                f"{bucket_url}/{file_name}",
                data=reader,
                params={"access_token": TOKEN},
                timeout=None, # Crucial for large uploads
            )
            r.raise_for_status()
            success = True
            break
        except requests.exceptions.RequestException as e:
            print(f"\n\nUpload failed on attempt {attempt}/{MAX_RETRIES}.")
            print(f"Error: {e}")
            if attempt < MAX_RETRIES:
                print("Zenodo occasionally drops long connections. Retrying in 10 seconds...")
                time.sleep(10)
            else:
                print("Max retries reached. Upload aborted.")
                reader.close()
                return
    
    reader.close()
    
    if not success:
        return

    print("\nUpload finished! Validating checksum...")
    response_data = r.json()
    zenodo_checksum = response_data.get("checksum", "")
    
    # Zenodo checksums are sometimes formatted as "md5:..."
    if zenodo_checksum.startswith("md5:"):
        zenodo_checksum = zenodo_checksum[4:]

    print(f"Zenodo MD5: {zenodo_checksum}")
    
    if local_md5 == zenodo_checksum:
        print("\n✅ SUCCESS: Uploaded file checksum matches the local file.")
    else:
        print("\n❌ ERROR: Checksum mismatch! The upload may be corrupted.")

if __name__ == "__main__":
    main()

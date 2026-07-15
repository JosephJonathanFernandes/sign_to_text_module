"""
Jump Calibration Logger — records raw landmark jump values during real signing
WITHOUT rejecting any frames, so you can compute the empirical distribution
and validate/tune LANDMARK_JUMP_THRESHOLD.

Usage:
    python scratch/calibrate_jump_threshold.py

Output:
    results/api_audit/jump_calibration.json  — full distribution stats
    results/api_audit/jump_calibration.csv   — raw jump values for plotting

Run while signing normally for 5-10 minutes. Then check:
    - If 99th-percentile < 0.20:  threshold is safe
    - If fast gestures hit 0.18+: raise threshold to 0.25 or 0.30
"""

import asyncio
import json
import csv
import numpy as np
import websockets
import time
import os

WS_URL = "ws://localhost:8000/ws/translate"
OUTPUT_JSON = os.path.join("results", "api_audit", "jump_calibration.json")
OUTPUT_CSV  = os.path.join("results", "api_audit", "jump_calibration.csv")

# ── Passive monitoring client ─────────────────────────────────────────────────

async def monitor(duration_seconds: int = 60):
    """
    Connect as a passive monitoring client. Intercepts all incoming prediction
    messages and logs the 'jump' value from debug payloads if present.

    NOTE: This requires DEBUG=true to be set on the API server, OR
    you can use the standalone mode below that reads from a saved .npy file.
    """
    print(f"[Calibrator] Connecting to {WS_URL} ...")
    print(f"[Calibrator] Will collect for {duration_seconds}s. Sign normally.")

    jumps = []
    start = time.time()

    try:
        async with websockets.connect(WS_URL) as ws:
            while time.time() - start < duration_seconds:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
                    msg = json.loads(raw)
                    if "jump" in msg.get("debug", {}):
                        jumps.append(float(msg["debug"]["jump"]))
                except asyncio.TimeoutError:
                    continue
    except Exception as e:
        print(f"[Calibrator] Connection error: {e}")

    return jumps


# ── Standalone mode: compute jumps from a saved sequence .npy ─────────────────

def compute_jumps_from_npy(npy_path: str) -> list[float]:
    """
    Load a saved sequence file (.npy, shape: [N, 506]) and compute
    frame-to-frame jump values using the same logic as api/app.py.
    """
    seq = np.load(npy_path).astype(np.float32)  # (N, 506)
    jumps = []
    prev = None
    for frame in seq:
        raw = frame[:126]
        if prev is not None:
            j = float(np.mean(np.abs(raw - prev)))
            jumps.append(j)
        prev = raw.copy()
    return jumps


def report(jumps: list[float], source: str = ""):
    if not jumps:
        print("[Calibrator] No jump data collected.")
        return

    arr = np.array(jumps, dtype=np.float32)
    stats = {
        "source": source,
        "n_frames": len(arr),
        "mean":   float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std":    float(np.std(arr)),
        "p90":    float(np.percentile(arr, 90)),
        "p95":    float(np.percentile(arr, 95)),
        "p99":    float(np.percentile(arr, 99)),
        "max":    float(np.max(arr)),
        "current_threshold": 0.20,
        "safe_margin_at_p99": round(0.20 - float(np.percentile(arr, 99)), 4),
    }

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(stats, f, indent=2)

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_index", "jump"])
        for i, j in enumerate(jumps):
            writer.writerow([i, round(j, 6)])

    print("\n[Calibrator] ── Jump Distribution Report ──────────────────────")
    print(f"  Source     : {source}")
    print(f"  Frames     : {stats['n_frames']}")
    print(f"  Mean       : {stats['mean']:.4f}")
    print(f"  Median     : {stats['median']:.4f}")
    print(f"  P90        : {stats['p90']:.4f}")
    print(f"  P95        : {stats['p95']:.4f}")
    print(f"  P99        : {stats['p99']:.4f}")
    print(f"  Max        : {stats['max']:.4f}")
    print(f"  Threshold  : 0.20")
    print(f"  Safety margin at P99: {stats['safe_margin_at_p99']:.4f}")
    if stats['safe_margin_at_p99'] < 0.02:
        print("  ⚠️  Threshold is very tight. Consider raising to 0.25.")
    elif stats['safe_margin_at_p99'] < 0.05:
        print("  ⚠️  Threshold may be tight. Monitor during fast signing.")
    else:
        print("  ✅ Threshold has comfortable safety margin.")
    print(f"\n  Saved stats → {OUTPUT_JSON}")
    print(f"  Saved raw   → {OUTPUT_CSV}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Standalone mode: python calibrate_jump_threshold.py path/to/file.npy
        npy_path = sys.argv[1]
        print(f"[Calibrator] Standalone mode — loading {npy_path}")
        jumps = compute_jumps_from_npy(npy_path)
        report(jumps, source=npy_path)
    else:
        # Live mode: connect to running API (requires DEBUG=true)
        print("[Calibrator] Live mode — collecting jump values for 120s")
        jumps = asyncio.run(monitor(duration_seconds=120))
        report(jumps, source="live_ws")

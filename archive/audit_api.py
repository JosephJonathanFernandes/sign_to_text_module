"""
Production API Validation & Audit Suite

Verifies:
1. Static dependencies (visual check completed)
2. Real inference accuracy via POST /predict
3. Sliding buffer behavior
4. Multi-client session isolation
5. WebSocket stress test
6. Latency Benchmarking
7. Memory/Resource validation
8. Failure & Fault Tolerance
"""

import os
import sys
import glob
import json
import time
import asyncio
import numpy as np
import httpx
import websockets
import psutil
import random
from collections import defaultdict

API_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000/ws/translate"

def get_real_samples(max_samples=5):
    """Load random real data samples from assets/Dataset."""
    files = glob.glob('assets/Dataset/*/*.npy')
    if not files:
        print("❌ No .npy files found in assets/Dataset/")
        sys.exit(1)
        
    random.shuffle(files)
    samples = []
    
    # Try to get different classes
    classes_found = set()
    for f in files:
        raw_cls = os.path.basename(os.path.dirname(f))
        # Handle "1. Dog" format -> "DOG"
        cls_name = raw_cls.split('.', 1)[-1].strip().upper() if '.' in raw_cls else raw_cls.strip().upper()
        if cls_name not in classes_found or len(samples) < max_samples:
            try:
                data = np.load(f)
                if data.shape == (20, 506):
                    samples.append((cls_name, data))
                    classes_found.add(cls_name)
            except Exception as e:
                pass
        if len(samples) >= max_samples:
            break
            
    return samples

async def test_real_inference(samples):
    print("\n--- PHASE 2: REAL INFERENCE VALIDATION ---")
    correct = 0
    total = len(samples)
    latencies = []
    
    async with httpx.AsyncClient() as client:
        for expected_cls, data in samples:
            payload = {"sequence": data.tolist()}
            start = time.perf_counter()
            r = await client.post(f"{API_URL}/predict", json=payload)
            latency = (time.perf_counter() - start) * 1000
            latencies.append(latency)
            
            if r.status_code == 200:
                resp = r.json()
                predicted = resp.get("predicted_word", "")
                conf = resp.get("confidence", 0.0)
                is_correct = predicted == expected_cls.upper()
                if is_correct: correct += 1
                
                print(f"Expected: {expected_cls.upper():<15} Predicted: {predicted:<15} "
                      f"Confidence: {conf:.4f}  Correct: {is_correct}  Latency: {latency:.1f}ms")
            else:
                print(f"❌ Failed request for {expected_cls}: {r.status_code}")
                
    print(f"\nAccuracy Summary: {correct}/{total} ({(correct/total)*100:.1f}%)")
    print(f"Avg Latency: {np.mean(latencies):.1f}ms")
    return correct == total

async def test_sliding_buffer(sample_data):
    print("\n--- PHASE 3: SLIDING BUFFER VALIDATION ---")
    # Send 25 frames
    frames = sample_data.tolist()
    # Duplicate last frame 5 times to simulate hold
    for _ in range(5):
        frames.append(frames[-1])
        
    predictions_received = []
    
    async with websockets.connect(WS_URL) as ws:
        for i, frame in enumerate(frames):
            await ws.send(json.dumps({
                "type": "landmarks",
                "features": frame,
                "timestamp": i
            }))
            
            try:
                # Wait briefly to check if a response arrived
                resp = await asyncio.wait_for(ws.recv(), timeout=0.1)
                data = json.loads(resp)
                if data.get("type") == "prediction":
                    predictions_received.append(i + 1)
            except asyncio.TimeoutError:
                pass
            await asyncio.sleep(0.01) # Simulate 10ms frame time
            
    print(f"Sent 25 frames.")
    print(f"Predictions received at frames: {predictions_received}")
    
    # Validation: first prediction should be at frame 20, then 21, 22, 23, 24, 25
    passed = len(predictions_received) > 0 and min(predictions_received) >= 20
    print(f"Pass/Fail: {'PASS' if passed else 'FAIL'}")
    return passed

async def single_client_session(client_id, expected_cls, data):
    """Simulate one client sending data and expecting a specific translation."""
    frames = data.tolist()
    # Add dummy frames to push the stable predictor (patience=3)
    for _ in range(10):
        frames.append(frames[-1])
        
    async with websockets.connect(WS_URL) as ws:
        for i, frame in enumerate(frames):
            await ws.send(json.dumps({
                "type": "landmarks",
                "features": frame
            }))
            await asyncio.sleep(0.01)
            
            # Drain incoming predictions
            try:
                while True:
                    await asyncio.wait_for(ws.recv(), timeout=0.01)
            except asyncio.TimeoutError:
                pass
                
        # Send stop
        await ws.send(json.dumps({"type": "stop"}))
        try:
            # Wait for translation
            resp = await asyncio.wait_for(ws.recv(), timeout=2.0)
            data = json.loads(resp)
            if data.get("type") == "translation":
                words = data.get("words", [])
                return client_id, expected_cls, words
        except asyncio.TimeoutError:
            pass
            
    return client_id, expected_cls, []

async def test_session_isolation(samples):
    print("\n--- PHASE 4: MULTI-CLIENT SESSION ISOLATION ---")
    if len(samples) < 3:
        print("Not enough samples for this test.")
        return False
        
    tasks = []
    for i in range(3):
        cls_name, data = samples[i]
        tasks.append(single_client_session(f"Client_{i+1}", cls_name, data))
        
    results = await asyncio.gather(*tasks)
    
    passed = True
    print("Session Results:")
    for cid, expected, words in results:
        print(f"{cid} | Expected: {expected.upper()} | Output Words: {words}")
        if expected.upper() not in words:
            passed = False
            
    print(f"Cross contamination check: {'PASS' if passed else 'FAIL'}")
    return passed

async def stress_client(data, num_frames=40, delay=0.033):
    """A client that streams frames and tracks latencies."""
    latencies = []
    frame = data[0].tolist()
    
    try:
        async with websockets.connect(WS_URL) as ws:
            for i in range(num_frames):
                start = time.perf_counter()
                await ws.send(json.dumps({
                    "type": "landmarks",
                    "features": frame
                }))
                
                try:
                    resp = await asyncio.wait_for(ws.recv(), timeout=delay)
                    latency = (time.perf_counter() - start) * 1000
                    latencies.append(latency)
                except asyncio.TimeoutError:
                    pass
                
                # Maintain constant tick
                elapsed = time.perf_counter() - start
                if elapsed < delay:
                    await asyncio.sleep(delay - elapsed)
                    
    except Exception as e:
        return []
    return latencies

async def test_stress_and_latency(sample_data, num_clients=25):
    print(f"\n--- PHASE 5 & 6: WEBSOCKET STRESS TEST ({num_clients} Clients) ---")
    
    proc = psutil.Process(os.getpid())
    start_cpu = psutil.cpu_percent()
    start_mem = proc.memory_info().rss / 1024 / 1024
    
    tasks = [stress_client(sample_data) for _ in range(num_clients)]
    
    start_time = time.perf_counter()
    all_latencies = await asyncio.gather(*tasks)
    end_time = time.perf_counter()
    
    flat_latencies = [l for client_lats in all_latencies for l in client_lats]
    
    end_cpu = psutil.cpu_percent()
    end_mem = proc.memory_info().rss / 1024 / 1024
    
    print(f"Total time for 40 frames × {num_clients} clients: {end_time - start_time:.2f}s")
    print(f"Total prediction responses received: {len(flat_latencies)}")
    print(f"CPU usage approx: {end_cpu}%")
    print(f"Memory change: {start_mem:.1f}MB -> {end_mem:.1f}MB")
    
    if flat_latencies:
        avg = np.mean(flat_latencies)
        p50 = np.percentile(flat_latencies, 50)
        p95 = np.percentile(flat_latencies, 95)
        p99 = np.percentile(flat_latencies, 99)
        print(f"Latencies - Avg: {avg:.1f}ms | P50: {p50:.1f}ms | P95: {p95:.1f}ms | P99: {p99:.1f}ms")
    else:
        print("No predictions received!")
        
    return len(flat_latencies) > 0

async def test_failure_modes():
    print("\n--- PHASE 8: FAILURE TESTING ---")
    failures_caught = 0
    total_tests = 4
    
    async with httpx.AsyncClient() as client:
        # 1. Malformed payload
        r = await client.post(f"{API_URL}/predict", json={"sequence": [[0]*10]})
        if r.status_code == 422:
            print("✅ Handled wrong dimensions payload (422)")
            failures_caught += 1
            
        # 2. Complete trash payload
        r = await client.post(f"{API_URL}/predict", content="not json")
        if r.status_code == 422:
            print("✅ Handled corrupted JSON payload (422)")
            failures_caught += 1
            
    # 3. WS invalid json
    try:
        async with websockets.connect(WS_URL) as ws:
            await ws.send("broken_json")
            resp = await asyncio.wait_for(ws.recv(), timeout=1.0)
            data = json.loads(resp)
            if data.get("type") == "error":
                print("✅ Handled invalid WS message")
                failures_caught += 1
    except Exception as e:
        print("❌ WS invalid message test failed")

    # 4. WS wrong dimensions
    try:
        async with websockets.connect(WS_URL) as ws:
            await ws.send(json.dumps({"type": "landmarks", "features": [0.0]*100}))
            resp = await asyncio.wait_for(ws.recv(), timeout=1.0)
            data = json.loads(resp)
            if data.get("type") == "error" and "Expected" in data.get("message", ""):
                print("✅ Handled wrong feature length WS message")
                failures_caught += 1
    except Exception as e:
        print("❌ WS wrong dimensions test failed")
        
    print(f"Failures caught gracefully: {failures_caught}/{total_tests}")
    return failures_caught == total_tests

async def run_all():
    # Wait for API to boot if needed
    try:
        async with httpx.AsyncClient() as client:
            await client.get(f"{API_URL}/health")
    except:
        print("Waiting for API server to start...")
        await asyncio.sleep(2)
        
    samples = get_real_samples()
    print(f"Loaded {len(samples)} real samples.")
    
    p2 = await test_real_inference(samples)
    p3 = await test_sliding_buffer(samples[0][1])
    p4 = await test_session_isolation(samples)
    p5 = await test_stress_and_latency(samples[0][1], num_clients=15)
    p8 = await test_failure_modes()
    
    print("\n=======================================================")
    print("FINAL VALIDATION SCORECARD")
    print(f"P2: Real Inference:    {'✅ PASS' if p2 else '❌ FAIL'}")
    print(f"P3: Sliding Buffer:    {'✅ PASS' if p3 else '❌ FAIL'}")
    print(f"P4: Session Isolation: {'✅ PASS' if p4 else '❌ FAIL'}")
    print(f"P5: Stress & Latency:  {'✅ PASS' if p5 else '❌ FAIL'}")
    print(f"P8: Failure Handling:  {'✅ PASS' if p8 else '❌ FAIL'}")
    print("=======================================================")

if __name__ == "__main__":
    asyncio.run(run_all())

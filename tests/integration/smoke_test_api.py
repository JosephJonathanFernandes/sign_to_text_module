"""
Smoke test script for the API.
Verifies GET /health, POST /predict, and WebSocket /ws/translate.

Run the API first: `python run_api.py`
Then run this script in another terminal: `python api/test_api.py`
"""

import requests
import asyncio
import websockets
import json
import numpy as np

API_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000/ws/translate"

def test_health():
    print("\n--- Testing GET /health ---")
    try:
        r = requests.get(f"{API_URL}/health")
        r.raise_for_status()
        data = r.json()
        print(f"✅ Health OK: {data}")
        return data["sequence_length"], data["feature_dimension"]
    except Exception as e:
        print(f"❌ Health Failed: {e}")
        return None, None

def test_predict_http(seq_len, feat_dim):
    print("\n--- Testing POST /predict ---")
    if not seq_len or not feat_dim:
        print("Skipping due to missing dimensions.")
        return
    
    # Create dummy data (all zeros)
    dummy_seq = np.zeros((seq_len, feat_dim)).tolist()
    
    payload = {"sequence": dummy_seq}
    try:
        r = requests.post(f"{API_URL}/predict", json=payload)
        r.raise_for_status()
        print(f"✅ Predict OK: {r.json()}")
    except Exception as e:
        print(f"❌ Predict Failed: {e}")
        if hasattr(e, 'response') and e.response:
            print(f"Response: {e.response.text}")

async def test_websocket(seq_len, feat_dim):
    print("\n--- Testing WS /ws/translate ---")
    if not seq_len or not feat_dim:
        print("Skipping due to missing dimensions.")
        return
        
    try:
        async with websockets.connect(WS_URL) as ws:
            print("Connected to WebSocket.")
            
            # Send seq_len frames
            for i in range(seq_len):
                frame = np.zeros(feat_dim).tolist()
                msg = {
                    "type": "landmarks",
                    "features": frame,
                    "timestamp": i
                }
                await ws.send(json.dumps(msg))
                # Small delay to simulate camera frame rate
                await asyncio.sleep(0.01)
                
            print(f"Sent {seq_len} frames. Waiting for prediction...")
            
            # Wait for prediction response
            resp = await asyncio.wait_for(ws.recv(), timeout=5.0)
            print(f"✅ WS Prediction Response: {resp}")
            
            # Send stop command
            stop_msg = {"type": "stop"}
            await ws.send(json.dumps(stop_msg))
            print("Sent stop command.")
            
            # Wait for final translation
            resp = await asyncio.wait_for(ws.recv(), timeout=5.0)
            print(f"✅ WS Translation Response: {resp}")
            
    except Exception as e:
        print(f"❌ WS Test Failed: {e}")

if __name__ == "__main__":
    print("Ensure the API is running at http://localhost:8000")
    seq_len, feat_dim = test_health()
    if seq_len:
        test_predict_http(seq_len, feat_dim)
        asyncio.run(test_websocket(seq_len, feat_dim))
    print("\nTests completed.")

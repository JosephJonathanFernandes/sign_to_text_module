import asyncio
import websockets
import json
import numpy as np
import time

WS_URL = "ws://localhost:8000/ws/translate"
NUM_FRAMES = 20
INPUT_SIZE = 506

async def send_valid_frame(ws):
    frame = np.random.rand(INPUT_SIZE).astype(np.float32).tolist()
    await ws.send(json.dumps({
        "type": "landmarks",
        "features": frame
    }))

async def test_normal_use():
    print("--- Test 1: Normal Use ---")
    async with websockets.connect(WS_URL) as ws:
        for _ in range(30):
            await send_valid_frame(ws)
            await asyncio.sleep(0.03) # 30 FPS
            
        # Wait for a prediction
        while True:
            try:
                resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=1.0))
                print(f"Received: {resp['type']}")
                if resp['type'] == "prediction":
                    print("Normal Use Passed!")
                    break
            except asyncio.TimeoutError:
                break
        await ws.send(json.dumps({"type": "stop"}))
        resp = json.loads(await ws.recv())
        print(f"Stop response: {resp['type']}")

async def test_disconnect():
    print("\n--- Test 2: Disconnect ---")
    async with websockets.connect(WS_URL) as ws:
        await send_valid_frame(ws)
        # Abrupt disconnect
    print("Disconnected abruptly. Server should clean up cleanly.")

async def test_bad_payload():
    print("\n--- Test 3: Bad Payload (NaN) ---")
    async with websockets.connect(WS_URL) as ws:
        frame = np.random.rand(INPUT_SIZE).astype(np.float32)
        frame[10] = float('nan') # Inject NaN
        await ws.send(json.dumps({
            "type": "landmarks",
            "features": frame.tolist()
        }))
        await asyncio.sleep(0.5)
        print("Sent NaN. Connection should remain open (dropped frame).")
        
        # Test large payload
        print("\n--- Test 4: Bad Payload (Large) ---")
        large_str = "a" * 60000
        await ws.send(large_str)
        try:
            resp = await ws.recv()
            print(f"Response: {resp}")
        except websockets.exceptions.ConnectionClosed as e:
            print(f"Connection closed as expected with code: {e.code}, reason: {e.reason}")

async def test_rapid_signing():
    print("\n--- Test 5: Rapid Burst ---")
    async with websockets.connect(WS_URL) as ws:
        # Send 100 frames as fast as possible
        for _ in range(100):
            frame = np.random.rand(INPUT_SIZE).astype(np.float32).tolist()
            await ws.send(json.dumps({
                "type": "landmarks",
                "features": frame
            }))
        
        print("Burst sent. Waiting for predictions to drain...")
        preds = 0
        while True:
            try:
                resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=2.0))
                if resp['type'] == 'prediction':
                    preds += 1
            except asyncio.TimeoutError:
                break
        print(f"Received {preds} predictions from burst.")

async def main():
    try:
        await test_normal_use()
        await test_disconnect()
        await test_bad_payload()
        await test_rapid_signing()
        print("\nAll automated smoke tests executed successfully!")
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())

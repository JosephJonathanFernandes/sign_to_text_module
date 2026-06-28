"""
Frontend Integration Simulator

This script simulates a JS browser client processing a video frame-by-frame,
building the feature vector using the shared Feature Extractor contract,
and sending it over WebSockets.
It verifies both `/validate_features` and `/ws/translate` endpoints.
"""

import asyncio
import json
import cv2
import numpy as np
import websockets
import httpx
import mediapipe as mp
import time
from typing import List, Optional

from mediapipe.tasks.python.vision import (
    HandLandmarker, HandLandmarkerOptions,
    FaceLandmarker, FaceLandmarkerOptions,
    RunningMode
)
from mediapipe.tasks.python import BaseOptions

# Relative imports from project root
from src.shared.feature_extractor import build_single_frame_features
from src.core.config import get_config

cfg = get_config()
HAND_MODEL = cfg.paths.hand_landmarker_model
FACE_MODEL = cfg.paths.face_landmarker_model

def get_landmarker():
    hand_options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=HAND_MODEL),
        running_mode=RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    hand = HandLandmarker.create_from_options(hand_options)
    
    face_options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=FACE_MODEL),
        running_mode=RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.3,
        min_face_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    face = FaceLandmarker.create_from_options(face_options)
    
    return hand, face

def get_raw_landmarks(hand_landmarker, face_landmarker, frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    
    h_res = hand_landmarker.detect(mp_image)
    f_res = face_landmarker.detect(mp_image)
    
    left_raw = None
    right_raw = None
    
    for hand, handedness_list in zip(h_res.hand_landmarks, h_res.handedness):
        label = handedness_list[0].display_name
        arr = np.zeros(63, dtype=np.float32)
        for i, lm in enumerate(hand):
            arr[i*3] = lm.x
            arr[i*3+1] = lm.y
            arr[i*3+2] = lm.z
        if label == "Left":
            left_raw = arr
        else:
            right_raw = arr
            
    face_raw = None
    if f_res.face_landmarks:
        flm = f_res.face_landmarks[0]
        face_raw = np.zeros(264 * 3, dtype=np.float32)
        for idx in [1, 33, 263]:
            if idx < len(flm):
                face_raw[idx*3] = flm[idx].x
                face_raw[idx*3+1] = flm[idx].y
                face_raw[idx*3+2] = flm[idx].z
                
    return left_raw, right_raw, face_raw

async def test_validate_features(left, right, face, features_253):
    print("[Simulator] Testing POST /validate_features...")
    payload = {
        "schema_version": "1.0",
        "raw_landmarks": {
            "left_hand": left.tolist() if left is not None else None,
            "right_hand": right.tolist() if right is not None else None,
            "face": face.tolist() if face is not None else None
        },
        "features": features_253.tolist()
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post("http://127.0.0.1:8000/validate_features", json=payload)
        data = response.json()
        assert response.status_code == 200, f"Error {response.status_code}: {data}"
        assert data["valid"] is True, f"Validation failed: {data}"
        print(f"  [OK] Validation passed! MAE: {data['mae']:.8f}")

async def test_websocket_stream(video_path: str):
    print(f"\n[Simulator] Testing WebSocket streaming with {video_path}...")
    hand_lm, face_lm = get_landmarker()
    
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frames.append(frame)
    cap.release()
    
    # Take evenly spaced 20 frames
    indices = np.linspace(0, len(frames)-1, 20, dtype=int)
    sampled = [frames[i] for i in indices]
    
    async with websockets.connect("ws://127.0.0.1:8000/ws/translate") as ws:
        prev_spatial = np.zeros(253, dtype=np.float32)
        
        for i, frame in enumerate(sampled):
            left, right, face = get_raw_landmarks(hand_lm, face_lm, frame)
            
            # Frontend builds spatial features
            spatial = build_single_frame_features(left, right, face)
            
            # Validation endpoint test on the very first frame
            if i == 0:
                await test_validate_features(left, right, face, spatial)
                
            # Frontend computes velocity
            if i == 0:
                velocity = np.zeros_like(spatial)
            else:
                velocity = spatial - prev_spatial
                
            prev_spatial = spatial.copy()
            
            # Combine 506 features
            combined = np.concatenate([spatial, velocity]).astype(np.float32)
            
            # Send frame
            msg = {
                "type": "landmarks",
                "schema_version": "1.0",
                "features": combined.tolist(),
                "timestamp": int(time.time() * 1000)
            }
            await ws.send(json.dumps(msg))
            
            # Discard any intermediate predictions if buffer fills
            # Actually we can just wait for them and print them
            
        # Stop signal
        print("  [Simulator] Sending 'stop' signal...")
        await ws.send(json.dumps({"type": "stop"}))
        
        # Collect all responses until translation
        while True:
            resp = await ws.recv()
            data = json.loads(resp)
            if data.get("type") == "prediction":
                word = data.get("word")
                if word:
                    print(f"  [WS] Intermediate prediction: {word} (conf: {data['confidence']:.2f})")
            elif data.get("type") == "translation":
                print(f"  [WS] Final Translation Output: {data['text']}")
                print(f"  [WS] Raw words: {data['words']}")
                break
            elif data.get("type") == "error":
                print(f"  [WS] ERROR: {data['message']}")
                break
                
    hand_lm.close()
    face_lm.close()
    print("[Simulator] Done.\n")

if __name__ == "__main__":
    video_file = r"assets\Dataset\40. I\MVI_0001.MOV"
    asyncio.run(test_websocket_stream(video_file))

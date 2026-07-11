"""
Thin synchronous wrapper around the existing ensemble_predict().

This module contains ZERO ML logic. It only:
  1. Calls the existing ensemble_predict() from src/inference/ensemble.py
  2. Formats the raw output into a response dict
  3. Optionally appends debug top-5 probabilities

It is designed to be called via asyncio.run_in_executor() so it
never blocks the FastAPI event loop.
"""

from __future__ import annotations

import numpy as np

from src.inference.ensemble import ensemble_predict, check_ood


def run_predict(
    models: list,
    sequence: np.ndarray,
    classes: list,
    debug: bool = False,
) -> tuple[dict, int, float, np.ndarray]:
    """
    Run synchronous ensemble inference and return a formatted response dict.

    Wraps the existing ensemble_predict() with NO changes to ML logic.
    Intended to run in a ThreadPoolExecutor via run_in_executor().

    Args:
        models:    List of loaded SignLanguageGRU models (app.state.models)
        sequence:  NumPy array of shape (NUM_FRAMES, INPUT_SIZE), dtype float32
        classes:   Ordered list of class name strings (app.state.classes)
        debug:     If True, include top-5 probabilities in returned dict

    Returns:
        Tuple of:
          - response_dict: {"predicted_word", "confidence", optionally "debug"}
          - pred_idx:      Raw argmax class index (before temporal smoothing)
          - confidence:    Raw max probability (before temporal smoothing)
          - all_probs:     Full softmax probability vector (num_classes,)
    """
    pred_idx, confidence, all_probs = ensemble_predict(models, sequence)

    is_ood, ood_reason = check_ood(all_probs)
    if is_ood:
        word = "__reject__"
        # We can keep the raw confidence in debug, but surface 0 to the user
        response_confidence = 0.0
    else:
        word = classes[pred_idx]
        response_confidence = round(float(confidence), 4)

    response: dict = {
        "predicted_word": word.upper(),
        "confidence": response_confidence,
    }
    
    if is_ood:
        response["reject_reason"] = ood_reason

    if debug:
        top5_idx = np.argsort(all_probs)[::-1][:5]
        response["debug"] = {
            "top5": [
                {
                    "word": classes[i].upper(),
                    "confidence": round(float(all_probs[i]), 4),
                }
                for i in top5_idx
            ],
            "raw_confidence": round(float(confidence), 4),
        }

    return response, int(pred_idx), float(confidence), all_probs

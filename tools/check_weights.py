import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from src.utils.quantization_utils import build_model_from_checkpoint

def check_keys():
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'model.pth')
    ckpt = torch.load(model_path, map_location='cpu')
    
    num_classes = int(ckpt["num_classes"])
    state_dict = ckpt["model_state_dict"]
    
    hidden_size = int(state_dict["input_proj.0.weight"].shape[0])
    bidirectional = any(key.endswith("_reverse") for key in state_dict.keys() if key.startswith("gru.weight_ih_l"))
    
    layer_indices = []
    for key in state_dict.keys():
        if key.startswith("gru.weight_ih_l") and not key.endswith("_reverse"):
            suffix = key.split("gru.weight_ih_l", 1)[1]
            try:
                layer_indices.append(int(suffix))
            except ValueError:
                continue
    num_layers = (max(layer_indices) + 1) if layer_indices else 1
    
    from src.training.model import SignLanguageGRU
    model = SignLanguageGRU(
        num_classes=num_classes,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bidirectional=bidirectional,
    )
    
    result = model.load_state_dict(state_dict, strict=False)
    
    print("\n--- MODEL WEIGHT LOAD DIAGNOSTICS ---")
    print(f"Number of MISSING keys (in model but not in checkpoint): {len(result.missing_keys)}")
    if result.missing_keys:
        print("First 10 missing keys:", result.missing_keys[:10])
        
    print(f"Number of UNEXPECTED keys (in checkpoint but not in model): {len(result.unexpected_keys)}")
    for k in result.unexpected_keys:
        print(f"  - {k}")
    
    print("\n--- DUMMY INFERENCE TEST ---")
    model.eval()
    
    from src.core.config import INPUT_SIZE
    print(f"Testing forward pass with RAW input dimension: {INPUT_SIZE}")
    
    dummy_input = torch.randn(1, 20, INPUT_SIZE)
    try:
        out = model(dummy_input)
        logits = out["sign_logits"] if isinstance(out, dict) else out
        print("Logits shape:", logits.shape)
        print("Logits contain NaN?", torch.isnan(logits).any().item())
        print("Logits contain Inf?", torch.isinf(logits).any().item())
        if torch.isnan(logits).any().item() or torch.isinf(logits).any().item():
            print("WARNING: Model outputs NaNs or Infs on dummy data!")
        else:
            print("Model forward pass successful, outputs look valid.")
        
        print("\n--- ZERO INFERENCE TEST ---")
        zero_input = torch.zeros(1, 20, INPUT_SIZE)
        zero_out = model(zero_input)
        zero_logits = zero_out["sign_logits"] if isinstance(zero_out, dict) else zero_out
        import torch.nn.functional as F
        probs = F.softmax(zero_logits, dim=1)[0]
        max_prob = torch.max(probs).item()
        print(f"Max confidence on ALL ZEROS input: {max_prob:.4f}")

    except Exception as e:
        print("Model forward pass crashed:")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    check_keys()

"""
Analyze conv frontend ablations: prints shapes, parameter counts, and activation statistics.
Run: python analyze_conv_frontend.py
"""
import torch
from config import get_config
from model import SignLanguageGRU


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    cfg = get_config()
    num_classes = 100  # placeholder; model will adapt head size
    model = SignLanguageGRU(num_classes=num_classes)
    model.eval()

    print("Model param count:", count_parameters(model))

    # Create dummy input: batch x seq_len x input_dim
    seq_len = cfg.preprocessing.num_frames
    input_dim = cfg.frame_features.input_sequence_dim
    x = torch.randn(2, seq_len, input_dim)

    # Forward hooks to capture conv frontend activations
    activations = {}

    def save_act(name):
        def hook(module, inp, out):
            activations[name] = out.detach()
        return hook

    if getattr(model, 'conv_pw', None) is not None:
        model.conv_pw.register_forward_hook(save_act('conv_pw'))
    if getattr(model, 'conv_dw', None) is not None:
        model.conv_dw.register_forward_hook(save_act('conv_dw'))
    if getattr(model, 'conv_pw2', None) is not None:
        model.conv_pw2.register_forward_hook(save_act('conv_pw2'))

    with torch.no_grad():
        logits = model(x)

    print('\n=== Conv Frontend Activation Shapes & Stats ===')
    for name, tensor in activations.items():
        print(f"Layer: {name}")
        print(f"  shape: {tuple(tensor.shape)}")
        print(f"  mean: {tensor.mean().item():.6f}")
        print(f"  std:  {tensor.std().item():.6f}")
        print(f"  min:  {tensor.min().item():.6f}")
        print(f"  max:  {tensor.max().item():.6f}")

    print('\n=== End ===')


if __name__ == '__main__':
    main()

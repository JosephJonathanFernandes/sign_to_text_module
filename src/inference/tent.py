import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

def setup_tent(model: nn.Module, lr: float = 1e-5) -> optim.Optimizer:
    """
    Setup Selective Test-Time Entropy Minimization (TENT) (Wang et al., 2021).
    Configures the model for test-time adaptation:
    - Keeps model in eval() mode so Dropout layers remain disabled at inference.
    - Freezes all parameters EXCEPT the affine parameters (weight & bias) of
      normalization layers (LayerNorm / BatchNorm / GroupNorm).
    """
    model.eval()

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    adaptable_params = []

    # Enable gradients only for normalization affine parameters (weight & bias)
    for _, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            module.train()
            if module.weight is not None:
                module.weight.requires_grad = True
                adaptable_params.append(module.weight)
            if module.bias is not None:
                module.bias.requires_grad = True
                adaptable_params.append(module.bias)

    optimizer = optim.Adam(adaptable_params, lr=lr)
    return optimizer


def tent_adapt_step(
    model: nn.Module,
    optimizer: optim.Optimizer,
    x: torch.Tensor,
    proximity: torch.Tensor = None,
    adapt_conf_thresh: float = 0.75,
    adapt_entropy_ratio: float = 0.25,
    anchor_state: dict = None,
    anchor_lambda: float = 1e-2,
) -> torch.Tensor:
    """
    Perform one step of Selective (Entropy-Gated) TENT adaptation.
    Only updates normalization affine weights on reliable, low-entropy samples
    to prevent high-entropy transition noise (epenthesis) from corrupting weights.
    """
    logits_dict = model(x, proximity=proximity)
    logits = logits_dict['sign_logits'] if isinstance(logits_dict, dict) else logits_dict

    probs = F.softmax(logits, dim=1)
    num_classes = probs.shape[1]
    max_entropy = float(np.log(num_classes)) if num_classes > 1 else 1.0

    sample_entropy = -(probs * torch.log(probs + 1e-6)).sum(dim=1)
    max_prob, _ = probs.max(dim=1)

    # Selective Entropy Gate: only adapt on confident, low-entropy in-distribution samples
    reliable_mask = (max_prob >= adapt_conf_thresh) & (sample_entropy <= adapt_entropy_ratio * max_entropy)

    if torch.any(reliable_mask):
        loss = sample_entropy[reliable_mask].mean()
        if anchor_state is not None and anchor_lambda > 0:
            reg = 0.0
            for name, param in model.named_parameters():
                if param.requires_grad and name in anchor_state:
                    reg = reg + torch.sum((param - anchor_state[name]) ** 2)
            loss = loss + anchor_lambda * reg

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return logits.detach()


class TentWrapper:
    """
    Stateful wrapper for online Selective TENT adaptation with anchor regularization.
    """
    def __init__(self, model: nn.Module, lr: float = 1e-5, reset_state: bool = False):
        self.model = model
        self.lr = lr
        self.reset_state = reset_state

        self.original_state = {
            k: v.detach().clone() for k, v in model.state_dict().items()
        }
        self.optimizer = setup_tent(self.model, self.lr)

    def adapt_and_predict(
        self,
        x: torch.Tensor,
        proximity: torch.Tensor = None,
        adapt: bool = True,
    ) -> torch.Tensor:
        if not adapt:
            with torch.no_grad():
                logits_dict = self.model(x, proximity=proximity)
                return logits_dict['sign_logits'] if isinstance(logits_dict, dict) else logits_dict

        with torch.enable_grad():
            logits = tent_adapt_step(
                self.model,
                self.optimizer,
                x,
                proximity=proximity,
                anchor_state=self.original_state,
            )
        return logits

    def reset(self):
        self.model.load_state_dict(self.original_state)
        self.optimizer = setup_tent(self.model, self.lr)

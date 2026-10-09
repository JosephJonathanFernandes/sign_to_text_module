import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

def setup_tent(model: nn.Module, lr: float = 1e-4) -> optim.Optimizer:
    """
    Setup Test-Time Entropy Minimization (TENT) (Wang et al., 2021).
    This function configures the model for test-time adaptation.
    It freezes all parameters EXCEPT the affine parameters of normalization layers.
    It also sets the normalization layers to train mode.
    
    Args:
        model: The PyTorch model to adapt.
        lr: Learning rate for the TENT optimizer.
        
    Returns:
        The optimizer for the adaptable parameters.
    """
    model.train() # Set to train mode so any BatchNorm stats update, though we mainly use LayerNorm
    
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
        
    adaptable_params = []
    
    # Enable gradients only for normalization affine parameters (weight & bias)
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            if module.weight is not None:
                module.weight.requires_grad = True
                adaptable_params.append(module.weight)
            if module.bias is not None:
                module.bias.requires_grad = True
                adaptable_params.append(module.bias)
                
    optimizer = optim.Adam(adaptable_params, lr=lr)
    return optimizer

def tent_adapt_step(model: nn.Module, optimizer: optim.Optimizer, x: torch.Tensor, proximity: torch.Tensor = None) -> torch.Tensor:
    """
    Perform one step of TENT adaptation on a batch of test data.
    
    Args:
        model: The PyTorch model.
        optimizer: The optimizer for the adaptable parameters.
        x: Input tensor (batch, seq_len, feat_dim).
        proximity: Optional proximity tensor.
        
    Returns:
        The logits from the forward pass.
    """
    # Forward pass
    logits_dict = model(x, proximity=proximity)
    logits = logits_dict['sign_logits'] if isinstance(logits_dict, dict) else logits_dict
    
    # Calculate entropy
    probs = F.softmax(logits, dim=1)
    entropy = -(probs * torch.log(probs + 1e-6)).sum(dim=1).mean()
    
    # Backward pass and optimize
    optimizer.zero_grad()
    entropy.backward()
    optimizer.step()
    
    return logits.detach()

class TentWrapper:
    """
    A stateful wrapper for a model that runs TENT online adaptation.
    """
    def __init__(self, model: nn.Module, lr: float = 1e-4, reset_state: bool = False):
        self.model = model
        self.lr = lr
        self.reset_state = reset_state
        
        # Save original state if we want to reset episodically
        if self.reset_state:
            self.original_state = {k: v.clone() for k, v in model.state_dict().items()}
            
        self.optimizer = setup_tent(self.model, self.lr)
        
    def adapt_and_predict(self, x: torch.Tensor, proximity: torch.Tensor = None) -> torch.Tensor:
        # In a real online streaming scenario, we adapt step-by-step
        with torch.enable_grad():
            logits = tent_adapt_step(self.model, self.optimizer, x, proximity)
        return logits
        
    def reset(self):
        if self.reset_state:
            self.model.load_state_dict(self.original_state)
            self.optimizer = setup_tent(self.model, self.lr)

"""
Lightweight adapter model for safe, user-specific live adaptation.

Instead of modifying the ensemble models, this adapter learns to correct
their outputs in real-time without corrupting the base models.

Architecture:
- Input: Ensemble probability vector (num_classes,)
- Dense(128) → ReLU → Dense(num_classes) → Softmax
- Trained only on high-confidence pseudo-labels
- Weights saved/restored for safety
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from datetime import datetime
from pathlib import Path


class AdapterModel(nn.Module):
    """Small MLP adapter for ensemble output correction."""
    
    def __init__(self, num_classes: int, hidden_dim: int = 128):
        """
        Args:
            num_classes: Number of output classes
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        
        # Small MLP: (num_classes,) -> (hidden_dim,) -> (num_classes,)
        self.fc1 = nn.Linear(num_classes, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.fc1.weight, a=0.01)
        nn.init.kaiming_uniform_(self.fc2.weight, a=0.01)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
    
    def forward(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Adapt ensemble probabilities.
        
        Args:
            probs: Ensemble probabilities (batch_size, num_classes)
        
        Returns:
            Adapted probabilities after softmax (batch_size, num_classes)
        """
        x = self.fc1(probs)
        x = self.relu(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1)
    
    def save_weights(self, path: str):
        """Save adapter weights to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)
    
    def load_weights(self, path: str):
        """Load adapter weights from disk."""
        if os.path.exists(path):
            self.load_state_dict(torch.load(path))
    
    def get_checkpoint(self) -> dict:
        """Get current weights as checkpoint."""
        return {
            'state_dict': self.state_dict(),
            'num_classes': self.num_classes,
            'hidden_dim': self.hidden_dim,
            'timestamp': datetime.now().isoformat(),
        }
    
    def restore_checkpoint(self, checkpoint: dict):
        """Restore weights from checkpoint."""
        self.load_state_dict(checkpoint['state_dict'])


class AdapterTrainer:
    """Trainer for the adapter model."""
    
    def __init__(
        self,
        num_classes: int,
        device: str = "cpu",
        learning_rate: float = 1e-4,
        hidden_dim: int = 128,
    ):
        """
        Args:
            num_classes: Number of output classes
            device: torch device
            learning_rate: Training learning rate
            hidden_dim: Hidden layer dimension
        """
        self.num_classes = num_classes
        self.device = device
        self.learning_rate = learning_rate
        self.hidden_dim = hidden_dim
        
        # Create model
        self.model = AdapterModel(num_classes, hidden_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.KLDivLoss(reduction='batchmean')
        
        # Training statistics
        self.training_history = []
    
    def create_dataset(
        self,
        ensemble_probs_list: list,
        class_indices_list: list,
        batch_size: int = 8,
        shuffle: bool = True,
    ):
        """
        Create training dataset from pseudo-labeled data.
        
        Args:
            ensemble_probs_list: List of ensemble probability vectors
            class_indices_list: List of target class indices
            batch_size: Batch size for training
            shuffle: Whether to shuffle data
        
        Returns:
            List of (probs_batch, targets_batch) tuples
        """
        if len(ensemble_probs_list) == 0:
            return []
        
        # Convert to tensors
        probs_tensor = torch.from_numpy(
            np.array(ensemble_probs_list, dtype=np.float32)
        ).to(self.device)
        
        targets_tensor = torch.zeros(
            len(class_indices_list), self.num_classes,
            dtype=torch.float32, device=self.device
        )
        for idx, class_idx in enumerate(class_indices_list):
            targets_tensor[idx, class_idx] = 1.0
        
        # Create batches
        dataset = []
        indices = np.arange(len(ensemble_probs_list))
        
        if shuffle:
            np.random.shuffle(indices)
        
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_probs = probs_tensor[batch_indices]
            batch_targets = targets_tensor[batch_indices]
            dataset.append((batch_probs, batch_targets))
        
        return dataset
    
    def train_epoch(self, dataset: list) -> float:
        """
        Train for one epoch.
        
        Args:
            dataset: List of (probs_batch, targets_batch) tuples
        
        Returns:
            Average loss
        """
        self.model.train()
        total_loss = 0.0
        
        for probs_batch, targets_batch in dataset:
            self.optimizer.zero_grad()
            
            # Forward pass
            adapted_probs = self.model(probs_batch)
            
            # KL divergence loss: how well adapted probs match targets
            # targets are one-hot encoded
            loss = self.criterion(
                F.log_softmax(adapted_probs, dim=1),
                targets_batch
            )
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataset) if dataset else 0.0
        return avg_loss
    
    def train(
        self,
        ensemble_probs_list: list,
        class_indices_list: list,
        epochs: int = 2,
        batch_size: int = 8,
        verbose: bool = True,
    ) -> dict:
        """
        Train adapter model.
        
        Args:
            ensemble_probs_list: List of ensemble probability vectors
            class_indices_list: List of target class indices
            epochs: Number of training epochs
            batch_size: Batch size
            verbose: Print training info
        
        Returns:
            Training history dict
        """
        if len(ensemble_probs_list) == 0:
            if verbose:
                print("[Adapter] No training data available")
            return {'success': False, 'reason': 'empty_dataset'}
        
        history = {
            'losses': [],
            'epochs': epochs,
            'batch_size': batch_size,
            'num_samples': len(ensemble_probs_list),
        }
        
        dataset = self.create_dataset(
            ensemble_probs_list,
            class_indices_list,
            batch_size=batch_size,
            shuffle=True,
        )
        
        if verbose:
            print(f"[Adapter] Training started: {len(ensemble_probs_list)} samples, {epochs} epochs")
        
        for epoch in range(epochs):
            loss = self.train_epoch(dataset)
            history['losses'].append(loss)
            
            if verbose and (epoch + 1) % max(1, epochs // 2) == 0:
                print(f"  Epoch {epoch+1}/{epochs}: loss = {loss:.4f}")
        
        self.training_history.append(history)
        return {'success': True, 'history': history}
    
    def evaluate_confidence(
        self,
        original_probs: np.ndarray,
    ) -> tuple:
        """
        Evaluate average confidence before/after adaptation.
        
        Args:
            original_probs: Ensemble probabilities (N, num_classes)
        
        Returns:
            (avg_conf_before, avg_conf_after)
        """
        self.model.eval()
        
        with torch.no_grad():
            probs_tensor = torch.from_numpy(
                original_probs.astype(np.float32)
            ).to(self.device)
            
            # Original confidence
            original_confidence = float(np.max(original_probs, axis=1).mean())
            
            # Adapted confidence
            adapted_probs = self.model(probs_tensor)
            adapted_confidence = float(
                adapted_probs.detach().cpu().numpy().max(axis=1).mean()
            )
        
        return original_confidence, adapted_confidence
    
    def save_model(self, path: str):
        """Save adapter model to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        if True:  # verbose
            print(f"[Adapter] Model saved to {path}")
    
    def load_model(self, path: str):
        """Load adapter model from disk."""
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            print(f"[Adapter] Model loaded from {path}")

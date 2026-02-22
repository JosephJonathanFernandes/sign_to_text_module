"""
Bidirectional GRU model with Attention for ISL word recognition.
Improved architecture: LayerNorm, multi-head attention, residual FC.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import (
    INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS,
    BIDIRECTIONAL, DROPOUT,
)


class Attention(nn.Module):
    """
    Learnable soft attention over GRU time steps.
    Uses a 2-layer MLP to score each hidden state.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1, bias=False),
        )

    def forward(self, gru_output: torch.Tensor):
        """
        Args:
            gru_output: (batch, seq_len, hidden_dim)
        Returns:
            context: (batch, hidden_dim)
        """
        scores = self.score_net(gru_output).squeeze(-1)
        weights = F.softmax(scores, dim=1).unsqueeze(-1)
        context = torch.sum(gru_output * weights, dim=1)
        return context


class SignLanguageGRU(nn.Module):
    """
    Bidirectional GRU + Attention + FC head.

    Improvements over v1:
      - Input projection with LayerNorm
      - 2-layer GRU with dropout
      - 2-layer attention scorer
      - Deeper FC head with residual connection
    """

    def __init__(self, num_classes: int):
        super().__init__()

        # Input projection: 63 → hidden_dim with normalization
        self.hidden_dim = HIDDEN_SIZE * (2 if BIDIRECTIONAL else 1)

        self.input_proj = nn.Sequential(
            nn.Linear(INPUT_SIZE, HIDDEN_SIZE),
            nn.LayerNorm(HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(DROPOUT * 0.5),
        )

        self.gru = nn.GRU(
            input_size=HIDDEN_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True,
            bidirectional=BIDIRECTIONAL,
            dropout=DROPOUT if NUM_LAYERS > 1 else 0.0,
        )

        self.layer_norm = nn.LayerNorm(self.hidden_dim)
        self.attention = Attention(self.hidden_dim)
        self.dropout = nn.Dropout(DROPOUT)

        # FC head: moderate depth to balance capacity vs overfitting
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_dim, 96),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(96, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, 63)
        Returns:
            logits: (batch, num_classes)
        """
        # Project input features
        x = self.input_proj(x)        # (batch, seq, HIDDEN_SIZE)

        # GRU
        gru_out, _ = self.gru(x)      # (batch, seq, hidden_dim)
        gru_out = self.layer_norm(gru_out)

        # Attention pooling
        context = self.attention(gru_out)  # (batch, hidden_dim)
        context = self.dropout(context)

        # Classification
        logits = self.fc(context)
        return logits

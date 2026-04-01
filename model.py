"""
Bidirectional GRU model with attention for ISL word recognition.
Includes optional face-proximity biased attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import (
    INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS,
    BIDIRECTIONAL, DROPOUT,
    FRAME_FEAT_DIM, PROXIMITY_FEAT_DIM, PROXIMITY_INDEX, USE_FACE_RELATIVE,
    USE_FACE_PROXIMITY_ATTENTION,
    PROXIMITY_SIGMA, LEARNABLE_PROXIMITY_SIGMA,
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
            alpha: (batch, seq_len)
        """
        scores = self.score_net(gru_output).squeeze(-1)
        alpha = F.softmax(scores, dim=1)
        context = torch.sum(gru_output * alpha.unsqueeze(-1), dim=1)
        return context, alpha


class FaceProximityAttention(nn.Module):
    """
    Attention with a multiplicative proximity prior.

    Frames with smaller face-relative hand distance receive larger bias.
    bias_t = exp(-proximity_t / sigma)
    """

    def __init__(
        self,
        hidden_dim: int,
        sigma_init: float = PROXIMITY_SIGMA,
        learnable_sigma: bool = LEARNABLE_PROXIMITY_SIGMA,
    ):
        super().__init__()
        self.score_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1, bias=False),
        )

        sigma_tensor = torch.tensor(float(sigma_init))
        if learnable_sigma:
            self.sigma = nn.Parameter(sigma_tensor)
        else:
            self.register_buffer("sigma", sigma_tensor)

    def forward(
        self,
        gru_output: torch.Tensor,
        proximity: torch.Tensor,
    ):
        """
        Args:
            gru_output: (batch, seq_len, hidden_dim)
            proximity:  (batch, seq_len) distances
        Returns:
            context: (batch, hidden_dim)
            alpha:   (batch, seq_len)
        """
        raw_scores = self.score_net(gru_output).squeeze(-1)
        sigma = torch.clamp(self.sigma, min=1e-4)
        bias = torch.exp(-proximity / sigma)
        scores = raw_scores * bias
        alpha = F.softmax(scores, dim=1)
        context = torch.sum(gru_output * alpha.unsqueeze(-1), dim=1)
        return context, alpha


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

        # Input projection: INPUT_SIZE -> hidden_dim with normalization
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
        self.use_face_proximity_attention = (
            USE_FACE_PROXIMITY_ATTENTION and USE_FACE_RELATIVE
        )
        if self.use_face_proximity_attention:
            self.attention = FaceProximityAttention(self.hidden_dim)
        else:
            self.attention = Attention(self.hidden_dim)
        self.dropout = nn.Dropout(DROPOUT)

        # FC head: moderate depth to balance capacity vs overfitting
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_dim, 96),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(96, num_classes),
        )

    @staticmethod
    def _frame_proximity_from_features(x: torch.Tensor) -> torch.Tensor:
        """
        Compute per-frame proximity from feature tensor.

        x can contain velocity-appended features; only the first FRAME_FEAT_DIM
        position block is used.
        """
        x_pos = x[:, :, :FRAME_FEAT_DIM]
        if PROXIMITY_FEAT_DIM > 0:
            return x_pos[:, :, PROXIMITY_INDEX]
        return torch.zeros(x_pos.shape[:2], device=x.device)

    def forward(
        self,
        x: torch.Tensor,
        proximity: torch.Tensor = None,
        return_attention: bool = False,
    ):
        """
        Args:
            x: (batch, seq_len, INPUT_SIZE)
            proximity: optional (batch, seq_len) frame-wise distances.
            return_attention: if True, also returns attention weights.
        Returns:
            logits: (batch, num_classes)
            optionally alpha: (batch, seq_len)
        """
        if self.use_face_proximity_attention:
            if proximity is None:
                proximity = self._frame_proximity_from_features(x)
            else:
                proximity = proximity.to(x.device)

        # Project input features
        x = self.input_proj(x)        # (batch, seq, HIDDEN_SIZE)

        # GRU
        gru_out, _ = self.gru(x)      # (batch, seq, hidden_dim)
        gru_out = self.layer_norm(gru_out)

        # Attention pooling
        if self.use_face_proximity_attention:
            context, alpha = self.attention(gru_out, proximity)
        else:
            context, alpha = self.attention(gru_out)
        context = self.dropout(context)

        # Classification
        logits = self.fc(context)
        if return_attention:
            return logits, alpha
        return logits

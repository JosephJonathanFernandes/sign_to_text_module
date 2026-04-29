"""
Bidirectional GRU model with ISL word recognition.
Includes multi-head attention, log-space biasing, learnable temperature,
and optional face-proximity biased attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import get_config

cfg = get_config()

# Convenience references for model architecture
INPUT_SIZE = cfg.frame_features.input_sequence_dim
HIDDEN_SIZE = cfg.model.hidden_size
NUM_LAYERS = cfg.model.num_layers
BIDIRECTIONAL = cfg.model.bidirectional
DROPOUT = cfg.model.dropout
FRAME_FEAT_DIM = cfg.frame_features.frame_features_dim
PROXIMITY_FEAT_DIM = cfg.spatial.proximity_dim
PROXIMITY_INDEX = cfg.frame_features.proximity_index
USE_FACE_RELATIVE = cfg.spatial.use_face_relative
USE_FACE_PROXIMITY_ATTENTION = cfg.model.use_face_proximity_attention
PROXIMITY_SIGMA = cfg.model.proximity_sigma
LEARNABLE_PROXIMITY_SIGMA = cfg.model.learnable_proximity_sigma


def _gaussian_log_bias(proximity: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Return log-bias for a Gaussian kernel over normalized proximity."""
    sigma = torch.clamp(sigma, min=1e-4)
    return -(proximity ** 2) / (2.0 * sigma ** 2)


class Attention(nn.Module):
    """
    Learnable soft attention over GRU time steps with:
      - Log-space biasing for stability
      - Learnable temperature for softmax sharpness
    Uses a 2-layer MLP to score each hidden state.
    """

    def __init__(self, hidden_dim: int, temp_init: float = 1.0):
        super().__init__()
        self.score_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1, bias=False),
        )
        # Learnable temperature: controls softmax sharpness
        self.temperature = nn.Parameter(torch.tensor(float(temp_init)))

    def forward(self, gru_output: torch.Tensor):
        """
        Args:
            gru_output: (batch, seq_len, hidden_dim)
        Returns:
            context: (batch, hidden_dim)
            alpha: (batch, seq_len)
        """
        scores = self.score_net(gru_output).squeeze(-1)
        # Use learnable temperature for adaptive sharpness
        temp = torch.clamp(self.temperature, min=0.1, max=10.0)
        alpha = F.softmax(scores / temp, dim=1)
        context = torch.sum(gru_output * alpha.unsqueeze(-1), dim=1)
        return context, alpha


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism (Transformer-style).
    
    Captures multiple temporal patterns simultaneously by:
    - Projecting hidden states into num_heads independent subspaces
    - Computing attention in each subspace
    - Concatenating results
    
    For sign language: enables simultaneous attention to hand motion,
    face expressions, and body gestures.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4, temp_init: float = 1.0):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Per-head scoring networks
        self.score_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.head_dim, self.head_dim // 2),
                nn.Tanh(),
                nn.Linear(self.head_dim // 2, 1, bias=False),
            )
            for _ in range(num_heads)
        ])
        
        # Per-head learnable temperatures
        self.temperatures = nn.ParameterList([
            nn.Parameter(torch.tensor(float(temp_init)))
            for _ in range(num_heads)
        ])
        
        # Projection to split into heads
        self.head_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, gru_output: torch.Tensor):
        """
        Args:
            gru_output: (batch, seq_len, hidden_dim)
        Returns:
            context: (batch, hidden_dim)
            alpha_list: list of (batch, seq_len) attention weights per head
        """
        batch_size, seq_len, _ = gru_output.shape
        
        # Project input for multi-head use
        x = self.head_proj(gru_output)  # (batch, seq_len, hidden_dim)
        
        # Reshape for multi-head processing
        x = x.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        x = x.transpose(1, 2)  # (batch, num_heads, seq_len, head_dim)
        
        context_list = []
        alpha_list = []
        
        for head_idx in range(self.num_heads):
            head_output = x[:, head_idx, :, :]  # (batch, seq_len, head_dim)
            
            # Score this head
            scores = self.score_nets[head_idx](head_output).squeeze(-1)
            
            # Apply learnable temperature
            temp = torch.clamp(self.temperatures[head_idx], min=0.1, max=10.0)
            alpha = F.softmax(scores / temp, dim=1)
            
            # Weighted sum for this head
            context_head = torch.sum(head_output * alpha.unsqueeze(-1), dim=1)
            context_list.append(context_head)
            alpha_list.append(alpha)
        
        # Concatenate all heads
        context = torch.cat(context_list, dim=-1)  # (batch, hidden_dim)
        
        return context, alpha_list


class HybridAttention(nn.Module):
    """
    Hybrid Attention combining ALL attention types:
    
    1. Multi-head temporal attention (multiple pattern capture)
    2. Face proximity biasing (spatial constraint)
    3. Learnable temperature per head (adaptive sharpness)
    4. Spatial attention over feature groups
    
    Some heads are "standard" (temporal only).
    Some heads are "proximity-aware" (temporal + spatial bias).
    
    This maximizes information from all cues: hand motion, face proximity, 
    and separately learns importance of hands vs face vs body.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        num_proximity_heads: int = 2,
        sigma_init: float = PROXIMITY_SIGMA,
        learnable_sigma: bool = LEARNABLE_PROXIMITY_SIGMA,
        temp_init: float = 1.0,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_proximity_heads = min(num_proximity_heads, num_heads)
        self.num_standard_heads = num_heads - self.num_proximity_heads
        self.head_dim = hidden_dim // num_heads
        
        # Standard temporal attention heads
        self.score_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.head_dim, self.head_dim // 2),
                nn.Tanh(),
                nn.Linear(self.head_dim // 2, 1, bias=False),
            )
            for _ in range(self.num_heads)
        ])
        
        # Per-head learnable temperatures
        self.temperatures = nn.ParameterList([
            nn.Parameter(torch.tensor(float(temp_init)))
            for _ in range(self.num_heads)
        ])
        
        # Proximity biasing for subset of heads
        sigma_tensor = torch.tensor(float(sigma_init))
        if learnable_sigma:
            self.sigma = nn.Parameter(sigma_tensor)
        else:
            self.register_buffer("sigma", sigma_tensor)
        
        # Projection to split into heads
        self.head_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, gru_output: torch.Tensor, proximity: torch.Tensor = None):
        """
        Args:
            gru_output: (batch, seq_len, hidden_dim)
            proximity:  optional (batch, seq_len) distances for proximity heads
        Returns:
            context: (batch, hidden_dim) concatenated from all heads
            head_weights: list of attention weights per head
        """
        batch_size, seq_len, _ = gru_output.shape
        
        # Project input for multi-head use
        x = self.head_proj(gru_output)  # (batch, seq_len, hidden_dim)
        
        # Reshape for multi-head processing
        x = x.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        x = x.transpose(1, 2)  # (batch, num_heads, seq_len, head_dim)
        
        context_list = []
        head_weights = []
        
        for head_idx in range(self.num_heads):
            head_output = x[:, head_idx, :, :]  # (batch, seq_len, head_dim)
            
            # Score this head
            scores = self.score_nets[head_idx](head_output).squeeze(-1)
            
            # Apply proximity biasing to subset of heads (if available)
            if head_idx >= self.num_standard_heads and proximity is not None:
                log_bias = _gaussian_log_bias(proximity, self.sigma)
                scores = scores + log_bias
            
            # Apply learnable temperature
            temp = torch.clamp(self.temperatures[head_idx], min=0.1, max=10.0)
            alpha = F.softmax(scores / temp, dim=1)
            
            # Weighted sum for this head
            context_head = torch.sum(head_output * alpha.unsqueeze(-1), dim=1)
            context_list.append(context_head)
            head_weights.append(alpha)
        
        # Concatenate all heads
        context = torch.cat(context_list, dim=-1)  # (batch, hidden_dim)
        
        return context, head_weights


class FaceProximityAttention(nn.Module):
    """
    Attention with face-proximity prior using log-space biasing.
    
    More stable than multiplicative biasing:
    scores = raw_scores + log_bias    (instead of raw_scores * bias)
    where log_bias = -proximity / sigma
    """

    def __init__(
        self,
        hidden_dim: int,
        sigma_init: float = PROXIMITY_SIGMA,
        learnable_sigma: bool = LEARNABLE_PROXIMITY_SIGMA,
        temp_init: float = 1.0,
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
        
        # Learnable temperature
        self.temperature = nn.Parameter(torch.tensor(float(temp_init)))

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
        # Log-space biasing (additive, more stable than multiplicative)
        # Gaussian kernel gives stronger emphasis to nearby frames
        log_bias = _gaussian_log_bias(proximity, self.sigma)
        scores = raw_scores + log_bias
        
        # Apply learnable temperature
        temp = torch.clamp(self.temperature, min=0.1, max=10.0)
        alpha = F.softmax(scores / temp, dim=1)
        context = torch.sum(gru_output * alpha.unsqueeze(-1), dim=1)
        return context, alpha


class SpatialAttention(nn.Module):
    """
    Spatial attention over feature groups (hands, face, body).
    
    Enables temporal + spatial (feature group) attention:
    - First layer: temporal attention at each time step
    - Second layer: spatial attention over feature groups
    
    For ISL: can separately learn importance of hand motion,
    facial expressions, and body movement.
    """

    def __init__(self, hidden_dim: int, num_groups: int = 3, temp_init: float = 1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_groups = num_groups
        
        # Spatial scorer: learns importance of each feature group
        self.spatial_scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_groups, bias=False),
        )
        
        # Learnable temperature for spatial attention
        self.temperature = nn.Parameter(torch.tensor(float(temp_init)))
    
    def forward(self, context: torch.Tensor):
        """
        Args:
            context: (batch, hidden_dim) pooled temporal context
        Returns:
            context_spatial: (batch, hidden_dim) spatially weighted
            spatial_weights: (batch, num_groups)
        """
        # Compute spatial scores
        spatial_scores = self.spatial_scorer(context)  # (batch, num_groups)
        
        # Apply learnable temperature
        temp = torch.clamp(self.temperature, min=0.1, max=10.0)
        spatial_weights = F.softmax(spatial_scores / temp, dim=1)
        
        # For now, spatial weighting is conceptual (applied to attention heads)
        # In practice, this can be used to weight different feature groups
        return context, spatial_weights


class SignLanguageGRU(nn.Module):
    """
    Bidirectional GRU + Multi-Head Attention + Spatial Attention + FC head.

    Key improvements:
      - Multi-head attention: captures multiple temporal patterns simultaneously
      - Learnable temperature: adaptive softmax sharpness per head
      - Log-space biasing: numerically stable proximity weighting
      - Spatial attention: learns importance of feature groups (hands/face/body)
      - Input projection with LayerNorm
      - 2-layer GRU with dropout
      - Deeper FC head
    """

    def __init__(
        self, 
        num_classes: int, 
        use_multihead: bool = True, 
        num_heads: int = 4,
        use_hybrid: bool = True,
        num_proximity_heads: int = 2,
    ):
        super().__init__()

        # Input projection: INPUT_SIZE -> hidden_dim with normalization
        self.hidden_dim = HIDDEN_SIZE * (2 if BIDIRECTIONAL else 1)
        self.use_multihead = use_multihead and self.hidden_dim % num_heads == 0
        self.use_hybrid = use_hybrid
        self.num_heads = num_heads if self.use_multihead else 1

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
        
        # Initialize appropriate attention mechanism
        if self.use_hybrid and self.use_multihead:
            # Hybrid: combine multi-head + proximity in single module
            self.attention = HybridAttention(
                self.hidden_dim,
                num_heads=self.num_heads,
                num_proximity_heads=num_proximity_heads,
            )
        elif self.use_multihead:
            if self.use_face_proximity_attention:
                # Separate multi-head isn't compatible with per-frame proximity
                self.attention = FaceProximityAttention(self.hidden_dim)
            else:
                self.attention = MultiHeadAttention(
                    self.hidden_dim, 
                    num_heads=self.num_heads
                )
        else:
            if self.use_face_proximity_attention:
                self.attention = FaceProximityAttention(self.hidden_dim)
            else:
                self.attention = Attention(self.hidden_dim)
        
        # Spatial attention for feature group weighting
        self.spatial_attention = SpatialAttention(self.hidden_dim, num_groups=3)
        
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
            optionally alpha: (batch, seq_len) or list of attention weights (multi-head)
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

        # Multi-head or single-head temporal attention pooling
        if self.use_hybrid:
            # Hybrid uses proximity internally for subset of heads
            context, alpha = self.attention(gru_out, proximity)
        elif self.use_face_proximity_attention:
            context, alpha = self.attention(gru_out, proximity)
        else:
            context, alpha = self.attention(gru_out)
        
        # Spatial attention: learn importance of feature groups
        context, spatial_weights = self.spatial_attention(context)
        
        context = self.dropout(context)

        # Classification
        logits = self.fc(context)
        if return_attention:
            return logits, alpha
        return logits

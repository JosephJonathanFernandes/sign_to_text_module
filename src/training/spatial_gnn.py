"""
Lightweight Spatial GNN for MediaPipe hand skeleton topology.

Design:
- Parallel branch to existing Conv1D frontend
- Explicit modeling of finger-joint connectivity
- 21 landmark nodes per hand with anatomical adjacency
- Weight-tying across left/right hands
- Minimal parameter count (<2K extra parameters)

Tensor Flow:
Input landmarks: (B, 20, 126)  [position-only, left+right raw coordinates]
→ Reshape: (B, 20, 2, 21, 3)  [batch, time, hands, nodes, coords]
→ GCN Layer 1: 3→16
→ GCN Layer 2: 16→8
→ Max pool across nodes: (B, 20, 2, 8)
→ Max pool across hands: (B, 20, 16)
→ Output: (B, 20, 16) concatenated with conv frontend features

Adjacency: Static MediaPipe hand skeleton (21×21, binary, undirected)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.core.config import get_config

cfg = get_config()


# ========================================================================
# MediaPipe Hand Skeleton Adjacency (21 landmarks)
# ========================================================================
# Landmark indices:
#   0:  WRIST
#   1:  THUMB_CMC,    2: THUMB_MCP,    3: THUMB_IP,     4: THUMB_TIP
#   5:  INDEX_FINGER_MCP,  6: INDEX_FINGER_PIP,  7: INDEX_FINGER_DIP,  8: INDEX_FINGER_TIP
#   9:  MIDDLE_FINGER_MCP, 10: MIDDLE_FINGER_PIP, 11: MIDDLE_FINGER_DIP, 12: MIDDLE_FINGER_TIP
#   13: RING_FINGER_MCP,   14: RING_FINGER_PIP,   15: RING_FINGER_DIP,   16: RING_FINGER_TIP
#   17: PINKY_MCP,         18: PINKY_PIP,         19: PINKY_DIP,         20: PINKY_TIP

# Anatomical connections: (i, j) pairs for undirected graph
HAND_SKELETON_EDGES = [
    # Wrist to finger bases
    (0, 1),   # Wrist → Thumb CMC
    (0, 5),   # Wrist → Index MCP
    (0, 9),   # Wrist → Middle MCP
    (0, 13),  # Wrist → Ring MCP
    (0, 17),  # Wrist → Pinky MCP
    # Thumb chain
    (1, 2),   # Thumb CMC → Thumb MCP
    (2, 3),   # Thumb MCP → Thumb IP
    (3, 4),   # Thumb IP → Thumb TIP
    # Index finger chain
    (5, 6),   # Index MCP → Index PIP
    (6, 7),   # Index PIP → Index DIP
    (7, 8),   # Index DIP → Index TIP
    # Middle finger chain
    (9, 10),  # Middle MCP → Middle PIP
    (10, 11), # Middle PIP → Middle DIP
    (11, 12), # Middle DIP → Middle TIP
    # Ring finger chain
    (13, 14), # Ring MCP → Ring PIP
    (14, 15), # Ring PIP → Ring DIP
    (15, 16), # Ring DIP → Ring TIP
    # Pinky chain
    (17, 18), # Pinky MCP → Pinky PIP
    (18, 19), # Pinky PIP → Pinky DIP
    (19, 20), # Pinky DIP → Pinky TIP
    # Cross-finger connections (knuckle row)
    (1, 5),   # Thumb CMC → Index MCP
    (5, 9),   # Index MCP → Middle MCP
    (9, 13),  # Middle MCP → Ring MCP
    (13, 17), # Ring MCP → Pinky MCP
]

NUM_LANDMARKS = 21


def _build_hand_adjacency() -> np.ndarray:
    """Build binary adjacency matrix (21×21) from anatomical edges.
    
    Returns:
        np.ndarray (21, 21) — undirected, unweighted adjacency
    """
    adj = np.zeros((NUM_LANDMARKS, NUM_LANDMARKS), dtype=np.float32)
    for i, j in HAND_SKELETON_EDGES:
        adj[i, j] = 1.0
        adj[j, i] = 1.0  # undirected
    # Self-loops for stability
    np.fill_diagonal(adj, 1.0)
    return adj


def _normalize_adjacency(adj: np.ndarray) -> torch.Tensor:
    """Symmetrically normalize adjacency: D^{-1/2} A D^{-1/2}
    
    Args:
        adj: (N, N) binary adjacency matrix
    
    Returns:
        torch.Tensor (N, N) normalized
    """
    adj = adj.copy()
    deg = np.sum(adj, axis=1)  # (N,)
    deg_inv_sqrt = np.power(deg, -0.5)
    deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.0
    deg_mat_inv_sqrt = np.diag(deg_inv_sqrt)
    adj_norm = deg_mat_inv_sqrt @ adj @ deg_mat_inv_sqrt
    return torch.from_numpy(adj_norm).float()


# Pre-compute adjacency matrix (module-level singleton)
_HAND_ADJACENCY: torch.Tensor = None


def get_hand_adjacency(device: torch.device = None) -> torch.Tensor:
    """Get globally cached, normalized hand adjacency matrix.
    
    Args:
        device: Target device (optional)
    
    Returns:
        torch.Tensor (21, 21) normalized adjacency
    """
    global _HAND_ADJACENCY
    if _HAND_ADJACENCY is None:
        adj_raw = _build_hand_adjacency()
        _HAND_ADJACENCY = _normalize_adjacency(adj_raw)
    if device is not None:
        return _HAND_ADJACENCY.to(device)
    return _HAND_ADJACENCY


# ========================================================================
# Graph Convolution Layer
# ========================================================================

class GraphConvLayer(nn.Module):
    """
    Simple Graph Convolution: X' = σ(A·X·W + b)
    
    Where:
    - A: normalized adjacency (N, N) — static, not learned
    - X: node features (B, N, F_in)
    - W: learned weight (F_in, F_out)
    - b: learned bias (F_out,)
    
    Includes residual connection if F_in == F_out.
    """
    
    def __init__(self, in_features: int, out_features: int, use_residual: bool = True, dropout: float = 0.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_residual = use_residual and (in_features == out_features)
        self.dropout = dropout
        
        self.linear = nn.Linear(in_features, out_features, bias=True)
        self.norm = nn.LayerNorm(out_features)
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, F_in) node features
            adj: (N, N) normalized adjacency matrix
        
        Returns:
            (B, N, F_out) processed node features
        """
        # Graph convolution: A @ X @ W
        # A @ X: message passing between neighbors
        # (B, N, F_in) -> (B, N, F_in) via adjacency propagation
        x_aggr = adj @ x  # (B, N, F_in)
        
        # Linear projection
        out = self.linear(x_aggr)  # (B, N, F_out)
        
        # Normalize and activate
        out = self.norm(out)
        out = F.relu(out)
        out = self.dropout_layer(out)
        
        # Residual connection (if dimensions match)
        if self.use_residual:
            out = out + x[..., :self.out_features] if x.shape[-1] == self.out_features else out
        
        return out


# ========================================================================
# Lightweight Spatial GNN (complete module)
# ========================================================================

class LightweightSpatialGNN(nn.Module):
    """
    Lightweight GNN for hand landmark spatial topology.
    
    Architecture:
    1. Extract raw landmark position features from input tensor
    2. Reshape to (B, T, 2, 21, 3) — hands × landmarks × coordinates
    3. Apply 1-2 GraphConvLayers with shared weights across hands
    4. Max-pool over landmarks → per-hand embedding
    5. Max-pool over hands → per-frame embedding
    6. Output: (B, T, output_dim)
    
    Config-driven: parameters from config.arch_improvements.gnn_*
    """
    
    def __init__(self):
        super().__init__()
        
        # Get config
        self.hidden_dim = cfg.arch_improvements.gnn_hidden_dim
        self.num_layers = cfg.arch_improvements.gnn_num_layers
        self.output_dim = cfg.arch_improvements.gnn_output_dim
        self.dropout = cfg.arch_improvements.gnn_dropout
        self.use_residual = cfg.arch_improvements.gnn_use_residual
        self.shared_weights = cfg.arch_improvements.gnn_shared_weights
        self.use_gnn = cfg.arch_improvements.use_gnn
        
        # Feature dimensions
        self.num_landmarks = 21
        self.num_coords = 3
        self.num_hands = 2
        
        # Input coordinate dimension (x, y, z)
        coord_in_features = self.num_coords  # 3
        
        # GNN layers
        if self.use_gnn:
            layers = []
            in_dim = coord_in_features
            
            for i in range(self.num_layers):
                out_dim = self.hidden_dim if i < self.num_layers - 1 else self.output_dim
                use_res = self.use_residual and (in_dim == out_dim)
                layers.append(GraphConvLayer(
                    in_features=in_dim,
                    out_features=out_dim,
                    use_residual=use_res,
                    dropout=self.dropout if i < self.num_layers - 1 else 0.0,
                ))
                in_dim = out_dim
            
            self.gcn_layers = nn.ModuleList(layers)
            
            # Final projection to combine pooled features from both hands
            # Per-hand output: (output_dim,) — after max pool over 21 landmarks
            # Both hands: (2 * output_dim,)
            # We project to per-frame output
            if self.shared_weights:
                # Both hands use same GNN → same output dim
                self.final_proj = nn.Linear(self.output_dim * self.num_hands, self.output_dim * 2)
            else:
                self.final_proj = nn.Linear(self.output_dim * self.num_hands, self.output_dim * 2)
        else:
            self.gcn_layers = nn.ModuleList()
            self.final_proj = None
    
    @staticmethod
    def _extract_landmarks(x: torch.Tensor) -> torch.Tensor:
        """
        Extract raw landmark position features from input tensor.
        
        Input x: (B, T, 504) — full feature vector [left_raw, right_raw, left_rel, right_rel, prox, velocity...]
        Position block: [left_raw(63), right_raw(63)] = 126 dims at indices 0:126
        
        Returns:
            (B, T, 2, 21, 3) — separated per hand, per landmark coordinates
        """
        batch_size, seq_len, _ = x.shape
        
        # Extract raw landmark position: first 126 dims = left_raw(63) + right_raw(63)
        # left_raw: indices 0..62, right_raw: indices 63..125
        left_raw = x[:, :, :63]   # (B, T, 63) left hand
        right_raw = x[:, :, 63:126]  # (B, T, 63) right hand
        
        # Reshape to per-landmark format
        left_landmarks = left_raw.reshape(batch_size, seq_len, 21, 3)  # (B, T, 21, 3)
        right_landmarks = right_raw.reshape(batch_size, seq_len, 21, 3)
        
        # Stack hands: (B, T, 2, 21, 3)
        landmarks = torch.stack([left_landmarks, right_landmarks], dim=2)
        
        return landmarks
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, 504) input features (full pipeline features including position blocks)
        
        Returns:
            gnn_features: (B, T, gnn_output_dim*2) spatial features
                          OR zeros if use_gnn=False
        """
        if not self.use_gnn or self.gcn_layers is None or len(self.gcn_layers) == 0:
            # Return zero features when GNN is disabled
            batch_size, seq_len = x.shape[0], x.shape[1]
            output_dim = getattr(cfg.arch_improvements, 'gnn_output_dim', 8) * 2
            return torch.zeros(batch_size, seq_len, output_dim, device=x.device, dtype=x.dtype)
        
        device = x.device
        batch_size, seq_len = x.shape[0], x.shape[1]
        
        # Extract landmarks: (B, T, 2, 21, 3)
        landmarks = self._extract_landmarks(x)
        
        # Get adjacency matrix
        adj = get_hand_adjacency(device)  # (21, 21)
        
        # Process each hand through GNN
        # Reshape to merge batch and time: (B*T, H, N, F) -> process -> unmerge
        # H=2 (hands), N=21 (landmarks), F=3 (coords)
        
        # Merge batch and time: (B*T, 2, 21, 3)
        landmarks_flat = landmarks.reshape(batch_size * seq_len, self.num_hands, self.num_landmarks, self.num_coords)
        
        hand_embeddings = []
        for h in range(self.num_hands):
            # Extract this hand: (B*T, 21, 3)
            hand_nodes = landmarks_flat[:, h, :, :]  # (B*T, N, F)
            
            # Through GNN layers
            gnn_out = hand_nodes
            for gcn_layer in self.gcn_layers:
                gnn_out = gcn_layer(gnn_out, adj)  # (B*T, N, out_dim)
            
            # Max pool over landmarks (nodes) -> per-hand embedding
            # (B*T, out_dim)
            hand_embedding = gnn_out.max(dim=1)[0]
            hand_embeddings.append(hand_embedding)
        
        # Concatenate both hands: (B*T, 2 * out_dim)
        combined = torch.cat(hand_embeddings, dim=-1)
        
        # Final projection: (B*T, 2*out_dim) -> (B*T, 2*out_dim)
        if self.final_proj is not None:
            combined = self.final_proj(combined)
            combined = F.relu(combined)
        
        # Reshape back: (B, T, 2*out_dim)
        gnn_output = combined.reshape(batch_size, seq_len, -1)
        
        return gnn_output
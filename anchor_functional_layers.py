import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AnchorFunctionalLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_anchors: int = 64, key_dim: int = 64, dropout: float = 0.1,
                 mix_anchors: bool = True):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_anchors = int(num_anchors)
        self.key_dim = int(key_dim)
        self.mix_anchors = bool(mix_anchors)

        # node -> key
        self.q_proj = nn.Linear(self.hidden_dim, self.key_dim, bias=False)
        # anchor keys
        self.anchor_keys = nn.Parameter(torch.randn(self.num_anchors, self.key_dim) * 0.02)

        if self.mix_anchors:
            self.anchor_mlp = nn.Sequential(
                nn.LayerNorm(self.hidden_dim),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.GELU(),
            )
        else:
            self.anchor_mlp = None

        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, A_unused=None) -> torch.Tensor:
        """
        x: [B, N, D]
        returns message: [B, N, D]
        """
        B, N, D = x.shape
        q = self.q_proj(x)  # [B,N,dk]
        # logits: [B,N,M]
        logits = torch.matmul(q, self.anchor_keys.t()) / math.sqrt(self.key_dim)
        A = torch.softmax(logits, dim=-1)

        # anchor values: [B,M,D] = A^T X
        anchor_vals = torch.matmul(A.transpose(1, 2), x)

        if self.anchor_mlp is not None:
            anchor_vals = self.anchor_mlp(anchor_vals)

        # back to nodes: [B,N,D] = A anchor_vals
        out = torch.matmul(A, anchor_vals)

        out = self.out_proj(out)
        out = self.norm(out)
        out = F.gelu(out)
        out = self.dropout(out)
        return out

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import math
import numpy as np


def sparse_row_sum(A: torch.Tensor) -> torch.Tensor:
    """Return row-wise sum (degree) for sparse COO/CSR, CPU/CUDA."""
    if not A.is_sparse:
        return A.sum(dim=1)

    if A.layout == torch.sparse_csr:
        crow = A.crow_indices()
        vals = A.values()
        n = A.size(0)
        counts = crow[1:] - crow[:-1]
        row_ids = torch.repeat_interleave(torch.arange(n, device=vals.device), counts)
        out = torch.zeros(n, device=vals.device, dtype=vals.dtype)
        out.scatter_add_(0, row_ids, vals)
        return out

    return torch.sparse.sum(A, dim=1).to_dense()


def gate_sparse_adj_by_zone(A: torch.Tensor, zone_probs_flat: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Gate sparse adjacency by soft zone affinity (batch-mean), then row-normalize."""
    if (A is None) or (not A.is_sparse):
        return A

    if A.layout == torch.sparse_csr:
        A = A.to_sparse_coo()

    A = A.coalesce()
    idx = A.indices()
    row = idx[0]
    col = idx[1]
    val = A.values()

    Z = zone_probs_flat.mean(dim=0)
    affinity = (Z.index_select(0, row) * Z.index_select(0, col)).sum(dim=-1).clamp(min=0.0)
    new_val = val * affinity

    N = A.size(0)
    deg = torch.zeros(N, device=new_val.device, dtype=new_val.dtype)
    deg.scatter_add_(0, row, new_val)
    new_val = new_val / (deg.index_select(0, row) + eps)

    return torch.sparse_coo_tensor(idx, new_val, A.size(), device=new_val.device, dtype=new_val.dtype).coalesce()


# =============================================================
# NEW: 2D Positional Encoding for Spatial Awareness
# =============================================================

class PositionalEncoding2D(nn.Module):
    """
    2D sinusoidal positional encoding for spatial grids.
    
    Gives each cell in the grid a unique position-dependent feature vector.
    This allows the model to generate spatially varying patterns.
    
    Based on "Attention is All You Need" (Vaswani et al., 2017) adapted for 2D.
    """
    
    def __init__(self, d_model, H=100, W=100, temperature=10000):
        super().__init__()
        self.d_model = d_model
        self.H = H
        self.W = W
        
        # Create coordinate grids
        y_pos = torch.arange(H, dtype=torch.float32).unsqueeze(1).repeat(1, W).flatten()  # [10000]
        x_pos = torch.arange(W, dtype=torch.float32).unsqueeze(0).repeat(H, 1).flatten()  # [10000]
        
        # Normalize to [-1, 1] range
        y_pos = (y_pos / max(H - 1, 1)) * 2 - 1
        x_pos = (x_pos / max(W - 1, 1)) * 2 - 1
        
        # Sinusoidal encoding with different frequencies
        pe = torch.zeros(H * W, d_model)
        
        # Create frequency bands
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * 
                             -(math.log(temperature) / d_model))

        for i in range(0, d_model // 4):
            pe[:, 4*i] = torch.sin(x_pos * div_term[i])
            pe[:, 4*i + 1] = torch.cos(x_pos * div_term[i])
            pe[:, 4*i + 2] = torch.sin(y_pos * div_term[i])
            pe[:, 4*i + 3] = torch.cos(y_pos * div_term[i])
        
        # Handle remaining dimensions if d_model not divisible by 4
        remainder = d_model % 4
        if remainder > 0:
            offset = d_model - remainder
            if remainder >= 1:
                pe[:, offset] = torch.sin(x_pos * div_term[-1])
            if remainder >= 2:
                pe[:, offset + 1] = torch.cos(x_pos * div_term[-1])
            if remainder >= 3:
                pe[:, offset + 2] = torch.sin(y_pos * div_term[-1])
        
        self.register_buffer('pe', pe)  # [H*W, d_model]
    
    def forward(self, x):
        """
        Add positional encoding to input.
        
        Args:
            x: [B, N, D] where N = H*W
        Returns:
            [B, N, D] with positional encoding added
        """
        return x + self.pe.unsqueeze(0)


# =============================================================
# Efficient Sparse GCN Layer
# =============================================================

class EfficientSparseGCN(nn.Module):
    """Memory-efficient GCN layer."""
    
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, A):
        """
        x: [B, N, D]
        A: sparse [N, N]
        """
        B, N, D = x.shape

        x = x.contiguous()

        with torch.amp.autocast('cuda', enabled=False):
            x_float = x.float()
            x2 = x_float.transpose(0, 1).reshape(N, B * D).contiguous()
            y2 = torch.sparse.mm(A, x2)
            out = y2.view(N, B, D).transpose(0, 1).contiguous()

        out = out.to(x.dtype)

        out = self.linear(out)
        out = self.norm(out)
        out = F.gelu(out)
        out = self.dropout(out)

        return out


class EfficientCoarseGenerator(nn.Module):
    def __init__(self, A_norm, cond_dim=20, noise_dim=16, hidden_dim=128, 
                 num_layers=3, num_zones=12, use_pos_encoding=True):
        super().__init__()

        self.A_norm = A_norm
        self.N = A_norm.size(0)
        self.use_pos_encoding = use_pos_encoding

        # NEW: Positional encoding module
        if use_pos_encoding:
            self.pos_encoding = PositionalEncoding2D(hidden_dim, H=100, W=100)
            print(f"  Coarse generator: Positional encoding ENABLED (fixes spatial bias)")
        else:
            self.pos_encoding = None
            print(f"  Coarse generator: Positional encoding DISABLED (may cause spatial bias)")

        self.input_proj = nn.Linear(cond_dim + noise_dim, hidden_dim)

        self.gcn_layers = nn.ModuleList([
            EfficientSparseGCN(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])

        self.out_proj_int = nn.Linear(hidden_dim, 1)
        self.out_proj_road = nn.Linear(hidden_dim, 1)
        self.num_zones = int(num_zones)
        self.out_proj_zone = nn.Linear(hidden_dim, self.num_zones)

        self.noise_dim = noise_dim

    def forward(self, cond_vec, noise=None):
        B = cond_vec.size(0)

        if noise is None:
            noise = torch.randn(B, self.noise_dim, device=cond_vec.device, dtype=cond_vec.dtype)
        else:
            if noise.dim() != 2 or noise.size(0) != B or noise.size(1) != self.noise_dim:
                raise ValueError(f"noise must have shape [B, {self.noise_dim}], got {tuple(noise.shape)}")
            noise = noise.to(device=cond_vec.device, dtype=cond_vec.dtype)

        z = torch.cat([noise, cond_vec], dim=-1)

        # Project global features
        h = self.input_proj(z).unsqueeze(1).expand(B, self.N, -1)  # [B, 10000, D]

        if self.use_pos_encoding and self.pos_encoding is not None:
            h = self.pos_encoding(h)  # [B, 10000, D]

        # GCN layers propagate spatially-aware features
        for gcn in self.gcn_layers:
            h = h + gcn(h, self.A_norm)

        # Output heads
        out_int = torch.sigmoid(self.out_proj_int(h))      # [B,N,1]
        out_road = torch.sigmoid(self.out_proj_road(h))    # [B,N,1]
        zone_logits = self.out_proj_zone(h)                # [B,N,K]

        # Reshape to 2D
        coarse_int = out_int.permute(0, 2, 1).contiguous().view(B, 1, 100, 100)
        coarse_road = out_road.permute(0, 2, 1).contiguous().view(B, 1, 100, 100)
        zone_logits = zone_logits.permute(0, 2, 1).contiguous().view(B, self.num_zones, 100, 100)

        return coarse_int, coarse_road, zone_logits


# =============================================================
# Cross-Attention and Mixture
# =============================================================

class EfficientCrossAttention(nn.Module):
    """Efficient cross-attention for dual-stream bridge."""
    
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, query, context):
        B, N, D = query.shape
        
        q = self.q_proj(query).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(context).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(context).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v).transpose(1, 2).contiguous().view(B, N, D)
        out = self.out_proj(out)
        out = self.norm(out)
        
        return out


class EfficientMixtureGate(nn.Module):
    """Learnable mixture gate for combining streams."""
    
    def __init__(self, hidden_dim, num_streams=2):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * num_streams, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_streams),
        )
        
    def forward(self, streams):
        B, N, D = streams[0].shape
        combined = torch.cat(streams, dim=-1)
        
        gates = F.softmax(self.gate(combined), dim=-1)
        
        mixed = sum(g.unsqueeze(-1) * s for g, s in zip(gates.unbind(-1), streams))
        
        return mixed, gates


# =============================================================
# Fine Generator
# =============================================================

class EfficientDualStreamFineGenerator(nn.Module):
    """Dual-stream fine generator (unchanged)."""
    
    def __init__(self, hidden_dim=192, cond_dim=20, poi_dim=21, num_zones=12,
                 num_layers=3, num_bridge_layers=2, heads=4, dropout=0.1,
                 zone_dim=64, temperature=0.7, use_checkpoint=False):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.poi_dim = poi_dim
        self.use_checkpoint = use_checkpoint
        
        self.coarse_proj = nn.Linear(2, hidden_dim)
        
        self.spatial_stream = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
        )
        
        self.func_layers = nn.ModuleList([
            EfficientSparseGCN(hidden_dim, hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        
        self.bridges = nn.ModuleList([
            EfficientCrossAttention(hidden_dim, heads, dropout)
            for _ in range(num_bridge_layers)
        ])
        
        self.bridge_gates = nn.ParameterList([
            nn.Parameter(torch.tensor([0.5, 0.5]))
            for _ in range(num_bridge_layers)
        ])
        
        self.mixture = EfficientMixtureGate(hidden_dim, 2)
        
        self.zone_table = nn.Parameter(torch.randn(num_zones, zone_dim) * 0.02)
        self.zone_to_hidden = nn.Linear(zone_dim, hidden_dim)
        
        self.cond_proj = nn.Linear(cond_dim, hidden_dim)
        
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, poi_dim),
        )
        self.temperature = temperature
        
    def forward(self, coarse_map, A_functional, zone_logits, cond_vec, zone_gate: bool = False, return_intermediates: bool = False):
        B, _, H, W = coarse_map.shape
        N = H * W
        device = coarse_map.device
        
        coarse_flat = coarse_map.permute(0,2,3,1).contiguous().view(B, N, 2)
        h = self.coarse_proj(coarse_flat)

        Bz, K, Hz, Wz = zone_logits.shape
        if Bz != B or Hz != H or Wz != W:
            raise ValueError(f"zone_logits must be [B,K,H,W] aligned with coarse_map")
        zone_probs = F.softmax(zone_logits, dim=1)
        zone_probs_flat = zone_probs.permute(0,2,3,1).contiguous().view(B, N, K)
        if zone_gate:
            A_functional = gate_sparse_adj_by_zone(A_functional, zone_probs_flat)
        
        h_spatial_2d = h.permute(0, 2, 1).contiguous().view(B, self.hidden_dim, H, W)
        h_spatial_2d = self.spatial_stream(h_spatial_2d)
        h_spatial = h_spatial_2d.view(B, self.hidden_dim, N).permute(0, 2, 1).contiguous()
        
        h_func = h
        for func_layer in self.func_layers:
            if self.use_checkpoint and self.training:
                h_func = h_func + checkpoint(func_layer, h_func, A_functional, use_reentrant=False)
            else:
                h_func = h_func + func_layer(h_func, A_functional)
        
        for i, (bridge, gate) in enumerate(zip(self.bridges, self.bridge_gates)):
            g = torch.softmax(gate, dim=0)
            
            if self.use_checkpoint and self.training:
                h_s2f = checkpoint(bridge, h_spatial, h_func, use_reentrant=False)
                h_f2s = checkpoint(bridge, h_func, h_spatial, use_reentrant=False)
            else:
                h_s2f = bridge(h_spatial, h_func)
                h_f2s = bridge(h_func, h_spatial)
            
            h_spatial = g[0] * h_spatial + g[1] * h_s2f
            h_func = g[0] * h_func + g[1] * h_f2s
        
        h_mixed, gates = self.mixture([h_spatial, h_func])
        
        if (self.zone_table is None) or (self.zone_table.shape[0] != K):
            self.zone_table = nn.Parameter(torch.randn(K, self.zone_table.shape[1], device=device, dtype=h_mixed.dtype) * 0.02)
        zone_feat = torch.matmul(zone_probs_flat, self.zone_table)
        h_mixed = h_mixed + self.zone_to_hidden(zone_feat)
        
        cond_hidden = self.cond_proj(cond_vec)
        h_mixed = h_mixed + cond_hidden.unsqueeze(1)
        
        logits = self.output_head(h_mixed)
        out_flat = F.softmax(logits / self.temperature, dim=-1)
        out = out_flat.view(B, H, W, self.poi_dim).permute(0, 3, 1, 2).contiguous()
        
        if return_intermediates:
            bridge_gate_softmax = [torch.softmax(g, dim=0).detach() for g in self.bridge_gates]
            info = {
                "mixture_gates": gates.detach(),
                "bridge_gate_softmax": bridge_gate_softmax,
                "h_spatial_norm": h_spatial.detach().norm(dim=-1).mean().item(),
                "h_func_norm": h_func.detach().norm(dim=-1).mean().item(),
            }
            return out, info

        return out


# =============================================================
# Discriminator
# =============================================================

class EfficientDiscriminator(nn.Module):
    """Lightweight discriminator using only spatial convolutions."""
    
    def __init__(self, in_channels=20, cond_dim=20, hidden_dim=128, num_layers=4):
        super().__init__()
        
        self.conv_in = nn.Conv2d(in_channels + cond_dim, hidden_dim, 3, padding=1)
        
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 4, stride=2, padding=1),
                nn.GroupNorm(8, hidden_dim),
                nn.LeakyReLU(0.2),
            ))
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, 1),
        )
        
    def forward(self, x, cond_vec):
        B, C, H, W = x.shape
        
        cond_map = cond_vec.unsqueeze(-1).unsqueeze(-1).expand(B, -1, H, W)
        h = torch.cat([x, cond_map], dim=1)
        
        h = self.conv_in(h)
        for layer in self.layers:
            h = layer(h)
        
        return self.classifier(h)

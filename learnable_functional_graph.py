import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LearnableFunctionalGraph(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        emb_dim: int = 32,
        k: int = 16,
        rebuild_every: int = 200,
        chunk_size: int = 256,
        temperature: float = 0.07,
        symmetric: bool = True,
        add_self_loop: bool = True,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.emb_dim = int(emb_dim)
        self.k = int(k)
        self.rebuild_every = int(rebuild_every)
        self.chunk_size = int(chunk_size)
        self.temperature = float(temperature)
        self.symmetric = bool(symmetric)
        self.add_self_loop = bool(add_self_loop)
        self.eps = float(eps)

        # Learnable node embeddings
        self.node_emb = nn.Parameter(torch.randn(self.num_nodes, self.emb_dim) * 0.02)

        # Cached neighbors (indices only). Stored as a buffer to be checkpointed.
        self.register_buffer("knn_idx", torch.empty(self.num_nodes, self.k, dtype=torch.long), persistent=True)
        self.register_buffer("last_rebuild_step", torch.tensor(-1, dtype=torch.long), persistent=True)

    @torch.no_grad()
    def _rebuild_knn(self, device: torch.device):
        """
        Exact kNN over all nodes using chunked matmul + topk.
        This is O(N^2) compute but O(chunk_size*N) memory and is feasible for N=10k if done periodically.
        """
        E = F.normalize(self.node_emb.detach().to(device), dim=1, eps=self.eps)  # [N, d]
        N = self.num_nodes
        k = self.k

        all_idx = torch.empty((N, k), device=device, dtype=torch.long)

        # Precompute transpose for matmul
        Et = E.t().contiguous()  # [d, N]

        for start in range(0, N, self.chunk_size):
            end = min(start + self.chunk_size, N)
            chunk = E[start:end]              # [B, d]
            sim = chunk @ Et                  # [B, N] cosine similarity

            # Exclude self from neighbors
            row_ids = torch.arange(start, end, device=device)
            sim[torch.arange(end-start, device=device), row_ids] = -1e9

            _, idx = torch.topk(sim, k=k, dim=1, largest=True, sorted=False)
            all_idx[start:end] = idx

        # Store on module buffer (keep on current device)
        self.knn_idx = all_idx

    def maybe_rebuild(self, step: int, device: torch.device):
        step = int(step)
        last = int(self.last_rebuild_step.item())
        
        # Always build on first call (last=-1)
        if last < 0:
            self._rebuild_knn(device=device)
            self.last_rebuild_step.fill_(step)
            return
        
        # If rebuild_every <= 0, never rebuild after first time
        if self.rebuild_every <= 0:
            return
        
        # Otherwise, rebuild periodically
        if (step - last) >= self.rebuild_every:
            self._rebuild_knn(device=device)
            self.last_rebuild_step.fill_(step)

    def build_sparse_adjacency(self, step: int, detach_weights: bool, device: torch.device):
        """
        Build sparse adjacency A in COO format on `device`.

        - Neighbor indices are cached (rebuild periodically).
        - Edge weights are attention softmax over k neighbors. If detach_weights=True, we detach
          the weights (useful for discriminator steps) to save memory.
        """
        self.maybe_rebuild(step, device=device)

        # Normalize embeddings (keep grad unless detach_weights)
        E = F.normalize(self.node_emb.to(device), dim=1, eps=self.eps)  # [N,d]
        idx = self.knn_idx.to(device)                                   # [N,k]

        # Gather neighbor embeddings: [N,k,d]
        En = E.index_select(0, idx.reshape(-1)).view(self.num_nodes, self.k, self.emb_dim)

        # Similarity per edge: [N,k]
        sim = (E.unsqueeze(1) * En).sum(dim=-1) / max(self.temperature, 1e-6)

        # Row-normalized attention weights: [N,k]
        w = F.softmax(sim, dim=1)

        if detach_weights:
            w = w.detach()

        # Build COO indices for edges i -> knn(i,j)
        row = torch.arange(self.num_nodes, device=device).unsqueeze(1).expand(-1, self.k).reshape(-1)
        col = idx.reshape(-1)
        val = w.reshape(-1)

        if self.add_self_loop:
            # Add self loop with a small fixed weight, then renormalize row-wise.
            # This helps stability early in training.
            self_row = torch.arange(self.num_nodes, device=device)
            self_col = self_row
            self_val = torch.full((self.num_nodes,), 1.0 / (self.k + 1), device=device, dtype=val.dtype)

            # scale existing vals to sum to k/(k+1)
            val = val * (self.k / (self.k + 1))

            row = torch.cat([row, self_row], dim=0)
            col = torch.cat([col, self_col], dim=0)
            val = torch.cat([val, self_val], dim=0)

        if self.symmetric:
            # Add reverse edges (j -> i) with same weight; then coalesce duplicates.
            row0, col0, val0 = row, col, val
            row = torch.cat([row0, col0], dim=0)
            col = torch.cat([col0, row0], dim=0)
            val = torch.cat([val0, val0], dim=0)

        indices = torch.stack([row, col], dim=0)

        A = torch.sparse_coo_tensor(
            indices=indices,
            values=val,
            size=(self.num_nodes, self.num_nodes),
            device=device,
            dtype=val.dtype
        ).coalesce()

        return A

    @torch.no_grad()
    def get_similarity_map(self, step: int, device: torch.device):
        """
        For interpretability: return per-node mean cosine similarity to its current kNN neighbors.
        Shape: [N] tensor.
        """
        self.maybe_rebuild(step, device=device)
        E = F.normalize(self.node_emb.detach().to(device), dim=1, eps=self.eps)  # [N,d]
        idx = self.knn_idx.to(device)  # [N,k]
        En = E.index_select(0, idx.reshape(-1)).view(self.num_nodes, self.k, self.emb_dim)
        sim = (E.unsqueeze(1) * En).sum(dim=-1)  # [N,k]
        return sim.mean(dim=1)  # [N]

import argparse
import os
import re
import json
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset import UrbanPlanDatasetWithGlobalPOI_Sparse, collate_dual_stream_sparse
from dual_stream_models import EfficientCoarseGenerator, EfficientDualStreamFineGenerator
from learnable_functional_graph import LearnableFunctionalGraph  # type: ignore

# ------------------------------
# Anchor functional layer
# ------------------------------
class AnchorFunctionalLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_anchors: int = 64, key_dim: int = 64, dropout: float = 0.1,
                 mix_anchors: bool = True):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_anchors = int(num_anchors)
        self.key_dim = int(key_dim)
        self.mix_anchors = bool(mix_anchors)

        self.q_proj = nn.Linear(self.hidden_dim, self.key_dim, bias=False)
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
        B, N, D = x.shape
        q = self.q_proj(x)  # [B,N,dk]
        logits = torch.matmul(q, self.anchor_keys.t()) / math.sqrt(self.key_dim)  # [B,N,M]
        A = torch.softmax(logits, dim=-1)

        anchor_vals = torch.matmul(A.transpose(1, 2), x)  # [B,M,D]
        if self.anchor_mlp is not None:
            anchor_vals = self.anchor_mlp(anchor_vals)

        out = torch.matmul(A, anchor_vals)  # [B,N,D]
        out = self.out_proj(out)
        out = self.norm(out)
        out = torch.nn.functional.gelu(out)
        out = self.dropout(out)
        return out


def any_prefix(sd: dict, prefix: str) -> bool:
    return any(k.startswith(prefix) for k in sd.keys())

def infer_max_index(sd: dict, prefix_regex: str) -> int:
    pat = re.compile(prefix_regex)
    mx = -1
    for k in sd.keys():
        m = pat.match(k)
        if m:
            try:
                mx = max(mx, int(m.group(1)))
            except Exception:
                pass
    return mx

def infer_num_layers(sd: dict, regex: str, default: int) -> int:
    mx = infer_max_index(sd, regex)
    return mx + 1 if mx >= 0 else default

def export_20ch(pred: np.ndarray, mode: str) -> np.ndarray:
    if pred.ndim == 3:
        pred = pred[None, ...]
    if pred.shape[1] == 20:
        return pred
    if pred.shape[1] != 21:
        raise ValueError(f"Expected 20 or 21 channels, got {pred.shape}")
    poi = pred[:, :20, :, :]
    if mode == "mass":
        return poi
    if mode == "renorm":
        p_empty = pred[:, 20:21, :, :]
        nonempty = 1.0 - p_empty
        return poi / (nonempty + 1e-8)
    raise ValueError(f"Unknown export_mode: {mode}")

def split_dataset(ds, fracs, seed):
    a, b, c = fracs
    s = a + b + c
    a, b, c = a/s, b/s, c/s
    n = len(ds)
    n_train = int(round(a * n))
    n_val = int(round(b * n))
    n_test = n - n_train - n_val
    g = torch.Generator().manual_seed(seed)
    return random_split(ds, [n_train, n_val, n_test], generator=g)

def identity_sparse(N: int, device):
    idx = torch.arange(N, device=device)
    indices = torch.stack([idx, idx], dim=0)
    values = torch.ones(N, device=device, dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, (N, N), device=device).coalesce()

def build_grid_adj_coo(H=100, W=100, device="cpu", diagonal=False):
    N = H * W
    dirs = [(-1,0),(1,0),(0,-1),(0,1)]
    if diagonal:
        dirs += [(-1,-1),(-1,1),(1,-1),(1,1)]

    src = []
    dst = []
    for i in range(H):
        for j in range(W):
            u = i * W + j
            for di, dj in dirs:
                ni, nj = i + di, j + dj
                if 0 <= ni < H and 0 <= nj < W:
                    v = ni * W + nj
                    src.extend([u, v])
                    dst.extend([v, u])

    # self-loop
    src.extend(range(N))
    dst.extend(range(N))

    indices = torch.tensor([src, dst], dtype=torch.long, device=device)
    values = torch.ones(indices.shape[1], dtype=torch.float32, device=device)
    A = torch.sparse_coo_tensor(indices, values, (N, N), device=device).coalesce()

    idx = A.indices()
    vals = A.values()
    deg = torch.zeros(N, device=device, dtype=torch.float32)
    deg.scatter_add_(0, idx[0], vals)
    deg_inv_sqrt = (deg + 1e-8).pow(-0.5)
    norm_vals = deg_inv_sqrt[idx[0]] * vals * deg_inv_sqrt[idx[1]]
    return torch.sparse_coo_tensor(idx, norm_vals, (N, N), device=device).coalesce()


def detect_backend(sd_fine: dict, ckpt: dict) -> str:
    if isinstance(ckpt, dict) and ckpt.get("func_graph") is not None:
        return "knn"
    if any("anchor_keys" in k for k in sd_fine.keys()):
        return "anchors"
    return "unknown"


def build_fine_model_from_ckpt(sd_f: dict, cond_dim: int, num_zones: int, device: torch.device,
                              temperature: float, force_backend: str = "auto"):
    if "output_head.1.weight" in sd_f:
        poi_dim = int(sd_f["output_head.1.weight"].shape[0])
    else:
        poi_dim = 21

    fine_num_layers = infer_num_layers(sd_f, r"^func_layers\.(\d+)\.", default=3)
    fine_num_bridge_layers = max(infer_max_index(sd_f, r"^bridges\.(\d+)\."), infer_max_index(sd_f, r"^bridge_gates\.(\d+)$")) + 1
    if fine_num_bridge_layers < 0:
        fine_num_bridge_layers = 2

    if "coarse_proj.weight" in sd_f:
        fine_hidden = int(sd_f["coarse_proj.weight"].shape[0])
    else:
        fine_hidden = 192

    zone_dim = 64
    if "zone_table" in sd_f and torch.is_tensor(sd_f["zone_table"]) and sd_f["zone_table"].ndim == 2:
        zone_dim = int(sd_f["zone_table"].shape[1])

    fine_has_cnn = any_prefix(sd_f, "spatial_stream.")

    has_anchor_keys = any("anchor_keys" in k for k in sd_f.keys())

    backend = "anchors" if has_anchor_keys else "standard"
    if force_backend == "anchors":
        backend = "anchors"
    elif force_backend == "knn":
        backend = "standard"

    G_fine = EfficientDualStreamFineGenerator(
        hidden_dim=fine_hidden,
        cond_dim=cond_dim,
        poi_dim=poi_dim,
        num_zones=num_zones,
        num_layers=fine_num_layers,
        num_bridge_layers=fine_num_bridge_layers,
        heads=4,
        dropout=0.1,
        zone_dim=zone_dim,
        temperature=temperature,
        use_checkpoint=False,
    ).to(device)

    if not fine_has_cnn:
        G_fine.spatial_stream = nn.Identity()

    if backend == "anchors":
        key_w = None
        for k, v in sd_f.items():
            if k.startswith("func_layers.0.q_proj.weight"):
                key_w = v
                break
        if key_w is not None:
            key_dim = int(key_w.shape[0])
        else:
            key_dim = 64

        anchor_k = None
        for k, v in sd_f.items():
            if k.startswith("func_layers.0.anchor_keys"):
                anchor_k = v
                break
        num_anchors = int(anchor_k.shape[0]) if anchor_k is not None else 64

        mix_anchors = any(k.startswith("func_layers.0.anchor_mlp") for k in sd_f.keys())

        G_fine.func_layers = nn.ModuleList([
            AnchorFunctionalLayer(hidden_dim=G_fine.hidden_dim, num_anchors=num_anchors, key_dim=key_dim,
                                  dropout=0.1, mix_anchors=mix_anchors).to(device)
            for _ in range(fine_num_layers)
        ])

    return G_fine


def build_func_graph_from_ckpt(ckpt: dict, device: torch.device) -> LearnableFunctionalGraph:
    sd = ckpt.get("func_graph", None)
    if sd is None:
        raise RuntimeError("Checkpoint has no func_graph")

    # infer emb_dim and k from state dict
    emb = sd.get("node_emb", None)
    if emb is None:
        # attempt to find by shape
        for k, v in sd.items():
            if torch.is_tensor(v) and v.ndim == 2 and v.shape[0] == 10000:
                emb = v
                break
    if emb is None:
        raise RuntimeError("Cannot infer emb_dim from func_graph state dict")

    emb_dim = int(emb.shape[1])
    knn_idx = sd.get("knn_idx", None)
    k = int(knn_idx.shape[1]) if knn_idx is not None and torch.is_tensor(knn_idx) and knn_idx.ndim == 2 else 16

    fg = LearnableFunctionalGraph(
        num_nodes=10000,
        emb_dim=emb_dim,
        k=k,
        rebuild_every=int(ckpt.get("args", {}).get("func_rebuild_every", 500)) if isinstance(ckpt.get("args", {}), dict) else 500,
        chunk_size=int(ckpt.get("args", {}).get("func_chunk_size", 256)) if isinstance(ckpt.get("args", {}), dict) else 256,
        temperature=float(ckpt.get("args", {}).get("func_temperature", 0.07)) if isinstance(ckpt.get("args", {}), dict) else 0.07,
        symmetric=bool(ckpt.get("args", {}).get("func_symmetric", True)) if isinstance(ckpt.get("args", {}), dict) else True,
        add_self_loop=bool(ckpt.get("args", {}).get("func_add_self_loop", True)) if isinstance(ckpt.get("args", {}), dict) else True,
    ).to(device)

    fg.load_state_dict(sd, strict=True)
    fg.eval()
    return fg


@torch.no_grad()
def generate_one(ckpt_path: str, out_npz: str, args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    sd_c = ckpt["G_coarse"]
    sd_f = ckpt["G_fine"]

    backend = detect_backend(sd_f, ckpt) if args.func_backend == "auto" else args.func_backend
    if backend == "unknown":
        backend = "knn" if ckpt.get("func_graph") is not None else "anchors"

    ds_all = UrbanPlanDatasetWithGlobalPOI_Sparse(args.data_dir, idx_start=args.idx_start, idx_end=args.idx_end,
                                                 log_transform_raw=False, add_empty_channel=True)
    cond_dim = int(ds_all.z.shape[1])

    use_pos_encoding = any("pos_encoding" in k for k in sd_c.keys())
    hidden_dim_coarse = int(sd_c["input_proj.weight"].shape[0])
    in_dim = int(sd_c["input_proj.weight"].shape[1])
    num_zones = int(sd_c["out_proj_zone.weight"].shape[0])
    noise_dim = in_dim - cond_dim
    if noise_dim <= 0:
        raise RuntimeError(f"Invalid noise_dim={noise_dim}; ckpt in_dim={in_dim}, cond_dim={cond_dim}")

    coarse_num_layers = infer_num_layers(sd_c, r"^gcn_layers\.(\d+)\.", default=3)

    # build coarse model
    A_spatial = build_grid_adj_coo(100, 100, device=device, diagonal=args.spatial_diag)
    G_coarse = EfficientCoarseGenerator(
        A_norm=A_spatial,
        cond_dim=cond_dim,
        noise_dim=noise_dim,
        hidden_dim=hidden_dim_coarse,
        num_layers=coarse_num_layers,
        num_zones=num_zones,
        use_pos_encoding=use_pos_encoding,
    ).to(device)

    # build fine model (handles anchors automatically)
    G_fine = build_fine_model_from_ckpt(sd_f, cond_dim=cond_dim, num_zones=num_zones, device=device,
                                        temperature=args.temperature, force_backend="auto")

    G_coarse.load_state_dict(sd_c, strict=True)
    G_fine.load_state_dict(sd_f, strict=True)

    G_coarse.eval(); G_fine.eval()

    if args.use_full_dataset:
        ds = ds_all
    else:
        _, _, test_set = split_dataset(ds_all, args.split_fracs, args.seed)
        ds = test_set

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0,
                        collate_fn=collate_dual_stream_sparse, drop_last=False)

    # Build functional adjacency
    N = 10000
    A_identity = identity_sparse(N, device)

    func_mode = args.func_mode
    if func_mode == "auto":
        func_mode = "cached" if backend == "knn" and ckpt.get("func_graph") is not None else "none"

    if func_mode == "none":
        A_func = A_identity
        func_info = {"backend": backend, "func_mode": "none"}
    else:
        if backend != "knn" or ckpt.get("func_graph") is None:
            # anchors backend ignores A anyway, but keep identity
            A_func = A_identity
            func_info = {"backend": backend, "func_mode": "identity_for_anchors"}
        else:
            fg = build_func_graph_from_ckpt(ckpt, device=device)
            # cached or rebuild
            if func_mode == "cached":
                last = int(fg.last_rebuild_step.item())
                if last < 0:
                    # ensure at least built once using step=0
                    A_func = fg.build_sparse_adjacency(step=0, detach_weights=True, device=device)
                    last_used = 0
                else:
                    A_func = fg.build_sparse_adjacency(step=last, detach_weights=True, device=device)
                    last_used = last
                func_info = {"backend": backend, "func_mode": "cached", "last_rebuild_step": int(fg.last_rebuild_step.item()), "step_used": int(last_used)}
            elif func_mode == "rebuild":
                # force one rebuild on this device
                fg.last_rebuild_step.fill_(-1)
                step = int(args.rebuild_step)
                A_func = fg.build_sparse_adjacency(step=step, detach_weights=True, device=device)
                func_info = {"backend": backend, "func_mode": "rebuild", "step_used": step}
            else:
                raise ValueError(f"Unknown func_mode: {func_mode}")

    outs = []
    for batch in loader:
        cond = batch["cond_vec"].to(device)
        B = cond.shape[0]
        noise = torch.randn(B, noise_dim, device=device)

        coarse_int, coarse_road, zone_logits = G_coarse(cond, noise)
        coarse_map = torch.cat([coarse_int, coarse_road], dim=1)

        fine_fake = G_fine(coarse_map, A_func, zone_logits, cond, zone_gate=args.zone_gate)
        outs.append(export_20ch(fine_fake.detach().cpu().numpy(), mode=args.export_mode))

    gen20 = np.concatenate(outs, axis=0)
    os.makedirs(os.path.dirname(out_npz) or ".", exist_ok=True)
    np.savez_compressed(out_npz, gen20=gen20)

    meta = {
        "ckpt": ckpt_path,
        "out_npz": out_npz,
        "export_mode": args.export_mode,
        "temperature": float(args.temperature),
        "zone_gate": bool(args.zone_gate),
        "split_fracs": list(args.split_fracs),
        "seed": int(args.seed),
        "use_full_dataset": bool(args.use_full_dataset),
        "backend_detected": backend,
        "func_info": func_info,
    }
    with open(os.path.splitext(out_npz)[0] + "_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[DONE] {backend} -> {out_npz} shape={gen20.shape}")


def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--ckpt", type=str, default=None, help="Single checkpoint path")
    ap.add_argument("--ckpts", type=str, default=None, help="Comma-separated checkpoints list")
    ap.add_argument("--data_dir", type=str, default="./data")

    ap.add_argument("--out_npz", type=str, default=None, help="Output NPZ for single mode")
    ap.add_argument("--out_root", type=str, default=None, help="Output root for multi mode")
    ap.add_argument("--out_name", type=str, default="generated_testset_20ch.npz")

    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--split_fracs", type=float, nargs=3, default=[0.8, 0.1, 0.1])
    ap.add_argument("--use_full_dataset", action="store_true")

    ap.add_argument("--idx_start", type=int, default=0)
    ap.add_argument("--idx_end", type=int, default=None)

    ap.add_argument("--export_mode", type=str, default="mass", choices=["mass", "renorm"])
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--zone_gate", action="store_true")

    ap.add_argument("--device", type=str, default="cuda")

    # functional behavior
    ap.add_argument("--func_backend", type=str, default="auto", choices=["auto", "knn", "anchors"],
                    help="Override detected backend if needed (usually leave auto).")
    ap.add_argument("--func_mode", type=str, default="auto", choices=["auto", "cached", "rebuild", "none"],
                    help="How to build functional adjacency for kNN backend.")
    ap.add_argument("--rebuild_step", type=int, default=0, help="Step used if --func_mode rebuild")

    # coarse adjacency option
    ap.add_argument("--spatial_diag", action="store_true", help="Use 8-neighbor spatial adjacency for coarse GCN")

    return ap.parse_args()


def main():
    args = parse_args()

    if args.ckpts:
        if not args.out_root:
            raise RuntimeError("--ckpts requires --out_root")
        ckpts = [c.strip() for c in args.ckpts.split(",") if c.strip()]
        if not ckpts:
            raise RuntimeError("Empty --ckpts")
        os.makedirs(args.out_root, exist_ok=True)
        for c in ckpts:
            tag = os.path.basename(os.path.dirname(c)) or "run"
            out_npz = os.path.join(args.out_root, tag, args.out_name)
            os.makedirs(os.path.dirname(out_npz), exist_ok=True)
            generate_one(c, out_npz, args)
        return

    if not args.ckpt or not args.out_npz:
        raise RuntimeError("Provide --ckpt and --out_npz (or use --ckpts + --out_root).")

    generate_one(args.ckpt, args.out_npz, args)


if __name__ == "__main__":
    main()

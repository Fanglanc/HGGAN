import os
import argparse
import json
import time

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dual_stream_models import (
    EfficientCoarseGenerator,
    EfficientDualStreamFineGenerator,
    EfficientDiscriminator,
)

from dataset import (
    UrbanPlanDatasetWithGlobalPOI_Sparse,
    collate_dual_stream_sparse,
)

from balanced_loss import UltraMinimalLossNoSpatial
from anchor_functional_layers import AnchorFunctionalLayer
from learnable_functional_graph import LearnableFunctionalGraph  # type: ignore


def set_seed(seed):
    if seed is None:
        return
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_spatial_adjacency(H=100, W=100):
    N = H * W
    edges = []
    for i in range(H):
        for j in range(W):
            idx = i * W + j
            if i > 0: edges.append((idx, (i-1)*W + j))
            if i < H-1: edges.append((idx, (i+1)*W + j))
            if j > 0: edges.append((idx, i*W + (j-1)))
            if j < W-1: edges.append((idx, i*W + (j+1)))

    row = [e[0] for e in edges]
    col = [e[1] for e in edges]
    data = np.ones(len(edges), dtype=np.float32)
    A = sp.coo_matrix((data, (row, col)), shape=(N, N))
    A = A + sp.eye(N)

    rowsum = np.array(A.sum(1)).flatten()
    d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    A_norm = (D_inv_sqrt @ A @ D_inv_sqrt).tocoo()

    idx = torch.LongTensor(np.vstack([A_norm.row, A_norm.col]))
    val = torch.FloatTensor(A_norm.data)
    return torch.sparse.FloatTensor(idx, val, torch.Size(A_norm.shape)).coalesce()


def identity_sparse(N: int, device):
    idx = torch.arange(N, device=device)
    indices = torch.stack([idx, idx], dim=0)
    values = torch.ones(N, device=device)
    return torch.sparse_coo_tensor(indices, values, (N, N), device=device).coalesce()


@torch.no_grad()
def build_poi_functional_graph(poi_mean_1x20hw: torch.Tensor, k=16, device="cpu", batch_size=500):
    """Fixed POI-based kNN functional graph (used ONLY for warmup in knn backend)."""
    poi_mean = poi_mean_1x20hw[0].to(device)  # [20,H,W]
    C, H, W = poi_mean.shape
    N = H * W

    poi_flat = poi_mean.view(C, N).T.contiguous()  # [N,20]
    poi_log = torch.log1p(poi_flat)
    poi_norm = torch.nn.functional.normalize(poi_log, p=2, dim=-1)

    all_neighbors = []
    for i in range(0, N, batch_size):
        end = min(i + batch_size, N)
        sim = poi_norm[i:end] @ poi_norm.T
        for j in range(end - i):
            sim[j, i + j] = -float("inf")
        _, idx = torch.topk(sim, k, dim=1)
        all_neighbors.append(idx)
    neighbors = torch.cat(all_neighbors, dim=0)  # [N,k]

    src = torch.arange(N, device=device).unsqueeze(1).expand(-1, k).reshape(-1)
    dst = neighbors.reshape(-1)
    w = torch.ones_like(src, dtype=torch.float32)

    src_all = torch.cat([src, dst, torch.arange(N, device=device)], dim=0)
    dst_all = torch.cat([dst, src, torch.arange(N, device=device)], dim=0)
    w_all = torch.cat([w, w, torch.ones(N, device=device)], dim=0)

    indices = torch.stack([src_all, dst_all], dim=0)
    A = torch.sparse_coo_tensor(indices, w_all, (N, N)).coalesce()

    vals = A.values()
    idx2 = A.indices()
    deg = torch.zeros(N, device=device)
    deg.scatter_add_(0, idx2[0], vals)
    deg_inv_sqrt = (deg + 1e-8).pow(-0.5)
    norm_vals = deg_inv_sqrt[idx2[0]] * vals * deg_inv_sqrt[idx2[1]]
    return torch.sparse_coo_tensor(idx2, norm_vals, (N, N)).coalesce()


def parse_args():
    p = argparse.ArgumentParser(description="Option B (kNN) OR Option 2 (anchors) functional stream; NO spatial loss")

    # Data
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--output_dir", type=str, default="./result/func_backend_compare")

    # Training
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr_g", type=float, default=2e-5)
    p.add_argument("--lr_d", type=float, default=1e-5)
    p.add_argument("--n_critic", type=int, default=1)

    # Architecture
    p.add_argument("--cond_dim", type=int, default=20)
    p.add_argument("--noise_dim", type=int, default=16)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--fine_hidden_dim", type=int, default=192)
    p.add_argument("--num_zones", type=int, default=12)
    p.add_argument("--use_pos_encoding", action="store_true", default=True)

    # Loss weights (spatial removed)
    p.add_argument("--dist_weight", type=float, default=0.3)
    p.add_argument("--road_weight", type=float, default=2.0)
    p.add_argument("--target_spacing", type=int, default=10)
    p.add_argument("--use_adversarial", action="store_true", default=False)

    # Choose functional backend
    p.add_argument("--func_backend", type=str, default="knn", choices=["knn", "anchors"],
                   help="knn: LearnableFunctionalGraph w/ periodic rebuild. anchors: low-rank anchors (no rebuild).")

    # kNN backend (Option B)
    p.add_argument("--func_emb_dim", type=int, default=32)
    p.add_argument("--func_k", type=int, default=16)
    p.add_argument("--func_rebuild_every", type=int, default=500)
    p.add_argument("--func_chunk_size", type=int, default=256)
    p.add_argument("--func_temperature", type=float, default=0.07)
    p.add_argument("--func_symmetric", action="store_true", default=True)
    p.add_argument("--func_add_self_loop", action="store_true", default=True)
    p.add_argument("--func_warmup_steps", type=int, default=1000)

    # anchors backend (Option 2)
    p.add_argument("--anchor_m", type=int, default=64)
    p.add_argument("--anchor_key_dim", type=int, default=64)
    p.add_argument("--anchor_mix_anchors", action="store_true", default=True)

    # Grad clipping
    p.add_argument("--g_clip_grad", type=float, default=2.0)
    p.add_argument("--d_clip_grad", type=float, default=1.0)

    # Logging / saving
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--save_interval", type=int, default=1000)
    p.add_argument("--val_interval", type=int, default=500)

    # Resume
    p.add_argument("--resume", type=str, default=None)

    # Device / seed
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=4)

    return p.parse_args()


def setup_directories(args):
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)


def create_models(args, A_spatial, device):
    G_coarse = EfficientCoarseGenerator(
        A_norm=A_spatial,
        cond_dim=args.cond_dim,
        noise_dim=args.noise_dim,
        hidden_dim=args.hidden_dim,
        num_layers=3,
        num_zones=args.num_zones,
        use_pos_encoding=args.use_pos_encoding,
    ).to(device)

    G_fine = EfficientDualStreamFineGenerator(
        hidden_dim=args.fine_hidden_dim,
        cond_dim=args.cond_dim,
        poi_dim=21,
        num_zones=args.num_zones,
        num_layers=3,
        num_bridge_layers=2,
        heads=4,
        dropout=0.1,
        zone_dim=64,
        temperature=0.7,
        use_checkpoint=False,
    ).to(device)
    # If anchors backend: replace functional layers (and move them to device)
    if args.func_backend == "anchors":
        import torch
        new_layers = [
            AnchorFunctionalLayer(
                hidden_dim=G_fine.hidden_dim,
                num_anchors=args.anchor_m,
                key_dim=args.anchor_key_dim,
                dropout=0.1,
                mix_anchors=args.anchor_mix_anchors
            ).to(device)
            for _ in range(len(G_fine.func_layers))
        ]
        G_fine.func_layers = torch.nn.ModuleList(new_layers)
        # Extra safety: ensure the whole module is on the target device
        G_fine.to(device)



    D = EfficientDiscriminator(in_channels=21, cond_dim=args.cond_dim).to(device)
    return G_coarse, G_fine, D


def create_loss_module(args, device):
    return UltraMinimalLossNoSpatial(
        dist_weight=args.dist_weight,
        road_weight=args.road_weight,
        target_spacing=args.target_spacing,
        use_adversarial=args.use_adversarial,
        empty_index=20,
        road_index=0,
    ).to(device)


def create_optimizers(args, G_coarse, G_fine, D, loss_mod, func_graph=None):
    g_params = list(G_coarse.parameters()) + list(G_fine.parameters())
    if func_graph is not None:
        g_params += list(func_graph.parameters())
    opt_g = torch.optim.Adam(g_params, lr=args.lr_g, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=args.lr_d, betas=(0.5, 0.999))
    opt_w = torch.optim.Adam([loss_mod.log_var_adv], lr=1e-4) if args.use_adversarial else None
    return opt_g, opt_d, opt_w


def load_data(args, num_workers=0, pin_memory=False):
    full = UrbanPlanDatasetWithGlobalPOI_Sparse(data_dir=args.data_dir, add_empty_channel=True)
    N = len(full)
    train_size = int(0.9 * N)

    train_dataset = UrbanPlanDatasetWithGlobalPOI_Sparse(args.data_dir, idx_start=0, idx_end=train_size, add_empty_channel=True)
    val_dataset = UrbanPlanDatasetWithGlobalPOI_Sparse(args.data_dir, idx_start=train_size, idx_end=N, add_empty_channel=True)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
        collate_fn=collate_dual_stream_sparse
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=max(1, num_workers//2), pin_memory=pin_memory,
        collate_fn=collate_dual_stream_sparse
    )
    return train_loader, val_loader, full


def create_grid_road_target(batch_size, H=100, W=100, spacing=10, device="cuda"):
    road_target = torch.zeros(batch_size, H, W, device=device)
    for i in range(0, H, spacing):
        road_target[:, i, :] = 1.0
    for j in range(0, W, spacing):
        road_target[:, :, j] = 1.0
    return road_target


@torch.no_grad()
def validate(G_coarse, G_fine, val_loader, A_func, device, args):
    G_coarse.eval(); G_fine.eval()
    val_mse = 0.0
    val_kl = 0.0
    nb = 0
    for batch in val_loader:
        x0 = batch["x0"].to(device)
        cond_vec = batch["cond_vec"].to(device)
        bs = x0.shape[0]
        noise = torch.randn(bs, args.noise_dim, device=device)

        coarse_int, coarse_road, zone_logits = G_coarse(cond_vec, noise)
        coarse_map = torch.cat([coarse_int, coarse_road], dim=1)
        fine_fake = G_fine(coarse_map, A_func, zone_logits, cond_vec)

        mse = F.mse_loss(fine_fake, x0)
        val_mse += mse.item()

        pred_poi = fine_fake[:, :20, :, :]
        target_poi = x0[:, :20, :, :]
        pred_dist = pred_poi.mean(dim=[0, 2, 3]); target_dist = target_poi.mean(dim=[0, 2, 3])
        pred_dist = pred_dist / (pred_dist.sum() + 1e-8); target_dist = target_dist / (target_dist.sum() + 1e-8)
        kl = (target_dist * torch.log((target_dist + 1e-8) / (pred_dist + 1e-8))).sum()
        val_kl += kl.item()
        nb += 1

    G_coarse.train(); G_fine.train()
    return val_mse / max(1, nb), val_kl / max(1, nb)


def train(args):
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    num_workers, pin_memory = (4, True) if device.type == "cuda" else (0, False)

    setup_directories(args)

    print("\nBuilding spatial adjacency (coarse GCN)...")
    A_spatial = build_spatial_adjacency(100, 100).to(device)
    print(f"✓ A_spatial nnz={A_spatial.values().numel():,}")

    G_coarse, G_fine, D = create_models(args, A_spatial, device)
    train_loader, val_loader, full_dataset = load_data(args, num_workers, pin_memory)

    # Build functional backend
    func_graph = None
    A_func_warmup = None
    A_identity = identity_sparse(10000, device)

    if args.func_backend == "knn":
        print("\n[FUNC BACKEND] kNN (LearnableFunctionalGraph)")

        print("Building POI functional graph for warmup...")
        poi_mean = full_dataset.get_global_poi_for_graph().to(device)  # [1,20,100,100]
        t0 = time.time()
        A_func_warmup = build_poi_functional_graph(poi_mean, k=args.func_k, device=device)
        print(f"✓ A_func_warmup built in {time.time()-t0:.1f}s  nnz={A_func_warmup.values().numel():,}")

        func_graph = LearnableFunctionalGraph(
            num_nodes=10000,
            emb_dim=args.func_emb_dim,
            k=args.func_k,
            rebuild_every=args.func_rebuild_every,
            chunk_size=args.func_chunk_size,
            temperature=args.func_temperature,
            symmetric=args.func_symmetric,
            add_self_loop=args.func_add_self_loop,
        ).to(device)

    else:
        print("\n[FUNC BACKEND] anchors (low-rank, no rebuild)")
        print(f"anchors: M={args.anchor_m}, key_dim={args.anchor_key_dim}, mix_anchors={args.anchor_mix_anchors}")

    loss_mod = create_loss_module(args, device)
    opt_g, opt_d, opt_w = create_optimizers(args, G_coarse, G_fine, D, loss_mod, func_graph=func_graph)

    start_epoch = 0
    global_step = 0
    best_val_mse = float("inf")

    if args.resume:
        print(f"\nResuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        G_coarse.load_state_dict(ckpt["G_coarse"])
        G_fine.load_state_dict(ckpt["G_fine"])
        D.load_state_dict(ckpt["D"])
        loss_mod.load_state_dict(ckpt["loss_mod"])
        opt_g.load_state_dict(ckpt["opt_g"])
        opt_d.load_state_dict(ckpt["opt_d"])
        if opt_w is not None and ckpt.get("opt_w") is not None:
            opt_w.load_state_dict(ckpt["opt_w"])
        if func_graph is not None and ckpt.get("func_graph") is not None:
            func_graph.load_state_dict(ckpt["func_graph"], strict=True)

        start_epoch = ckpt.get("epoch", 0) + 1
        global_step = ckpt.get("global_step", 0)
        best_val_mse = ckpt.get("best_val_mse", float("inf"))

    print("\n" + "="*80)
    print("START TRAINING (NO spatial loss) | func_backend =", args.func_backend)
    print("="*80 + "\n")

    for epoch in range(start_epoch, args.epochs):
        for batch in train_loader:
            x0 = batch["x0"].to(device)
            cond_vec = batch["cond_vec"].to(device)
            bs = x0.shape[0]

            do_g = ((global_step + 1) % args.n_critic == 0)

            # Functional adjacency for current step
            if args.func_backend == "anchors":
                A_func = A_identity
            else:
                if global_step < args.func_warmup_steps:
                    A_func = A_func_warmup
                else:
                    A_func = func_graph.build_sparse_adjacency(
                        step=global_step,
                        detach_weights=(not do_g),
                        device=device
                    )

            noise = torch.randn(bs, args.noise_dim, device=device)
            coarse_int, coarse_road, zone_logits = G_coarse(cond_vec, noise)
            coarse_map = torch.cat([coarse_int, coarse_road], dim=1)

            fine_fake = G_fine(coarse_map, A_func, zone_logits, cond_vec)

            d_fake = D(fine_fake.detach(), cond_vec)
            d_real = D(x0, cond_vec)

            opt_d.zero_grad()
            d_loss = loss_mod.forward_discriminator(d_real, d_fake)
            d_loss.backward()
            if args.d_clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(D.parameters(), args.d_clip_grad)
            opt_d.step()

            if do_g:
                d_fake_for_g = D(fine_fake, cond_vec) if args.use_adversarial else None
                road_target = create_grid_road_target(bs, 100, 100, args.target_spacing, device)

                g_losses = loss_mod.forward_generator(
                    pred=fine_fake,
                    target=x0,
                    coarse_road=coarse_road,
                    road_target=road_target,
                    d_fake=d_fake_for_g,
                )

                opt_g.zero_grad()
                if opt_w is not None:
                    opt_w.zero_grad()
                g_losses["total"].backward()

                if args.g_clip_grad > 0:
                    params = list(G_coarse.parameters()) + list(G_fine.parameters())
                    if func_graph is not None:
                        params += list(func_graph.parameters())
                    torch.nn.utils.clip_grad_norm_(params, args.g_clip_grad)

                opt_g.step()
                if opt_w is not None:
                    opt_w.step()
                    loss_mod.clamp_log_vars()
            else:
                with torch.no_grad():
                    road_target = create_grid_road_target(bs, 100, 100, args.target_spacing, device)
                    g_losses = loss_mod.forward_generator(fine_fake, x0, coarse_road, road_target, None)

            if global_step % args.log_interval == 0:
                msg = (
                    f"[Epoch {epoch}, Step {global_step}] "
                    f"MSE={g_losses['mse']:.4f} KL={g_losses['kl']:.4f} "
                    f"Road={g_losses['road']:.4f} TotalG={g_losses['total']:.4f} D={d_loss:.4f}"
                )
                if args.func_backend == "knn" and global_step >= args.func_warmup_steps:
                    msg += f" | func_last_rebuild={int(func_graph.last_rebuild_step.item())}"
                print(msg)

            if global_step % args.val_interval == 0 and global_step > 0:
                if args.func_backend == "anchors":
                    A_val = A_identity
                else:
                    if global_step < args.func_warmup_steps:
                        A_val = A_func_warmup
                    else:
                        A_val = func_graph.build_sparse_adjacency(step=global_step, detach_weights=True, device=device)

                val_mse, val_kl = validate(G_coarse, G_fine, val_loader, A_val, device, args)
                print(f"\n{'='*70}\nVALIDATION (Step {global_step})  ValMSE={val_mse:.4f}  ValKL={val_kl:.4f}")

                if val_mse < best_val_mse:
                    best_val_mse = val_mse
                    best_path = os.path.join(args.output_dir, "best_model.pt")
                    torch.save({
                        "epoch": epoch,
                        "global_step": global_step,
                        "G_coarse": G_coarse.state_dict(),
                        "G_fine": G_fine.state_dict(),
                        "D": D.state_dict(),
                        "loss_mod": loss_mod.state_dict(),
                        "opt_g": opt_g.state_dict(),
                        "opt_d": opt_d.state_dict(),
                        "opt_w": opt_w.state_dict() if opt_w is not None else None,
                        "func_graph": func_graph.state_dict() if func_graph is not None else None,
                        "val_mse": val_mse,
                        "val_kl": val_kl,
                        "best_val_mse": best_val_mse,
                        "args": vars(args),
                    }, best_path)
                    print(f"New best model saved: {best_path}")
                else:
                    print(f"Best MSE so far: {best_val_mse:.4f}")
                print(f"{'='*70}\n")

            if global_step % args.save_interval == 0 and global_step > 0:
                ckpt_path = os.path.join(args.output_dir, "checkpoints", f"checkpoint_step_{global_step:06d}.pt")
                torch.save({
                    "epoch": epoch,
                    "global_step": global_step,
                    "G_coarse": G_coarse.state_dict(),
                    "G_fine": G_fine.state_dict(),
                    "D": D.state_dict(),
                    "loss_mod": loss_mod.state_dict(),
                    "opt_g": opt_g.state_dict(),
                    "opt_d": opt_d.state_dict(),
                    "opt_w": opt_w.state_dict() if opt_w is not None else None,
                    "func_graph": func_graph.state_dict() if func_graph is not None else None,
                    "best_val_mse": best_val_mse,
                    "args": vars(args),
                }, ckpt_path)
                print(f"Saved checkpoint: {ckpt_path}")

            global_step += 1

        epoch_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch:03d}.pt")
        torch.save({
            "epoch": epoch,
            "global_step": global_step,
            "G_coarse": G_coarse.state_dict(),
            "G_fine": G_fine.state_dict(),
            "D": D.state_dict(),
            "loss_mod": loss_mod.state_dict(),
            "opt_g": opt_g.state_dict(),
            "opt_d": opt_d.state_dict(),
            "opt_w": opt_w.state_dict() if opt_w is not None else None,
            "func_graph": func_graph.state_dict() if func_graph is not None else None,
            "best_val_mse": best_val_mse,
            "args": vars(args),
        }, epoch_path)
        print(f"Epoch {epoch} complete. Saved: {epoch_path}")

    print("\nTRAINING COMPLETE")
    print(f"Best validation MSE: {best_val_mse:.4f}")


if __name__ == "__main__":
    args = parse_args()
    train(args)

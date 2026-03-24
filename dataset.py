import os
import numpy as np
import torch
from torch.utils.data import Dataset


def _ensure_poi_nchw(poi: np.ndarray) -> np.ndarray:
    if poi.ndim != 4:
        raise ValueError(f"poi must have 4 dims, got {poi.shape}")
    if poi.shape[-1] == 20:  # [N, H, W, 20] -> [N, 20, H, W]
        poi = np.transpose(poi, (0, 3, 1, 2))
    return poi


def build_sparse_target_with_empty(poi_raw: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    if poi_raw.ndim != 4 or poi_raw.shape[1] != 20:
        raise ValueError(f"poi_raw must be [N,20,H,W], got {poi_raw.shape}")

    N, C, H, W = poi_raw.shape
    # Empty if total count in the cell == 0
    cell_sum = poi_raw.sum(axis=1)  # [N,H,W]
    empty = (cell_sum <= 0).astype(np.float32)  # [N,H,W]

    # Normalize POIs ONLY for non-empty cells
    norm = poi_raw / (cell_sum[:, None, :, :] + eps)  # [N,20,H,W]
    norm = norm.astype(np.float32)

    # Zero out POI channels at empty cells
    norm = norm * (1.0 - empty[:, None, :, :])

    # Concatenate empty channel (last)
    x0 = np.concatenate([norm, empty[:, None, :, :]], axis=1)  # [N,21,H,W]
    return x0


class UrbanPlanDatasetWithGlobalPOI_Sparse(Dataset):
    def __init__(self, data_dir="./data", idx_start=0, idx_end=None,
                 log_transform_raw: bool = False,
                 add_empty_channel: bool = True,
                 road_threshold: float = 0.0):
        func = np.load(os.path.join(data_dir, "func1_100.npz"))["arr_0"].astype(np.int64)
        poi = np.load(os.path.join(data_dir, "100_poi_dis.npz"))["arr_0"].astype(np.float32)
        sc = np.load(os.path.join(data_dir, "surround_context_emb.npz"))["arr_0"].astype(np.float32)
        hg = np.load(os.path.join(data_dir, "human_guide_emb.npz"))["arr_0"].astype(np.float32)

        poi = _ensure_poi_nchw(poi)  # [N,20,100,100]

        road_raw = poi[:, 0:1, :, :].copy()  # [N,1,H,W] - raw road counts
        self.road_binary = (road_raw > road_threshold).astype(np.float32)
        print(f"  Extracted binary road masks: {self.road_binary.shape}")
        print(f"    Road presence: {self.road_binary.mean():.2%} of cells")
        print(f"    Using threshold: {road_threshold}")

        poi_raw = poi.copy()

        if log_transform_raw:
            poi_raw = np.log1p(poi_raw)

        self.global_poi_mean = poi_raw.mean(axis=0).astype(np.float32)  # [20,H,W]
        self.global_poi_std = (poi_raw.std(axis=0) + 1e-8).astype(np.float32)

        if add_empty_channel:
            x0 = build_sparse_target_with_empty(poi_raw)  # [N,21,H,W]
        else:
            cell_sum = poi_raw.sum(axis=1, keepdims=True) + 1e-8
            x0 = (poi_raw / cell_sum).astype(np.float32)  # [N,20,H,W]

        # Condition vector
        z = np.hstack([sc, hg]).astype(np.float32)

        N = func.shape[0]
        if idx_end is None:
            idx_end = N

        self.func = func[idx_start:idx_end]
        self.x0 = x0[idx_start:idx_end]
        self.poi_raw = poi_raw[idx_start:idx_end]
        self.z = z[idx_start:idx_end]
        self.road_binary = self.road_binary[idx_start:idx_end]

        print(f"Sparse DualStream Dataset loaded: {self.func.shape[0]} samples")
        print(f"  x0 shape: {self.x0.shape} (poi_dim={self.x0.shape[1]})")
        print(f"  poi_raw shape: {self.poi_raw.shape}")
        print(f"  zones shape: {self.func.shape}")
        print(f"  cond shape: {self.z.shape}")
        print(f"  road_binary shape: {self.road_binary.shape}")

    def get_global_poi_for_graph(self):
        return torch.from_numpy(self.global_poi_mean).float().unsqueeze(0)

    def __len__(self):
        return self.func.shape[0]

    def __getitem__(self, i):
        return {
            "x0": torch.from_numpy(self.x0[i]).float(),
            "zones": torch.from_numpy(self.func[i]).long(),
            "cond_vec": torch.from_numpy(self.z[i]).float(),
            "poi_raw": torch.from_numpy(self.poi_raw[i]).float(),
            "road_binary": torch.from_numpy(self.road_binary[i]).float(),  # NEW
        }


def collate_dual_stream_sparse(batch):
    return {
        "x0": torch.stack([b["x0"] for b in batch]),
        "zones": torch.stack([b["zones"] for b in batch]),
        "cond_vec": torch.stack([b["cond_vec"] for b in batch]),
        "poi_raw": torch.stack([b["poi_raw"] for b in batch]),
        "road_binary": torch.stack([b["road_binary"] for b in batch]),  # NEW
    }

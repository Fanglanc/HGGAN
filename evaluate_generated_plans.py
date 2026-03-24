import os
import json
import math
import argparse
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from gensim.matutils import kullback_leibler, hellinger
from scipy.stats import wasserstein_distance
from scipy.spatial import distance
from collections import defaultdict

try:
    from urban_plan_evaluator import UrbanPlanEvaluator
    LLM_ENHANCED_AVAILABLE = True
except ImportError:
    print("WARNING: urban_plan_evaluator.py not found.")
    print("Urban Plan Evaluator will be disabled.")
    print("To enable: ensure urban_plan_evaluator.py is in the same directory.")
    LLM_ENHANCED_AVAILABLE = False
    UrbanPlanEvaluator = None


# ---------------------------- load helpers ----------------------------
def _load_npz_arr(path: str) -> np.ndarray:
    d = np.load(path, allow_pickle=True)
    for k in ["generated", "predictions", "gen_samples", "samples", "arr_0"]:
        if k in d:
            return d[k]
    return d[d.files[0]]


def _ensure_nchw(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if x.ndim != 4:
        raise ValueError(f"Expected 4D array, got {x.shape}")
    # NCHW
    if x.shape[1] == 20 and x.shape[2] == 100 and x.shape[3] == 100:
        return x
    # NHWC
    if x.shape[-1] == 20 and x.shape[1] == 100 and x.shape[2] == 100:
        return np.transpose(x, (0, 3, 1, 2))
    # flattened
    if x.ndim == 2 and x.shape[1] == 100 * 100 * 20:
        x = x.reshape((x.shape[0], 100, 100, 20))
        return np.transpose(x, (0, 3, 1, 2))
    raise ValueError(f"Unsupported shape (expect NCHW or NHWC or flattened): {x.shape}")


def _ensure_zone_hw(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z)
    z = np.squeeze(z)
    if z.ndim == 2:
        return z.astype(np.int32)
    if z.ndim == 3:
        # one-hot or channels-first
        if z.shape[0] > 1:
            return np.argmax(z, axis=0).astype(np.int32)
        return z[0].astype(np.int32)
    raise ValueError(f"Unsupported zone shape: {z.shape}")


def find_generated_file(model_name: str, model_dir: str, tag: str, results_dir: str) -> Optional[str]:
    cands = [
        os.path.join(model_dir, "generated", f"generated_{tag}.npz"),
        os.path.join(model_dir, "generated", "generated_testset.npz"),
        os.path.join(model_dir, f"generated_{tag}.npz"),
        # TF baseline output
        os.path.join(results_dir, f"{model_name}_generate_result_test.npz"),
        os.path.join(results_dir, f"{model_name}_generate_result_testset.npz"),
        os.path.join(results_dir, f"{model_name.lower()}_generate_result_test.npz"),
        os.path.join(results_dir, f"{model_name.lower()}_generate_result_testset.npz"),
    ]
    for p in cands:
        if os.path.isfile(p):
            return p
    return None


# ---------------------------- dataset alignment (matches generator scripts) ----------------------------
def load_green_standards(data_dir: str):
    import pickle
    
    data_dir = os.path.expanduser(data_dir)
    standards_path = os.path.join(data_dir, "green_standards.pkl")
    
    if not os.path.isfile(standards_path):
        print(f"[INFO] green_standards.pkl not found at {standards_path}")
        print("[INFO] Will use test-vs-test comparison instead of test-vs-standards")
        return None
    
    try:
        with open(standards_path, "rb") as f:
            green_standards = pickle.load(f)
        
        print(f"[INFO] Loaded green_standards.pkl")
        print(f"       Green levels: {sorted(green_standards.keys())}")
        for level, standard in sorted(green_standards.items()):
            print(f"       Level {level}: shape={standard.shape}, range=[{standard.min():.4f}, {standard.max():.4f}]")
        
        return green_standards
    except Exception as e:
        print(f"[WARNING] Failed to load green_standards.pkl: {e}")
        return None


def load_canonical_test_subset(data_dir: str, ratio: float = 0.9, test_size: int = 299, seed: int = 0):
    data_dir = os.path.expanduser(data_dir)
    poi_path = os.path.join(data_dir, "100_poi_dis.npz")
    zone_path = os.path.join(data_dir, "func1_100.npz")
    label_path = os.path.join(data_dir, "con_label.npz")

    if not os.path.isfile(poi_path):
        raise FileNotFoundError(f"Missing {poi_path}")
    if not os.path.isfile(zone_path):
        raise FileNotFoundError(f"Missing {zone_path}")

    good_plan = _ensure_nchw(_load_npz_arr(poi_path))        # (N,20,100,100)
    zones_all = _load_npz_arr(zone_path)                     # (N,100,100) typical

    N = int(good_plan.shape[0])
    split = int(N * float(ratio))
    tail_n = max(0, N - split)

    rng = np.random.default_rng(int(seed))

    if tail_n >= int(test_size):
        idx = np.arange(split, split + int(test_size), dtype=np.int64)
    else:
        if tail_n > 0:
            pick = rng.integers(0, tail_n, size=int(test_size), endpoint=False)
            idx = split + pick
        else:
            idx = rng.integers(0, N, size=int(test_size), endpoint=False)

    real = good_plan[idx]

    zones_list = []
    for k in idx:
        zones_list.append(_ensure_zone_hw(zones_all[int(k)]))
    zones_hw = np.stack(zones_list, axis=0)

    con_label = None
    if os.path.isfile(label_path):
        try:
            # Load the raw data first (don't index yet!)
            con_label_raw = _load_npz_arr(label_path)
            
            # Check size and handle accordingly
            if len(con_label_raw) == test_size:
                con_label = con_label_raw
                print(f"[INFO] Loaded con_label: {len(con_label)} samples (already test subset)")
                print(f"      Unique green levels: {np.unique(con_label)}")
                
            elif len(con_label_raw) == N:
                con_label = con_label_raw[idx]
                print(f"[INFO] Loaded con_label: extracted {len(con_label)} test samples from full dataset")
                print(f"      Unique green levels: {np.unique(con_label)}")
                
            else:
                print(f"[WARNING] con_label has unexpected size: {len(con_label_raw)}")
                print(f"           Expected {test_size} (test) or {N} (full)")
                print(f"           Will try to use first {test_size} samples...")
                con_label = con_label_raw[:test_size]
                
        except Exception as e:
            print(f"[WARNING] Failed to load con_label.npz: {e}")
            import traceback
            traceback.print_exc()
            con_label = None
    else:
        print(f"[INFO] con_label.npz not found at {label_path}")
        print(f"[INFO] Will use ungrouped evaluation (lower KL values expected)")

    return real, zones_hw, con_label


# ---------------------------- quantitative metrics ----------------------------
def _normalize_nonneg(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Normalize to a probability distribution.
    Used for TV distance, L1, L2 metrics that need proper normalization.
    """
    v = np.asarray(v, dtype=np.float64).flatten()
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    v = np.clip(v, 0.0, None)
    v[v <= 0] = 0.00001
    s = float(v.sum())
    if s <= eps:
        return np.full_like(v, 1.0 / max(1, v.size), dtype=np.float64)
    return v / s


def _replace_zeros(v: np.ndarray, eps_val: float = 0.00001) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).flatten()
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    v = np.clip(v, 0.0, None)
    v[v <= 0] = eps_val
    return v


def _kl(p, q, eps=1e-8):
    p = _replace_zeros(p, eps_val=0.00001)
    q = _replace_zeros(q, eps_val=0.00001)
    return float(kullback_leibler(p, q))


def _js(p, q, eps=1e-8):
    p = _replace_zeros(p, eps_val=0.00001)
    q = _replace_zeros(q, eps_val=0.00001)
    return float(distance.jensenshannon(p, q))


def _tv(p, q, eps=1e-8):
    p = _replace_zeros(p, eps_val=0.00001)
    q = _replace_zeros(q, eps_val=0.00001)
    p_norm = p / (np.sum(p) + eps)
    q_norm = q / (np.sum(q) + eps)
    return float(0.5 * np.sum(np.abs(p_norm - q_norm)))


def _wd(p, q, eps=1e-8):
    p = _replace_zeros(p, eps_val=0.00001)
    q = _replace_zeros(q, eps_val=0.00001)
    return float(wasserstein_distance(p, q))


def _cos_dist(p, q, eps=1e-8):
    p = _replace_zeros(p, eps_val=0.00001)
    q = _replace_zeros(q, eps_val=0.00001)
    return float(distance.cosine(p, q))


def _hellinger(p, q, eps=1e-8):
    p = _replace_zeros(p, eps_val=0.00001)
    q = _replace_zeros(q, eps_val=0.00001)
    return float(hellinger(p, q))


def _bhattacharyya_dist(p, q, eps=1e-8):
    p = _replace_zeros(p, eps_val=0.00001)
    q = _replace_zeros(q, eps_val=0.00001)
    # Normalize for Bhattacharyya (mathematically required)
    p_norm = p / (np.sum(p) + eps)
    q_norm = q / (np.sum(q) + eps)
    bc = np.sum(np.sqrt(p_norm * q_norm))  # Bhattacharyya coefficient
    return float(-np.log(bc + eps))


def _distribution_iou(p, q, eps=1e-8):
    p = _replace_zeros(p, eps_val=0.00001)
    q = _replace_zeros(q, eps_val=0.00001)
    p_norm = p / (np.sum(p) + eps)
    q_norm = q / (np.sum(q) + eps)
    intersection = np.sum(np.minimum(p_norm, q_norm))
    union = np.sum(np.maximum(p_norm, q_norm))
    return float(1.0 - intersection / (union + eps))  # Return as distance


def _gini(x: np.ndarray, eps: float = 1e-12) -> float:
    x = np.asarray(x, dtype=np.float64).flatten()
    x = np.clip(np.nan_to_num(x, nan=0.0), 0.0, None)
    if x.size == 0:
        return float("nan")
    s = float(x.sum())
    if s <= eps:
        return 0.0
    x = np.sort(x)
    n = x.size
    idx = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * np.sum(idx * x) / (n * s)) - (n + 1.0) / n)


def _land_mix_entropy(plan_chw: np.ndarray) -> float:
    comp = plan_chw.mean(axis=(1, 2))  # 20
    p = _normalize_nonneg(comp)
    return float(-np.sum(p * np.log(p + 1e-12)) / (math.log(20.0) + 1e-12))


def _edge_density(dom: np.ndarray) -> float:
    dom = np.asarray(dom)
    if dom.ndim != 2:
        return float("nan")
    dh = float((dom[1:, :] != dom[:-1, :]).mean()) if dom.shape[0] > 1 else 0.0
    dw = float((dom[:, 1:] != dom[:, :-1]).mean()) if dom.shape[1] > 1 else 0.0
    return float(0.5 * (dh + dw))


def _spcorr01(tot: np.ndarray, eps: float = 1e-8) -> float:
    tot = np.asarray(tot, dtype=np.float64)
    tot = np.nan_to_num(tot, nan=0.0, posinf=0.0, neginf=0.0)
    if tot.ndim != 2:
        return float("nan")
    pairs = []
    if tot.shape[0] > 1:
        pairs.append((tot[1:, :].reshape(-1), tot[:-1, :].reshape(-1)))
    if tot.shape[1] > 1:
        pairs.append((tot[:, 1:].reshape(-1), tot[:, :-1].reshape(-1)))
    if not pairs:
        return 0.5
    rs = []
    for a, b in pairs:
        a = a - float(np.mean(a))
        b = b - float(np.mean(b))
        denom = float(np.linalg.norm(a) * np.linalg.norm(b) + eps)
        r = float(np.dot(a, b) / denom) if denom > 0 else 0.0
        r = max(-1.0, min(1.0, r))
        rs.append((r + 1.0) / 2.0)
    return float(np.mean(rs))


# ========== URBAN STRUCTURE METRICS ==========

def _shannon_diversity_by_zone(plan_chw: np.ndarray, zones_hw: np.ndarray) -> float:
    zones_hw = zones_hw.astype(int)
    diversities = []
    for zone_id in range(int(zones_hw.max()) + 1):
        mask = (zones_hw == zone_id)
        if mask.sum() == 0:
            continue
        
        # Get POI distribution in this zone
        zone_pois = plan_chw[:, mask].sum(axis=1)  # Sum across spatial locations
        zone_pois = zone_pois / (zone_pois.sum() + 1e-8)
        
        # Shannon entropy
        entropy = -np.sum(zone_pois * np.log(zone_pois + 1e-8))
        diversities.append(entropy / np.log(20))  # Normalize by max entropy
    
    return float(np.mean(diversities)) if diversities else 0.0


def _simpson_diversity(plan_chw: np.ndarray) -> float:
    comp = plan_chw.mean(axis=(1, 2))  # Average composition
    p = comp / (comp.sum() + 1e-8)
    return float(1.0 - np.sum(p ** 2))


def _jobs_housing_balance(plan_chw: np.ndarray, job_channels=[1, 2, 3, 15, 16], housing_channels=[11]) -> float:
    if plan_chw.shape[0] <= max(max(job_channels), max(housing_channels)):
        return 0.5  # Can't compute, return neutral value
    
    jobs = plan_chw[job_channels].sum()
    housing = plan_chw[housing_channels].sum()
    ratio = jobs / (housing + 1e-8)
    
    # Penalize deviation from ideal range [0.8, 1.5]
    if 0.8 <= ratio <= 1.5:
        balance_score = 1.0
    else:
        balance_score = 1.0 / (1.0 + abs(ratio - 1.15))
    
    return float(balance_score)


def _aggregation_index(plan_dominant_zones: np.ndarray) -> float:
    H, W = plan_dominant_zones.shape
    max_like_adjacencies = 0
    actual_like_adjacencies = 0
    
    for i in range(H):
        for j in range(W):
            current = plan_dominant_zones[i, j]
            neighbors = []
            if i > 0: neighbors.append(plan_dominant_zones[i-1, j])
            if i < H-1: neighbors.append(plan_dominant_zones[i+1, j])
            if j > 0: neighbors.append(plan_dominant_zones[i, j-1])
            if j < W-1: neighbors.append(plan_dominant_zones[i, j+1])
            
            actual_like_adjacencies += sum(n == current for n in neighbors)
            max_like_adjacencies += len(neighbors)
    
    return float(actual_like_adjacencies / max_like_adjacencies if max_like_adjacencies > 0 else 0)


def _mixing_index(plan_chw: np.ndarray, window_size=10) -> float:
    C, H, W = plan_chw.shape
    diversities = []
    
    step = window_size // 2
    for i in range(0, H - window_size, step):
        for j in range(0, W - window_size, step):
            window = plan_chw[:, i:i+window_size, j:j+window_size]
            local_comp = window.sum(axis=(1, 2))
            local_comp = local_comp / (local_comp.sum() + 1e-8)
            
            # Shannon entropy
            entropy = -np.sum(local_comp * np.log(local_comp + 1e-8))
            diversity = entropy / np.log(C)  # Normalize
            diversities.append(diversity)
    
    return float(np.mean(diversities)) if diversities else 0.0



# ========== SPATIAL FRAGMENTATION METRICS ==========


def _patch_density(plan_dominant_zones: np.ndarray) -> float:
    try:
        from skimage.measure import label
    except ImportError:
        # Fallback: estimate fragmentation from transitions
        H, W = plan_dominant_zones.shape
        transitions = 0
        if H > 1:
            transitions += (plan_dominant_zones[1:, :] != plan_dominant_zones[:-1, :]).sum()
        if W > 1:
            transitions += (plan_dominant_zones[:, 1:] != plan_dominant_zones[:, :-1]).sum()
        # Rough approximation: more transitions = more patches
        return float(transitions / plan_dominant_zones.size * 10000 * 0.5)
    
    # Label connected components for each land use type
    n_patches = 0
    for zone_id in range(int(plan_dominant_zones.max()) + 1):
        binary = (plan_dominant_zones == zone_id).astype(int)
        if binary.sum() > 0:  # Only if this zone exists
            labeled = label(binary, connectivity=2)
            n_patches += int(labeled.max())
    
    # Normalize by area
    area = plan_dominant_zones.size
    return float(n_patches / area * 10000)  # Per 10,000 pixels



def _compactness_ratio(plan_chw: np.ndarray, threshold_density=0.1) -> float:
    try:
        from skimage.measure import label, regionprops
    except ImportError:
        # Fallback: Use edge density as inverse proxy for compactness
        total_density = plan_chw.sum(axis=0)
        developed = (total_density > threshold_density * total_density.max()).astype(int)
        
        # Calculate edge ratio
        H, W = developed.shape
        edges = 0
        if H > 1:
            edges += (developed[1:, :] != developed[:-1, :]).sum()
        if W > 1:
            edges += (developed[:, 1:] != developed[:, :-1]).sum()
        
        edge_ratio = edges / (developed.sum() + 1e-8)
        # Convert to compactness (lower edges = more compact)
        return float(max(0.0, min(1.0, 1.0 - edge_ratio)))
    
    # Get developed area
    total_density = plan_chw.sum(axis=0)
    developed = (total_density > threshold_density * total_density.max()).astype(int)
    
    if developed.sum() == 0:
        return 0.0
    
    labeled = label(developed)
    
    if labeled.max() == 0:
        return 0.0
    
    # Get largest connected component
    props = regionprops(labeled)
    if not props:
        return 0.0
    
    largest = max(props, key=lambda x: x.area)
    
    # Compactness = 4π * area / perimeter^2 (circle = 1.0, line = 0.0)
    if largest.perimeter > 0:
        compactness = 4 * np.pi * largest.area / (largest.perimeter ** 2)
    else:
        compactness = 1.0
    
    return float(min(1.0, compactness))



def _shopping_clustering_index(plan_chw: np.ndarray, shopping_channels=[5]) -> float:
    try:
        from skimage.measure import label
    except ImportError:
        # Fallback: Use simple neighborhood analysis
        if plan_chw.shape[0] <= max(shopping_channels):
            return 0.0
        
        shopping = plan_chw[shopping_channels].sum(axis=0)
        if shopping.sum() == 0:
            return 0.0
        
        # Count shopping pixels with shopping neighbors
        binary = (shopping > shopping.mean()).astype(int)
        H, W = binary.shape
        clustered = 0
        total = binary.sum()
        
        for i in range(H):
            for j in range(W):
                if binary[i, j]:
                    neighbors = 0
                    if i > 0 and binary[i-1, j]: neighbors += 1
                    if i < H-1 and binary[i+1, j]: neighbors += 1
                    if j > 0 and binary[i, j-1]: neighbors += 1
                    if j < W-1 and binary[i, j+1]: neighbors += 1
                    if neighbors >= 2:  # Has at least 2 shopping neighbors
                        clustered += 1
        
        return float(clustered / (total + 1e-8))
    
    if plan_chw.shape[0] <= max(shopping_channels):
        return 0.0
    
    shopping = plan_chw[shopping_channels].sum(axis=0)
    
    if shopping.sum() == 0:
        return 0.0
    
    # Use adaptive threshold - shopping areas above median density
    threshold = max(shopping.mean() * 0.5, 1e-6)
    binary_shopping = (shopping > threshold).astype(int)
    
    if binary_shopping.sum() == 0:
        return 0.0
    
    labeled = label(binary_shopping, connectivity=2)
    n_clusters = labeled.max()
    
    if n_clusters == 0:
        return 0.0
    
    # Optimal clustering: not too fragmented, not too concentrated
    total_shopping = binary_shopping.sum()
    avg_cluster_size = total_shopping / (n_clusters + 1e-8)
    
    # Normalize to [0, 1], optimal around 50-100 pixels per cluster
    # Too small = fragmented, too large = monopolistic
    optimal_size = 75
    if avg_cluster_size < optimal_size:
        clustering_score = avg_cluster_size / optimal_size
    else:
        clustering_score = optimal_size / avg_cluster_size
    
    return float(min(1.0, clustering_score))


PRESENCE_MODE_DEFAULT = "argmax"   # "argmax" or "quantile"
PRESENCE_QUANTILE_DEFAULT = 99.5   # used when mode="quantile"

ARGMAX_INTENSITY_QUANTILE_V5_DEFAULT = 90.0   # gate argmax masks by top intensity for the facility group
LOCAL_MIX_WINDOW_V5_DEFAULT = 10              # block size for neighborhood mix entropy
COMPLETENESS_RADIUS_V5_DEFAULT = 30.0         # "15-min" proxy on 100x100 grids (matches v3/v4)
OPPORTUNITY_QUANTILE_V5_DEFAULT = 90.0        # gate gravity opportunities by top intensity
OPPORTUNITY_GAMMA_V5_DEFAULT = 1.10           # mild peak-emphasis (1.0=no change)
BOTTOM_TAIL_PCT_V5_DEFAULT = 10.0             # bottom tail (equity) in percent

ARGMAX_INTENSITY_QUANTILE_V6_DEFAULT = 92.0        # slightly stricter facility gating
LOCAL_MIX_WINDOW_V6_DEFAULT = 12                  # neighborhood window for mix
COMPLETENESS_RADIUS_V6_SMALL_DEFAULT = 20.0       # "local" access radius (cells)
COMPLETENESS_RADIUS_V6_LARGE_DEFAULT = 35.0       # "district" access radius (cells)
OPPORTUNITY_QUANTILE_V6_DEFAULT = 92.0            # gate gravity opportunities a bit more
OPPORTUNITY_GAMMA_V6_DEFAULT = 1.20               # mild peak emphasis (still close to 1)

LOCAL_MIX_WINDOW_V7_DEFAULT = 11               # neighborhood window for mix
ARGMAX_INTENSITY_QUANTILE_V7_DEFAULT = 90.0    # facility gating (between v5 and v6)
OPPORTUNITY_QUANTILE_V7_DEFAULT = 90.0         # opportunity gating (between v5 and v6)
OPPORTUNITY_GAMMA_V7_DEFAULT = 1.12            # mild peak emphasis

FUNCTIONAL_COMPLETENESS_Q_V7 = 85.0            # softer footprint threshold for essential groups
FUNCTIONAL_COMPLETENESS_MINCELLS_V7 = 6        # softer minimum footprint size

BOTTOM_TAIL_PCT_V6_DEFAULT = 10.0                 # bottom tail in percent
BUFFER_CONFLICT_RADIUS_V6_DEFAULT = 3.0           # nuisance buffer radius (cells)

MIN_FACILITY_CELLS_DEFAULT = 5

HOUSING_CHANNELS_DEFAULT = (11,)
RESIDENTIAL_CHANNELS_DEFAULT = (11,)

EQUITY_DIST_CAP_DEFAULT = 25.0      # pixels; neighborhood-scale cap for 100x100 grids
EQUITY_ACCESS_FLOOR_DEFAULT = 0.20  # floor on access score to reduce extreme inequality

def _presence_mask(plan_chw: np.ndarray,
                   channels: List[int],
                   mode: str = PRESENCE_MODE_DEFAULT,
                   q: float = PRESENCE_QUANTILE_DEFAULT,
                   eps: float = 1e-12) -> np.ndarray:
    """Return boolean mask (H,W) indicating where the facility exists."""
    C, H, W = plan_chw.shape
    chs = [int(c) for c in channels if 0 <= int(c) < C]
    if not chs:
        return np.zeros((H, W), dtype=bool)

    if mode == "argmax":
        dom = np.argmax(plan_chw, axis=0)
        return np.isin(dom, chs)

    # mode == "quantile"
    x = plan_chw[chs].sum(axis=0).astype(np.float64, copy=False)
    mx = float(np.max(x))
    if not np.isfinite(mx) or mx <= eps:
        return np.zeros((H, W), dtype=bool)
    thr = float(np.percentile(x, q))
    # Ensure we don't include pure zeros when thr==0 on sparse maps
    return (x >= thr) & (x > eps)

def _distance_to_facility(facility_mask_hw: np.ndarray) -> np.ndarray:
    """Euclidean distance (pixels) to nearest facility cell."""
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError:
        return np.full_like(facility_mask_hw, fill_value=1e6, dtype=np.float64)

    return distance_transform_edt(~facility_mask_hw)

def _weighted_coverage(dist_hw: np.ndarray,
                       pop_hw: np.ndarray,
                       max_dist: float) -> float:
    pop = np.asarray(pop_hw, dtype=np.float64)
    total = float(np.sum(pop)) + 1e-8
    if total <= 1e-8:
        return 0.0
    covered = (dist_hw <= max_dist).astype(np.float64)
    return float(np.sum(covered * pop) / total)

def _weighted_mean(dist_hw: np.ndarray, pop_hw: np.ndarray) -> float:
    pop = np.asarray(pop_hw, dtype=np.float64)
    total = float(np.sum(pop)) + 1e-8
    if total <= 1e-8:
        return 0.0
    return float(np.sum(dist_hw * pop) / total)

def _gini_weighted(values: np.ndarray, weights: np.ndarray, eps: float = 1e-12) -> float:
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    m = (w > eps) & np.isfinite(v)
    if not np.any(m):
        return 0.0
    v = v[m]
    w = w[m]


    if float(np.sum(v * w)) <= eps:
        return 0.0
    # Sort by value
    idx = np.argsort(v)
    v = v[idx]
    w = w[idx]
    cw = np.cumsum(w)
    cw = cw / (cw[-1] + eps)
    cv = np.cumsum(v * w)
    cv = cv / (cv[-1] + eps)
    # Area under Lorenz curve
    B = float(np.trapz(cv, cw))
    g = 1.0 - 2.0 * B
    return float(max(0.0, min(1.0, g)))

def _cosine_similarity(a_hw: np.ndarray, b_hw: np.ndarray, eps: float = 1e-12) -> float:
    """Cosine similarity in [0,1] for nonnegative fields."""
    a = np.asarray(a_hw, dtype=np.float64).reshape(-1)
    b = np.asarray(b_hw, dtype=np.float64).reshape(-1)
    a = np.maximum(a, 0.0); b = np.maximum(b, 0.0)
    na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
    if na <= eps or nb <= eps:
        return 0.0
    c = float(np.dot(a, b) / (na * nb + eps))
    # map [-1,1] -> [0,1] but with nonneg inputs it is already >=0
    return float(max(0.0, min(1.0, c)))

def _normalize01_field(x_hw: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x_hw, dtype=np.float64)
    mx = float(np.max(x)) if x.size else 0.0
    if not np.isfinite(mx) or mx <= eps:
        return np.zeros_like(x, dtype=np.float64)
    return np.clip(x / (mx + eps), 0.0, 1.0)


def _resident_weight_map(plan_chw: np.ndarray,
                         residential_channels: List[int] = list(RESIDENTIAL_CHANNELS_DEFAULT),
                         fallback: str = "activity",
                         eps: float = 1e-12) -> np.ndarray:
    plan = np.asarray(plan_chw, dtype=np.float64)
    plan = np.nan_to_num(plan, nan=0.0, posinf=0.0, neginf=0.0)
    plan = np.clip(plan, 0.0, None)

    C, H, W = plan.shape
    rchs = [c for c in (residential_channels or []) if 0 <= int(c) < C]
    if rchs:
        pop = plan[rchs].sum(axis=0)
        if float(np.sum(pop)) > eps:
            return pop

    if fallback == "uniform":
        return np.ones((H, W), dtype=np.float64)

    # fallback == "activity"
    tot = plan.sum(axis=0)
    s = float(np.sum(tot))
    if s <= eps:
        return np.ones((H, W), dtype=np.float64)
    return tot



def _gravity_accessibility(plan_chw: np.ndarray,
                           target_channels: List[int],
                           residential_channels: List[int],
                           sigma: float = 7.0,
                           eps: float = 1e-12,
                           residential_fallback: str = "activity") -> Dict[str, Any]:
    plan = np.asarray(plan_chw, dtype=np.float64)
    plan = np.nan_to_num(plan, nan=0.0, posinf=0.0, neginf=0.0)
    plan = np.clip(plan, 0.0, None)

    C = plan.shape[0]
    tchs = [c for c in (target_channels or []) if 0 <= int(c) < C]
    if not tchs:
        return {"access_mean": 0.0, "access_gini": 0.0, "access_match": 0.0,
                "access_surface_stats": {"max": 0.0, "mean": 0.0}}

    opp = plan[tchs].sum(axis=0).astype(np.float64, copy=False)
    pop = _resident_weight_map(plan, residential_channels=residential_channels,
                               fallback=residential_fallback, eps=eps)
    pop_sum = float(np.sum(pop))
    if pop_sum <= eps:
        # truly empty / degenerate plan
        return {"access_mean": 0.0, "access_gini": 0.0, "access_match": 0.0,
                "access_surface_stats": {"max": float(np.max(opp)), "mean": float(np.mean(opp))}}

    try:
        from scipy.ndimage import gaussian_filter
        acc = gaussian_filter(opp, sigma=float(sigma))
    except Exception:
        acc = opp

    acc01 = _normalize01_field(acc, eps=eps)

    # Demand-weighted mean access in [0,1]
    access_mean = float(np.sum(acc01 * pop) / (pop_sum + eps))

    # Inequality of access among residents in [0,1] (0=perfect equality)
    access_gini = float(_gini_weighted(acc01, pop, eps=eps))

    # Spatial match between demand and access hot spots
    access_match = float(_cosine_similarity(pop, acc01, eps=eps))

    return {
        "access_mean": float(max(0.0, min(1.0, access_mean))),
        "access_gini": float(max(0.0, min(1.0, access_gini))),
        "access_match": float(max(0.0, min(1.0, access_match))),
        "access_surface_stats": {"max": float(np.max(acc01)), "mean": float(np.mean(acc01))}
    }


def _weighted_percentile(values: np.ndarray,
                         weights: np.ndarray,
                         pct: float,
                         eps: float = 1e-12) -> float:
    """Weighted percentile (0..100) for 1D arrays."""
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    m = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not np.any(m):
        return float("nan")
    v = v[m]
    w = w[m]
    order = np.argsort(v)
    v = v[order]
    w = w[order]
    cw = np.cumsum(w)
    tot = float(cw[-1])
    if tot <= eps:
        return float("nan")
    target = (float(pct) / 100.0) * tot
    idx = int(np.searchsorted(cw, target, side="left"))
    idx = min(max(idx, 0), v.size - 1)
    return float(v[idx])


def _presence_mask_v5(plan_chw: np.ndarray,
                      channels: List[int],
                      mode: str = "argmax",
                      q: float = PRESENCE_QUANTILE_DEFAULT,
                      argmax_intensity_q: float = ARGMAX_INTENSITY_QUANTILE_V5_DEFAULT,
                      eps: float = 1e-12) -> np.ndarray:
    plan = np.asarray(plan_chw, dtype=np.float64)
    plan = np.nan_to_num(plan, nan=0.0, posinf=0.0, neginf=0.0)
    plan = np.clip(plan, 0.0, None)

    C, H, W = plan.shape
    chs = [int(c) for c in (channels or []) if 0 <= int(c) < C]
    if not chs:
        return np.zeros((H, W), dtype=bool)

    if mode != "argmax":
        return _presence_mask(plan, chs, mode=mode, q=q, eps=eps)

    dom = np.argmax(plan, axis=0)
    x = plan[chs].sum(axis=0).astype(np.float64, copy=False)
    vals = x[x > eps]
    if vals.size == 0:
        return np.zeros((H, W), dtype=bool)
    thr = float(np.percentile(vals, float(argmax_intensity_q)))
    return (np.isin(dom, chs)) & (x >= thr) & (x > eps)


def _cell_crispness(plan_chw: np.ndarray,
                    exclude_channels: Optional[List[int]] = None,
                    eps: float = 1e-12) -> float:
    plan = np.asarray(plan_chw, dtype=np.float64)
    plan = np.nan_to_num(plan, nan=0.0, posinf=0.0, neginf=0.0)
    plan = np.clip(plan, 0.0, None)
    C, H, W = plan.shape
    excl = set(int(c) for c in (exclude_channels or [0]) if 0 <= int(c) < C)
    idx = [c for c in range(C) if c not in excl]
    if not idx:
        return 0.0
    x = plan[idx]
    s = np.sum(x, axis=0)
    m = s > eps
    if not np.any(m):
        return 0.0
    pmax = np.max(x, axis=0) / (s + eps)
    return float(np.mean(pmax[m]))


def _functional_completeness(plan_chw: np.ndarray,
                             group_channels: Dict[str, List[int]],
                             q: float = 90.0,
                             min_cells: int = 10,
                             eps: float = 1e-12) -> float:
    plan = np.asarray(plan_chw, dtype=np.float64)
    plan = np.nan_to_num(plan, nan=0.0, posinf=0.0, neginf=0.0)
    plan = np.clip(plan, 0.0, None)
    C, H, W = plan.shape

    ok = 0
    total = 0
    for _, chs in (group_channels or {}).items():
        chs2 = [int(c) for c in chs if 0 <= int(c) < C]
        if not chs2:
            continue
        total += 1
        x = plan[chs2].sum(axis=0).astype(np.float64, copy=False)
        vals = x[x > eps]
        if vals.size == 0:
            continue
        thr = float(np.percentile(vals, float(q)))
        mask = (x >= thr) & (x > eps)
        if int(mask.sum()) >= int(min_cells):
            ok += 1
    if total <= 0:
        return 0.0
    return float(ok) / float(total)


def _local_mix_entropy(plan_chw: np.ndarray,
                       window: int = LOCAL_MIX_WINDOW_V5_DEFAULT,
                       exclude_channels: Optional[List[int]] = None,
                       eps: float = 1e-12) -> Dict[str, float]:
    plan = np.asarray(plan_chw, dtype=np.float64)
    plan = np.nan_to_num(plan, nan=0.0, posinf=0.0, neginf=0.0)
    plan = np.clip(plan, 0.0, None)
    C, H, W = plan.shape

    excl = set(int(c) for c in (exclude_channels or [0]) if 0 <= int(c) < C)
    idx = [c for c in range(C) if c not in excl]
    if not idx:
        return {"mean": 0.0, "high_share": 0.0}

    x = plan[idx]
    w = int(window)
    if H % w != 0 or W % w != 0:
        # fallback: coarse resize by truncation
        Hb = H // w
        Wb = W // w
        x = x[:, :Hb*w, :Wb*w]
        H = Hb*w
        W = Wb*w

    Hb = H // w
    Wb = W // w
    # (C, Hb, w, Wb, w) -> sum over block pixels -> (C, Hb, Wb)
    xb = x.reshape((x.shape[0], Hb, w, Wb, w)).sum(axis=(2, 4))
    s = np.sum(xb, axis=0)  # (Hb, Wb)
    m = s > eps
    if not np.any(m):
        return {"mean": 0.0, "high_share": 0.0}
    p = xb / (s[None, :, :] + eps)
    # entropy normalized by log(K)
    K = float(p.shape[0])
    ent = -np.sum(p * np.log(p + eps), axis=0) / (math.log(K + eps))
    ent = np.clip(ent, 0.0, 1.0)
    ent_m = ent[m]
    mean_ent = float(np.mean(ent_m)) if ent_m.size else 0.0
    # "very high" entropy threshold: 0.85 is mild (not too strict)
    high_share = float(np.mean(ent_m > 0.85)) if ent_m.size else 0.0
    return {"mean": mean_ent, "high_share": high_share}


def _largest_component_share(mask_hw: np.ndarray) -> float:
    m = np.asarray(mask_hw, dtype=bool)
    tot = int(m.sum())
    if tot <= 0:
        return 0.0
    try:
        from scipy.ndimage import label
        lab, num = label(m)
        if num <= 0:
            return 0.0
        # sizes
        sizes = np.bincount(lab.reshape(-1))
        # label 0 is background
        if sizes.size <= 1:
            return 0.0
        lcc = int(np.max(sizes[1:]))
        return float(lcc) / float(tot)
    except Exception:
        from collections import deque
        H, W = m.shape
        visited = np.zeros((H, W), dtype=bool)
        best = 0
        for y in range(H):
            for x in range(W):
                if (not m[y, x]) or visited[y, x]:
                    continue
                q = deque()
                q.append((y, x))
                visited[y, x] = True
                cnt = 0
                while q:
                    cy, cx = q.popleft()
                    cnt += 1
                    for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                        if 0 <= ny < H and 0 <= nx < W and m[ny, nx] and (not visited[ny, nx]):
                            visited[ny, nx] = True
                            q.append((ny, nx))
                if cnt > best:
                    best = cnt
        return float(best) / float(tot)


def _green_connectivity(plan_chw: np.ndarray,
                        green_channels: List[int],
                        q: float = 90.0,
                        min_cells: int = 25,
                        eps: float = 1e-12) -> float:
    plan = np.asarray(plan_chw, dtype=np.float64)
    plan = np.nan_to_num(plan, nan=0.0, posinf=0.0, neginf=0.0)
    plan = np.clip(plan, 0.0, None)
    C, H, W = plan.shape
    chs = [int(c) for c in (green_channels or []) if 0 <= int(c) < C]
    if not chs:
        return 0.0
    x = plan[chs].sum(axis=0)
    vals = x[x > eps]
    if vals.size == 0:
        return 0.0
    thr = float(np.percentile(vals, float(q)))
    mask = (x >= thr) & (x > eps)
    if int(mask.sum()) < int(min_cells):
        # too few cells to form a meaningful network
        return 0.0
    return float(_largest_component_share(mask))


def _co_location_corr(plan_chw: np.ndarray,
                      group_a: List[int],
                      group_b: List[int],
                      sigma: float = 3.0,
                      eps: float = 1e-12) -> float:
    plan = np.asarray(plan_chw, dtype=np.float64)
    plan = np.nan_to_num(plan, nan=0.0, posinf=0.0, neginf=0.0)
    plan = np.clip(plan, 0.0, None)
    C, H, W = plan.shape
    a = [int(c) for c in (group_a or []) if 0 <= int(c) < C]
    b = [int(c) for c in (group_b or []) if 0 <= int(c) < C]
    if not a or not b:
        return 0.5
    xa = plan[a].sum(axis=0)
    xb = plan[b].sum(axis=0)
    try:
        from scipy.ndimage import gaussian_filter
        xa = gaussian_filter(xa, sigma=float(sigma))
        xb = gaussian_filter(xb, sigma=float(sigma))
    except Exception:
        pass
    va = xa.reshape(-1)
    vb = xb.reshape(-1)
    ma = np.isfinite(va) & np.isfinite(vb)
    if not np.any(ma):
        return 0.5
    va = va[ma]
    vb = vb[ma]
    sa = float(np.std(va))
    sb = float(np.std(vb))
    if sa <= eps or sb <= eps:
        return 0.5
    r = float(np.corrcoef(va, vb)[0, 1])
    if not np.isfinite(r):
        return 0.5
    # map [-1,1] -> [0,1]
    return float(max(0.0, min(1.0, 0.5 * (r + 1.0))))


def _conflict_rate(plan_chw: np.ndarray,
                   eps: float = 1e-12) -> float:
    plan = np.asarray(plan_chw, dtype=np.float64)
    plan = np.nan_to_num(plan, nan=0.0, posinf=0.0, neginf=0.0)
    plan = np.clip(plan, 0.0, None)
    C, H, W = plan.shape

    # Coarse groups (Beijing POI mapping)
    ROAD = [0]
    HOUSING = [11]
    SCHOOLS = [13]
    HEALTH = [8]
    TRANSIT = [14]
    GREEN = [7, 10]
    SERVICES = [4, 5, 6, 12, 19]
    AUTO_NUIS = [1, 2, 3]         # car/motorbike service/repair
    JOBS_OTHER = [15, 16]         # finance/company as office-like

    groups = [
        ("road", ROAD),
        ("housing", HOUSING),
        ("schools", SCHOOLS),
        ("health", HEALTH),
        ("green", GREEN),
        ("services", SERVICES),
        ("transit", TRANSIT),
        ("auto_nuis", AUTO_NUIS),
        ("jobs_other", JOBS_OTHER),
    ]

    maps = []
    for _, chs in groups:
        chs2 = [int(c) for c in chs if 0 <= int(c) < C]
        if not chs2:
            maps.append(np.zeros((H, W), dtype=np.float64))
        else:
            maps.append(plan[chs2].sum(axis=0))
    maps = np.stack(maps, axis=0)  # (G,H,W)
    lab = np.argmax(maps, axis=0).astype(np.int32)

    auto_idx = 7
    incompatible = {(auto_idx, 1), (auto_idx, 2), (auto_idx, 3)}
    incompatible |= {(b, a) for (a, b) in incompatible}

    a = lab[:, :-1]
    b = lab[:, 1:]
    c = lab[:-1, :]
    d = lab[1:, :]

    total_pairs = float(a.size + c.size)
    if total_pairs <= eps:
        return 0.0

    bad = 0.0
    for u, v in incompatible:
        bad += float(np.sum((a == u) & (b == v)))
        bad += float(np.sum((c == u) & (d == v)))
    return float(bad / total_pairs)


def _resident_completeness_metrics(plan_chw: np.ndarray,
                                  zones_hw: Optional[np.ndarray],
                                  radius: float = COMPLETENESS_RADIUS_V5_DEFAULT,
                                  facility_q: float = ARGMAX_INTENSITY_QUANTILE_V5_DEFAULT,
                                  eps: float = 1e-12) -> Dict[str, float]:
    plan = np.asarray(plan_chw, dtype=np.float64)
    plan = np.nan_to_num(plan, nan=0.0, posinf=0.0, neginf=0.0)
    plan = np.clip(plan, 0.0, None)
    C, H, W = plan.shape

    RES = [11]
    JOBS = [1, 2, 3, 15, 16]
    SERVICES = [4, 5, 6, 12, 19]
    SCHOOL = [13]
    HEALTH = [8]
    TRANSIT = [14]
    GREEN = [7, 10]

    pop = _resident_weight_map(plan, residential_channels=RES, fallback="activity", eps=eps)
    pop_sum = float(np.sum(pop))
    if pop_sum <= eps:
        return {"completeness_mean": 0.0, "complete_coverage": 0.0,
                "svc_dist_p90": float("nan"), "job_dist_p90": float("nan"),
                "zone_cv": float("nan")}

    svc_m = _presence_mask_v5(plan, SERVICES, mode="argmax", argmax_intensity_q=facility_q, eps=eps)
    sch_m = _presence_mask_v5(plan, SCHOOL, mode="argmax", argmax_intensity_q=facility_q, eps=eps)
    hlt_m = _presence_mask_v5(plan, HEALTH, mode="argmax", argmax_intensity_q=facility_q, eps=eps)
    trn_m = _presence_mask_v5(plan, TRANSIT, mode="argmax", argmax_intensity_q=facility_q, eps=eps)
    grn_m = _presence_mask_v5(plan, GREEN, mode="argmax", argmax_intensity_q=facility_q, eps=eps)
    job_m = _presence_mask_v5(plan, JOBS, mode="argmax", argmax_intensity_q=facility_q, eps=eps)

    # distance fields
    svc_d = _distance_to_facility(svc_m)
    job_d = _distance_to_facility(job_m)
    sch_d = _distance_to_facility(sch_m)
    hlt_d = _distance_to_facility(hlt_m)
    trn_d = _distance_to_facility(trn_m)
    grn_d = _distance_to_facility(grn_m)

    R = float(radius)
    within = [
        (svc_d <= R),
        (sch_d <= R),
        (hlt_d <= R),
        (trn_d <= R),
        (grn_d <= R),
    ]
    K = float(len(within))
    count = np.zeros((H, W), dtype=np.float64)
    for w in within:
        count += w.astype(np.float64)
    frac = count / max(K, 1.0)  # (H,W)
    all_ok = (count >= K - 1e-6)

    completeness_mean = float(np.sum(frac * pop) / (pop_sum + eps))
    complete_coverage = float(np.sum(all_ok.astype(np.float64) * pop) / (pop_sum + eps))

    svc_p90 = _weighted_percentile(svc_d, pop, 90.0)
    job_p90 = _weighted_percentile(job_d, pop, 90.0)

    zone_cv = float("nan")
    if zones_hw is not None:
        z = np.asarray(zones_hw)
        if z.shape == (H, W):
            zs = [int(v) for v in np.unique(z) if np.isfinite(v)]
            means = []
            for zz in zs:
                m = (z == zz)
                w = pop[m]
                sw = float(np.sum(w))
                if sw <= eps:
                    continue
                means.append(float(np.sum(frac[m] * w) / (sw + eps)))
            if len(means) >= 2:
                mu = float(np.mean(means))
                sd = float(np.std(means))
                if mu > eps:
                    zone_cv = float(sd / mu)

    return {"completeness_mean": completeness_mean,
            "complete_coverage": complete_coverage,
            "svc_dist_p90": float(svc_p90),
            "job_dist_p90": float(job_p90),
            "zone_cv": float(zone_cv)}


def _tod_synergy(plan_chw: np.ndarray,
                 radius: float = COMPLETENESS_RADIUS_V5_DEFAULT,
                 facility_q: float = ARGMAX_INTENSITY_QUANTILE_V5_DEFAULT,
                 eps: float = 1e-12) -> float:
    plan = np.asarray(plan_chw, dtype=np.float64)
    plan = np.nan_to_num(plan, nan=0.0, posinf=0.0, neginf=0.0)
    plan = np.clip(plan, 0.0, None)
    C, H, W = plan.shape

    HOUSING = [11]
    JOBS = [1, 2, 3, 15, 16]
    TRANSIT = [14]

    trn_m = _presence_mask_v5(plan, TRANSIT, mode="argmax", argmax_intensity_q=facility_q, eps=eps)
    trn_d = _distance_to_facility(trn_m)
    near = (trn_d <= float(radius))

    hj = plan[HOUSING].sum(axis=0) + plan[JOBS].sum(axis=0)
    tot = float(np.sum(hj))
    if tot <= eps:
        return 0.0
    return float(np.sum(hj * near.astype(np.float64)) / (tot + eps))


def _gravity_accessibility_v5(plan_chw: np.ndarray,
                              target_channels: List[int],
                              residential_channels: List[int],
                              sigma: float = 7.0,
                              opp_q: float = OPPORTUNITY_QUANTILE_V5_DEFAULT,
                              gamma: float = OPPORTUNITY_GAMMA_V5_DEFAULT,
                              bottom_pct: float = BOTTOM_TAIL_PCT_V5_DEFAULT,
                              eps: float = 1e-12,
                              residential_fallback: str = "activity") -> Dict[str, Any]:
    plan = np.asarray(plan_chw, dtype=np.float64)
    plan = np.nan_to_num(plan, nan=0.0, posinf=0.0, neginf=0.0)
    plan = np.clip(plan, 0.0, None)
    C = plan.shape[0]
    tchs = [c for c in (target_channels or []) if 0 <= int(c) < C]
    if not tchs:
        return {"access_mean": 0.0, "access_gini": 0.0, "access_match": 0.0, "access_bottom": 0.0}

    opp = plan[tchs].sum(axis=0).astype(np.float64, copy=False)
    pop = _resident_weight_map(plan, residential_channels=residential_channels,
                               fallback=residential_fallback, eps=eps)
    pop_sum = float(np.sum(pop))
    if pop_sum <= eps:
        return {"access_mean": 0.0, "access_gini": 0.0, "access_match": 0.0, "access_bottom": 0.0}

    opp01 = _normalize01_field(opp, eps=eps)
    gam = float(gamma) if (gamma is not None) else 1.0
    if gam > 1.0:
        opp01 = np.power(opp01, gam)

    vals = opp01[opp01 > eps]
    if vals.size > 0:
        thr = float(np.percentile(vals, float(opp_q)))
        opp01 = opp01 * (opp01 >= thr)

    try:
        from scipy.ndimage import gaussian_filter
        acc = gaussian_filter(opp01, sigma=float(sigma))
    except Exception:
        acc = opp01

    acc01 = _normalize01_field(acc, eps=eps)

    access_mean = float(np.sum(acc01 * pop) / (pop_sum + eps))
    access_gini = float(_gini_weighted(acc01.reshape(-1), pop.reshape(-1)))
    access_match = float(_cosine_similarity(acc01, pop, eps=eps))

    # bottom-tail mean access among residents
    v = acc01.reshape(-1)
    w = pop.reshape(-1)
    m = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not np.any(m):
        bottom_mean = 0.0
    else:
        v = v[m]
        w = w[m]
        order = np.argsort(v)
        v = v[order]
        w = w[order]
        cw = np.cumsum(w)
        tot = float(cw[-1])
        cut = (float(bottom_pct) / 100.0) * tot
        sel = cw <= cut
        if not np.any(sel):
            sel = np.zeros_like(cw, dtype=bool)
            sel[0] = True
        bottom_mean = float(np.sum(v[sel] * w[sel]) / (float(np.sum(w[sel])) + eps))

    return {"access_mean": access_mean, "access_gini": access_gini,
            "access_match": access_match, "access_bottom": float(bottom_mean)}


def _weighted_bottom_mean(values: np.ndarray,
                          weights: np.ndarray,
                          bottom_pct: float,
                          eps: float = 1e-12) -> float:
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    m = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not np.any(m):
        return float("nan")
    v = v[m]
    w = w[m]
    thr = _weighted_percentile(v, w, float(bottom_pct), eps=eps)
    if not np.isfinite(thr):
        return float("nan")
    sel = v <= thr
    sw = float(np.sum(w[sel]))
    if sw <= eps:
        return float("nan")
    return float(np.sum(v[sel] * w[sel]) / (sw + eps))


def _resident_completeness_metrics_v6(plan_chw: np.ndarray,
                                      zones_hw: Optional[np.ndarray],
                                      radius: float,
                                      facility_q: float = ARGMAX_INTENSITY_QUANTILE_V6_DEFAULT,
                                      bottom_pct: float = BOTTOM_TAIL_PCT_V6_DEFAULT,
                                      eps: float = 1e-12) -> Dict[str, float]:
    plan = np.asarray(plan_chw, dtype=np.float64)
    plan = np.nan_to_num(plan, nan=0.0, posinf=0.0, neginf=0.0)
    plan = np.clip(plan, 0.0, None)
    C, H, W = plan.shape

    RES = [11]
    JOBS = [1, 2, 3, 15, 16]
    SERVICES = [4, 5, 6, 12, 19]
    SCHOOL = [13]
    HEALTH = [8]
    TRANSIT = [14]
    GREEN = [7, 10]

    pop = _resident_weight_map(plan, residential_channels=RES, fallback="activity", eps=eps)
    pop_sum = float(np.sum(pop))
    if pop_sum <= eps:
        return {"completeness_mean": 0.0, "complete_coverage": 0.0, "completeness_bottom": 0.0,
                "svc_dist_p90": float("nan"), "svc_dist_p95": float("nan"),
                "job_dist_p90": float("nan"), "job_dist_p95": float("nan"),
                "zone_cv": float("nan")}

    svc_m = _presence_mask_v5(plan, SERVICES, mode="argmax", argmax_intensity_q=facility_q, eps=eps)
    sch_m = _presence_mask_v5(plan, SCHOOL, mode="argmax", argmax_intensity_q=facility_q, eps=eps)
    hlt_m = _presence_mask_v5(plan, HEALTH, mode="argmax", argmax_intensity_q=facility_q, eps=eps)
    trn_m = _presence_mask_v5(plan, TRANSIT, mode="argmax", argmax_intensity_q=facility_q, eps=eps)
    grn_m = _presence_mask_v5(plan, GREEN, mode="argmax", argmax_intensity_q=facility_q, eps=eps)
    job_m = _presence_mask_v5(plan, JOBS, mode="argmax", argmax_intensity_q=facility_q, eps=eps)

    svc_d = _distance_to_facility(svc_m)
    job_d = _distance_to_facility(job_m)
    sch_d = _distance_to_facility(sch_m)
    hlt_d = _distance_to_facility(hlt_m)
    trn_d = _distance_to_facility(trn_m)
    grn_d = _distance_to_facility(grn_m)

    R = float(radius)
    within = [(svc_d <= R), (sch_d <= R), (hlt_d <= R), (trn_d <= R), (grn_d <= R)]
    K = float(len(within))
    count = np.zeros((H, W), dtype=np.float64)
    for w in within:
        count += w.astype(np.float64)

    frac = count / max(K, 1.0)
    all_ok = (count >= K - 1e-6)

    completeness_mean = float(np.sum(frac * pop) / (pop_sum + eps))
    complete_coverage = float(np.sum(all_ok.astype(np.float64) * pop) / (pop_sum + eps))
    completeness_bottom = float(_weighted_bottom_mean(frac, pop, bottom_pct=float(bottom_pct), eps=eps))

    k2_coverage = float(np.sum((count >= 2.0).astype(np.float64) * pop) / (pop_sum + eps))

    k3_coverage = float(np.sum((count >= 3.0).astype(np.float64) * pop) / (pop_sum + eps))
    k4_coverage = float(np.sum((count >= 4.0).astype(np.float64) * pop) / (pop_sum + eps))
    underserved_share = float(np.sum((count <= 2.0).astype(np.float64) * pop) / (pop_sum + eps))

    svc_p75 = _weighted_percentile(svc_d, pop, 75.0)
    svc_p90 = _weighted_percentile(svc_d, pop, 90.0)
    svc_p95 = _weighted_percentile(svc_d, pop, 95.0)
    job_p75 = _weighted_percentile(job_d, pop, 75.0)
    job_p90 = _weighted_percentile(job_d, pop, 90.0)
    job_p95 = _weighted_percentile(job_d, pop, 95.0)

    zone_cv = float("nan")
    if zones_hw is not None:
        z = np.asarray(zones_hw)
        if z.shape == (H, W):
            zs = [int(v) for v in np.unique(z) if np.isfinite(v)]
            means = []
            for zz in zs:
                mm = (z == zz)
                ww = pop[mm]
                sw = float(np.sum(ww))
                if sw <= eps:
                    continue
                means.append(float(np.sum(frac[mm] * ww) / (sw + eps)))
            if len(means) >= 2:
                mu = float(np.mean(means))
                sd = float(np.std(means))
                if mu > eps:
                    zone_cv = float(sd / mu)

    return {"completeness_mean": completeness_mean,
            "complete_coverage": complete_coverage,
            "completeness_bottom": completeness_bottom,
            "k2_coverage": float(k2_coverage),
            "k3_coverage": float(k3_coverage),
            "k4_coverage": float(k4_coverage),
            "underserved_share": float(underserved_share),
            "svc_dist_p75": float(svc_p75),
            "svc_dist_p90": float(svc_p90),
            "svc_dist_p95": float(svc_p95),
            "job_dist_p75": float(job_p75),
            "job_dist_p90": float(job_p90),
            "job_dist_p95": float(job_p95),
            "zone_cv": float(zone_cv)}


def _facility_area_share_v6(plan_chw: np.ndarray,
                            channels: List[int],
                            facility_q: float = ARGMAX_INTENSITY_QUANTILE_V6_DEFAULT,
                            eps: float = 1e-12) -> float:
    plan = np.asarray(plan_chw, dtype=np.float64)
    plan = np.nan_to_num(plan, nan=0.0, posinf=0.0, neginf=0.0)
    plan = np.clip(plan, 0.0, None)
    mask = _presence_mask_v5(plan, channels, mode="argmax", argmax_intensity_q=facility_q, eps=eps)
    return float(np.mean(mask.astype(np.float64)))


def _supply_demand_ratio_v6(plan_chw: np.ndarray,
                            supply_channels: List[int],
                            demand_channels: List[int],
                            eps: float = 1e-12) -> float:
    plan = np.asarray(plan_chw, dtype=np.float64)
    plan = np.nan_to_num(plan, nan=0.0, posinf=0.0, neginf=0.0)
    plan = np.clip(plan, 0.0, None)
    C = plan.shape[0]
    sup = [int(c) for c in (supply_channels or []) if 0 <= int(c) < C]
    dem = [int(c) for c in (demand_channels or []) if 0 <= int(c) < C]
    if not sup or not dem:
        return float("nan")
    s = float(np.sum(plan[sup]))
    d = float(np.sum(plan[dem]))
    return float(s / (d + eps))


def _weighted_centroid_from_map(x_hw: np.ndarray,
                                eps: float = 1e-12) -> Tuple[float, float]:
    x = np.asarray(x_hw, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.clip(x, 0.0, None)
    H, W = x.shape
    tot = float(np.sum(x))
    if tot <= eps:
        return (0.5 * (H - 1), 0.5 * (W - 1))
    ys, xs = np.mgrid[0:H, 0:W]
    cy = float(np.sum(ys * x) / (tot + eps))
    cx = float(np.sum(xs * x) / (tot + eps))
    return (cy, cx)


def _centroid_distance_v6(plan_chw: np.ndarray,
                          channels_a: List[int],
                          channels_b: List[int],
                          eps: float = 1e-12) -> float:
    """Distance between weighted centroids of two groups (cells)."""
    plan = np.asarray(plan_chw, dtype=np.float64)
    plan = np.nan_to_num(plan, nan=0.0, posinf=0.0, neginf=0.0)
    plan = np.clip(plan, 0.0, None)
    C, H, W = plan.shape
    a = [int(c) for c in (channels_a or []) if 0 <= int(c) < C]
    b = [int(c) for c in (channels_b or []) if 0 <= int(c) < C]
    if not a or not b:
        return float("nan")
    xa = plan[a].sum(axis=0)
    xb = plan[b].sum(axis=0)
    ya, xa0 = _weighted_centroid_from_map(xa, eps=eps)
    yb, xb0 = _weighted_centroid_from_map(xb, eps=eps)
    return float(math.sqrt((ya - yb) ** 2 + (xa0 - xb0) ** 2))


def _polycentricity_v6(plan_chw: np.ndarray,
                       channels: List[int],
                       bins: int = 10,
                       topk: int = 5,
                       eps: float = 1e-12) -> float:
    plan = np.asarray(plan_chw, dtype=np.float64)
    plan = np.nan_to_num(plan, nan=0.0, posinf=0.0, neginf=0.0)
    plan = np.clip(plan, 0.0, None)
    C, H, W = plan.shape
    chs = [int(c) for c in (channels or []) if 0 <= int(c) < C]
    if not chs:
        return 0.0
    x = plan[chs].sum(axis=0)
    tot = float(np.sum(x))
    if tot <= eps:
        return 0.0
    b = int(bins)
    bh = max(1, H // b)
    bw = max(1, W // b)
    H2 = (H // bh) * bh
    W2 = (W // bw) * bw
    x = x[:H2, :W2]
    xb = x.reshape((H2 // bh, bh, W2 // bw, bw)).sum(axis=(1, 3))
    v = xb.reshape(-1)
    v = v[np.isfinite(v)]
    if v.size <= 0:
        return 0.0
    k = int(max(1, min(topk, v.size)))
    vv = np.sort(v)[::-1][:k]
    s = float(np.sum(vv))
    if s <= eps:
        return 0.0
    return float(max(0.0, min(1.0, 1.0 - float(vv[0]) / (s + eps))))


def _buffer_conflict_rate_v6(plan_chw: np.ndarray,
                             sensitive_channels: List[int],
                             nuisance_channels: List[int],
                             radius: float = BUFFER_CONFLICT_RADIUS_V6_DEFAULT,
                             nuisance_q: float = ARGMAX_INTENSITY_QUANTILE_V6_DEFAULT,
                             eps: float = 1e-12) -> float:
    plan = np.asarray(plan_chw, dtype=np.float64)
    plan = np.nan_to_num(plan, nan=0.0, posinf=0.0, neginf=0.0)
    plan = np.clip(plan, 0.0, None)
    C, H, W = plan.shape

    sens = [int(c) for c in (sensitive_channels or []) if 0 <= int(c) < C]
    nuis = [int(c) for c in (nuisance_channels or []) if 0 <= int(c) < C]
    if not sens or not nuis:
        return 0.0

    s_map = plan[sens].sum(axis=0)
    s_tot = float(np.sum(s_map))
    if s_tot <= eps:
        return 0.0

    nuis_mask = _presence_mask_v5(plan, nuis, mode="argmax", argmax_intensity_q=float(nuisance_q), eps=eps)
    d = _distance_to_facility(nuis_mask)
    near = (d <= float(radius)).astype(np.float64)
    return float(np.sum(s_map * near) / (s_tot + eps))


def _access_inequality(plan_chw: np.ndarray,
                       facility_channels: List[int],
                       residential_channels: List[int],
                       presence_mode: str = PRESENCE_MODE_DEFAULT,
                       q: float = PRESENCE_QUANTILE_DEFAULT,
                       min_facility_cells: int = MIN_FACILITY_CELLS_DEFAULT) -> float:
    C = plan_chw.shape[0]
    if C <= max(max(facility_channels or [0]), max(residential_channels or [0])):
        return 1.0

    fac = _presence_mask(plan_chw, facility_channels, mode=presence_mode, q=q)

    fac_n = int(np.sum(fac))
    if fac_n < int(min_facility_cells):
        q_fallback = float(min(99.5, max(97.5, q - 1.0)))
        fac_fb = _presence_mask(plan_chw, facility_channels, mode="quantile", q=q_fallback)
        if int(np.sum(fac_fb)) > fac_n:
            fac = fac_fb

    res = plan_chw[residential_channels].sum(axis=0).astype(np.float64, copy=False)
    dist = _distance_to_facility(fac).astype(np.float64, copy=False)

    dist_cap = float(EQUITY_DIST_CAP_DEFAULT)
    dist = np.minimum(dist, dist_cap)

    access_lin = 1.0 - (dist / (dist_cap + 1e-12))
    access_lin = np.clip(access_lin, 0.0, 1.0)
    floor = float(EQUITY_ACCESS_FLOOR_DEFAULT)
    access = floor + (1.0 - floor) * access_lin
    return _gini_weighted(access, res)



def _service_coverage(plan_chw: np.ndarray,
                      service_channels=[4, 6],
                      radius=10,
                      presence_mode: str = PRESENCE_MODE_DEFAULT,
                      q: float = PRESENCE_QUANTILE_DEFAULT,
                      min_facility_cells: int = MIN_FACILITY_CELLS_DEFAULT) -> float:
    if plan_chw.shape[0] <= max(service_channels):
        return 0.0

    fac = _presence_mask(plan_chw, service_channels, mode=presence_mode, q=q)
    if int(np.sum(fac)) < int(min_facility_cells):
        return 0.0

    dist = _distance_to_facility(fac)
    covered = (dist <= radius)
    return float(np.mean(covered))


def _spatial_gini(plan_chw: np.ndarray,
                  service_channels=[4, 6],
                  presence_mode: str = PRESENCE_MODE_DEFAULT,
                  q: float = PRESENCE_QUANTILE_DEFAULT) -> float:
    if plan_chw.shape[0] <= max(service_channels):
        return 0.5

    x = plan_chw[service_channels].sum(axis=0).astype(np.float64, copy=False)
    x = np.maximum(x, 0.0).reshape(-1)
    s = float(np.sum(x))
    if s <= 1e-12:
        return 1.0

    x = np.sort(x)
    n = x.size
    idx = np.arange(1, n + 1, dtype=np.float64)
    g = (2.0 * float(np.sum(idx * x))) / (n * s + 1e-12) - (n + 1.0) / n
    return float(max(0.0, min(1.0, g)))


def _avg_nearest_service_distance(plan_chw: np.ndarray,
                                  service_channels=[4, 6],
                                  residential_channels=[11],
                                  presence_mode: str = PRESENCE_MODE_DEFAULT,
                                  q: float = PRESENCE_QUANTILE_DEFAULT,
                                  min_facility_cells: int = MIN_FACILITY_CELLS_DEFAULT) -> float:
    if plan_chw.shape[0] <= max(max(service_channels), max(residential_channels)):
        return 0.0

    fac = _presence_mask(plan_chw, service_channels, mode=presence_mode, q=q)
    if int(np.sum(fac)) < int(min_facility_cells):
        # no services -> very poor walkability
        return 50.0

    res = plan_chw[residential_channels].sum(axis=0)
    dist = _distance_to_facility(fac)
    return _weighted_mean(dist, res)

def _spcorr01(tot: np.ndarray, eps: float = 1e-8) -> float:
    tot = np.asarray(tot, dtype=np.float64)
    tot = np.nan_to_num(tot, nan=0.0, posinf=0.0, neginf=0.0)
    if tot.ndim != 2:
        return float("nan")
    pairs = []
    if tot.shape[0] > 1:
        pairs.append((tot[1:, :].reshape(-1), tot[:-1, :].reshape(-1)))
    if tot.shape[1] > 1:
        pairs.append((tot[:, 1:].reshape(-1), tot[:, :-1].reshape(-1)))
    if not pairs:
        return 0.5
    rs = []
    for a, b in pairs:
        a = a - float(np.mean(a))
        b = b - float(np.mean(b))
        denom = float(np.linalg.norm(a) * np.linalg.norm(b) + eps)
        r = float(np.dot(a, b) / denom) if denom > 0 else 0.0
        r = max(-1.0, min(1.0, r))
        rs.append((r + 1.0) / 2.0)
    return float(np.mean(rs))


def _recreation_space_ratio(plan_chw: np.ndarray, green_channels=[7, 10]) -> float:
    """Ratio of recreation/park space to total developed area
    Channels: [7,10]=recreation service, tourist attraction
    Urban Planning: WHO recommends 9m² green space per capita, higher = healthier city"""
    if plan_chw.shape[0] <= max(green_channels):
        return 0.0
    
    green = plan_chw[green_channels].sum()
    total = plan_chw.sum() + 1e-8
    return float(green / total)



def _recreation_space_accessibility(plan_chw: np.ndarray,
                                    green_channels=[7, 10],
                                    residential_channels=[11],
                                    max_distance=30,
                                    presence_mode: str = PRESENCE_MODE_DEFAULT,
                                    q: float = PRESENCE_QUANTILE_DEFAULT,
                                    min_facility_cells: int = MIN_FACILITY_CELLS_DEFAULT) -> float:
    if plan_chw.shape[0] <= max(max(green_channels), max(residential_channels)):
        return 0.0

    fac = _presence_mask(plan_chw, green_channels, mode=presence_mode, q=q)
    if int(np.sum(fac)) < int(min_facility_cells):
        return 0.0

    res = plan_chw[residential_channels].sum(axis=0)
    dist = _distance_to_facility(fac)
    return _weighted_coverage(dist, res, max_distance)

def _street_network_connectivity(plan_chw: np.ndarray, street_channels=[0, 17]) -> float:
    if plan_chw.shape[0] <= max(street_channels):
        return 0.0
    
    dom = np.argmax(plan_chw, axis=0)
    binary_streets = np.isin(dom, street_channels).astype(int)
    
    H, W = binary_streets.shape
    intersections = 0
    
    for i in range(1, H-1):
        for j in range(1, W-1):
            if binary_streets[i, j]:
                neighbors = (binary_streets[i-1, j] + binary_streets[i+1, j] + 
                           binary_streets[i, j-1] + binary_streets[i, j+1])
                if neighbors >= 3:
                    intersections += 1
    
    street_area = binary_streets.sum() + 1e-8
    return float(intersections / street_area)


def _network_density(plan_chw: np.ndarray, street_channels=[0, 17]) -> float:
    if plan_chw.shape[0] <= max(street_channels):
        return 0.0
    
    dom = np.argmax(plan_chw, axis=0)
    street_mask = np.isin(dom, street_channels)
    street_length = street_mask.sum()
    total_area = street_mask.size
    
    return float(street_length / total_area)


def _building_density(plan_chw: np.ndarray, building_channels=[5, 6, 15, 16]) -> float:
    if plan_chw.shape[0] <= max(building_channels):
        return 0.0
    
    built = plan_chw[building_channels].sum()
    total = plan_chw.sum() + 1e-8
    return float(built / total)


def _floor_area_ratio(plan_chw: np.ndarray, building_channels=[5, 6, 15, 16]) -> float:
    if plan_chw.shape[0] <= max(building_channels):
        return 0.0
    
    dom = np.argmax(plan_chw, axis=0)
    built_mask = np.isin(dom, building_channels)
    if built_mask.sum() == 0:
        return 0.0
    
    building_intensity = plan_chw[building_channels].sum(axis=0)
    built_area_intensity = float(np.mean(building_intensity[built_mask]))
    
    total_intensity = float(np.mean(plan_chw.sum(axis=0)))
    if total_intensity <= 1e-8:
        return 0.0
    
    far_proxy = built_area_intensity / (total_intensity + 1e-8)
    return float(min(far_proxy, 5.0))


def _mixed_use_intensity(plan_chw: np.ndarray, window_size=20) -> float:
    C, H, W = plan_chw.shape
    intensities = []
    
    for i in range(0, H - window_size, window_size // 2):
        for j in range(0, W - window_size, window_size // 2):
            window = plan_chw[:, i:i+window_size, j:j+window_size]
            composition = window.mean(axis=(1, 2))
            
            n_types = (composition > 0.05 * composition.max()).sum()
            intensities.append(float(n_types))
    
    return float(np.mean(intensities) if intensities else 0.0)


def _walkability_score(plan_chw: np.ndarray, residential_channels=[11],
                        service_channels=[4, 6], street_channels=[0, 17]) -> float:
    if plan_chw.shape[0] <= max(max(residential_channels), max(service_channels), max(street_channels)):
        return 0.0
    
    dom = np.argmax(plan_chw, axis=0)
    walkable_channels = list(set(residential_channels + service_channels))
    density = float(np.isin(dom, walkable_channels).mean())
    
    diversity = _land_mix_entropy(plan_chw)
    
    design = _street_network_connectivity(plan_chw, street_channels)
    
    walkability = (density * 0.35 + diversity * 0.40 + design * 0.25)
    return float(min(walkability, 1.0))



def _fifteen_minute_city_coverage(plan_chw: np.ndarray,
                                  essential_services=[4, 6, 8, 13],
                                  residential_channels=[11],
                                  max_distance=30,
                                  presence_mode: str = PRESENCE_MODE_DEFAULT,
                                  q: float = PRESENCE_QUANTILE_DEFAULT,
                                  min_facility_cells: int = MIN_FACILITY_CELLS_DEFAULT) -> float:
    if plan_chw.shape[0] <= max(max(essential_services), max(residential_channels)):
        return 0.0

    fac = _presence_mask(plan_chw, essential_services, mode=presence_mode, q=q)
    if int(np.sum(fac)) < int(min_facility_cells):
        return 0.0

    res = plan_chw[residential_channels].sum(axis=0)
    dist = _distance_to_facility(fac)
    return _weighted_coverage(dist, res, max_distance)


def _transit_coverage(plan_chw: np.ndarray,
                      transit_channels=[14],
                      residential_channels=[11],
                      buffer_distance=10,
                      presence_mode: str = PRESENCE_MODE_DEFAULT,
                      q: float = PRESENCE_QUANTILE_DEFAULT,
                      min_facility_cells: int = MIN_FACILITY_CELLS_DEFAULT) -> float:
    if not transit_channels:
        return 0.0
    if plan_chw.shape[0] <= max(max(transit_channels), max(residential_channels)):
        return 0.0

    fac = _presence_mask(plan_chw, transit_channels, mode=presence_mode, q=q)
    if int(np.sum(fac)) < int(min_facility_cells):
        return 0.0

    res = plan_chw[residential_channels].sum(axis=0)
    dist = _distance_to_facility(fac)
    return _weighted_coverage(dist, res, buffer_distance)


def _transit_oriented_density(plan_chw: np.ndarray,
                               transit_channels=[14],
                               building_channels=[5, 6, 15, 16],
                               buffer_distance=10,
                               presence_mode: str = PRESENCE_MODE_DEFAULT,
                               q: float = PRESENCE_QUANTILE_DEFAULT,
                               min_facility_cells: int = MIN_FACILITY_CELLS_DEFAULT) -> float:
    if plan_chw.shape[0] <= max(max(transit_channels or [0]), max(building_channels)):
        return 0.0

    if not transit_channels or all(ch >= plan_chw.shape[0] for ch in transit_channels):
        return _building_density(plan_chw, building_channels)

    fac = _presence_mask(plan_chw, transit_channels, mode=presence_mode, q=q)
    if int(np.sum(fac)) < int(min_facility_cells):
        return _building_density(plan_chw, building_channels)

    buildings = plan_chw[building_channels].sum(axis=0)
    dist = _distance_to_facility(fac)
    near = dist <= buffer_distance

    tod_density = float(np.mean(buildings[near])) if int(np.sum(near)) > 0 else 0.0
    overall_density = float(np.mean(buildings))
    ratio = tod_density / (overall_density + 1e-8)
    return float(min(ratio / 1.5, 1.0))

def _urban_heat_island_risk(plan_chw: np.ndarray, impervious_channels=[0, 5, 6, 11, 15, 16, 17],
                            green_channels=[7, 10]) -> float:
    if plan_chw.shape[0] <= max(max(impervious_channels), max(green_channels)):
        return 0.5
    
    impervious = plan_chw[impervious_channels].sum()
    green = plan_chw[green_channels].sum()
    total = plan_chw.sum() + 1e-8
    
    impervious_ratio = impervious / total
    recreation_ratio = green / total
    
    heat_risk = impervious_ratio * (1 - recreation_ratio)
    return float(heat_risk)


def _flood_resilience_score(plan_chw: np.ndarray, green_channels=[7, 10],
                            impervious_channels=[0, 5, 6, 11, 15, 16, 17],
                            water_channels=[]) -> float:
    C = plan_chw.shape[0]
    total = plan_chw.sum() + 1e-8
    
    green_chs = [c for c in green_channels if 0 <= c < C]
    imperv_chs = [c for c in impervious_channels if 0 <= c < C]
    
    green = plan_chw[green_chs].sum() if green_chs else 0.0
    impervious = plan_chw[imperv_chs].sum() if imperv_chs else 0.0
    
    green_imperv_ratio = float(green / (impervious + green + 1e-8))
    
    dom = np.argmax(plan_chw, axis=0)
    green_spread = float(np.isin(dom, green_chs).mean()) if green_chs else 0.0

    green_mask = np.isin(dom, green_chs) if green_chs else np.zeros(dom.shape, dtype=bool)
    if green_mask.sum() > 5:
        lcs = _largest_component_share(green_mask)
        connectivity_score = 1.0 - abs(lcs - 0.5) * 2.0  # peaks at lcs=0.5
        connectivity_score = max(0.0, connectivity_score)
    else:
        connectivity_score = 0.0
    
    resilience = 0.40 * green_imperv_ratio + 0.30 * green_spread + 0.30 * connectivity_score
    return float(min(resilience, 1.0))



def _school_accessibility(plan_chw: np.ndarray,
                          school_channels=[13],
                          residential_channels=[11],
                          max_distance=20,
                          presence_mode: str = PRESENCE_MODE_DEFAULT,
                          q: float = PRESENCE_QUANTILE_DEFAULT,
                          min_facility_cells: int = MIN_FACILITY_CELLS_DEFAULT) -> float:

    if plan_chw.shape[0] <= max(max(school_channels), max(residential_channels)):
        return 0.0

    fac = _presence_mask(plan_chw, school_channels, mode=presence_mode, q=q)
    if int(np.sum(fac)) < int(min_facility_cells):
        return 0.0

    res = plan_chw[residential_channels].sum(axis=0)
    dist = _distance_to_facility(fac)
    return _weighted_coverage(dist, res, max_distance)


def _healthcare_accessibility(plan_chw: np.ndarray,
                              healthcare_channels=[8],
                              residential_channels=[11],
                              max_distance=40,
                              presence_mode: str = PRESENCE_MODE_DEFAULT,
                              q: float = PRESENCE_QUANTILE_DEFAULT,
                              min_facility_cells: int = MIN_FACILITY_CELLS_DEFAULT) -> float:
    if plan_chw.shape[0] <= max(max(healthcare_channels), max(residential_channels)):
        return 0.0

    fac = _presence_mask(plan_chw, healthcare_channels, mode=presence_mode, q=q)
    if int(np.sum(fac)) < int(min_facility_cells):
        return 0.0

    res = plan_chw[residential_channels].sum(axis=0)
    dist = _distance_to_facility(fac)
    return _weighted_coverage(dist, res, max_distance)




def _service_access_inequality(plan_chw: np.ndarray,
                               service_channels=[4, 6],
                               residential_channels=[11],
                               presence_mode: str = PRESENCE_MODE_DEFAULT,
                               q: float = PRESENCE_QUANTILE_DEFAULT,
                               min_facility_cells: int = MIN_FACILITY_CELLS_DEFAULT) -> float:
    return _access_inequality(plan_chw, service_channels, residential_channels,
                              presence_mode=presence_mode, q=q, min_facility_cells=min_facility_cells)

def _school_access_inequality(plan_chw: np.ndarray,
                              school_channels=[13],
                              residential_channels=[11],
                              presence_mode: str = PRESENCE_MODE_DEFAULT,
                              q: float = PRESENCE_QUANTILE_DEFAULT,
                              min_facility_cells: int = MIN_FACILITY_CELLS_DEFAULT) -> float:
    return _access_inequality(plan_chw, school_channels, residential_channels,
                              presence_mode=presence_mode, q=q, min_facility_cells=min_facility_cells)

def _healthcare_access_inequality(plan_chw: np.ndarray,
                                  healthcare_channels=[8],
                                  residential_channels=[11],
                                  presence_mode: str = PRESENCE_MODE_DEFAULT,
                                  q: float = PRESENCE_QUANTILE_DEFAULT,
                                  min_facility_cells: int = MIN_FACILITY_CELLS_DEFAULT) -> float:
    return _access_inequality(plan_chw, healthcare_channels, residential_channels,
                              presence_mode=presence_mode, q=q, min_facility_cells=min_facility_cells)

def _transit_access_inequality(plan_chw: np.ndarray,
                               transit_channels=[14],
                               residential_channels=[11],
                               presence_mode: str = PRESENCE_MODE_DEFAULT,
                               q: float = PRESENCE_QUANTILE_DEFAULT,
                               min_facility_cells: int = MIN_FACILITY_CELLS_DEFAULT) -> float:
    return _access_inequality(plan_chw, transit_channels, residential_channels,
                              presence_mode=presence_mode, q=q, min_facility_cells=min_facility_cells)

def _green_access_inequality(plan_chw: np.ndarray,
                             green_channels=[7, 10],
                             residential_channels=[11],
                             presence_mode: str = PRESENCE_MODE_DEFAULT,
                             q: float = PRESENCE_QUANTILE_DEFAULT,
                             min_facility_cells: int = MIN_FACILITY_CELLS_DEFAULT) -> float:
    return _access_inequality(plan_chw, green_channels, residential_channels,
                              presence_mode=presence_mode, q=q, min_facility_cells=min_facility_cells)


def _employment_centrality(plan_chw: np.ndarray, employment_channels=[1, 2, 3, 15, 16]) -> float:
    if plan_chw.shape[0] <= max(employment_channels):
        return 0.5
    
    employment = plan_chw[employment_channels].sum(axis=0)
    
    H, W = employment.shape
    y_coords, x_coords = np.mgrid[0:H, 0:W]
    
    total_emp = employment.sum() + 1e-8
    center_y = (employment * y_coords).sum() / total_emp
    center_x = (employment * x_coords).sum() / total_emp
    
    distances_from_center = np.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)
    weighted_distances = distances_from_center * employment
    
    avg_distance = weighted_distances.sum() / total_emp
    max_distance = np.sqrt(H**2 + W**2) / 2
    
    centrality = 1.0 - (avg_distance / max_distance)
    return float(centrality)


def _density_gradient(plan_chw: np.ndarray, eps: float = 1e-12) -> float:
    C, H, W = plan_chw.shape
    tot = plan_chw.sum(axis=0)  # (H, W) total activity
    total_mass = float(np.sum(tot))
    if total_mass <= eps:
        return 1.0  # uniform (no activity)
    
    ys, xs = np.mgrid[0:H, 0:W]
    cy = float(np.sum(ys * tot) / (total_mass + eps))
    cx = float(np.sum(xs * tot) / (total_mass + eps))
    
    dist = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
    max_dist = float(np.max(dist)) + eps
    
    inner_mask = dist <= 0.33 * max_dist
    outer_mask = dist >= 0.66 * max_dist
    
    inner_density = float(np.mean(tot[inner_mask])) if inner_mask.sum() > 0 else eps
    outer_density = float(np.mean(tot[outer_mask])) if outer_mask.sum() > 0 else eps
    
    return float(inner_density / (outer_density + eps))


def _land_use_compatibility_score(plan_chw: np.ndarray, eps: float = 1e-12) -> float:
    C, H, W = plan_chw.shape
    dom = np.argmax(plan_chw, axis=0).astype(np.int32)

    good_pairs = {
        (11, 4), (11, 6), (11, 19),  # housing + food/daily life/public service
        (11, 13),                      # housing + education
        (13, 7), (13, 10),             # education + recreation/tourist (parks)
        (14, 15), (14, 16),            # transit + finance/company (TOD)
        (14, 11),                      # transit + housing
        (7, 11), (10, 11),             # recreation/tourist + housing
        (8, 14),                       # health + transit (accessibility)
    }
    good_pairs |= {(b, a) for (a, b) in good_pairs}

    bad_pairs = {
        (1, 11), (2, 11), (3, 11),    # auto near housing
        (1, 13), (2, 13), (3, 13),    # auto near education
        (1, 8), (2, 8), (3, 8),       # auto near health
        (5, 13),                       # heavy shopping directly next to schools
    }
    bad_pairs |= {(b, a) for (a, b) in bad_pairs}
    
    a_r = dom[:, :-1]; b_r = dom[:, 1:]  # horizontal pairs
    a_d = dom[:-1, :]; b_d = dom[1:, :]  # vertical pairs
    
    total_pairs = float(a_r.size + a_d.size)
    if total_pairs <= eps:
        return 0.5
    
    good_count = 0.0
    bad_count = 0.0
    for u, v in good_pairs:
        good_count += float(np.sum((a_r == u) & (b_r == v)))
        good_count += float(np.sum((a_d == u) & (b_d == v)))
    for u, v in bad_pairs:
        bad_count += float(np.sum((a_r == u) & (b_r == v)))
        bad_count += float(np.sum((a_d == u) & (b_d == v)))
    
    compatibility = 0.5 + 0.5 * (good_count - bad_count) / (total_pairs + eps)
    return float(max(0.0, min(1.0, compatibility)))


def _service_hierarchy_score(plan_chw: np.ndarray, eps: float = 1e-12) -> float:
    C, H, W = plan_chw.shape
    dom = np.argmax(plan_chw, axis=0)

    hierarchy_groups = [
        [4, 6],        # most spread (neighborhood services)
        [5, 19],       # moderate spread
        [13, 8],       # district scale
        [12, 15],      # regional scale
    ]
    
    spreads = []
    for group in hierarchy_groups:
        chs = [c for c in group if 0 <= c < C]
        if not chs:
            spreads.append(0.0)
            continue
        mask = np.isin(dom, chs)
        spreads.append(float(mask.mean()))
    
    if len(spreads) < 2 or sum(spreads) <= eps:
        return 0.5  # can't determine
    
    expected_rank = list(range(len(spreads), 0, -1))  # [4,3,2,1]
    actual_rank_order = np.argsort(np.argsort([-s for s in spreads])) + 1
    
    n = len(spreads)
    d_sq = sum((expected_rank[i] - actual_rank_order[i]) ** 2 for i in range(n))
    rho = 1.0 - (6.0 * d_sq) / (n * (n ** 2 - 1) + eps)
    
    return float(max(0.0, min(1.0, 0.5 * (rho + 1.0))))


def _open_space_distribution(plan_chw: np.ndarray, green_channels=[7, 10],
                              eps: float = 1e-12) -> float:
    C, H, W = plan_chw.shape
    chs = [c for c in green_channels if 0 <= c < C]
    if not chs:
        return 0.0
    
    green = plan_chw[chs].sum(axis=0)
    total_green = float(np.sum(green))
    if total_green <= eps:
        return 0.0
    
    n_div = 4
    bh = H // n_div
    bw = W // n_div
    if bh < 2 or bw < 2:
        return 0.5
    
    block_means = []
    for i in range(n_div):
        for j in range(n_div):
            block = green[i*bh:(i+1)*bh, j*bw:(j+1)*bw]
            block_means.append(float(np.mean(block)))
    
    arr = np.array(block_means)
    mu = float(np.mean(arr))
    if mu <= eps:
        return 0.0
    cv = float(np.std(arr) / mu)

    return float(max(0.0, min(1.0, math.exp(-cv))))


def _emergency_response_coverage(plan_chw: np.ndarray,
                                  emergency_channels=[8],
                                  residential_channels=[11],
                                  max_response_dist: float = 25.0,
                                  facility_q: float = 90.0,
                                  eps: float = 1e-12) -> float:
    C = plan_chw.shape[0]
    plan = np.asarray(plan_chw, dtype=np.float64)
    plan = np.nan_to_num(plan, nan=0.0, posinf=0.0, neginf=0.0)
    plan = np.clip(plan, 0.0, None)
    
    echs = [c for c in emergency_channels if 0 <= c < C]
    if not echs:
        return 0.0
    
    pop = _resident_weight_map(plan, residential_channels=residential_channels,
                               fallback="activity", eps=eps)
    pop_sum = float(np.sum(pop))
    if pop_sum <= eps:
        return 0.0
    
    fac_mask = _presence_mask_v5(plan, echs, mode="argmax",
                                 argmax_intensity_q=facility_q, eps=eps)
    if fac_mask.sum() < 3:
        return 0.0
    
    dist = _distance_to_facility(fac_mask)
    covered = (dist <= float(max_response_dist)).astype(np.float64)
    return float(np.sum(covered * pop) / (pop_sum + eps))


def _intensity_transition_smoothness(plan_chw: np.ndarray, eps: float = 1e-12) -> float:
    tot = plan_chw.sum(axis=0).astype(np.float64)
    tot = np.nan_to_num(tot, nan=0.0, posinf=0.0, neginf=0.0)
    H, W = tot.shape
    
    intensity_range = float(np.max(tot) - np.min(tot))
    if intensity_range <= eps:
        return 1.0  # perfectly uniform = perfectly smooth
    
    diff_h = np.abs(tot[:, 1:] - tot[:, :-1])
    diff_v = np.abs(tot[1:, :] - tot[:-1, :])
    mean_diff = float(np.mean(diff_h) + np.mean(diff_v)) / 2.0
    
    roughness = mean_diff / (intensity_range + eps)
    
    return float(max(0.0, min(1.0, math.exp(-3.0 * roughness))))


def _infrastructure_demand_alignment(plan_chw: np.ndarray,
                                      road_channels=[0, 17],
                                      activity_channels=[4, 5, 6, 8, 11, 13, 15, 16],
                                      sigma: float = 5.0,
                                      eps: float = 1e-12) -> float:
    C = plan_chw.shape[0]
    plan = np.asarray(plan_chw, dtype=np.float64)
    plan = np.nan_to_num(plan, nan=0.0, posinf=0.0, neginf=0.0)
    plan = np.clip(plan, 0.0, None)
    
    road_chs = [c for c in road_channels if 0 <= c < C]
    act_chs = [c for c in activity_channels if 0 <= c < C]
    if not road_chs or not act_chs:
        return 0.5
    
    road_map = plan[road_chs].sum(axis=0)
    activity_map = plan[act_chs].sum(axis=0)
    
    try:
        from scipy.ndimage import gaussian_filter
        road_smooth = gaussian_filter(road_map, sigma=float(sigma))
        activity_smooth = gaussian_filter(activity_map, sigma=float(sigma))
    except Exception:
        road_smooth = road_map
        activity_smooth = activity_map
    
    return float(_cosine_similarity(road_smooth, activity_smooth, eps=eps))


def _education_green_proximity(plan_chw: np.ndarray,
                                education_channels=[13],
                                green_channels=[7, 10],
                                max_distance: float = 15.0,
                                facility_q: float = 90.0,
                                eps: float = 1e-12) -> float:
    C = plan_chw.shape[0]
    plan = np.asarray(plan_chw, dtype=np.float64)
    plan = np.nan_to_num(plan, nan=0.0, posinf=0.0, neginf=0.0)
    plan = np.clip(plan, 0.0, None)
    
    edu_chs = [c for c in education_channels if 0 <= c < C]
    grn_chs = [c for c in green_channels if 0 <= c < C]
    if not edu_chs or not grn_chs:
        return 0.0
    
    edu_mask = _presence_mask_v5(plan, edu_chs, mode="argmax",
                                 argmax_intensity_q=facility_q, eps=eps)
    grn_mask = _presence_mask_v5(plan, grn_chs, mode="argmax",
                                  argmax_intensity_q=facility_q, eps=eps)
    
    if edu_mask.sum() < 2 or grn_mask.sum() < 2:
        return 0.0
    
    dist_to_green = _distance_to_facility(grn_mask)
    edu_near_green = (dist_to_green[edu_mask] <= float(max_distance))
    
    return float(np.mean(edu_near_green)) if edu_near_green.size > 0 else 0.0


def _residential_noise_exposure(plan_chw: np.ndarray,
                                 residential_channels=[11],
                                 noise_channels=[0, 1, 2, 3, 14, 17],
                                 buffer_dist: float = 5.0,
                                 facility_q: float = 90.0,
                                 eps: float = 1e-12) -> float:
    C = plan_chw.shape[0]
    plan = np.asarray(plan_chw, dtype=np.float64)
    plan = np.nan_to_num(plan, nan=0.0, posinf=0.0, neginf=0.0)
    plan = np.clip(plan, 0.0, None)
    
    res_chs = [c for c in residential_channels if 0 <= c < C]
    noi_chs = [c for c in noise_channels if 0 <= c < C]
    if not res_chs or not noi_chs:
        return 0.0
    
    res_map = plan[res_chs].sum(axis=0)
    res_total = float(np.sum(res_map))
    if res_total <= eps:
        return 0.0
    
    noise_mask = _presence_mask_v5(plan, noi_chs, mode="argmax",
                                    argmax_intensity_q=facility_q, eps=eps)
    if noise_mask.sum() < 2:
        return 0.0
    
    dist_to_noise = _distance_to_facility(noise_mask)
    exposed = (dist_to_noise <= float(buffer_dist)).astype(np.float64)
    
    return float(np.sum(exposed * res_map) / (res_total + eps))


def compute_distribution_metrics_grouped(
    gen: np.ndarray,
    con_label: np.ndarray,
    green_standards: Dict[int, np.ndarray]
) -> Dict[str, Any]:
    mappings = defaultdict(list)
    for ind, pred_solu in enumerate(gen):
        mappings[con_label[ind]].append(pred_solu)
    
    predict_dict = {}
    for green_level, preds in mappings.items():
        tmp_emb = np.array(preds).reshape(len(preds), -1)  # (N_group, 200000)
        avg_emb = np.mean(tmp_emb, axis=0)                  # (200000,)
        
        avg_emb[avg_emb <= 0] = 0.00001
        
        predict_dict[green_level] = avg_emb
    
    eval_results = []
    
    for green_level, standard_solution in sorted(green_standards.items()):
        if green_level not in predict_dict:
            continue
        
        predict_solution = predict_dict[green_level]
        
        if standard_solution.shape != predict_solution.shape:
            continue

        hd_dis = hellinger(standard_solution, predict_solution)
        
        kl_dis = kullback_leibler(standard_solution, predict_solution)
        
        js_dis = distance.jensenshannon(standard_solution, predict_solution)
        
        wd_dis = wasserstein_distance(standard_solution, predict_solution)
        cos_dis = distance.cosine(standard_solution, predict_solution)
        
        tv_dis = _tv(standard_solution, predict_solution)
        bhat_dis = _bhattacharyya_dist(standard_solution, predict_solution)
        iou_dis = _distribution_iou(standard_solution, predict_solution)
        
        eval_results.append([kl_dis, js_dis, wd_dis, hd_dis, cos_dis, tv_dis, bhat_dis, iou_dis])
    
    eval_results = np.array(eval_results)
    final_results = np.mean(eval_results, axis=0)
    
    return {
        "kl": float(final_results[0]),
        "js": float(final_results[1]),
        "wd": float(final_results[2]),
        "hellinger": float(final_results[3]),
        "cos": float(final_results[4]),
        "tv": float(final_results[5]),
        "bhattacharyya": float(final_results[6]),
        "iou": float(final_results[7]),
        "method": "grouped_with_standards",
        "n_groups": len(eval_results)
    }


def compute_distribution_metrics_test_vs_test(
    real: np.ndarray,
    gen: np.ndarray,
    con_label: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    N = min(real.shape[0], gen.shape[0])
    real = real[:N]
    gen = gen[:N]
    
    if con_label is not None and len(con_label) >= N:
        con_label = con_label[:N]
        
        real_groups = defaultdict(list)
        gen_groups = defaultdict(list)
        
        for i in range(N):
            level = int(con_label[i])
            real_groups[level].append(real[i])
            gen_groups[level].append(gen[i])
        
        eval_results = []
        
        for level in sorted(real_groups.keys()):
            if level not in gen_groups:
                continue
            
            real_tmp = np.array(real_groups[level]).reshape(len(real_groups[level]), -1)
            gen_tmp = np.array(gen_groups[level]).reshape(len(gen_groups[level]), -1)
            
            real_avg = real_tmp.mean(axis=0)
            gen_avg = gen_tmp.mean(axis=0)
            
            real_avg[real_avg <= 0] = 0.00001
            gen_avg[gen_avg <= 0] = 0.00001

            hd_dis = hellinger(real_avg, gen_avg)
            
            kl_dis = kullback_leibler(real_avg, gen_avg)
            
            js_dis = distance.jensenshannon(real_avg, gen_avg)
            
            wd_dis = wasserstein_distance(real_avg, gen_avg)
            cos_dis = distance.cosine(real_avg, gen_avg)
            
            tv_dis = _tv(real_avg, gen_avg)
            bhat_dis = _bhattacharyya_dist(real_avg, gen_avg)
            iou_dis = _distribution_iou(real_avg, gen_avg)
            
            eval_results.append([kl_dis, js_dis, wd_dis, hd_dis, cos_dis, tv_dis, bhat_dis, iou_dis])
        
        eval_results = np.array(eval_results)
        final_results = np.mean(eval_results, axis=0)
        
        return {
            "kl": float(final_results[0]),
            "js": float(final_results[1]),
            "wd": float(final_results[2]),
            "hellinger": float(final_results[3]),
            "cos": float(final_results[4]),
            "tv": float(final_results[5]),
            "bhattacharyya": float(final_results[6]),
            "iou": float(final_results[7]),
            "method": "grouped_test_vs_test",
            "n_groups": len(eval_results)
        }
    
    else:
        print("[WARNING] Using global averaging (no con_label). KL will be lower than grouped method.")
        
        real_flat = real.reshape(N, -1)
        gen_flat = gen.reshape(N, -1)
        
        real_avg = real_flat.mean(axis=0)
        gen_avg = gen_flat.mean(axis=0)
        
        # Replace zeros
        real_avg[real_avg <= 0] = 0.00001
        gen_avg[gen_avg <= 0] = 0.00001

        hd_value = hellinger(real_avg, gen_avg)
        
        tv_value = _tv(real_avg, gen_avg)
        bhat_value = _bhattacharyya_dist(real_avg, gen_avg)
        iou_value = _distribution_iou(real_avg, gen_avg)
        
        return {
            "kl": float(kullback_leibler(real_avg, gen_avg)),  # Unnormalized (original)
            "js": float(distance.jensenshannon(real_avg, gen_avg)),  # Normalizes internally
            "wd": float(wasserstein_distance(real_avg, gen_avg)),  # Unnormalized
            "hellinger": float(hd_value),  # Unnormalized (original)
            "cos": float(distance.cosine(real_avg, gen_avg)),  # Unnormalized
            "tv": float(tv_value),  # Normalizes internally
            "bhattacharyya": float(bhat_value),  # Normalizes internally
            "iou": float(iou_value),  # Normalizes internally
            "method": "global_averaging",
            "n_groups": 1
        }


def compute_quant_metrics(
    real: np.ndarray,
    gen: np.ndarray,
    zones_hw: np.ndarray,
    con_label: Optional[np.ndarray] = None,
    green_standards: Optional[Dict[int, np.ndarray]] = None,
    max_pairs: int = 200,
    presence_mode: str = "argmax",
    presence_quantile: float = 99.5,
    min_facility_cells: int = 25,
    dimension_profile: str = "planning_6dimension"
) -> Dict[str, Any]:
    N = min(real.shape[0], gen.shape[0], zones_hw.shape[0])
    real = real[:N]
    gen = gen[:N]
    zones_hw = zones_hw[:N]
    if con_label is not None:
        con_label = con_label[:N]
    
    # ============================================================================
    # DISTRIBUTION METRICS (using grouped averaging)
    # ============================================================================
    
    if green_standards is not None and con_label is not None:
        # Method 1: Compare against green_standards (EXACT original evaluate.py)
        print("[INFO] Using grouped averaging with green_standards (original method)")
        dist = compute_distribution_metrics_grouped(gen, con_label, green_standards)
    elif con_label is not None:
        # Method 2: Test vs test with grouping (correct approach without standards)
        print("[INFO] Using grouped averaging for test-vs-test comparison")
        dist = compute_distribution_metrics_test_vs_test(real, gen, con_label)
    else:
        # Method 3: Fallback to global averaging (legacy, gives lower values)
        print("[WARNING] No con_label - using global averaging (KL will be lower)")
        dist = compute_distribution_metrics_test_vs_test(real, gen, None)


    # ============================================================================
    # SPATIAL ACCURACY METRICS
    # ============================================================================
    
    # Spatial accuracy: per-pixel on total density map
    real_tot = real.sum(axis=1)  # (N,100,100)
    gen_tot = gen.sum(axis=1)
    diff = gen_tot - real_tot
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    rmse = float(math.sqrt(max(0.0, mse)))

    # Total-map correlation (global Pearson)
    a = real_tot.reshape(N, -1).mean(axis=0)
    b = gen_tot.reshape(N, -1).mean(axis=0)
    a = a - float(np.mean(a))
    b = b - float(np.mean(b))
    denom = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
    corr = float(np.dot(a, b) / denom) if denom > 0 else 0.0

    spatial = {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "rmse_ref": float(np.std(real_tot) + 1e-8),
        "corr_total": corr,
        "spcorr_local01": float(np.mean([_spcorr01(gen_tot[i]) for i in range(N)])),
    }

    # ============================================================================
    # URBAN PLANNING DOMAIN METRICS
    # ============================================================================
    
    # Land use diversity
    ent = float(np.mean([_land_mix_entropy(gen[i]) for i in range(N)]))
    simp_div = float(np.mean([_simpson_diversity(gen[i]) for i in range(N)]))
    shannon_zone = float(np.mean([_shannon_diversity_by_zone(gen[i], zones_hw[i]) for i in range(N)]))
    
    # Spatial structure
    gini = float(np.mean([_gini(gen_tot[i]) for i in range(N)]))
    edge = float(np.mean([_edge_density(np.argmax(gen[i], axis=0)) for i in range(N)]))
    patch_dens = float(np.mean([_patch_density(np.argmax(gen[i], axis=0)) for i in range(N)]))
    compact = float(np.mean([_compactness_ratio(gen[i]) for i in range(N)]))
    aggreg = float(np.mean([_aggregation_index(np.argmax(gen[i], axis=0)) for i in range(N)]))
    mixing = float(np.mean([_mixing_index(gen[i]) for i in range(N)]))
    
    # Jobs-housing balance
    jh_balance = float(np.mean([_jobs_housing_balance(gen[i]) for i in range(N)]))
    
    # Accessibility and equity
    svc_coverage = float(np.mean([_service_coverage(gen[i], presence_mode=presence_mode, q=presence_quantile, min_facility_cells=min_facility_cells) for i in range(N)]))
    spatial_gini_val = float(np.mean([_spatial_gini(gen[i]) for i in range(N)]))
    avg_svc_dist = float(np.mean([_avg_nearest_service_distance(gen[i], presence_mode=presence_mode, q=presence_quantile, min_facility_cells=min_facility_cells) for i in range(N)]))

    # Equity (distributional): inequality of access distances (0=equal,1=unequal)
    service_access_inequality = float(np.mean([_service_access_inequality(gen[i], presence_mode=presence_mode, q=presence_quantile, min_facility_cells=min_facility_cells) for i in range(N)]))
    school_access_inequality = float(np.mean([_school_access_inequality(gen[i], presence_mode=presence_mode, q=presence_quantile, min_facility_cells=min_facility_cells) for i in range(N)]))
    healthcare_access_inequality = float(np.mean([_healthcare_access_inequality(gen[i], presence_mode=presence_mode, q=presence_quantile, min_facility_cells=min_facility_cells) for i in range(N)]))
    transit_access_inequality = float(np.mean([_transit_access_inequality(gen[i], presence_mode=presence_mode, q=presence_quantile, min_facility_cells=min_facility_cells) for i in range(N)]))
    green_access_inequality = float(np.mean([_green_access_inequality(gen[i], presence_mode=presence_mode, q=presence_quantile, min_facility_cells=min_facility_cells) for i in range(N)]))

    # NEW: Additional Urban Planning Metrics
    
    # Recreation space metrics
    recreation_ratio = float(np.mean([_recreation_space_ratio(gen[i]) for i in range(N)]))
    recreation_accessibility = float(np.mean([_recreation_space_accessibility(gen[i], presence_mode=presence_mode, q=presence_quantile, min_facility_cells=min_facility_cells) for i in range(N)]))
    
    # Connectivity and network metrics
    street_connectivity = float(np.mean([_street_network_connectivity(gen[i]) for i in range(N)]))
    network_density = float(np.mean([_network_density(gen[i]) for i in range(N)]))
    
    # Density and intensity
    building_density = float(np.mean([_building_density(gen[i]) for i in range(N)]))
    floor_area_ratio = float(np.mean([_floor_area_ratio(gen[i]) for i in range(N)]))
    mixed_use_intensity = float(np.mean([_mixed_use_intensity(gen[i]) for i in range(N)]))
    
    # Walkability and 15-minute city
    walkability_score = float(np.mean([_walkability_score(gen[i]) for i in range(N)]))
    fifteen_min_coverage = float(np.mean([_fifteen_minute_city_coverage(gen[i], presence_mode=presence_mode, q=presence_quantile, min_facility_cells=min_facility_cells) for i in range(N)]))
    
    # Transit-oriented development
    transit_coverage = float(np.mean([_transit_coverage(gen[i], presence_mode=presence_mode, q=presence_quantile, min_facility_cells=min_facility_cells) for i in range(N)]))
    transit_density = float(np.mean([_transit_oriented_density(gen[i], presence_mode=presence_mode, q=presence_quantile, min_facility_cells=min_facility_cells) for i in range(N)]))
    
    # Environmental metrics
    urban_heat_risk = float(np.mean([_urban_heat_island_risk(gen[i]) for i in range(N)]))
    flood_resilience = float(np.mean([_flood_resilience_score(gen[i]) for i in range(N)]))
    
    # Social infrastructure
    school_accessibility = float(np.mean([_school_accessibility(gen[i], presence_mode=presence_mode, q=presence_quantile, min_facility_cells=min_facility_cells) for i in range(N)]))
    healthcare_access = float(np.mean([_healthcare_accessibility(gen[i], presence_mode=presence_mode, q=presence_quantile, min_facility_cells=min_facility_cells) for i in range(N)]))


    # ------------------------------------------------------------------------
    # Continuous / gravity-based accessibility signals (robust for soft maps)
    # ------------------------------------------------------------------------
    # Channel groups (Beijing mapping)
    RES_CH = [11]
    SCHOOL_CH = [13]
    HEALTH_CH = [8]
    TRANSIT_CH = [14]
    GREEN_CH = [7, 10]  # recreation service + tourist attraction as proxy green/amenity
    SERVICES_CH = [4, 5, 6, 12, 19]  # food/shopping/daily life/government/public service

    # NOTE: sigma is a distance-decay proxy. 6-10 works well for 100x100 grids.
    _ga_svc = [_gravity_accessibility(gen[i], SERVICES_CH, RES_CH, sigma=7.0) for i in range(N)]
    _ga_sch = [_gravity_accessibility(gen[i], SCHOOL_CH, RES_CH, sigma=7.0) for i in range(N)]
    _ga_hlt = [_gravity_accessibility(gen[i], HEALTH_CH, RES_CH, sigma=7.0) for i in range(N)]
    _ga_trn = [_gravity_accessibility(gen[i], TRANSIT_CH, RES_CH, sigma=7.0) for i in range(N)]
    _ga_grn = [_gravity_accessibility(gen[i], GREEN_CH, RES_CH, sigma=7.0) for i in range(N)]

    svc_access_gravity = float(np.mean([d["access_mean"] for d in _ga_svc]))
    sch_access_gravity = float(np.mean([d["access_mean"] for d in _ga_sch]))
    hlt_access_gravity = float(np.mean([d["access_mean"] for d in _ga_hlt]))
    trn_access_gravity = float(np.mean([d["access_mean"] for d in _ga_trn]))
    grn_access_gravity = float(np.mean([d["access_mean"] for d in _ga_grn]))

    svc_access_gini = float(np.mean([d["access_gini"] for d in _ga_svc]))
    sch_access_gini = float(np.mean([d["access_gini"] for d in _ga_sch]))
    hlt_access_gini = float(np.mean([d["access_gini"] for d in _ga_hlt]))
    trn_access_gini = float(np.mean([d["access_gini"] for d in _ga_trn]))
    grn_access_gini = float(np.mean([d["access_gini"] for d in _ga_grn]))

    svc_access_match = float(np.mean([d["access_match"] for d in _ga_svc]))
    sch_access_match = float(np.mean([d["access_match"] for d in _ga_sch]))
    hlt_access_match = float(np.mean([d["access_match"] for d in _ga_hlt]))
    trn_access_match = float(np.mean([d["access_match"] for d in _ga_trn]))
    grn_access_match = float(np.mean([d["access_match"] for d in _ga_grn]))

    real_density_mean = float(np.mean(real_tot))
    gen_density_mean = float(np.mean(gen_tot))
    density_ratio = float(gen_density_mean / (real_density_mean + 1e-8))
    density_l1 = float(abs(gen_density_mean - real_density_mean) / (real_density_mean + 1e-8))

    shopping_clustering = float(np.mean([_shopping_clustering_index(gen[i]) for i in range(N)]))
    employment_centrality = float(np.mean([_employment_centrality(gen[i]) for i in range(N)]))

    density_gradient = float(np.mean([_density_gradient(gen[i]) for i in range(N)]))
    land_use_compatibility = float(np.mean([_land_use_compatibility_score(gen[i]) for i in range(N)]))
    service_hierarchy = float(np.mean([_service_hierarchy_score(gen[i]) for i in range(N)]))
    open_space_distribution = float(np.mean([_open_space_distribution(gen[i]) for i in range(N)]))
    emergency_response_cov = float(np.mean([_emergency_response_coverage(gen[i]) for i in range(N)]))
    intensity_transition = float(np.mean([_intensity_transition_smoothness(gen[i]) for i in range(N)]))
    infra_demand_alignment = float(np.mean([_infrastructure_demand_alignment(gen[i]) for i in range(N)]))
    education_green_prox = float(np.mean([_education_green_proximity(gen[i]) for i in range(N)]))
    residential_noise_exp = float(np.mean([_residential_noise_exposure(gen[i]) for i in range(N)]))

    zmin = int(np.min(zones_hw))
    zmax = int(np.max(zones_hw))
    if zmin >= 0 and zmax < 20:
        zone_match = float(np.mean([np.mean(np.argmax(gen[i], axis=0) == zones_hw[i]) for i in range(N)]))
    else:
        zone_match = None

    
    functional_completeness_v7 = float('nan')
    cell_crispness_mean = float('nan')
    functional_completeness = float('nan')
    local_mix_entropy_mean = float('nan')
    local_mix_high_share = float('nan')
    green_connectivity_v5 = float('nan')
    coloc_housing_services = float('nan')
    coloc_jobs_transit = float('nan')
    coloc_housing_green = float('nan')
    conflict_rate = float('nan')
    tod_synergy = float('nan')
    resident_completeness_mean = float('nan')
    resident_complete_coverage = float('nan')
    svc_dist_p90 = float('nan')
    job_dist_p90 = float('nan')
    zone_cv_completeness = float('nan')
    svc_access_gravity_v5 = float('nan')
    svc_access_gini_v5 = float('nan')
    svc_access_match_v5 = float('nan')
    svc_access_bottom10_v5 = float('nan')
    job_access_gravity_v5 = float('nan')
    job_access_gini_v5 = float('nan')
    job_access_match_v5 = float('nan')
    job_access_bottom10_v5 = float('nan')
    trn_access_gravity_v5 = float('nan')
    trn_access_gini_v5 = float('nan')
    trn_access_bottom10_v5 = float('nan')

    # V6/V7 extended metrics defaults
    resident_completeness_mean_r20 = float('nan')
    resident_complete_coverage_r20 = float('nan')
    completeness_bottom10_r20 = float('nan')
    zone_cv_completeness_r20 = float('nan')
    resident_completeness_mean_r35 = float('nan')
    resident_complete_coverage_r35 = float('nan')
    completeness_bottom10_r35 = float('nan')
    zone_cv_completeness_r35 = float('nan')
    svc_dist_p95 = float('nan')
    job_dist_p95 = float('nan')
    svc_dist_p75 = float('nan')
    job_dist_p75 = float('nan')
    resident_k3_coverage_r20 = float('nan')
    resident_k4_coverage_r20 = float('nan')
    resident_k2_coverage_r20 = float('nan')
    resident_k3_coverage_r35 = float('nan')
    resident_k4_coverage_r35 = float('nan')
    resident_k2_coverage_r35 = float('nan')
    underserved_share_r20 = float('nan')
    underserved_share_r35 = float('nan')
    svc_access_bottom20_v7 = float('nan')
    job_access_bottom20_v7 = float('nan')
    buffer_conflict_r3 = float('nan')
    svc_area_share = float('nan')
    trn_area_share = float('nan')
    green_area_share = float('nan')
    jobs_area_share = float('nan')
    svc_supply_per_res = float('nan')
    sch_supply_per_res = float('nan')
    hlt_supply_per_res = float('nan')
    trn_supply_per_res = float('nan')
    grn_supply_per_res = float('nan')
    job_supply_per_res = float('nan')
    jobs_housing_centroid_dist = float('nan')
    jobs_polycentricity = float('nan')

    svc_access_gravity_v6 = float('nan')
    svc_access_gini_v6 = float('nan')
    svc_access_match_v6 = float('nan')
    svc_access_bottom10_v6 = float('nan')
    job_access_gravity_v6 = float('nan')
    job_access_gini_v6 = float('nan')
    job_access_match_v6 = float('nan')
    job_access_bottom10_v6 = float('nan')
    trn_access_gravity_v6 = float('nan')
    trn_access_gini_v6 = float('nan')
    trn_access_bottom10_v6 = float('nan')

    svc_access_gravity_v7 = float('nan')
    svc_access_gini_v7 = float('nan')
    svc_access_match_v7 = float('nan')
    svc_access_bottom10_v7 = float('nan')
    job_access_gravity_v7 = float('nan')
    job_access_gini_v7 = float('nan')
    job_access_match_v7 = float('nan')
    job_access_bottom10_v7 = float('nan')
    trn_access_gravity_v7 = float('nan')
    trn_access_gini_v7 = float('nan')
    trn_access_bottom10_v7 = float('nan')

    real_q: Dict[str, Dict[str, float]] = {}
    if str(dimension_profile) == "planning_6dimension":
        RES_CH = [11]
        JOBS_CH = [1, 2, 3, 15, 16]
        SERVICES_CH = [4, 5, 6, 12, 19]
        SCHOOL_CH = [13]
        HEALTH_CH = [8]
        TRANSIT_CH = [14]
        GREEN_CH = [7, 10]

        essential_groups = {
            "services": SERVICES_CH,
            "schools": SCHOOL_CH,
            "health": HEALTH_CH,
            "transit": TRANSIT_CH,
            "green": GREEN_CH,
            "jobs": JOBS_CH,
            "housing": RES_CH,
        }

        crisp_g, crisp_r = [], []
        func_comp_g, func_comp_r = [], []
        func_comp_v7_g, func_comp_v7_r = [], []  # softer completeness for v7
        mix_local_g, mix_local_r = [], []
        mix_high_g, mix_high_r = [], []
        green_conn_g, green_conn_r = [], []
        coloc_hs_g, coloc_hs_r = [], []
        coloc_jt_g, coloc_jt_r = [], []
        coloc_hg_g, coloc_hg_r = [], []
        conflict_g, conflict_r = [], []
        tod_g, tod_r = [], []
        comp_g, comp_r = [], []  # dicts

        ga_svc_g, ga_svc_r = [], []
        ga_job_g, ga_job_r = [], []
        ga_trn_g, ga_trn_r = [], []

        ga_svc_g20, ga_svc_r20 = [], []
        ga_job_g20, ga_job_r20 = [], []

        profile = str(dimension_profile)
        is_v6 = True
        is_v7 = True
        is_v9 = True

        mix_window = LOCAL_MIX_WINDOW_V7_DEFAULT
        facility_q = ARGMAX_INTENSITY_QUANTILE_V7_DEFAULT
        opp_q = OPPORTUNITY_QUANTILE_V7_DEFAULT
        opp_gamma = OPPORTUNITY_GAMMA_V7_DEFAULT
        bottom_pct = BOTTOM_TAIL_PCT_V6_DEFAULT

        comp_small_g, comp_small_r = [], []   # dicts at small radius
        comp_large_g, comp_large_r = [], []   # dicts at large radius
        buffer_conf_g, buffer_conf_r = [], []
        svc_area_g, svc_area_r = [], []
        trn_area_g, trn_area_r = [], []
        green_area_g, green_area_r = [], []
        jobs_area_g, jobs_area_r = [], []
        svc_per_res_g, svc_per_res_r = [], []
        sch_per_res_g, sch_per_res_r = [], []
        hlt_per_res_g, hlt_per_res_r = [], []
        trn_per_res_g, trn_per_res_r = [], []

        v11_density_gradient_g, v11_density_gradient_r = [], []
        v11_land_use_compat_g, v11_land_use_compat_r = [], []
        v11_service_hier_g, v11_service_hier_r = [], []
        v11_open_space_dist_g, v11_open_space_dist_r = [], []
        v11_emerg_response_g, v11_emerg_response_r = [], []
        v11_intensity_trans_g, v11_intensity_trans_r = [], []
        v11_infra_demand_g, v11_infra_demand_r = [], []
        v11_edu_green_g, v11_edu_green_r = [], []
        v11_noise_exp_g, v11_noise_exp_r = [], []
        grn_per_res_g, grn_per_res_r = [], []
        job_per_res_g, job_per_res_r = [], []
        jh_centroid_dist_g, jh_centroid_dist_r = [], []
        jobs_poly_g, jobs_poly_r = [], []

        for i in range(N):
            pg = gen[i]
            pr = real[i]

            crisp_g.append(_cell_crispness(pg, exclude_channels=[0]))
            crisp_r.append(_cell_crispness(pr, exclude_channels=[0]))

            func_comp_g.append(_functional_completeness(pg, essential_groups, q=90.0, min_cells=10))
            func_comp_r.append(_functional_completeness(pr, essential_groups, q=90.0, min_cells=10))
            func_comp_v7_g.append(_functional_completeness(pg, essential_groups, q=FUNCTIONAL_COMPLETENESS_Q_V7, min_cells=FUNCTIONAL_COMPLETENESS_MINCELLS_V7))
            func_comp_v7_r.append(_functional_completeness(pr, essential_groups, q=FUNCTIONAL_COMPLETENESS_Q_V7, min_cells=FUNCTIONAL_COMPLETENESS_MINCELLS_V7))

            lm_g = _local_mix_entropy(pg, window=mix_window, exclude_channels=[0])
            lm_r = _local_mix_entropy(pr, window=mix_window, exclude_channels=[0])
            mix_local_g.append(float(lm_g.get("mean", 0.0)))
            mix_local_r.append(float(lm_r.get("mean", 0.0)))
            mix_high_g.append(float(lm_g.get("high_share", 0.0)))
            mix_high_r.append(float(lm_r.get("high_share", 0.0)))

            green_conn_g.append(_green_connectivity(pg, GREEN_CH, q=90.0, min_cells=25))
            green_conn_r.append(_green_connectivity(pr, GREEN_CH, q=90.0, min_cells=25))

            coloc_hs_g.append(_co_location_corr(pg, RES_CH, SERVICES_CH, sigma=3.0))
            coloc_hs_r.append(_co_location_corr(pr, RES_CH, SERVICES_CH, sigma=3.0))
            coloc_jt_g.append(_co_location_corr(pg, JOBS_CH, TRANSIT_CH, sigma=3.0))
            coloc_jt_r.append(_co_location_corr(pr, JOBS_CH, TRANSIT_CH, sigma=3.0))
            coloc_hg_g.append(_co_location_corr(pg, RES_CH, GREEN_CH, sigma=3.0))
            coloc_hg_r.append(_co_location_corr(pr, RES_CH, GREEN_CH, sigma=3.0))

            conflict_g.append(_conflict_rate(pg))
            conflict_r.append(_conflict_rate(pr))

            tod_g.append(_tod_synergy(pg, radius=(COMPLETENESS_RADIUS_V6_LARGE_DEFAULT if is_v6 else COMPLETENESS_RADIUS_V5_DEFAULT),
                                      facility_q=facility_q))
            tod_r.append(_tod_synergy(pr, radius=(COMPLETENESS_RADIUS_V6_LARGE_DEFAULT if is_v6 else COMPLETENESS_RADIUS_V5_DEFAULT),
                                      facility_q=facility_q))

            comp_g.append(_resident_completeness_metrics(pg, zones_hw[i] if zones_hw is not None else None,
                                                         radius=COMPLETENESS_RADIUS_V5_DEFAULT,
                                                         facility_q=facility_q))
            comp_r.append(_resident_completeness_metrics(pr, zones_hw[i] if zones_hw is not None else None,
                                                         radius=COMPLETENESS_RADIUS_V5_DEFAULT,
                                                         facility_q=facility_q))

            ga_svc_g.append(_gravity_accessibility_v5(pg, SERVICES_CH, RES_CH, sigma=7.0, opp_q=opp_q, gamma=opp_gamma, bottom_pct=bottom_pct))
            ga_svc_r.append(_gravity_accessibility_v5(pr, SERVICES_CH, RES_CH, sigma=7.0, opp_q=opp_q, gamma=opp_gamma, bottom_pct=bottom_pct))
            ga_job_g.append(_gravity_accessibility_v5(pg, JOBS_CH, RES_CH, sigma=7.0, opp_q=opp_q, gamma=opp_gamma, bottom_pct=bottom_pct))
            ga_job_r.append(_gravity_accessibility_v5(pr, JOBS_CH, RES_CH, sigma=7.0, opp_q=opp_q, gamma=opp_gamma, bottom_pct=bottom_pct))
            ga_trn_g.append(_gravity_accessibility_v5(pg, TRANSIT_CH, RES_CH, sigma=7.0, opp_q=opp_q, gamma=opp_gamma, bottom_pct=bottom_pct))
            ga_trn_r.append(_gravity_accessibility_v5(pr, TRANSIT_CH, RES_CH, sigma=7.0, opp_q=opp_q, gamma=opp_gamma, bottom_pct=bottom_pct))

            if is_v9:
                ga_svc_g20.append(_gravity_accessibility_v5(pg, SERVICES_CH, RES_CH, sigma=7.0, opp_q=opp_q, gamma=opp_gamma, bottom_pct=20.0))
                ga_svc_r20.append(_gravity_accessibility_v5(pr, SERVICES_CH, RES_CH, sigma=7.0, opp_q=opp_q, gamma=opp_gamma, bottom_pct=20.0))
                ga_job_g20.append(_gravity_accessibility_v5(pg, JOBS_CH, RES_CH, sigma=7.0, opp_q=opp_q, gamma=opp_gamma, bottom_pct=20.0))
                ga_job_r20.append(_gravity_accessibility_v5(pr, JOBS_CH, RES_CH, sigma=7.0, opp_q=opp_q, gamma=opp_gamma, bottom_pct=20.0))

            # NEW v11 per-plan metrics (gen + real)
            v11_density_gradient_g.append(_density_gradient(pg))
            v11_density_gradient_r.append(_density_gradient(pr))
            v11_land_use_compat_g.append(_land_use_compatibility_score(pg))
            v11_land_use_compat_r.append(_land_use_compatibility_score(pr))
            v11_service_hier_g.append(_service_hierarchy_score(pg))
            v11_service_hier_r.append(_service_hierarchy_score(pr))
            v11_open_space_dist_g.append(_open_space_distribution(pg))
            v11_open_space_dist_r.append(_open_space_distribution(pr))
            v11_emerg_response_g.append(_emergency_response_coverage(pg))
            v11_emerg_response_r.append(_emergency_response_coverage(pr))
            v11_intensity_trans_g.append(_intensity_transition_smoothness(pg))
            v11_intensity_trans_r.append(_intensity_transition_smoothness(pr))
            v11_infra_demand_g.append(_infrastructure_demand_alignment(pg))
            v11_infra_demand_r.append(_infrastructure_demand_alignment(pr))
            v11_edu_green_g.append(_education_green_proximity(pg))
            v11_edu_green_r.append(_education_green_proximity(pr))
            v11_noise_exp_g.append(_residential_noise_exposure(pg))
            v11_noise_exp_r.append(_residential_noise_exposure(pr))

            if is_v6:
                zone_i = zones_hw[i] if zones_hw is not None else None

                comp_small_g.append(_resident_completeness_metrics_v6(pg, zone_i,
                                                                     radius=COMPLETENESS_RADIUS_V6_SMALL_DEFAULT,
                                                                     facility_q=facility_q, bottom_pct=bottom_pct))
                comp_small_r.append(_resident_completeness_metrics_v6(pr, zone_i,
                                                                     radius=COMPLETENESS_RADIUS_V6_SMALL_DEFAULT,
                                                                     facility_q=facility_q, bottom_pct=bottom_pct))
                comp_large_g.append(_resident_completeness_metrics_v6(pg, zone_i,
                                                                     radius=COMPLETENESS_RADIUS_V6_LARGE_DEFAULT,
                                                                     facility_q=facility_q, bottom_pct=bottom_pct))
                comp_large_r.append(_resident_completeness_metrics_v6(pr, zone_i,
                                                                     radius=COMPLETENESS_RADIUS_V6_LARGE_DEFAULT,
                                                                     facility_q=facility_q, bottom_pct=bottom_pct))
                buffer_conf_g.append(_buffer_conflict_rate_v6(pg, sensitive_channels=[11, 13, 8], nuisance_channels=[1, 2, 3],
                                                             radius=BUFFER_CONFLICT_RADIUS_V6_DEFAULT, nuisance_q=facility_q))
                buffer_conf_r.append(_buffer_conflict_rate_v6(pr, sensitive_channels=[11, 13, 8], nuisance_channels=[1, 2, 3],
                                                             radius=BUFFER_CONFLICT_RADIUS_V6_DEFAULT, nuisance_q=facility_q))
                svc_area_g.append(_facility_area_share_v6(pg, SERVICES_CH, facility_q=facility_q))
                svc_area_r.append(_facility_area_share_v6(pr, SERVICES_CH, facility_q=facility_q))
                trn_area_g.append(_facility_area_share_v6(pg, TRANSIT_CH, facility_q=facility_q))
                trn_area_r.append(_facility_area_share_v6(pr, TRANSIT_CH, facility_q=facility_q))
                green_area_g.append(_facility_area_share_v6(pg, GREEN_CH, facility_q=facility_q))
                green_area_r.append(_facility_area_share_v6(pr, GREEN_CH, facility_q=facility_q))
                jobs_area_g.append(_facility_area_share_v6(pg, JOBS_CH, facility_q=facility_q))
                jobs_area_r.append(_facility_area_share_v6(pr, JOBS_CH, facility_q=facility_q))
                svc_per_res_g.append(_supply_demand_ratio_v6(pg, SERVICES_CH, RES_CH))
                svc_per_res_r.append(_supply_demand_ratio_v6(pr, SERVICES_CH, RES_CH))
                sch_per_res_g.append(_supply_demand_ratio_v6(pg, SCHOOL_CH, RES_CH))
                sch_per_res_r.append(_supply_demand_ratio_v6(pr, SCHOOL_CH, RES_CH))
                hlt_per_res_g.append(_supply_demand_ratio_v6(pg, HEALTH_CH, RES_CH))
                hlt_per_res_r.append(_supply_demand_ratio_v6(pr, HEALTH_CH, RES_CH))
                trn_per_res_g.append(_supply_demand_ratio_v6(pg, TRANSIT_CH, RES_CH))
                trn_per_res_r.append(_supply_demand_ratio_v6(pr, TRANSIT_CH, RES_CH))
                grn_per_res_g.append(_supply_demand_ratio_v6(pg, GREEN_CH, RES_CH))
                grn_per_res_r.append(_supply_demand_ratio_v6(pr, GREEN_CH, RES_CH))
                job_per_res_g.append(_supply_demand_ratio_v6(pg, JOBS_CH, RES_CH))
                job_per_res_r.append(_supply_demand_ratio_v6(pr, JOBS_CH, RES_CH))
                jh_centroid_dist_g.append(_centroid_distance_v6(pg, JOBS_CH, RES_CH))
                jh_centroid_dist_r.append(_centroid_distance_v6(pr, JOBS_CH, RES_CH))
                jobs_poly_g.append(_polycentricity_v6(pg, JOBS_CH, bins=10, topk=5))
                jobs_poly_r.append(_polycentricity_v6(pr, JOBS_CH, bins=10, topk=5))

        cell_crispness_mean = float(np.mean(crisp_g))
        functional_completeness = float(np.mean(func_comp_g))
        functional_completeness_v7 = float(np.mean(func_comp_v7_g))
        local_mix_entropy_mean = float(np.mean(mix_local_g))
        local_mix_high_share = float(np.mean(mix_high_g))
        green_connectivity_v5 = float(np.mean(green_conn_g))
        coloc_housing_services = float(np.mean(coloc_hs_g))
        coloc_jobs_transit = float(np.mean(coloc_jt_g))
        coloc_housing_green = float(np.mean(coloc_hg_g))
        conflict_rate = float(np.mean(conflict_g))
        tod_synergy = float(np.mean(tod_g))

        resident_completeness_mean = float(np.mean([d.get("completeness_mean", 0.0) for d in comp_g]))
        resident_complete_coverage = float(np.mean([d.get("complete_coverage", 0.0) for d in comp_g]))
        svc_dist_p90 = float(np.nanmean([d.get("svc_dist_p90", float("nan")) for d in comp_g]))
        job_dist_p90 = float(np.nanmean([d.get("job_dist_p90", float("nan")) for d in comp_g]))
        zone_cv_completeness = float(np.nanmean([d.get("zone_cv", float("nan")) for d in comp_g]))

        svc_access_gravity_v5 = float(np.mean([d.get("access_mean", 0.0) for d in ga_svc_g]))
        svc_access_gini_v5 = float(np.mean([d.get("access_gini", 0.0) for d in ga_svc_g]))
        svc_access_match_v5 = float(np.mean([d.get("access_match", 0.0) for d in ga_svc_g]))
        svc_access_bottom10_v5 = float(np.mean([d.get("access_bottom", 0.0) for d in ga_svc_g]))

        job_access_gravity_v5 = float(np.mean([d.get("access_mean", 0.0) for d in ga_job_g]))
        job_access_gini_v5 = float(np.mean([d.get("access_gini", 0.0) for d in ga_job_g]))
        job_access_match_v5 = float(np.mean([d.get("access_match", 0.0) for d in ga_job_g]))
        job_access_bottom10_v5 = float(np.mean([d.get("access_bottom", 0.0) for d in ga_job_g]))

        trn_access_gravity_v5 = float(np.mean([d.get("access_mean", 0.0) for d in ga_trn_g]))
        trn_access_gini_v5 = float(np.mean([d.get("access_gini", 0.0) for d in ga_trn_g]))
        trn_access_bottom10_v5 = float(np.mean([d.get("access_bottom", 0.0) for d in ga_trn_g]))


        if is_v6:
            resident_completeness_mean_r20 = float(np.mean([d.get("completeness_mean", 0.0) for d in comp_small_g]))
            resident_complete_coverage_r20 = float(np.mean([d.get("complete_coverage", 0.0) for d in comp_small_g]))
            completeness_bottom10_r20 = float(np.mean([d.get("completeness_bottom", 0.0) for d in comp_small_g]))
            zone_cv_completeness_r20 = float(np.nanmean([d.get("zone_cv", float("nan")) for d in comp_small_g]))

            resident_k3_coverage_r20 = float(np.mean([d.get("k3_coverage", 0.0) for d in comp_small_g]))
            resident_k4_coverage_r20 = float(np.mean([d.get("k4_coverage", 0.0) for d in comp_small_g]))
            resident_k2_coverage_r20 = float(np.mean([d.get("k2_coverage", 0.0) for d in comp_small_g]))
            underserved_share_r20 = float(np.mean([d.get("underserved_share", 0.0) for d in comp_small_g]))

            resident_completeness_mean_r35 = float(np.mean([d.get("completeness_mean", 0.0) for d in comp_large_g]))
            resident_complete_coverage_r35 = float(np.mean([d.get("complete_coverage", 0.0) for d in comp_large_g]))
            completeness_bottom10_r35 = float(np.mean([d.get("completeness_bottom", 0.0) for d in comp_large_g]))
            zone_cv_completeness_r35 = float(np.nanmean([d.get("zone_cv", float("nan")) for d in comp_large_g]))

            resident_k3_coverage_r35 = float(np.mean([d.get("k3_coverage", 0.0) for d in comp_large_g]))
            resident_k4_coverage_r35 = float(np.mean([d.get("k4_coverage", 0.0) for d in comp_large_g]))
            resident_k2_coverage_r35 = float(np.mean([d.get("k2_coverage", 0.0) for d in comp_large_g]))
            underserved_share_r35 = float(np.mean([d.get("underserved_share", 0.0) for d in comp_large_g]))

            svc_dist_p95 = float(np.nanmean([d.get("svc_dist_p95", float("nan")) for d in comp_large_g]))
            job_dist_p95 = float(np.nanmean([d.get("job_dist_p95", float("nan")) for d in comp_large_g]))

            svc_dist_p75 = float(np.nanmean([d.get("svc_dist_p75", float("nan")) for d in comp_large_g]))
            job_dist_p75 = float(np.nanmean([d.get("job_dist_p75", float("nan")) for d in comp_large_g]))

            buffer_conflict_r3 = float(np.mean(buffer_conf_g))
            svc_area_share = float(np.mean(svc_area_g))
            trn_area_share = float(np.mean(trn_area_g))
            green_area_share = float(np.mean(green_area_g))
            jobs_area_share = float(np.mean(jobs_area_g))

            svc_supply_per_res = float(np.nanmean(svc_per_res_g))
            sch_supply_per_res = float(np.nanmean(sch_per_res_g))
            hlt_supply_per_res = float(np.nanmean(hlt_per_res_g))
            trn_supply_per_res = float(np.nanmean(trn_per_res_g))
            grn_supply_per_res = float(np.nanmean(grn_per_res_g))
            job_supply_per_res = float(np.nanmean(job_per_res_g))

            jobs_housing_centroid_dist = float(np.nanmean(jh_centroid_dist_g))
            jobs_polycentricity = float(np.nanmean(jobs_poly_g))

            svc_access_gravity_v7 = float(svc_access_gravity_v5)
            svc_access_gini_v7 = float(svc_access_gini_v5)
            svc_access_match_v7 = float(svc_access_match_v5)
            svc_access_bottom10_v7 = float(svc_access_bottom10_v5)

            job_access_gravity_v7 = float(job_access_gravity_v5)
            job_access_gini_v7 = float(job_access_gini_v5)
            job_access_match_v7 = float(job_access_match_v5)
            job_access_bottom10_v7 = float(job_access_bottom10_v5)

            trn_access_gravity_v7 = float(trn_access_gravity_v5)
            trn_access_gini_v7 = float(trn_access_gini_v5)
            trn_access_bottom10_v7 = float(trn_access_bottom10_v5)

            if len(ga_svc_g20) > 0:
                svc_access_bottom20_v7 = float(np.mean([d.get("access_bottom", 0.0) for d in ga_svc_g20]))
            if len(ga_job_g20) > 0:
                job_access_bottom20_v7 = float(np.mean([d.get("access_bottom", 0.0) for d in ga_job_g20]))

            svc_access_gravity_v6 = float(svc_access_gravity_v5)
            svc_access_gini_v6 = float(svc_access_gini_v5)
            svc_access_match_v6 = float(svc_access_match_v5)
            svc_access_bottom10_v6 = float(svc_access_bottom10_v5)

            job_access_gravity_v6 = float(job_access_gravity_v5)
            job_access_gini_v6 = float(job_access_gini_v5)
            job_access_match_v6 = float(job_access_match_v5)
            job_access_bottom10_v6 = float(job_access_bottom10_v5)

            trn_access_gravity_v6 = float(trn_access_gravity_v5)
            trn_access_gini_v6 = float(trn_access_gini_v5)
            trn_access_bottom10_v6 = float(trn_access_bottom10_v5)
        else:
            resident_completeness_mean_r20 = float("nan")
            resident_complete_coverage_r20 = float("nan")
            completeness_bottom10_r20 = float("nan")
            zone_cv_completeness_r20 = float("nan")
            resident_completeness_mean_r35 = float("nan")
            resident_complete_coverage_r35 = float("nan")
            completeness_bottom10_r35 = float("nan")
            zone_cv_completeness_r35 = float("nan")
            svc_dist_p95 = float("nan")
            job_dist_p95 = float("nan")
            svc_dist_p75 = float("nan")
            job_dist_p75 = float("nan")
            resident_k3_coverage_r20 = float("nan")
            resident_k4_coverage_r20 = float("nan")
            resident_k3_coverage_r35 = float("nan")
            resident_k4_coverage_r35 = float("nan")
            underserved_share_r20 = float("nan")
            underserved_share_r35 = float("nan")
            svc_access_bottom20_v7 = float("nan")
            job_access_bottom20_v7 = float("nan")
            buffer_conflict_r3 = float("nan")
            svc_area_share = float("nan")
            trn_area_share = float("nan")
            green_area_share = float("nan")
            jobs_area_share = float("nan")
            svc_supply_per_res = float("nan")
            sch_supply_per_res = float("nan")
            hlt_supply_per_res = float("nan")
            trn_supply_per_res = float("nan")
            grn_supply_per_res = float("nan")
            job_supply_per_res = float("nan")
            jobs_housing_centroid_dist = float("nan")
            jobs_polycentricity = float("nan")
            svc_access_gravity_v6 = float("nan")
            svc_access_gini_v6 = float("nan")
            svc_access_match_v6 = float("nan")
            svc_access_bottom10_v6 = float("nan")
            job_access_gravity_v6 = float("nan")
            job_access_gini_v6 = float("nan")
            job_access_match_v6 = float("nan")
            job_access_bottom10_v6 = float("nan")
            trn_access_gravity_v6 = float("nan")
            trn_access_gini_v6 = float("nan")
            trn_access_bottom10_v6 = float("nan")
            functional_completeness_v7 = float("nan")
            svc_access_gravity_v7 = float("nan")
            svc_access_gini_v7 = float("nan")
            svc_access_match_v7 = float("nan")
            svc_access_bottom10_v7 = float("nan")
            job_access_gravity_v7 = float("nan")
            job_access_gini_v7 = float("nan")
            job_access_match_v7 = float("nan")
            job_access_bottom10_v7 = float("nan")
            trn_access_gravity_v7 = float("nan")
            trn_access_gini_v7 = float("nan")
            trn_access_bottom10_v7 = float("nan")

        real_series = {
            "cell_crispness_mean": crisp_r,
            "functional_completeness": func_comp_r,
            "functional_completeness_v7": func_comp_v7_r,
            "local_mix_entropy_mean": mix_local_r,
            "local_mix_high_share": mix_high_r,
            "green_connectivity_v5": green_conn_r,
            "coloc_housing_services": coloc_hs_r,
            "coloc_jobs_transit": coloc_jt_r,
            "coloc_housing_green": coloc_hg_r,
            "conflict_rate": conflict_r,
            "tod_synergy": tod_r,
            "resident_completeness_mean": [d.get("completeness_mean", 0.0) for d in comp_r],
            "resident_complete_coverage": [d.get("complete_coverage", 0.0) for d in comp_r],
            "svc_dist_p90": [d.get("svc_dist_p90", float("nan")) for d in comp_r],
            "job_dist_p90": [d.get("job_dist_p90", float("nan")) for d in comp_r],
            "zone_cv_completeness": [d.get("zone_cv", float("nan")) for d in comp_r],
            "svc_access_gravity_v5": [d.get("access_mean", 0.0) for d in ga_svc_r],
            "svc_access_gini_v5": [d.get("access_gini", 0.0) for d in ga_svc_r],
            "svc_access_bottom10_v5": [d.get("access_bottom", 0.0) for d in ga_svc_r],
            "job_access_gravity_v5": [d.get("access_mean", 0.0) for d in ga_job_r],
            "job_access_gini_v5": [d.get("access_gini", 0.0) for d in ga_job_r],
            "job_access_bottom10_v5": [d.get("access_bottom", 0.0) for d in ga_job_r],
            "trn_access_gravity_v5": [d.get("access_mean", 0.0) for d in ga_trn_r],

            "resident_completeness_mean_r20": [d.get("completeness_mean", 0.0) for d in comp_small_r],
            "resident_complete_coverage_r20": [d.get("complete_coverage", 0.0) for d in comp_small_r],
            "resident_k3_coverage_r20": [d.get("k3_coverage", 0.0) for d in comp_small_r],
            "resident_k4_coverage_r20": [d.get("k4_coverage", 0.0) for d in comp_small_r],
            "resident_k2_coverage_r20": [d.get("k2_coverage", 0.0) for d in comp_small_r],
            "underserved_share_r20": [d.get("underserved_share", 0.0) for d in comp_small_r],
            "completeness_bottom10_r20": [d.get("completeness_bottom", 0.0) for d in comp_small_r],
            "zone_cv_completeness_r20": [d.get("zone_cv", float("nan")) for d in comp_small_r],

            "resident_completeness_mean_r35": [d.get("completeness_mean", 0.0) for d in comp_large_r],
            "resident_complete_coverage_r35": [d.get("complete_coverage", 0.0) for d in comp_large_r],
            "resident_k3_coverage_r35": [d.get("k3_coverage", 0.0) for d in comp_large_r],
            "resident_k4_coverage_r35": [d.get("k4_coverage", 0.0) for d in comp_large_r],
            "resident_k2_coverage_r35": [d.get("k2_coverage", 0.0) for d in comp_large_r],
            "underserved_share_r35": [d.get("underserved_share", 0.0) for d in comp_large_r],
            "completeness_bottom10_r35": [d.get("completeness_bottom", 0.0) for d in comp_large_r],
            "zone_cv_completeness_r35": [d.get("zone_cv", float("nan")) for d in comp_large_r],

            "svc_dist_p95": [d.get("svc_dist_p95", float("nan")) for d in comp_large_r],
            "job_dist_p95": [d.get("job_dist_p95", float("nan")) for d in comp_large_r],

            "svc_dist_p75": [d.get("svc_dist_p75", float("nan")) for d in comp_large_r],
            "job_dist_p75": [d.get("job_dist_p75", float("nan")) for d in comp_large_r],

            "buffer_conflict_r3": buffer_conf_r,
            "svc_area_share": svc_area_r,
            "trn_area_share": trn_area_r,
            "green_area_share": green_area_r,
            "jobs_area_share": jobs_area_r,

            "svc_supply_per_res": svc_per_res_r,
            "sch_supply_per_res": sch_per_res_r,
            "hlt_supply_per_res": hlt_per_res_r,
            "trn_supply_per_res": trn_per_res_r,
            "grn_supply_per_res": grn_per_res_r,
            "job_supply_per_res": job_per_res_r,

            "jobs_housing_centroid_dist": jh_centroid_dist_r,
            "jobs_polycentricity": jobs_poly_r,

            "svc_access_gravity_v6": [d.get("access_mean", 0.0) for d in ga_svc_r],
            "svc_access_gini_v6": [d.get("access_gini", 0.0) for d in ga_svc_r],
            "svc_access_bottom10_v6": [d.get("access_bottom", 0.0) for d in ga_svc_r],

            "job_access_gravity_v6": [d.get("access_mean", 0.0) for d in ga_job_r],
            "job_access_gini_v6": [d.get("access_gini", 0.0) for d in ga_job_r],
            "job_access_bottom10_v6": [d.get("access_bottom", 0.0) for d in ga_job_r],

            "trn_access_gravity_v6": [d.get("access_mean", 0.0) for d in ga_trn_r],
            "trn_access_gini_v6": [d.get("access_gini", 0.0) for d in ga_trn_r],
            "trn_access_bottom10_v6": [d.get("access_bottom", 0.0) for d in ga_trn_r],

            "svc_access_gravity_v7": [d.get("access_mean", 0.0) for d in ga_svc_r],
            "svc_access_gini_v7": [d.get("access_gini", 0.0) for d in ga_svc_r],
            "svc_access_bottom10_v7": [d.get("access_bottom", 0.0) for d in ga_svc_r],
            "svc_access_bottom20_v7": [d.get("access_bottom", 0.0) for d in ga_svc_r20],

            "job_access_gravity_v7": [d.get("access_mean", 0.0) for d in ga_job_r],
            "job_access_gini_v7": [d.get("access_gini", 0.0) for d in ga_job_r],
            "job_access_bottom10_v7": [d.get("access_bottom", 0.0) for d in ga_job_r],
            "job_access_bottom20_v7": [d.get("access_bottom", 0.0) for d in ga_job_r20],

            "trn_access_gravity_v7": [d.get("access_mean", 0.0) for d in ga_trn_r],
            "trn_access_gini_v7": [d.get("access_gini", 0.0) for d in ga_trn_r],
            "trn_access_bottom10_v7": [d.get("access_bottom", 0.0) for d in ga_trn_r],

            "density_gradient": v11_density_gradient_r,
            "land_use_compatibility": v11_land_use_compat_r,
            "service_hierarchy": v11_service_hier_r,
            "open_space_distribution": v11_open_space_dist_r,
            "emergency_response_coverage": v11_emerg_response_r,
            "intensity_transition_smoothness": v11_intensity_trans_r,
            "infrastructure_demand_alignment": v11_infra_demand_r,
            "education_green_proximity": v11_edu_green_r,
            "residential_noise_exposure": v11_noise_exp_r,
        }
        for k, arr in real_series.items():
            try:
                real_q[k] = _quantiles(list(arr))
            except Exception:
                real_q[k] = {"q05": float("nan"), "q50": float("nan"), "q95": float("nan")}

        try:
            real_q["land_mix_entropy"] = _quantiles([_land_mix_entropy(real[i]) for i in range(N)])
            real_q["simpson_diversity"] = _quantiles([_simpson_diversity(real[i]) for i in range(N)])
            real_q["mixing_index"] = _quantiles([_mixing_index(real[i]) for i in range(N)])
            real_q["mixed_use_intensity"] = _quantiles([_mixed_use_intensity(real[i]) for i in range(N)])
            real_q["walkability_score"] = _quantiles([_walkability_score(real[i]) for i in range(N)])
            real_q["compactness_ratio"] = _quantiles([_compactness_ratio(real[i]) for i in range(N)])
            real_q["transit_density"] = _quantiles([
                _transit_oriented_density(real[i], presence_mode=presence_mode,
                                          q=presence_quantile, min_facility_cells=min_facility_cells)
                for i in range(N)
            ])
            real_q["urban_heat_risk"] = _quantiles([_urban_heat_island_risk(real[i]) for i in range(N)])
            real_q["flood_resilience"] = _quantiles([_flood_resilience_score(real[i]) for i in range(N)])
            if zones_hw is not None:
                real_q["shannon_diversity_by_zone"] = _quantiles([
                    _shannon_diversity_by_zone(real[i], zones_hw[i]) for i in range(N)
                ])
            # NEW v11 core metric references
            real_q["street_connectivity"] = _quantiles([_street_network_connectivity(real[i]) for i in range(N)])
            real_q["recreation_space_accessibility"] = _quantiles([
                _recreation_space_accessibility(real[i], presence_mode=presence_mode,
                                                 q=presence_quantile, min_facility_cells=min_facility_cells)
                for i in range(N)
            ])
        except Exception:
            pass
    else:
        # Defaults (present but not used)
        cell_crispness_mean = float("nan")
        functional_completeness = float("nan")
        local_mix_entropy_mean = float("nan")
        local_mix_high_share = float("nan")
        green_connectivity_v5 = float("nan")
        coloc_housing_services = float("nan")
        coloc_jobs_transit = float("nan")
        coloc_housing_green = float("nan")
        conflict_rate = float("nan")
        tod_synergy = float("nan")
        resident_completeness_mean = float("nan")
        resident_complete_coverage = float("nan")
        svc_dist_p90 = float("nan")
        job_dist_p90 = float("nan")
        zone_cv_completeness = float("nan")
        svc_access_gravity_v5 = float("nan")
        svc_access_gini_v5 = float("nan")
        svc_access_match_v5 = float("nan")
        svc_access_bottom10_v5 = float("nan")
        job_access_gravity_v5 = float("nan")
        job_access_gini_v5 = float("nan")
        job_access_match_v5 = float("nan")
        job_access_bottom10_v5 = float("nan")
        trn_access_gravity_v5 = float("nan")
        trn_access_gini_v5 = float("nan")
        trn_access_bottom10_v5 = float("nan")
    urban = {
        # Land use diversity
        "land_mix_entropy": ent,
        "simpson_diversity": simp_div,
        "shannon_diversity_by_zone": shannon_zone,
        # Spatial structure
        "gini_total": gini,
        "edge_density": edge,
        "compactness_ratio": compact,
        # REMOVED: "patch_density": patch_dens,
        # REMOVED: "compactness_ratio": compact,
        "aggregation_index": aggreg,
        "mixing_index": mixing,
        # Jobs-housing balance
        "jobs_housing_balance": jh_balance,
        # Accessibility and equity
        "service_coverage": svc_coverage,
        "spatial_gini": spatial_gini_val,
        "avg_service_distance": avg_svc_dist,
        "service_access_inequality": service_access_inequality,
        "school_access_inequality": school_access_inequality,
        "healthcare_access_inequality": healthcare_access_inequality,
        "transit_access_inequality": transit_access_inequality,
        "green_access_inequality": green_access_inequality,
        # NEW: Recreation space
        "recreation_space_ratio": recreation_ratio,
        "recreation_space_accessibility": recreation_accessibility,
        # NEW: Connectivity
        "street_connectivity": street_connectivity,
        "network_density": network_density,
        # NEW: Density
        "building_density": building_density,
        "floor_area_ratio": floor_area_ratio,
        "mixed_use_intensity": mixed_use_intensity,
        # NEW: Walkability
        "walkability_score": walkability_score,
        "fifteen_min_coverage": fifteen_min_coverage,
        # NEW: Transit
        "transit_coverage": transit_coverage,
        "transit_density": transit_density,
        # NEW: Environmental
        "urban_heat_risk": urban_heat_risk,
        "flood_resilience": flood_resilience,
        # NEW: Social infrastructure
        "school_accessibility": school_accessibility,
        "healthcare_accessibility": healthcare_access,
"svc_access_gravity": svc_access_gravity,
"sch_access_gravity": sch_access_gravity,
"hlt_access_gravity": hlt_access_gravity,
"trn_access_gravity": trn_access_gravity,
"grn_access_gravity": grn_access_gravity,
"svc_access_gini": svc_access_gini,
"sch_access_gini": sch_access_gini,
"hlt_access_gini": hlt_access_gini,
"trn_access_gini": trn_access_gini,
"grn_access_gini": grn_access_gini,
"svc_access_match": svc_access_match,
"sch_access_match": sch_access_match,
"hlt_access_match": hlt_access_match,
"trn_access_match": trn_access_match,
"grn_access_match": grn_access_match,
        "cell_crispness_mean": cell_crispness_mean,
        "functional_completeness": functional_completeness,
        "functional_completeness_v7": functional_completeness_v7,
        "local_mix_entropy_mean": local_mix_entropy_mean,
        "local_mix_high_share": local_mix_high_share,
        "green_connectivity_v5": green_connectivity_v5,
        "coloc_housing_services": coloc_housing_services,
        "coloc_jobs_transit": coloc_jobs_transit,
        "coloc_housing_green": coloc_housing_green,
        "conflict_rate": conflict_rate,
        "tod_synergy": tod_synergy,
        "resident_completeness_mean": resident_completeness_mean,
        "resident_complete_coverage": resident_complete_coverage,
        "svc_dist_p90": svc_dist_p90,
        "job_dist_p90": job_dist_p90,
        "zone_cv_completeness": zone_cv_completeness,
        "svc_access_gravity_v5": svc_access_gravity_v5,
        "svc_access_gini_v5": svc_access_gini_v5,
        "svc_access_match_v5": svc_access_match_v5,
        "svc_access_bottom10_v5": svc_access_bottom10_v5,
        "job_access_gravity_v5": job_access_gravity_v5,
        "job_access_gini_v5": job_access_gini_v5,
        "job_access_match_v5": job_access_match_v5,
        "job_access_bottom10_v5": job_access_bottom10_v5,
        "trn_access_gravity_v5": trn_access_gravity_v5,
        "trn_access_gini_v5": trn_access_gini_v5,
        "trn_access_bottom10_v5": trn_access_bottom10_v5,
        "resident_completeness_mean_r20": resident_completeness_mean_r20,
        "resident_complete_coverage_r20": resident_complete_coverage_r20,
        "resident_k3_coverage_r20": resident_k3_coverage_r20,
        "resident_k4_coverage_r20": resident_k4_coverage_r20,
        "resident_k2_coverage_r20": resident_k2_coverage_r20,
        "underserved_share_r20": underserved_share_r20,
        "completeness_bottom10_r20": completeness_bottom10_r20,
        "zone_cv_completeness_r20": zone_cv_completeness_r20,

        "resident_completeness_mean_r35": resident_completeness_mean_r35,
        "resident_complete_coverage_r35": resident_complete_coverage_r35,
        "resident_k3_coverage_r35": resident_k3_coverage_r35,
        "resident_k4_coverage_r35": resident_k4_coverage_r35,
        "resident_k2_coverage_r35": resident_k2_coverage_r35,
        "underserved_share_r35": underserved_share_r35,
        "completeness_bottom10_r35": completeness_bottom10_r35,
        "zone_cv_completeness_r35": zone_cv_completeness_r35,

        "svc_dist_p95": svc_dist_p95,
        "job_dist_p95": job_dist_p95,

        "svc_dist_p75": svc_dist_p75,
        "job_dist_p75": job_dist_p75,

        "buffer_conflict_r3": buffer_conflict_r3,
        "svc_area_share": svc_area_share,
        "trn_area_share": trn_area_share,
        "green_area_share": green_area_share,
        "jobs_area_share": jobs_area_share,

        "svc_supply_per_res": svc_supply_per_res,
        "sch_supply_per_res": sch_supply_per_res,
        "hlt_supply_per_res": hlt_supply_per_res,
        "trn_supply_per_res": trn_supply_per_res,
        "grn_supply_per_res": grn_supply_per_res,
        "job_supply_per_res": job_supply_per_res,

        "jobs_housing_centroid_dist": jobs_housing_centroid_dist,
        "jobs_polycentricity": jobs_polycentricity,

        "svc_access_gravity_v6": svc_access_gravity_v6,
        "svc_access_gini_v6": svc_access_gini_v6,
        "svc_access_match_v6": svc_access_match_v6,
        "svc_access_bottom10_v6": svc_access_bottom10_v6,

        "job_access_gravity_v6": job_access_gravity_v6,
        "job_access_gini_v6": job_access_gini_v6,
        "job_access_match_v6": job_access_match_v6,
        "job_access_bottom10_v6": job_access_bottom10_v6,

        "trn_access_gravity_v6": trn_access_gravity_v6,
        "trn_access_gini_v6": trn_access_gini_v6,
        "trn_access_bottom10_v6": trn_access_bottom10_v6,

        "svc_access_gravity_v7": svc_access_gravity_v7,
        "svc_access_gini_v7": svc_access_gini_v7,
        "svc_access_match_v7": svc_access_match_v7,
        "svc_access_bottom10_v7": svc_access_bottom10_v7,
        "svc_access_bottom20_v7": svc_access_bottom20_v7,

        "job_access_gravity_v7": job_access_gravity_v7,
        "job_access_gini_v7": job_access_gini_v7,
        "job_access_match_v7": job_access_match_v7,
        "job_access_bottom10_v7": job_access_bottom10_v7,
        "job_access_bottom20_v7": job_access_bottom20_v7,

        "trn_access_gravity_v7": trn_access_gravity_v7,
        "trn_access_gini_v7": trn_access_gini_v7,
        "trn_access_bottom10_v7": trn_access_bottom10_v7,

        "density_ratio": density_ratio,
        "density_l1": density_l1,
        "employment_centrality": employment_centrality,
        "density_gradient": density_gradient,
        "land_use_compatibility": land_use_compatibility,
        "service_hierarchy": service_hierarchy,
        "open_space_distribution": open_space_distribution,
        "emergency_response_coverage": emergency_response_cov,
        "intensity_transition_smoothness": intensity_transition,
        "infrastructure_demand_alignment": infra_demand_alignment,
        "education_green_proximity": education_green_prox,
        "residential_noise_exposure": residential_noise_exp,
        "zone_match_accuracy": zone_match
    }

    real_tot = real.sum(axis=1)  # (N,100,100)
    gen_tot = gen.sum(axis=1)
    diff = gen_tot - real_tot
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    rmse = float(math.sqrt(max(0.0, mse)))

    a = real_tot.reshape(N, -1).mean(axis=0)
    b = gen_tot.reshape(N, -1).mean(axis=0)
    a = a - float(np.mean(a))
    b = b - float(np.mean(b))
    denom = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
    corr = float(np.dot(a, b) / denom) if denom > 0 else 0.0

    spatial = {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "corr_total": corr,
        "spcorr_local01": float(np.mean([_spcorr01(gen_tot[i]) for i in range(N)])),
    }


    ent = float(np.mean([_land_mix_entropy(gen[i]) for i in range(N)]))
    simp_div = float(np.mean([_simpson_diversity(gen[i]) for i in range(N)]))
    shannon_zone = float(np.mean([_shannon_diversity_by_zone(gen[i], zones_hw[i]) for i in range(N)]))
    
    gini = float(np.mean([_gini(gen_tot[i]) for i in range(N)]))
    edge = float(np.mean([_edge_density(np.argmax(gen[i], axis=0)) for i in range(N)]))
    patch_dens = float(np.mean([_patch_density(np.argmax(gen[i], axis=0)) for i in range(N)]))
    compact = float(np.mean([_compactness_ratio(gen[i]) for i in range(N)]))
    aggreg = float(np.mean([_aggregation_index(np.argmax(gen[i], axis=0)) for i in range(N)]))
    mixing = float(np.mean([_mixing_index(gen[i]) for i in range(N)]))
    
    jh_balance = float(np.mean([_jobs_housing_balance(gen[i]) for i in range(N)]))
    
    svc_coverage = float(np.mean([_service_coverage(gen[i], presence_mode=presence_mode, q=presence_quantile, min_facility_cells=min_facility_cells) for i in range(N)]))
    spatial_gini_val = float(np.mean([_spatial_gini(gen[i]) for i in range(N)]))
    avg_svc_dist = float(np.mean([_avg_nearest_service_distance(gen[i], presence_mode=presence_mode, q=presence_quantile, min_facility_cells=min_facility_cells) for i in range(N)]))

    recreation_ratio = float(np.mean([_recreation_space_ratio(gen[i]) for i in range(N)]))
    recreation_accessibility = float(np.mean([_recreation_space_accessibility(gen[i], presence_mode=presence_mode, q=presence_quantile, min_facility_cells=min_facility_cells) for i in range(N)]))
    
    street_connectivity = float(np.mean([_street_network_connectivity(gen[i]) for i in range(N)]))
    network_density = float(np.mean([_network_density(gen[i]) for i in range(N)]))
    
    building_density = float(np.mean([_building_density(gen[i]) for i in range(N)]))
    floor_area_ratio = float(np.mean([_floor_area_ratio(gen[i]) for i in range(N)]))
    mixed_use_intensity = float(np.mean([_mixed_use_intensity(gen[i]) for i in range(N)]))
    
    walkability_score = float(np.mean([_walkability_score(gen[i]) for i in range(N)]))
    fifteen_min_coverage = float(np.mean([_fifteen_minute_city_coverage(gen[i], presence_mode=presence_mode, q=presence_quantile, min_facility_cells=min_facility_cells) for i in range(N)]))
    
    transit_coverage = float(np.mean([_transit_coverage(gen[i], presence_mode=presence_mode, q=presence_quantile, min_facility_cells=min_facility_cells) for i in range(N)]))
    transit_density = float(np.mean([_transit_oriented_density(gen[i], presence_mode=presence_mode, q=presence_quantile, min_facility_cells=min_facility_cells) for i in range(N)]))
    
    urban_heat_risk = float(np.mean([_urban_heat_island_risk(gen[i]) for i in range(N)]))
    flood_resilience = float(np.mean([_flood_resilience_score(gen[i]) for i in range(N)]))
    
    school_accessibility = float(np.mean([_school_accessibility(gen[i], presence_mode=presence_mode, q=presence_quantile, min_facility_cells=min_facility_cells) for i in range(N)]))
    healthcare_access = float(np.mean([_healthcare_accessibility(gen[i], presence_mode=presence_mode, q=presence_quantile, min_facility_cells=min_facility_cells) for i in range(N)]))
    
    shopping_clustering = float(np.mean([_shopping_clustering_index(gen[i]) for i in range(N)]))
    employment_centrality = float(np.mean([_employment_centrality(gen[i]) for i in range(N)]))

    zmin = int(np.min(zones_hw))
    zmax = int(np.max(zones_hw))
    if zmin >= 0 and zmax < 20:
        zone_match = float(np.mean([np.mean(np.argmax(gen[i], axis=0) == zones_hw[i]) for i in range(N)]))
    else:
        zone_match = None
    
    comp_by_plan = gen.mean(axis=(2, 3))  # (N,20)
    feat_std = float(np.mean(np.std(comp_by_plan, axis=0)))
    ent_mean = float(np.mean([_land_mix_entropy(gen[i]) for i in range(N)]))

    rng = np.random.default_rng(0)
    pairs = []
    if N >= 2:
        for _ in range(min(max_pairs, N * (N - 1) // 2)):
            i = int(rng.integers(0, N))
            j = int(rng.integers(0, N - 1))
            if j >= i:
                j += 1
            pairs.append((i, j))
    cos_div = []
    l2_div = []
    for i, j in pairs:
        cos_div.append(_cos_dist(comp_by_plan[i], comp_by_plan[j]))
        l2_div.append(float(np.linalg.norm(_normalize_nonneg(comp_by_plan[i]) - _normalize_nonneg(comp_by_plan[j]))))
    diversity = {
        "feat_std": feat_std,
        "ent_mean": ent_mean,
        "pair_cos_dist": float(np.mean(cos_div)) if cos_div else None,
        "pair_l2": float(np.mean(l2_div)) if l2_div else None,
        "unique_dom_ratio": float(len({tuple(np.argmax(gen[i], axis=0).reshape(-1)[:200].tolist()) for i in range(N)}) / N) if N > 0 else None,
    }
    
    # ========================================================================
    # COMPUTE DIMENSION SCORES FROM METRICS
    # ========================================================================
    
    result = {
        "distribution": dist, 
        "spatial": spatial, 
        "urban": urban, 
        "diversity": diversity, 
        "n": int(N),
        "real_q": real_q
    }
    
    try:
        dimension_scores = compute_dimension_scores_from_metrics(result, profile=dimension_profile)
        result["dimensions"] = dimension_scores
        # Helpful when debugging downstream summary reports
        result["dimensions_error"] = None
    except Exception as e:
        print(f"[WARNING] Could not compute dimension scores: {e}")
        result["dimensions"] = {}
        result["dimensions_error"] = str(e)
    
    return result


# ---------------------------- free "LLM-style" ablation ----------------------------
def _compactness_score(tot: np.ndarray) -> float:
    tot = np.asarray(tot, dtype=np.float64)
    H, W = tot.shape
    flat = tot.reshape(-1)
    if flat.size == 0:
        return 0.0
    thr = np.percentile(flat, 90.0)
    mask = tot >= thr
    if not np.any(mask):
        return 0.0
    ys, xs = np.where(mask)
    cy = float(np.mean(ys))
    cx = float(np.mean(xs))
    d = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
    mean_d = float(np.mean(d))
    scale = 0.3 * math.sqrt(H * H + W * W) + 1e-6
    score = math.exp(-mean_d / scale)
    return float(max(0.0, min(1.0, score)))


# ============================================================================
# DIMENSION SCORING
# ============================================================================



def _quantiles(x: List[float], qs=(5, 50, 95)) -> Dict[str, float]:
    arr = np.asarray([v for v in x if v is not None and np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {"q05": float("nan"), "q50": float("nan"), "q95": float("nan")}
    q = np.percentile(arr, qs)
    return {"q05": float(q[0]), "q50": float(q[1]), "q95": float(q[2])}

def _score_similarity_to_real(x: float, q05: float, q50: float, q95: float,
                              floor: float = 0.05, eps: float = 1e-12) -> float:
    """
    Score in [floor,1] based on closeness to the *real* median, using a robust scale derived
    from (q95-q05). This is 'realism-aligned' and works well for GAN/MSE models.
    """
    if not np.isfinite(x) or not np.isfinite(q50) or not np.isfinite(q05) or not np.isfinite(q95):
        return 0.5
    scale = 0.5 * (q95 - q05)
    if scale <= eps:
        return 1.0
    d = abs(float(x) - float(q50))
    s = math.exp(-d / (scale + eps))
    return float(max(floor, min(1.0, s)))

def _score_higher_is_better(x: float, q05: float, q50: float, q95: float,
                            floor: float = 0.05, eps: float = 1e-12) -> float:
    """Monotone score using real quantiles (q05->low, q95->high) with a soft sigmoid."""
    if not np.isfinite(x) or not np.isfinite(q50) or not np.isfinite(q05) or not np.isfinite(q95):
        return 0.5
    scale = 0.5 * (q95 - q05)
    if scale <= eps:
        return 0.5
    z = (float(x) - float(q50)) / (scale + eps)
    s = 1.0 / (1.0 + math.exp(-z))
    return float(max(floor, min(1.0, s)))

def _score_lower_is_better(x: float, q05: float, q50: float, q95: float,
                           floor: float = 0.05, eps: float = 1e-12) -> float:
    """Monotone inverted score using real quantiles."""
    return _score_higher_is_better(-float(x), -float(q95), -float(q50), -float(q05), floor=floor, eps=eps)


def compute_dimension_scores_from_metrics(metrics_dict: Dict[str, Any], profile: str = "planning_6dimension") -> Dict[str, float]:
    """Aggregate metrics into the single retained six-dimension planning profile."""

    m = metrics_dict.get("urban", metrics_dict)

    def clamp01(x: float) -> float:
        try:
            x = float(x)
        except Exception:
            return 0.0
        if not np.isfinite(x):
            return 0.0
        return float(max(0.0, min(1.0, x)))

    def normalize(value: float, min_val: float, max_val: float) -> float:
        if max_val <= min_val:
            return 0.0
        return clamp01((float(value) - min_val) / (max_val - min_val))

    def normalize_balance(value: float, acceptable_min: float, acceptable_max: float) -> float:
        v = float(value)
        if acceptable_min <= v <= acceptable_max:
            denom = max(abs(acceptable_min - 1.0), abs(acceptable_max - 1.0), 1e-8)
            return clamp01(1.0 - abs(v - 1.0) / denom)
        if v < acceptable_min:
            return clamp01(0.5 * (v / (acceptable_min + 1e-8)))
        return clamp01(0.5 * (acceptable_max / (v + 1e-8)))

    def safe_metric(name: str, default: float = float("nan")) -> float:
        try:
            return float(m.get(name, default))
        except Exception:
            return float(default)

    def q(name: str) -> Tuple[float, float, float]:
        real_q = metrics_dict.get("real_q", {}) or {}
        d = real_q.get(name, {}) if isinstance(real_q, dict) else {}
        return (
            float(d.get("q05", float("nan"))),
            float(d.get("q50", float("nan"))),
            float(d.get("q95", float("nan"))),
        )

    def _soft_factory():
        scale_mult = 4.2
        power_soft = 0.46
        floor = 0.24

        def _soft(s: float) -> float:
            s = clamp01(s)
            if not np.isfinite(s):
                return 0.5
            return float(max(floor, min(1.0, s ** power_soft)))

        def _wide_scale(q05: float, q95: float, eps: float = 1e-12) -> float:
            if not (np.isfinite(q05) and np.isfinite(q95)):
                return float("nan")
            base = 0.5 * (q95 - q05)
            if base <= eps:
                return float("nan")
            return float(scale_mult * base)

        return _soft, _wide_scale

    _soft, _wide_scale = _soft_factory()

    def score_hi(name: str, fallback_min: float = 0.0, fallback_max: float = 1.0) -> float:
        x = safe_metric(name)
        q05, q50, q95 = q(name)
        sc = _wide_scale(q05, q95)
        if np.isfinite(x) and np.isfinite(q50) and np.isfinite(sc):
            z = (x - q50) / (sc + 1e-12)
            return _soft(1.0 / (1.0 + math.exp(-z)))
        return _soft(normalize(x, fallback_min, fallback_max))

    def score_lo(name: str, fallback_min: float = 0.0, fallback_max: float = 1.0) -> float:
        x = safe_metric(name)
        q05, q50, q95 = q(name)
        sc = _wide_scale(q05, q95)
        if np.isfinite(x) and np.isfinite(q50) and np.isfinite(sc):
            z = (q50 - x) / (sc + 1e-12)
            return _soft(1.0 / (1.0 + math.exp(-z)))
        return _soft(1.0 - normalize(x, fallback_min, fallback_max))

    def score_mod(name: str, fallback_min: float = 0.0, fallback_max: float = 1.0) -> float:
        x = safe_metric(name)
        q05, q50, q95 = q(name)
        sc = _wide_scale(q05, q95)
        if np.isfinite(x) and np.isfinite(q50) and np.isfinite(sc):
            return _soft(math.exp(-abs(x - q50) / (sc + 1e-12)))
        return _soft(normalize(x, fallback_min, fallback_max))

    def comp(name: str) -> float:
        return _soft(safe_metric(name, 0.0))

    def gate_cap(base_score: float, caps: List[Tuple[bool, float]]) -> float:
        s = clamp01(base_score)
        for cond, cap in caps:
            if cond:
                s = min(s, cap)
        return float(s)

    def gate_multiplier(conditions: List[Tuple[bool, float]]) -> float:
        mult = 1.0
        for cond, factor in conditions:
            if cond:
                mult *= float(factor)
        return float(clamp01(mult))

    def polycentric_band_score() -> float:
        parts = [
            score_mod("jobs_polycentricity", 0.0, 1.0),
            score_mod("employment_centrality", 0.0, 1.0),
            score_mod("jobs_housing_centroid_dist", 0.0, 130.0),
        ]
        parts = [max(0.03, clamp01(v)) for v in parts]
        return float(np.exp(np.mean(np.log(np.asarray(parts, dtype=np.float64)))))

    residential_adequacy = clamp01(
        0.34 * score_hi("resident_k2_coverage_r35", 0.0, 1.0) +
        0.28 * score_hi("resident_completeness_mean_r35", 0.0, 1.0) +
        0.18 * score_hi("resident_complete_coverage_r35", 0.0, 1.0) +
        0.20 * score_mod("cell_crispness_mean", 0.0, 1.0)
    )
    essential_adequacy = clamp01(
        0.26 * score_hi("svc_access_gravity_v7", 0.0, 1.0) +
        0.18 * score_hi("trn_access_gravity_v7", 0.0, 1.0) +
        0.12 * score_hi("job_access_gravity_v7", 0.0, 1.0) +
        0.22 * score_hi("resident_k2_coverage_r35", 0.0, 1.0) +
        0.22 * comp("resident_complete_coverage_r35")
    )
    anti_diffuse_guard = clamp01(
        0.24 * score_mod("cell_crispness_mean", 0.0, 1.0) +
        0.20 * score_lo("buffer_conflict_r3", 0.0, 0.5) +
        0.18 * score_mod("local_mix_high_share", 0.0, 0.8) +
        0.18 * score_mod("green_area_share", 0.0, 0.5) +
        0.20 * score_mod("aggregation_index", 0.0, 1.0)
    )

    spatial_coherence = clamp01(
        0.18 * score_mod("aggregation_index", 0.0, 1.0) +
        0.16 * score_mod("cell_crispness_mean", 0.0, 1.0) +
        0.13 * score_lo("buffer_conflict_r3", 0.0, 0.5) +
        0.13 * score_hi("green_connectivity_v5", 0.0, 1.0) +
        0.10 * score_mod("street_connectivity", 0.0, 1.0) +
        0.10 * score_hi("intensity_transition_smoothness", 0.0, 1.0) +
        0.08 * score_hi("infrastructure_demand_alignment", 0.0, 1.0) +
        0.12 * anti_diffuse_guard
    )

    land_use_balance = clamp01(
        0.12 * score_mod("land_mix_entropy", 0.0, 1.0) +
        0.09 * score_hi("simpson_diversity", 0.0, 1.0) +
        0.09 * score_hi("shannon_diversity_by_zone", 0.0, 1.0) +
        0.08 * score_hi("mixing_index", 0.0, 1.0) +
        0.08 * score_mod("mixed_use_intensity", 0.0, 1.0) +
        0.10 * score_hi("functional_completeness_v7", 0.0, 1.0) +
        0.09 * score_mod("local_mix_high_share", 0.0, 0.8) +
        0.08 * score_hi("land_use_compatibility", 0.0, 1.0) +
        0.08 * score_hi("service_hierarchy", 0.0, 1.0) +
        0.19 * normalize_balance(safe_metric("jobs_housing_balance", 1.0), 0.45, 2.00)
    )

    mobility_integration = clamp01(
        0.20 * score_hi("walkability_score", 0.0, 1.0) +
        0.15 * score_hi("transit_density", 0.0, 1.0) +
        0.16 * score_hi("tod_synergy", 0.0, 1.0) +
        0.12 * score_mod("coloc_jobs_transit", 0.0, 1.0) +
        0.14 * score_hi("trn_access_gravity_v7", 0.0, 1.0) +
        0.11 * score_mod("street_connectivity", 0.0, 1.0) +
        0.12 * score_lo("svc_dist_p75", 0.0, 100.0)
    )

    service_adequacy = clamp01(
        0.18 * comp("resident_completeness_mean_r35") +
        0.14 * comp("resident_complete_coverage_r35") +
        0.15 * score_hi("resident_k2_coverage_r35", 0.0, 1.0) +
        0.10 * score_hi("resident_k3_coverage_r35", 0.0, 1.0) +
        0.15 * score_hi("svc_access_gravity_v7", 0.0, 1.0) +
        0.10 * score_hi("svc_access_match_v7", 0.0, 1.0) +
        0.09 * score_lo("svc_dist_p75", 0.0, 100.0) +
        0.09 * score_lo("underserved_share_r35", 0.0, 1.0)
    )

    balanced_service_distribution = clamp01(
        0.30 * score_lo("svc_access_gini_v7", 0.0, 1.0) +
        0.25 * score_lo("underserved_share_r35", 0.0, 1.0) +
        0.20 * score_lo("zone_cv_completeness_r35", 0.0, 1.0) +
        0.15 * score_hi("resident_complete_coverage_r35", 0.0, 1.0) +
        0.10 * score_hi("open_space_distribution", 0.0, 1.0)
    )

    environmental_resilience = clamp01(
        0.18 * score_hi("green_connectivity_v5", 0.0, 1.0) +
        0.13 * score_mod("green_area_share", 0.0, 0.50) +
        0.13 * score_lo("urban_heat_risk", 0.0, 1.0) +
        0.13 * score_hi("flood_resilience", 0.0, 1.0) +
        0.10 * score_mod("coloc_housing_green", 0.0, 1.0) +
        0.10 * score_hi("recreation_space_accessibility", 0.0, 1.0) +
        0.10 * score_hi("open_space_distribution", 0.0, 1.0) +
        0.06 * score_hi("education_green_proximity", 0.0, 1.0) +
        0.07 * anti_diffuse_guard
    )

    safety_preparedness = clamp01(
        0.28 * score_hi("emergency_response_coverage", 0.0, 1.0) +
        0.22 * score_lo("residential_noise_exposure", 0.0, 1.0) +
        0.18 * score_lo("buffer_conflict_r3", 0.0, 0.5) +
        0.16 * score_hi("land_use_compatibility", 0.0, 1.0) +
        0.16 * score_hi("infrastructure_demand_alignment", 0.0, 1.0)
    )

    infrastructure_coherence = clamp01(
        0.22 * score_hi("infrastructure_demand_alignment", 0.0, 1.0) +
        0.18 * score_hi("intensity_transition_smoothness", 0.0, 1.0) +
        0.16 * score_mod("density_gradient", 1.0, 5.0) +
        0.16 * score_mod("street_connectivity", 0.0, 1.0) +
        0.14 * score_hi("tod_synergy", 0.0, 1.0) +
        0.14 * score_mod("aggregation_index", 0.0, 1.0)
    )

    implementation_readiness = clamp01(
        0.22 * anti_diffuse_guard +
        0.18 * score_mod("cell_crispness_mean", 0.0, 1.0) +
        0.16 * score_hi("functional_completeness_v7", 0.0, 1.0) +
        0.14 * score_lo("buffer_conflict_r3", 0.0, 0.5) +
        0.15 * score_mod("aggregation_index", 0.0, 1.0) +
        0.15 * score_mod("street_connectivity", 0.0, 1.0)
    )

    service_proximity = clamp01(
        0.30 * score_hi("svc_access_gravity_v7", 0.0, 1.0) +
        0.24 * score_lo("svc_dist_p75", 0.0, 100.0) +
        0.18 * score_hi("resident_k2_coverage_r35", 0.0, 1.0) +
        0.14 * score_hi("trn_access_gravity_v7", 0.0, 1.0) +
        0.14 * comp("resident_complete_coverage_r35")
    )

    amenity_diversity = clamp01(
        0.24 * score_hi("land_mix_entropy", 0.0, 1.0) +
        0.20 * score_hi("simpson_diversity", 0.0, 1.0) +
        0.18 * score_hi("shannon_diversity_by_zone", 0.0, 1.0) +
        0.16 * score_mod("mixed_use_intensity", 0.0, 1.0) +
        0.12 * score_mod("local_mix_high_share", 0.0, 0.8) +
        0.10 * score_hi("service_hierarchy", 0.0, 1.0)
    )

    activity_center_strength = clamp01(
        0.30 * polycentric_band_score() +
        0.18 * score_hi("tod_synergy", 0.0, 1.0) +
        0.16 * score_hi("transit_density", 0.0, 1.0) +
        0.14 * score_hi("service_hierarchy", 0.0, 1.0) +
        0.12 * score_mod("aggregation_index", 0.0, 1.0) +
        0.10 * score_mod("jobs_housing_centroid_dist", 0.0, 130.0)
    )

    green_access = clamp01(
        0.30 * score_hi("recreation_space_accessibility", 0.0, 1.0) +
        0.24 * score_hi("green_connectivity_v5", 0.0, 1.0) +
        0.18 * score_hi("open_space_distribution", 0.0, 1.0) +
        0.16 * score_mod("green_area_share", 0.0, 0.50) +
        0.12 * score_hi("education_green_proximity", 0.0, 1.0)
    )

    public_space_access = clamp01(
        0.34 * score_hi("recreation_space_accessibility", 0.0, 1.0) +
        0.24 * score_hi("open_space_distribution", 0.0, 1.0) +
        0.18 * score_hi("green_connectivity_v5", 0.0, 1.0) +
        0.14 * score_mod("green_area_share", 0.0, 0.50) +
        0.10 * score_lo("urban_heat_risk", 0.0, 1.0)
    )

    development_compactness = clamp01(
        0.28 * score_mod("aggregation_index", 0.0, 1.0) +
        0.22 * score_mod("cell_crispness_mean", 0.0, 1.0) +
        0.18 * score_mod("density_gradient", 1.0, 5.0) +
        0.16 * score_hi("infrastructure_demand_alignment", 0.0, 1.0) +
        0.16 * score_lo("buffer_conflict_r3", 0.0, 0.5)
    )

    public_service_readiness = clamp01(
        0.22 * score_hi("emergency_response_coverage", 0.0, 1.0) +
        0.20 * score_hi("resident_k2_coverage_r35", 0.0, 1.0) +
        0.18 * score_hi("svc_access_gravity_v7", 0.0, 1.0) +
        0.16 * score_hi("trn_access_gravity_v7", 0.0, 1.0) +
        0.12 * score_hi("infrastructure_demand_alignment", 0.0, 1.0) +
        0.12 * score_hi("service_hierarchy", 0.0, 1.0)
    )

    resident_wellbeing = clamp01(
        0.24 * clamp01(0.14 * comp("resident_completeness_mean_r35") + 0.12 * score_hi("walkability_score", 0.0, 1.0) + 0.12 * score_hi("recreation_space_accessibility", 0.0, 1.0) + 0.10 * score_hi("green_connectivity_v5", 0.0, 1.0) + 0.09 * score_mod("green_area_share", 0.0, 0.50) + 0.09 * score_lo("svc_dist_p75", 0.0, 100.0) + 0.07 * score_lo("urban_heat_risk", 0.0, 1.0) + 0.07 * score_lo("buffer_conflict_r3", 0.0, 0.5) + 0.07 * score_mod("coloc_housing_services", 0.0, 1.0) + 0.06 * score_lo("residential_noise_exposure", 0.0, 1.0) + 0.07 * score_hi("open_space_distribution", 0.0, 1.0)) +
        0.18 * service_proximity +
        0.16 * green_access +
        0.16 * score_lo("residential_noise_exposure", 0.0, 1.0) +
        0.14 * score_lo("urban_heat_risk", 0.0, 1.0) +
        0.12 * safety_preparedness
    )

    urban_resilience = clamp01(
        0.26 * environmental_resilience +
        0.18 * safety_preparedness +
        0.16 * public_service_readiness +
        0.14 * green_access +
        0.14 * balanced_service_distribution +
        0.12 * activity_center_strength
    )

    community_convenience = clamp01(
        0.28 * resident_wellbeing +
        0.22 * service_proximity +
        0.18 * amenity_diversity +
        0.16 * public_space_access +
        0.16 * score_lo("svc_dist_p75", 0.0, 100.0)
    )

    healthy_environment = clamp01(
        0.24 * score_lo("urban_heat_risk", 0.0, 1.0) +
        0.22 * score_lo("residential_noise_exposure", 0.0, 1.0) +
        0.18 * score_lo("buffer_conflict_r3", 0.0, 0.5) +
        0.18 * score_hi("green_connectivity_v5", 0.0, 1.0) +
        0.18 * score_hi("land_use_compatibility", 0.0, 1.0)
    )

    spatial_coherence = gate_cap(spatial_coherence, [
        (safe_metric("cell_crispness_mean", 0.0) < 0.14, 0.88),
        (safe_metric("buffer_conflict_r3", 1.0) > 0.48, 0.90),
    ])
    land_use_balance = gate_cap(land_use_balance, [
        (safe_metric("functional_completeness_v7", 0.0) < 0.16, 0.88),
        (safe_metric("local_mix_high_share", 1.0) > 0.93, 0.90),
    ])
    environmental_resilience = gate_cap(environmental_resilience, [
        (safe_metric("green_connectivity_v5", 0.0) < 0.08, 0.89),
        (safe_metric("urban_heat_risk", 1.0) > 0.95, 0.89),
    ])
    safety_preparedness = gate_cap(safety_preparedness, [
        (safe_metric("emergency_response_coverage", 0.0) < 0.10, 0.86),
        (safe_metric("buffer_conflict_r3", 1.0) > 0.50, 0.88),
    ])
    infrastructure_coherence = gate_cap(infrastructure_coherence, [
        (safe_metric("infrastructure_demand_alignment", 0.0) < 0.15, 0.88),
        (safe_metric("intensity_transition_smoothness", 0.0) < 0.10, 0.90),
    ])
    service_proximity = gate_cap(service_proximity, [
        (essential_adequacy < 0.12, 0.90),
        (safe_metric("svc_dist_p75", 999.0) > 85.0, 0.90),
    ])
    balanced_service_distribution = gate_cap(balanced_service_distribution, [
        (safe_metric("underserved_share_r35", 1.0) > 0.82, 0.88),
        (safe_metric("zone_cv_completeness_r35", 1.0) > 0.95, 0.90),
    ])
    green_access = gate_cap(green_access, [
        (safe_metric("green_connectivity_v5", 0.0) < 0.08, 0.90),
        (safe_metric("open_space_distribution", 0.0) < 0.08, 0.90),
    ])
    development_compactness = gate_cap(development_compactness, [
        (safe_metric("infrastructure_demand_alignment", 0.0) < 0.08, 0.88),
        (safe_metric("density_gradient", 0.0) <= 0.2, 0.92),
    ])
    public_service_readiness = gate_cap(public_service_readiness, [
        (safe_metric("emergency_response_coverage", 0.0) < 0.08, 0.88),
        (essential_adequacy < 0.10, 0.90),
    ])
    resident_wellbeing = gate_cap(resident_wellbeing, [
        (safe_metric("residential_noise_exposure", 1.0) > 0.92, 0.90),
        (safe_metric("urban_heat_risk", 1.0) > 0.95, 0.90),
    ])
    healthy_environment = gate_cap(healthy_environment, [
        (safe_metric("urban_heat_risk", 1.0) > 0.96, 0.88),
        (safe_metric("buffer_conflict_r3", 1.0) > 0.50, 0.90),
    ])
    urban_resilience = gate_cap(urban_resilience, [
        (safe_metric("flood_resilience", 0.0) < 0.08, 0.90),
        (safe_metric("urban_heat_risk", 1.0) > 0.96, 0.88),
    ])
    community_convenience = gate_cap(community_convenience, [
        (safe_metric("svc_dist_p75", 999.0) > 90.0, 0.90),
        (essential_adequacy < 0.10, 0.90),
    ])

    validity_multiplier = gate_multiplier([
        (residential_adequacy < 0.10, 0.94),
        (essential_adequacy < 0.10, 0.95),
        (anti_diffuse_guard < 0.08, 0.97),
    ])

    spatial_coherence *= validity_multiplier
    land_use_balance *= validity_multiplier
    development_compactness *= max(validity_multiplier, 0.98)
    healthy_environment *= validity_multiplier
    community_convenience *= validity_multiplier
    urban_resilience *= validity_multiplier

    return {
        "spatial_coherence": float(clamp01(spatial_coherence)),
        "development_compactness": float(clamp01(development_compactness)),
        "healthy_environment": float(clamp01(healthy_environment)),
        "land_use_balance": float(clamp01(land_use_balance)),
        "community_convenience": float(clamp01(community_convenience)),
        "urban_resilience": float(clamp01(urban_resilience)),
    }

def score_plan_free(plan_chw: np.ndarray, zone_hw: Optional[np.ndarray], rubric: str) -> Dict[str, float]:
    plan = np.nan_to_num(plan_chw, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)
    plan = np.clip(plan, 0.0, None)
    tot = plan.sum(axis=0)
    dom = np.argmax(plan, axis=0)

    edge = _edge_density(dom)
    neigh = _spcorr01(tot)  # 0..1

    balance = _land_mix_entropy(plan_chw)
    access = _compactness_score(tot)

    zoning = float("nan")
    if zone_hw is not None:
        zmin = int(np.min(zone_hw))
        zmax = int(np.max(zone_hw))
        if zmin >= 0 and zmax < 20:
            zoning = float(np.mean(dom == zone_hw))

    spatial = max(0.0, min(1.0, 0.5 * (1.0 - edge) + 0.5 * neigh))

    if rubric == "standard":
        weights = dict(zoning=0.25, spatial=0.35, access=0.20, balance=0.20)
    elif rubric == "detailed":
        weights = dict(zoning=0.20, spatial=0.40, access=0.15, balance=0.25)
    elif rubric == "critical":
        weights = dict(zoning=0.30, spatial=0.40, access=0.10, balance=0.20)
    else:
        weights = dict(zoning=0.25, spatial=0.35, access=0.20, balance=0.20)

    scores = dict(zoning=zoning, spatial=spatial, access=access, balance=balance)
    if not math.isfinite(scores["zoning"]):
        weights["zoning"] = 0.0
    wsum = sum(weights.values()) + 1e-8
    overall = sum(weights[k] * (scores[k] if math.isfinite(scores[k]) else 0.0) for k in weights) / wsum
    scores["overall"] = float(max(0.0, min(1.0, overall)))
    return scores


def _uncertain(samples: List[float]) -> Dict[str, Any]:
    arr = np.asarray(samples, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=np.nan)
    m = float(np.nanmean(arr)) if arr.size else float("nan")
    s = float(np.nanstd(arr)) if arr.size else float("nan")
    lo = float(np.nanpercentile(arr, 2.5)) if arr.size else float("nan")
    hi = float(np.nanpercentile(arr, 97.5)) if arr.size else float("nan")
    rel = float(1.0 / (1.0 + (s if math.isfinite(s) else 1e6)))
    rel = max(0.0, min(1.0, rel))
    return {"mean": (m if math.isfinite(m) else None),
            "std": (s if math.isfinite(s) else None),
            "ci95": [(lo if math.isfinite(lo) else None), (hi if math.isfinite(hi) else None)],
            "reliability": rel,
            "samples": [float(x) if math.isfinite(x) else None for x in arr.tolist()]}


def run_free_ablation(plan_chw: np.ndarray, zone_hw: Optional[np.ndarray], rubrics: List[str], n_samples: int, noise_std: float, seed: int) -> Dict[str, Any]:
    rng = np.random.default_rng(int(seed))

    def eval_setting(setting_rubrics: List[str], setting_samples: int) -> Dict[str, Any]:
        pooled = {k: [] for k in ["zoning", "spatial", "access", "balance", "overall"]}
        per_rubric_means = []
        for rv in setting_rubrics:
            over = []
            for _ in range(setting_samples):
                noise = rng.normal(0.0, noise_std, size=plan_chw.shape).astype(np.float32)
                plan_noisy = np.clip(plan_chw * (1.0 + noise), 0.0, None)
                sc = score_plan_free(plan_noisy, zone_hw, rubric=rv)
                for k in pooled:
                    pooled[k].append(float(sc.get(k, float("nan"))))
                over.append(float(sc.get("overall", float("nan"))))
            per_rubric_means.append(float(np.nanmean(over)) if over else float("nan"))

        scores = {k: _uncertain(v) for k, v in pooled.items()}

        arr = np.asarray([x for x in pooled["overall"] if x is not None], dtype=np.float64)
        std = float(np.nanstd(arr)) if arr.size else float("nan")
        agreement = float(max(0.0, min(1.0, 1.0 / (1.0 + (std if math.isfinite(std) else 1e6)))))

        pr = np.asarray([x for x in per_rubric_means if math.isfinite(x)], dtype=np.float64)
        pr_std = float(np.std(pr)) if pr.size else float("nan")
        robustness = float(max(0.0, min(1.0, 1.0 / (1.0 + (pr_std if math.isfinite(pr_std) else 1e6)))))

        overall_rel = float(scores["overall"]["reliability"])
        rel_cat = "high" if overall_rel >= 0.75 else ("medium" if overall_rel >= 0.55 else "low")

        return {"scores": scores,
                "diagnostics": {"inter_sample_agreement": agreement,
                                "prompt_robustness": robustness,
                                "overall_reliability": overall_rel,
                                "reliability_category": rel_cat}}

    return {
        "no_uncertainty": eval_setting(["standard"], 1),
        "multi_sample_only": eval_setting(["standard"], int(n_samples)),
        "prompt_ensemble_only": eval_setting(list(rubrics), 1),
        "full_uncertainty": eval_setting(list(rubrics), int(n_samples)),
    }


def run_free_ablation(plan_chw: np.ndarray, zone_hw: Optional[np.ndarray], rubrics: List[str], n_samples: int, noise_std: float, seed: int) -> Dict[str, Any]:
    rng = np.random.default_rng(int(seed))

    def eval_setting(setting_rubrics: List[str], setting_samples: int) -> Dict[str, Any]:
        pooled = {k: [] for k in ["zoning", "spatial", "access", "balance", "overall"]}
        per_rubric_means = []
        for rv in setting_rubrics:
            over = []
            for _ in range(setting_samples):
                noise = rng.normal(0.0, noise_std, size=plan_chw.shape).astype(np.float32)
                plan_noisy = np.clip(plan_chw * (1.0 + noise), 0.0, None)
                sc = score_plan_free(plan_noisy, zone_hw, rubric=rv)
                for k in pooled:
                    pooled[k].append(float(sc.get(k, float("nan"))))
                over.append(float(sc.get("overall", float("nan"))))
            per_rubric_means.append(float(np.nanmean(over)) if over else float("nan"))

        scores = {k: _uncertain(v) for k, v in pooled.items()}

        arr = np.asarray([x for x in pooled["overall"] if x is not None], dtype=np.float64)
        std = float(np.nanstd(arr)) if arr.size else float("nan")
        agreement = float(max(0.0, min(1.0, 1.0 / (1.0 + (std if math.isfinite(std) else 1e6)))))

        pr = np.asarray([x for x in per_rubric_means if math.isfinite(x)], dtype=np.float64)
        pr_std = float(np.std(pr)) if pr.size else float("nan")
        robustness = float(max(0.0, min(1.0, 1.0 / (1.0 + (pr_std if math.isfinite(pr_std) else 1e6)))))

        overall_rel = float(scores["overall"]["reliability"])
        rel_cat = "high" if overall_rel >= 0.75 else ("medium" if overall_rel >= 0.55 else "low")

        return {"scores": scores,
                "diagnostics": {"inter_sample_agreement": agreement,
                                "prompt_robustness": robustness,
                                "overall_reliability": overall_rel,
                                "reliability_category": rel_cat}}

    return {
        "no_uncertainty": eval_setting(["standard"], 1),
        "multi_sample_only": eval_setting(["standard"], int(n_samples)),
        "prompt_ensemble_only": eval_setting(list(rubrics), 1),
        "full_uncertainty": eval_setting(list(rubrics), int(n_samples)),
    }


# ======================== ENHANCED LLM AGGREGATION ========================

def aggregate_llm_quant_results(llm_results: List[Dict]) -> Dict[str, Any]:
    """Aggregate enhanced LLM quantitative results across multiple plans"""
    
    dimensions = ["overall", "spatial_quality", "functional_diversity", "sustainability",
                 "accessibility", "equity", "economic_viability", "livability"]
    
    aggregated = {
        "n_plans": len(llm_results),
        "method": llm_results[0].get("method", "unknown") if llm_results else "unknown",
        "dimensions": {},
        "confidence_summary": {},
        "diagnostics_summary": {}
    }
    
    for dim in dimensions:
        dim_scores = []
        dim_confidences = []
        dim_stds = []
        
        for result in llm_results:
            if dim in result.get("dimensions", {}):
                dim_scores.append(result["dimensions"][dim].get("mean", 0.5))
                dim_stds.append(result["dimensions"][dim].get("std", 0.1))
            
            if dim in result.get("uncertainty", {}):
                dim_confidences.append(result["uncertainty"][dim].get("confidence", 0.5))
        
        if dim_scores:
            dim_scores_arr = np.array(dim_scores)
            aggregated["dimensions"][dim] = {
                "mean_across_plans": float(np.mean(dim_scores_arr)),
                "std_across_plans": float(np.std(dim_scores_arr)),
                "median_across_plans": float(np.median(dim_scores_arr)),
                "min": float(np.min(dim_scores_arr)),
                "max": float(np.max(dim_scores_arr)),
                "ci_95_across_plans": [
                    float(np.percentile(dim_scores_arr, 2.5)),
                    float(np.percentile(dim_scores_arr, 97.5))
                ]
            }
            
            if dim_confidences:
                aggregated["confidence_summary"][dim] = {
                    "mean_confidence": float(np.mean(dim_confidences)),
                    "min_confidence": float(np.min(dim_confidences)),
                    "max_confidence": float(np.max(dim_confidences))
                }
            
            if dim_stds:
                aggregated["confidence_summary"][dim] = aggregated["confidence_summary"].get(dim, {})
                aggregated["confidence_summary"][dim]["mean_within_plan_std"] = float(np.mean(dim_stds))
    
    # Aggregate diagnostics
    if llm_results and "diagnostics" in llm_results[0]:
        diag_keys = ["inter_dimension_consistency", "ensemble_agreement"]
        for key in diag_keys:
            vals = [r["diagnostics"].get(key, 0.5) for r in llm_results 
                   if "diagnostics" in r and key in r["diagnostics"]]
            if vals:
                aggregated["diagnostics_summary"][key] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals))
                }
    
    return aggregated


def aggregate_llm_qual_results(llm_results: List[Dict]) -> Dict[str, Any]:
    """Aggregate enhanced LLM qualitative results across multiple plans"""
    
    aggregated = {
        "n_plans": len(llm_results),
        "quality_distribution": {},
        "strengths_frequency": {},
        "weaknesses_frequency": {},
        "pattern_distribution": {},
        "recommendation_priorities": {"high": 0, "medium": 0, "low": 0}
    }
    
    # Quality levels
    quality_counts = {}
    for result in llm_results:
        if "overall_assessment" in result:
            quality = result["overall_assessment"].get("quality_level", "unknown")
            quality_counts[quality] = quality_counts.get(quality, 0) + 1
    
    aggregated["quality_distribution"] = quality_counts
    
    # Strength dimensions
    strength_dims = {}
    for result in llm_results:
        for strength in result.get("strengths", []):
            dim = strength.get("dimension", "unknown")
            strength_dims[dim] = strength_dims.get(dim, 0) + 1
    
    aggregated["strengths_frequency"] = strength_dims
    
    # Weakness dimensions
    weakness_dims = {}
    for result in llm_results:
        for weakness in result.get("weaknesses", []):
            dim = weakness.get("dimension", "unknown")
            weakness_dims[dim] = weakness_dims.get(dim, 0) + 1
    
    aggregated["weaknesses_frequency"] = weakness_dims
    
    # Pattern types
    pattern_types = {}
    for result in llm_results:
        if "pattern_recognition" in result:
            dev_type = result["pattern_recognition"].get("development_type", {}).get("type", "unknown")
            pattern_types[dev_type] = pattern_types.get(dev_type, 0) + 1
    
    aggregated["pattern_distribution"] = pattern_types
    
    # Recommendation priorities
    for result in llm_results:
        for rec in result.get("recommendations", []):
            priority = rec.get("priority", "medium")
            aggregated["recommendation_priorities"][priority] = \
                aggregated["recommendation_priorities"].get(priority, 0) + 1
    
    return aggregated


def generate_summary_report(
    model_name: str,
    quant_results: Optional[Dict] = None,
    llm_quant_agg: Optional[Dict] = None,
    llm_qual_agg: Optional[Dict] = None
) -> str:
    """Generate human-readable summary report combining all evaluation types"""
    
    lines = []
    lines.append("=" * 80)
    lines.append(f"EVALUATION SUMMARY: {model_name}")
    lines.append("=" * 80)
    lines.append("")
    
    # Standard Quantitative
    if quant_results:
        lines.append("STANDARD QUANTITATIVE METRICS")
        lines.append("-" * 80)
        
        if "distribution" in quant_results:
            dist = quant_results["distribution"]
            lines.append("Distribution Similarity:")
            lines.append(f"  KL Divergence:        {dist.get('kl', 0):.4f}")
            lines.append(f"  JS Divergence:        {dist.get('js', 0):.4f}")
            lines.append(f"  Wasserstein Distance: {dist.get('wd', 0):.4f}")
            lines.append(f"  Hellinger Distance:   {dist.get('hellinger', 0):.4f}")
        
        if "spatial" in quant_results:
            spatial = quant_results["spatial"]
            lines.append(f"\nSpatial Accuracy:")
            lines.append(f"  RMSE:        {spatial.get('rmse', 0):.4f}")
            lines.append(f"  Correlation: {spatial.get('corr_total', 0):.4f}")
        
        if "urban" in quant_results:
            urban = quant_results["urban"]
            lines.append(f"\nUrban Structure:")
            lines.append(f"  Land Mix Entropy:    {urban.get('land_mix_entropy', 0):.4f}")
            lines.append(f"  Jobs-Housing Bal.:   {urban.get('jobs_housing_balance', 0):.4f}")
            lines.append(f"  Service Coverage:    {urban.get('service_coverage', 0):.4f}")
        
        # Dimension Scores (if available)
        if "dimensions" in quant_results:
            dims = quant_results["dimensions"]
            lines.append("")
            lines.append("DIMENSION SCORES (aggregated from 42 metrics)")
            lines.append("-" * 80)
            lines.append(f"{'Dimension':<30} {'Score':<10} {'Rating'}")
            lines.append("-" * 80)

            if isinstance(dims, dict) and len(dims) == 0:
                err = quant_results.get("dimensions_error")
                if err:
                    lines.append(f"(No dimension scores computed: {err})")
                else:
                    lines.append("(No dimension scores computed.)")
            
            # Display in consistent order
            dim_order = [
                "spatial_quality",
                "functional_diversity",
                "sustainability",
                "accessibility",
                "equity",
                "economic_viability",
                "livability",
                "spatial_coherence",
                "land_use_balance",
                "mobility_integration",
                "service_adequacy",
                "equity_inclusion",
                "environmental_resilience",
                "economic_structure",
                "neighborhood_livability",
                "implementation_readiness",
                "safety_preparedness",
                "infrastructure_coherence",
                "service_proximity",
                "amenity_diversity",
                "balanced_service_distribution",
                "local_job_access",
                "activity_center_strength",
                "neighborhood_identity",
                "green_access",
                "transit_convenience",
                "public_space_access",
                "development_compactness",
                "public_service_readiness",
                "resident_wellbeing",
                "complete_neighborhoods",
                "service_fairness",
                "jobs_housing_synergy",
                "green_public_realm",
                "healthy_environment",
                "family_friendliness",
                "center_hierarchy",
                "growth_readiness",
                "mobility_choice",
                "urban_resilience",
                "community_convenience",
                "infrastructure_alignment",
                "validity_guard",
                "residential_adequacy",
                "essential_adequacy",
            ]
            seen_dims = set()
            for dim_name in dim_order + [k for k in dims.keys() if k not in dim_order]:
                if dim_name in dims:
                    seen_dims.add(dim_name)
                    score = dims[dim_name]
                    
                    # Determine rating
                    if score >= 0.8:
                        rating = "⭐⭐⭐⭐⭐ Excellent"
                    elif score >= 0.6:
                        rating = "⭐⭐⭐⭐ Good"
                    elif score >= 0.4:
                        rating = "⭐⭐⭐ Moderate"
                    elif score >= 0.2:
                        rating = "⭐⭐ Needs Improvement"
                    else:
                        rating = "⭐ Poor"
                    
                    display_name = dim_name.replace('_', ' ').title()
                    lines.append(f"{display_name:<30} {score:<10.3f} {rating}")
        
        lines.append("")
    
    # Enhanced LLM Quantitative
    if llm_quant_agg:
        lines.append("ENHANCED LLM EVALUATION - QUANTITATIVE (with Uncertainty)")
        lines.append("-" * 80)
        lines.append(f"Method: {llm_quant_agg.get('method', 'unknown')}")
        lines.append(f"Number of plans: {llm_quant_agg.get('n_plans', 0)}")
        lines.append("")
        
        dims = llm_quant_agg.get("dimensions", {})
        conf = llm_quant_agg.get("confidence_summary", {})
        
        lines.append(f"{'Dimension':<25} {'Mean':<8} {'±Std':<8} {'95% CI':<20} {'Confidence'}")
        lines.append("-" * 80)
        
        for dim_name in ["overall", "spatial_quality", "functional_diversity", "sustainability",
                         "accessibility", "equity", "economic_viability", "livability"]:
            if dim_name in dims:
                dim_data = dims[dim_name]
                mean = dim_data.get("mean_across_plans", 0)
                std = dim_data.get("std_across_plans", 0)
                ci = dim_data.get("ci_95_across_plans", [0, 0])
                
                conf_val = "N/A"
                if dim_name in conf:
                    conf_val = f"{conf[dim_name].get('mean_confidence', 0):.3f}"
                
                ci_str = f"[{ci[0]:.3f}, {ci[1]:.3f}]"
                lines.append(f"{dim_name:<25} {mean:>7.3f} {std:>7.3f} {ci_str:<20} {conf_val}")
        
        lines.append("")
    
    # Enhanced LLM Qualitative
    if llm_qual_agg:
        lines.append("ENHANCED LLM EVALUATION - QUALITATIVE")
        lines.append("-" * 80)
        lines.append(f"Number of plans: {llm_qual_agg.get('n_plans', 0)}")
        lines.append("")
        
        if "quality_distribution" in llm_qual_agg:
            lines.append("Quality Level Distribution:")
            for quality, count in sorted(llm_qual_agg["quality_distribution"].items()):
                pct = 100 * count / llm_qual_agg["n_plans"]
                lines.append(f"  {quality:<20} {count:>3} plans ({pct:>5.1f}%)")
        
        lines.append("")
        
        if "strengths_frequency" in llm_qual_agg:
            strengths = llm_qual_agg["strengths_frequency"]
            if strengths:
                lines.append("Most Common Strengths:")
                sorted_strengths = sorted(strengths.items(), key=lambda x: x[1], reverse=True)[:5]
                for dim, count in sorted_strengths:
                    pct = 100 * count / llm_qual_agg["n_plans"]
                    lines.append(f"  {dim:<30} {count:>3} plans ({pct:>5.1f}%)")
        
        lines.append("")
        
        if "weaknesses_frequency" in llm_qual_agg:
            weaknesses = llm_qual_agg["weaknesses_frequency"]
            if weaknesses:
                lines.append("Most Common Weaknesses:")
                sorted_weak = sorted(weaknesses.items(), key=lambda x: x[1], reverse=True)[:5]
                for dim, count in sorted_weak:
                    pct = 100 * count / llm_qual_agg["n_plans"]
                    lines.append(f"  {dim:<30} {count:>3} plans ({pct:>5.1f}%)")
        
        lines.append("")
    
    lines.append("=" * 80)
    
    return "\n".join(lines)


# ---------------------------- main ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_dir", type=str, default="./result/baselines")
    ap.add_argument("--results_dir", type=str, default="./results")
    ap.add_argument("--data_dir", type=str, default="./data")
    ap.add_argument("--models", type=str, default="all")
    ap.add_argument("--tag", type=str, default="testset")
    ap.add_argument("--max_plans", type=int, default=299)
    ap.add_argument("--presence_mode", type=str, default="argmax", choices=["argmax","quantile"],
                    help="How to derive facility presence on soft maps.")
    ap.add_argument("--presence_quantile", type=float, default=99.5,
                    help="Quantile for facility presence when presence_mode=quantile (e.g., 99.5).")
    ap.add_argument("--min_facility_cells", type=int, default=25,
                    help="Guardrail: minimum facility cells required for accessibility metrics.")
    ap.add_argument("--dimension_profile", type=str, default="planning_6dimension",
                    choices=["planning_6dimension"],
                    help="Dimension aggregation profile. The only supported option is planning_6dimension, which returns the six selected dimensions: spatial_coherence, development_compactness, healthy_environment, land_use_balance, community_convenience, and urban_resilience.")


    ap.add_argument("--do_quant", action="store_true")
    ap.add_argument("--do_free_llm", action="store_true")
    ap.add_argument("--save_all", action="store_true")
    
    # NEW: Enhanced LLM evaluation
    ap.add_argument("--do_llm_enhanced", action="store_true",
                   help="Run enhanced LLM quantitative evaluation with uncertainty")
    ap.add_argument("--llm_qualitative", action="store_true",
                   help="Run enhanced LLM qualitative analysis (requires --do_llm_enhanced)")
    ap.add_argument("--llm_quick", action="store_true",
                   help="Quick mode: fewer samples, faster evaluation")

    # dataset split alignment
    ap.add_argument("--ratio", type=float, default=0.9)
    ap.add_argument("--test_size", type=int, default=299)
    ap.add_argument("--align_seed", type=int, default=0)

    # free llm params (original)
    ap.add_argument("--rubrics", type=str, default="standard,detailed,critical")
    ap.add_argument("--n_samples", type=int, default=8)
    ap.add_argument("--noise_std", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    
    # Enhanced LLM quantitative parameters
    ap.add_argument("--llm_uncertainty_method", type=str, default="ensemble",
                   choices=["ensemble", "bootstrap", "bayesian"],
                   help="Uncertainty estimation method")
    ap.add_argument("--llm_n_samples", type=int, default=20,
                   help="Number of samples for uncertainty estimation")
    ap.add_argument("--llm_noise_levels", type=str, default="0.01,0.03,0.05",
                   help="Comma-separated noise levels")
    ap.add_argument("--llm_rubric_variants", type=str, default="standard,sustainability,equity,economic",
                   help="Comma-separated rubric variants")
    ap.add_argument("--llm_seed", type=int, default=42)
    
    # Enhanced LLM qualitative parameters
    ap.add_argument("--llm_analysis_depth", type=str, default="standard",
                   choices=["basic", "standard", "comprehensive"],
                   help="Depth of qualitative analysis")
    ap.add_argument("--llm_perspectives", type=str, default="planner,resident,developer,policymaker",
                   help="Comma-separated stakeholder perspectives")

    args = ap.parse_args()
    result_tag = f"{args.tag}_{args.dimension_profile}" if getattr(args, "dimension_profile", None) else args.tag
    
    # Check enhanced LLM availability
    if (args.do_llm_enhanced or args.llm_qualitative) and not LLM_ENHANCED_AVAILABLE:
        print("ERROR: Enhanced LLM evaluation requested but urban_plan_evaluator.py not found!")
        print("Please ensure urban_plan_evaluator.py is in the same directory.")
        sys.exit(1)
    
    # Quick mode overrides
    if args.llm_quick:
        args.llm_n_samples = 10
        args.llm_uncertainty_method = "bootstrap"
        args.llm_analysis_depth = "basic"
        print("Quick mode enabled: n_samples=10, method=bootstrap, depth=basic")
    
    # Parse LLM parameters
    llm_noise_levels = [float(x.strip()) for x in args.llm_noise_levels.split(",")]
    llm_rubric_variants = [x.strip() for x in args.llm_rubric_variants.split(",")]
    llm_perspectives = [x.strip() for x in args.llm_perspectives.split(",")]

    base_dir = os.path.expanduser(args.baseline_dir)
    results_dir = os.path.expanduser(args.results_dir)

    if args.models.lower() == "all":
        model_names = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
    else:
        model_names = [m.strip() for m in args.models.split(",") if m.strip()]

    rubrics = [r.strip() for r in args.rubrics.split(",") if r.strip()]

    real_eval = None
    zones_eval = None
    con_label_eval = None
    green_standards = None
    
    need_dataset = (args.do_quant or args.do_free_llm or args.do_llm_enhanced or getattr(args, "llm_qualitative", False))
    if need_dataset:
        try:
            real_eval, zones_eval, con_label_eval = load_canonical_test_subset(
                args.data_dir, ratio=args.ratio, test_size=args.test_size, seed=args.align_seed
            )
            if real_eval is not None and zones_eval is not None:
                print(f"Loaded test dataset: real={real_eval.shape}, zones={zones_eval.shape}")
            elif zones_eval is not None:
                print(f"Loaded test dataset: zones={zones_eval.shape} (real plans not available)")
            else:
                print("Loaded test dataset: (no real/zones returned)")

            if con_label_eval is not None:
                print(f"  con_label={con_label_eval.shape}, unique levels={np.unique(con_label_eval)}")

            # Load green_standards.pkl if available
            green_standards = load_green_standards(args.data_dir)
        except FileNotFoundError as e:
            if args.do_quant:
                raise
            print(f"[WARNING] Dataset files not found for data_dir={args.data_dir}: {e}")
            print("[WARNING] Continuing in gen-only mode for qualitative/free-form evaluation.")
            real_eval, zones_eval, con_label_eval, green_standards = None, None, None, None
    
    # Initialize enhanced LLM evaluator
    evaluator = None
    if args.do_llm_enhanced and LLM_ENHANCED_AVAILABLE:
        evaluator = UrbanPlanEvaluator()
        print("Initialized enhanced LLM evaluator")

    all_quant: Dict[str, Any] = {}
    all_free: Dict[str, Any] = {}
    all_llm_quant_agg: Dict[str, Any] = {}
    all_llm_qual_agg: Dict[str, Any] = {}

    for model_name in model_names:
        model_dir = os.path.join(base_dir, model_name)
        gen_path = find_generated_file(model_name, model_dir, args.tag, results_dir)
        if gen_path is None:
            print(f"[{model_name}] No generated file found. Skipping.")
            continue

        try:
            gen = _ensure_nchw(_load_npz_arr(gen_path))
        except Exception as e:
            print(f"[{model_name}] Failed to load/shape generated file {gen_path}: {e}")
            continue

        max_plans = int(args.max_plans)
        if args.do_quant:
            if real_eval is None or zones_eval is None:
                raise RuntimeError(
                    "Quantitative evaluation (--do_quant) requires the canonical test set. "
                    "Please set --data_dir to a folder containing 100_poi_dis.npz and func1_100.npz (and optional con_label.npz)."
                )
            N = min(max_plans, gen.shape[0], real_eval.shape[0], zones_eval.shape[0])
            gen = gen[:N]
            real = real_eval[:N]
            zones = zones_eval[:N]
            con_label = con_label_eval[:N] if con_label_eval is not None else None
            print(f"[{model_name}] Loaded {gen_path} -> gen={gen.shape} real={real.shape}")
        else:
            if zones_eval is not None:
                N = min(max_plans, gen.shape[0], zones_eval.shape[0])
                zones = zones_eval[:N]
            else:
                N = min(max_plans, gen.shape[0])
                zones = None
            gen = gen[:N]
            real = None
            if con_label_eval is not None and len(con_label_eval) >= N:
                con_label = con_label_eval[:N]
            else:
                con_label = None
            print(f"[{model_name}] Loaded {gen_path} -> gen={gen.shape} (gen-only)")

        # Quantitative
        if args.do_quant:
            quant_res = compute_quant_metrics(
                real, gen, zones,
                con_label=con_label,
                green_standards=green_standards,
                max_pairs=200,
                presence_mode=str(args.presence_mode),
                presence_quantile=float(args.presence_quantile),
                min_facility_cells=int(args.min_facility_cells),
                dimension_profile=str(getattr(args, "dimension_profile", "planning_6dimension"))
            )
            outp = os.path.join(model_dir, f"eval_quant_{result_tag}.json")
            with open(outp, "w", encoding="utf-8") as f:
                json.dump(quant_res, f, indent=2)
            all_quant[model_name] = quant_res
            print(f"[{model_name}] Saved quant -> {outp}")

        # Free LLM-style (original)
        if args.do_free_llm:
            per_plan = []
            for i in range(N):
                ab = run_free_ablation(gen[i], zones[i], rubrics=rubrics, n_samples=int(args.n_samples),
                                       noise_std=float(args.noise_std), seed=int(args.seed * 100000 + i))
                per_plan.append({"index": int(i), "ablation": ab})

            def _summ(setting: str) -> Dict[str, Any]:
                vals = []
                for p in per_plan:
                    v = p["ablation"][setting]["scores"]["overall"]["mean"]
                    if v is not None:
                        vals.append(float(v))
                if not vals:
                    return {"n": 0, "overall_mean_mean": None}
                arr = np.asarray(vals, dtype=np.float64)
                return {"n": int(arr.size),
                        "overall_mean_mean": float(np.mean(arr)),
                        "overall_mean_std": float(np.std(arr)),
                        "overall_mean_ci95": [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]}

            summary = {s: _summ(s) for s in ["no_uncertainty", "multi_sample_only", "prompt_ensemble_only", "full_uncertainty"]}
            free_res = {"model": model_name, "tag": args.tag, "n_plans": int(N),
                        "summary": summary, "per_plan": per_plan,
                        "config": {"rubrics": rubrics, "n_samples": int(args.n_samples), "noise_std": float(args.noise_std)}}
            outp = os.path.join(model_dir, f"eval_free_llm_{result_tag}.json")
            with open(outp, "w", encoding="utf-8") as f:
                json.dump(free_res, f, indent=2)
            all_free[model_name] = free_res
            print(f"[{model_name}] Saved free-llm -> {outp}")
        
        # ======================================================================
        # ENHANCED LLM QUANTITATIVE (with Uncertainty)
        # ======================================================================
        llm_quant_results = []
        llm_quant_agg = None
        
        if args.do_llm_enhanced:
            print(f"[{model_name}] Running enhanced LLM quantitative evaluation...")
            print(f"  Method: {args.llm_uncertainty_method}, Samples: {args.llm_n_samples}")
            
            for i in range(N):
                if (i + 1) % 50 == 0:
                    print(f"  Progress: {i+1}/{N} plans...")
                
                quant_result = evaluator.quantitative_eval(
                    gen[i], zones[i],
                    n_samples=args.llm_n_samples,
                    uncertainty_method=args.llm_uncertainty_method,
                    noise_levels=llm_noise_levels,
                    rubric_variants=llm_rubric_variants,
                    seed=args.llm_seed + i
                )
                
                llm_quant_results.append({
                    "plan_index": i,
                    **quant_result
                })
            
            # Aggregate results
            llm_quant_agg = aggregate_llm_quant_results(llm_quant_results)
            
            # Save results
            outp_per_plan = os.path.join(model_dir, f"eval_llm_quant_per_plan_{result_tag}.json")
            with open(outp_per_plan, "w", encoding="utf-8") as f:
                json.dump(llm_quant_results, f, indent=2)
            
            outp_agg = os.path.join(model_dir, f"eval_llm_quant_summary_{result_tag}.json")
            with open(outp_agg, "w", encoding="utf-8") as f:
                json.dump(llm_quant_agg, f, indent=2)
            
            all_llm_quant_agg[model_name] = llm_quant_agg
            
            print(f"[{model_name}] Saved LLM quant per-plan -> {outp_per_plan}")
            print(f"[{model_name}] Saved LLM quant summary -> {outp_agg}")
            print(f"[{model_name}] Overall: {llm_quant_agg['dimensions']['overall']['mean_across_plans']:.3f} "
                  f"(confidence: {llm_quant_agg['confidence_summary']['overall']['mean_confidence']:.3f})")
        
        # ======================================================================
        # ENHANCED LLM QUALITATIVE ANALYSIS
        # ======================================================================
        llm_qual_results = []
        llm_qual_agg = None
        
        if args.llm_qualitative and args.do_llm_enhanced:
            print(f"[{model_name}] Running enhanced LLM qualitative analysis...")
            print(f"  Depth: {args.llm_analysis_depth}, Perspectives: {len(llm_perspectives)}")
            
            for i in range(N):
                if (i + 1) % 50 == 0:
                    print(f"  Progress: {i+1}/{N} plans...")
                
                qual_result = evaluator.qualitative_eval(
                    gen[i], zones[i],
                    reference_plan=real[i],
                    analysis_depth=args.llm_analysis_depth,
                    perspectives=llm_perspectives
                )
                
                llm_qual_results.append({
                    "plan_index": i,
                    **qual_result
                })
            
            # Aggregate results
            llm_qual_agg = aggregate_llm_qual_results(llm_qual_results)
            
            # Save results
            outp_per_plan = os.path.join(model_dir, f"eval_llm_qual_per_plan_{result_tag}.json")
            with open(outp_per_plan, "w", encoding="utf-8") as f:
                json.dump(llm_qual_results, f, indent=2)
            
            outp_agg = os.path.join(model_dir, f"eval_llm_qual_summary_{result_tag}.json")
            with open(outp_agg, "w", encoding="utf-8") as f:
                json.dump(llm_qual_agg, f, indent=2)
            
            all_llm_qual_agg[model_name] = llm_qual_agg
            
            print(f"[{model_name}] Saved LLM qual per-plan -> {outp_per_plan}")
            print(f"[{model_name}] Saved LLM qual summary -> {outp_agg}")
            
            # Print quality distribution
            if "quality_distribution" in llm_qual_agg:
                print(f"[{model_name}] Quality distribution:")
                for quality, count in sorted(llm_qual_agg["quality_distribution"].items()):
                    pct = 100 * count / N
                    print(f"    {quality}: {count} ({pct:.1f}%)")
        
        # ======================================================================
        # GENERATE COMBINED SUMMARY REPORT
        # ======================================================================
        if args.do_quant or args.do_llm_enhanced:
            print(f"[{model_name}] Generating summary report...")
            
            summary_report = generate_summary_report(
                model_name,
                quant_results=quant_res if args.do_quant else None,
                llm_quant_agg=llm_quant_agg,
                llm_qual_agg=llm_qual_agg
            )
            
            # Save report
            report_path = os.path.join(model_dir, f"eval_summary_report_{result_tag}.txt")
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(summary_report)
            
            print(f"[{model_name}] Saved summary report -> {report_path}")
            print("")
            print(summary_report)
            print("")

    if args.save_all:
        if args.do_quant and all_quant:
            outp = os.path.join(base_dir, f"eval_all_quant_{result_tag}.json")
            with open(outp, "w", encoding="utf-8") as f:
                json.dump(all_quant, f, indent=2)
            print(f"[ALL] Saved combined quant -> {outp}")
        
        if args.do_free_llm and all_free:
            outp = os.path.join(base_dir, f"eval_all_free_llm_{result_tag}.json")
            with open(outp, "w", encoding="utf-8") as f:
                json.dump(all_free, f, indent=2)
            print(f"[ALL] Saved combined free-llm -> {outp}")
        
        if args.do_llm_enhanced and all_llm_quant_agg:
            outp = os.path.join(base_dir, f"eval_all_llm_quant_{result_tag}.json")
            with open(outp, "w", encoding="utf-8") as f:
                json.dump(all_llm_quant_agg, f, indent=2)
            print(f"[ALL] Saved combined LLM quant -> {outp}")
        
        if args.llm_qualitative and all_llm_qual_agg:
            outp = os.path.join(base_dir, f"eval_all_llm_qual_{result_tag}.json")
            with open(outp, "w", encoding="utf-8") as f:
                json.dump(all_llm_qual_agg, f, indent=2)
            print(f"[ALL] Saved combined LLM qual -> {outp}")
        
        # Cross-model comparison
        if all_llm_quant_agg:
            print("\n" + "=" * 80)
            print("CROSS-MODEL COMPARISON")
            print("=" * 80)
            print(f"\n{'Model':<20} {'Overall':<10} {'Confidence':<12} {'Best Dimension'}")
            print("-" * 80)
            
            for model_name, agg in all_llm_quant_agg.items():
                overall = agg['dimensions']['overall']['mean_across_plans']
                conf = agg['confidence_summary']['overall']['mean_confidence']
                
                # Find best dimension
                best_dim = max(
                    [(d, agg['dimensions'][d]['mean_across_plans']) 
                     for d in agg['dimensions'] if d != 'overall'],
                    key=lambda x: x[1]
                )
                
                print(f"{model_name:<20} {overall:>9.3f} {conf:>11.3f} {best_dim[0]} ({best_dim[1]:.3f})")
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()

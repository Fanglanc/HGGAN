import ast
import json
import os
import random
import re
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import json5  # type: ignore
except Exception:
    json5 = None

EPS = 1e-8

SELECTED_DIMENSIONS = [
    "spatial_coherence",
    "development_compactness",
    "healthy_environment",
    "land_use_balance",
    "community_convenience",
    "urban_resilience",
]

RUN_LENSES = [
    "morphology-first: pay slightly more attention to structure, continuity, clustering, and fragmentation.",
    "resident-daily-life-first: pay slightly more attention to convenience, essential-service proximity, and neighborhood usability.",
    "public-health-first: pay slightly more attention to exposure, residential protection, greenery, and incompatible adjacencies.",
    "balanced-growth-first: pay slightly more attention to compact development, land-use balance, and sprawl control.",
    "resilience-first: pay slightly more attention to redundancy, green continuity, distributed services, and robustness to localized failure.",
    "planning-synthesis: use a balanced interpretation across all six dimensions with no strong bias.",
]

POI_NAMES = [
    "road",
    "car service",
    "car repair",
    "motorbike service",
    "food service",
    "shopping",
    "daily life service",
    "recreation service",
    "medical service",
    "lodging",
    "tourist attraction",
    "real estate",
    "government place",
    "education",
    "transportation",
    "finance",
    "company",
    "road furniture",
    "specific address",
    "public service",
]

# Correct Beijing mapping (same direction as fixed planning evaluators)
IDX_RES = [11]
IDX_JOBS = [1, 2, 3, 15, 16]
IDX_SERVICE = [4, 5, 6, 12, 19]
IDX_ESSENTIAL = [4, 6, 8, 13, 19]
IDX_SCHOOL = [13]
IDX_MEDICAL = [8]
IDX_TRANSIT = [14]
IDX_GREEN = [7, 10]
IDX_STREET = [0, 17]
IDX_IMPACT = [0, 1, 2, 3, 17]  # infrastructure / nuisance-ish


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------

def _safe_prob(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = np.clip(x, 0.0, None)
    s = float(x.sum())
    if s <= 0:
        return np.ones_like(x) / max(len(x), 1)
    return x / (s + EPS)


def _normalized_entropy(x: np.ndarray) -> float:
    p = _safe_prob(x)
    h = float(-(p * np.log(p + EPS)).sum())
    hmax = float(np.log(len(p) + EPS))
    return float(h / (hmax + EPS))


def _simpson_diversity(x: np.ndarray) -> float:
    p = _safe_prob(x)
    return float(1.0 - np.sum(p ** 2))


def _dominant_map(plan: np.ndarray) -> np.ndarray:
    return np.argmax(plan, axis=0).astype(np.int32)


def _developed_mask(plan: np.ndarray, thr: float = 0.0) -> np.ndarray:
    tot = plan.sum(axis=0)
    return (tot > thr).astype(np.uint8)


def _edge_density(dom: np.ndarray) -> float:
    H, W = dom.shape
    diff = 0
    tot = 0
    if H > 1:
        diff += int(np.sum(dom[1:, :] != dom[:-1, :]))
        tot += (H - 1) * W
    if W > 1:
        diff += int(np.sum(dom[:, 1:] != dom[:, :-1]))
        tot += H * (W - 1)
    return float(diff / (tot + EPS))


def _aggregation_index(dom: np.ndarray) -> float:
    H, W = dom.shape
    same = 0
    tot = 0
    if H > 1:
        same += int(np.sum(dom[1:, :] == dom[:-1, :]))
        tot += (H - 1) * W
    if W > 1:
        same += int(np.sum(dom[:, 1:] == dom[:, :-1]))
        tot += H * (W - 1)
    return float(same / (tot + EPS))


def _compactness_ratio(mask: np.ndarray) -> float:
    mask = (mask > 0).astype(np.uint8)
    A = float(mask.sum())
    if A <= 0:
        return 0.0
    P = 0.0
    P += float(np.sum(mask[1:, :] != mask[:-1, :]))
    P += float(np.sum(mask[:, 1:] != mask[:, :-1]))
    P += float(np.sum(mask[0, :])) + float(np.sum(mask[-1, :])) + float(np.sum(mask[:, 0])) + float(np.sum(mask[:, -1]))
    if P <= 0:
        return 0.0
    return float((4.0 * np.pi * A) / (P * P + EPS))


def _label_components(binary: np.ndarray) -> int:
    binary = (binary > 0).astype(np.uint8)
    H, W = binary.shape
    vis = np.zeros_like(binary, dtype=np.uint8)
    from collections import deque
    comps = 0
    for i in range(H):
        for j in range(W):
            if binary[i, j] and not vis[i, j]:
                comps += 1
                q = deque([(i, j)])
                vis[i, j] = 1
                while q:
                    x, y = q.popleft()
                    for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < H and 0 <= ny < W and binary[nx, ny] and not vis[nx, ny]:
                            vis[nx, ny] = 1
                            q.append((nx, ny))
    return comps


def _patch_density(dom: np.ndarray) -> float:
    H, W = dom.shape
    patches = 0
    for k in np.unique(dom):
        patches += _label_components(dom == k)
    return float(patches / ((H * W) / 10000.0 + EPS))


def _window_composition(plan: np.ndarray, x0: int, y0: int, win: int) -> np.ndarray:
    return plan[:, x0:x0+win, y0:y0+win].sum(axis=(1, 2))


def _mixing_index(plan: np.ndarray, win: int = 10, stride: int = 10) -> float:
    C, H, W = plan.shape
    vals = []
    for x0 in range(0, H - win + 1, stride):
        for y0 in range(0, W - win + 1, stride):
            vals.append(_normalized_entropy(_window_composition(plan, x0, y0, win)))
    return float(np.mean(vals)) if vals else 0.0


def _distance_to_nearest(facility_mask: np.ndarray, cap: Optional[float] = None) -> np.ndarray:
    facility_mask = (facility_mask > 0).astype(np.uint8)
    H, W = facility_mask.shape
    if cap is None:
        cap = float(np.hypot(H, W))
    dist = np.full((H, W), fill_value=float(cap), dtype=float)
    from collections import deque
    q = deque()
    xs, ys = np.where(facility_mask)
    for x, y in zip(xs.tolist(), ys.tolist()):
        dist[x, y] = 0.0
        q.append((x, y))
    if not q:
        return dist
    while q:
        x, y = q.popleft()
        nd = dist[x, y] + 1.0
        for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < H and 0 <= ny < W and nd < dist[nx, ny]:
                dist[nx, ny] = nd
                q.append((nx, ny))
    return np.clip(np.nan_to_num(dist, nan=float(cap), posinf=float(cap), neginf=0.0), 0.0, float(cap))


def _weighted_mean(x: np.ndarray, w: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    s = float(np.sum(w))
    if s <= EPS:
        return 0.0
    return float(np.sum(x * w) / (s + EPS))


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=float).reshape(-1)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    m = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(m):
        return float("nan")
    values = values[m]
    weights = weights[m]
    idx = np.argsort(values)
    values = values[idx]
    weights = weights[idx]
    cw = np.cumsum(weights)
    cutoff = (q / 100.0) * cw[-1]
    return float(values[min(len(values)-1, int(np.searchsorted(cw, cutoff)))])


def _gini(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).flatten()
    x = np.clip(x, 0.0, None)
    if x.size == 0:
        return 0.0
    s = float(x.sum())
    if s <= 0:
        return 0.0
    x_sorted = np.sort(x)
    n = x_sorted.size
    idx = np.arange(1, n + 1, dtype=float)
    return float((2.0 * np.sum(idx * x_sorted) / (n * s)) - (n + 1) / n)


def _presence_mask(plan: np.ndarray, channels: List[int], mode: str = "argmax", q: float = 99.0) -> np.ndarray:
    C, H, W = plan.shape
    chs = [c for c in channels if 0 <= c < C]
    if not chs:
        return np.zeros((H, W), dtype=bool)
    x = np.asarray(plan[chs].sum(axis=0), dtype=float)
    vals = x[x > EPS]
    if vals.size == 0:
        return np.zeros((H, W), dtype=bool)
    if mode == "argmax":
        dom = np.argmax(plan, axis=0)
        base = np.isin(dom, chs) & (x > EPS)
        if int(base.sum()) == 0:
            for qq in (97.5, 95.0, 90.0):
                thr = float(np.percentile(vals, qq))
                m = (x >= thr) & (x > EPS)
                if int(m.sum()) > 0:
                    return m
            return np.zeros((H, W), dtype=bool)
        base_vals = x[base]
        thr = float(np.percentile(base_vals, 85.0)) if base_vals.size > 0 else float(np.percentile(vals, 95.0))
        gated = base & (x >= thr)
        if int(gated.sum()) >= 3:
            return gated
        if int(base.sum()) >= 3:
            return base
        for qq in (97.5, 95.0, 90.0):
            thr2 = float(np.percentile(vals, qq))
            m = base | ((x >= thr2) & (x > EPS))
            if int(m.sum()) >= 3:
                return m
        return base
    thr = float(np.percentile(vals, q))
    m = (x >= thr) & (x > EPS)
    if int(m.sum()) == 0:
        thr = float(np.percentile(vals, 95.0))
        m = (x >= thr) & (x > EPS)
    return m


def _coarse_block_sum(channel: np.ndarray, k: int = 10) -> np.ndarray:
    H, W = channel.shape
    bh = max(1, H // k)
    bw = max(1, W // k)
    out = np.zeros((k, k), dtype=float)
    for i in range(k):
        for j in range(k):
            out[i, j] = float(channel[i*bh:(i+1)*bh, j*bw:(j+1)*bw].sum())
    return out


def _mask_centroid(mask: np.ndarray) -> Optional[Tuple[float, float]]:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def describe_spatial_patterns(plan: np.ndarray, max_items: int = 14) -> List[str]:
    plan = np.nan_to_num(plan, nan=0.0).astype(float)
    plan = np.clip(plan, 0.0, None)
    if plan.ndim != 3:
        return []
    C, H, W = plan.shape
    obs: List[str] = []
    totals = np.asarray(plan.sum(axis=(1, 2)), dtype=float)
    if float(totals.sum()) > 0:
        order = np.argsort(totals)[::-1]
        top_idx = [int(i) for i in order[:5] if totals[int(i)] > 0]
        if top_idx:
            obs.append("Top POI categories by total intensity: " + ", ".join([f"{POI_NAMES[i]}({totals[i]:.3g})" for i in top_idx]) + ".")

    jobs = plan[IDX_JOBS].sum(axis=0)
    ctr = _mask_centroid(jobs > 0)
    if ctr is not None:
        obs.append(f"Jobs centroid near (x={ctr[0]:.1f}, y={ctr[1]:.1f}); inspect monocentric vs distributed employment geography.")

    housing = plan[IDX_RES].sum(axis=0)
    ctr = _mask_centroid(housing > 0)
    if ctr is not None:
        obs.append(f"Residential centroid near (x={ctr[0]:.1f}, y={ctr[1]:.1f}); compare to essential services and green amenities.")

    road_mask = (plan[0] + plan[17]) > 0 if C >= 18 else (plan[0] > 0)
    for i in [14, 5, 4, 6, 16, 15]:
        if i >= C:
            continue
        m = (plan[i] > 0)
        if not m.any():
            continue
        overlap = float(np.sum(m & road_mask)) / max(1.0, float(np.sum(m)))
        if overlap >= 0.6:
            obs.append(f"{POI_NAMES[i]} strongly aligns with road corridors (overlap ~{overlap:.2f}).")
        elif overlap <= 0.2:
            obs.append(f"{POI_NAMES[i]} is mostly off-corridor (overlap ~{overlap:.2f}).")

    for i in [4, 6, 8, 13, 14, 5]:
        if i >= C or totals[i] <= 0:
            continue
        heat = _coarse_block_sum(plan[i], k=10)
        flat = heat.ravel()
        if flat.max() <= 0:
            continue
        top_cells = np.argsort(flat)[::-1][:2]
        cells = [f"({int(idx//10)},{int(idx%10)})={flat[idx]:.3g}" for idx in top_cells]
        obs.append(f"{POI_NAMES[i]} hotspots (coarse 10x10 cells row,col): " + ", ".join(cells) + ".")

    dedup = []
    seen = set()
    for x in obs:
        if x not in seen:
            dedup.append(x)
            seen.add(x)
        if len(dedup) >= max_items:
            break
    return dedup


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------
class LLMProvider(ABC):
    def __init__(self, api_key: Optional[str] = None, temperature: float = 0.7, model: Optional[str] = None):
        self.api_key = api_key
        self.temperature = temperature
        self.name = "base"
        self.model = model

    @abstractmethod
    def call_api(self, prompt: str) -> str:
        pass


class GeminiProvider(LLMProvider):
    DEFAULT_MODEL = "gemini-2.5-flash"

    def __init__(self, api_key: Optional[str] = None, temperature: float = 0.7, model: Optional[str] = None):
        super().__init__(api_key, temperature, model=model)
        self.name = "gemini"
        self.model = model or self.DEFAULT_MODEL
        if not self.api_key:
            self.api_key = os.environ.get("GOOGLE_API_KEY")

    def call_api(self, prompt: str) -> str:
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("Install: pip install google-generativeai")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not set")
        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(self.model)
        generation_config = {
            "temperature": self.temperature,
            "max_output_tokens": 8192,
            "response_mime_type": "application/json",
        }
        try:
            resp = model.generate_content(prompt, generation_config=generation_config)
        except TypeError:
            generation_config.pop("response_mime_type", None)
            resp = model.generate_content(prompt, generation_config=generation_config)
        return resp.text


class ClaudeProvider(LLMProvider):
    DEFAULT_MODEL = "claude-sonnet-4-20250514"

    def __init__(self, api_key: Optional[str] = None, temperature: float = 0.7, model: Optional[str] = None):
        super().__init__(api_key, temperature, model=model)
        self.name = "claude"
        self.model = model or self.DEFAULT_MODEL
        if not self.api_key:
            self.api_key = os.environ.get("ANTHROPIC_API_KEY")

    def call_api(self, prompt: str) -> str:
        try:
            import anthropic
        except ImportError:
            raise ImportError("Install: pip install anthropic")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        client = anthropic.Anthropic(api_key=self.api_key)
        msg = client.messages.create(
            model=self.model,
            max_tokens=3000,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text


class GPTProvider(LLMProvider):
    DEFAULT_MODEL = "gpt-4o"

    def __init__(self, api_key: Optional[str] = None, temperature: float = 0.7, model: Optional[str] = None):
        super().__init__(api_key, temperature, model=model)
        self.name = "gpt"
        self.model = model or self.DEFAULT_MODEL
        if not self.api_key:
            self.api_key = os.environ.get("OPENAI_API_KEY")

    def call_api(self, prompt: str) -> str:
        try:
            import openai
        except ImportError:
            raise ImportError("Install: pip install openai")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not set")
        client = openai.OpenAI(api_key=self.api_key)
        resp = client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=3000,
            messages=[
                {"role": "system", "content": "You are an expert urban planner and evaluator."},
                {"role": "user", "content": prompt},
            ],
        )
        return resp.choices[0].message.content


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------
class RobustLLMEvaluator:
    def __init__(self, provider: str = "gemini", api_key: Optional[str] = None, temperature: float = 0.7, model: Optional[str] = None, allow_mock: bool = False):
        self.allow_mock = allow_mock
        provider = (provider or "gemini").strip().lower()
        if provider == "openai":
            provider = "gpt"
        if provider == "anthropic":
            provider = "claude"
        if provider == "gemini":
            self.llm = GeminiProvider(api_key, temperature, model=model)
        elif provider == "claude":
            self.llm = ClaudeProvider(api_key, temperature, model=model)
        elif provider == "gpt":
            self.llm = GPTProvider(api_key, temperature, model=model)
        else:
            raise ValueError(f"Unknown provider: {provider}")
        self.poi_names = POI_NAMES

    def _extract_plan_characteristics(self, plan: np.ndarray) -> Dict[str, Any]:
        plan = np.nan_to_num(plan, nan=0.0).astype(float)
        plan = np.clip(plan, 0.0, None)
        if plan.ndim != 3:
            raise ValueError(f"Plan must be (C,H,W), got {plan.shape}")
        C, H, W = plan.shape
        ch_totals = plan.sum(axis=(1, 2))
        total_mass = float(ch_totals.sum()) + EPS
        poi_totals = {self.poi_names[i] if i < len(self.poi_names) else f"channel_{i}": float(ch_totals[i]) for i in range(C)}
        poi_shares = {self.poi_names[i] if i < len(self.poi_names) else f"channel_{i}": float(ch_totals[i] / total_mass) for i in range(C)}
        dom = _dominant_map(plan)
        developed = _developed_mask(plan)

        res_mass = plan[IDX_RES].sum(axis=0)
        res_w = np.maximum(res_mass, 0.0)
        if float(res_w.sum()) <= EPS:
            res_w = np.maximum(plan.sum(axis=0), 0.0)
        if float(res_w.sum()) <= EPS:
            res_w = np.ones((H, W), dtype=float)

        jobs_mask = _presence_mask(plan, IDX_JOBS, mode="argmax")
        service_mask = _presence_mask(plan, IDX_SERVICE, mode="argmax")
        essential_mask = _presence_mask(plan, IDX_ESSENTIAL, mode="argmax")
        school_mask = _presence_mask(plan, IDX_SCHOOL, mode="argmax")
        medical_mask = _presence_mask(plan, IDX_MEDICAL, mode="argmax")
        transit_mask = _presence_mask(plan, IDX_TRANSIT, mode="argmax")
        green_mask = _presence_mask(plan, IDX_GREEN, mode="argmax")
        impact_mask = _presence_mask(plan, IDX_IMPACT, mode="argmax")
        road_mask = _presence_mask(plan, [0, 17], mode="argmax")

        dist_service = _distance_to_nearest(service_mask)
        dist_essential = _distance_to_nearest(essential_mask)
        dist_school = _distance_to_nearest(school_mask)
        dist_medical = _distance_to_nearest(medical_mask)
        dist_transit = _distance_to_nearest(transit_mask)
        dist_green = _distance_to_nearest(green_mask)
        dist_jobs = _distance_to_nearest(jobs_mask)
        dist_impact = _distance_to_nearest(impact_mask)
        dist_road = _distance_to_nearest(road_mask)

        # morphology evidence
        aggregation = _aggregation_index(dom)
        edge = _edge_density(dom)
        patch = _patch_density(dom)
        compactness = _compactness_ratio(developed)
        mixing = _mixing_index(plan, win=10, stride=10)
        entropy = _normalized_entropy(ch_totals)
        simpson = _simpson_diversity(ch_totals)
        jobs_housing_ratio = float(plan[IDX_JOBS].sum() / (plan[IDX_RES].sum() + EPS))

        # resident-facing access evidence (finite, weighted)
        def cov(dist, thr):
            return float(np.sum(((dist <= thr).astype(float)) * res_w) / (np.sum(res_w) + EPS))
        service_cov_10 = cov(dist_service, 10)
        essential_cov_12 = cov(dist_essential, 12)
        school_cov_15 = cov(dist_school, 15)
        medical_cov_15 = cov(dist_medical, 15)
        transit_cov_12 = cov(dist_transit, 12)
        green_cov_15 = cov(dist_green, 15)

        mean_service_dist = _weighted_mean(dist_service, res_w)
        mean_essential_dist = _weighted_mean(dist_essential, res_w)
        mean_green_dist = _weighted_mean(dist_green, res_w)
        mean_jobs_dist = _weighted_mean(dist_jobs, res_w)
        p75_service_dist = _weighted_quantile(dist_service, res_w, 75.0)
        p90_service_dist = _weighted_quantile(dist_service, res_w, 90.0)
        p90_essential_dist = _weighted_quantile(dist_essential, res_w, 90.0)

        # health / exposure proxies aligned to selected-6 metrics
        mean_impact_dist = _weighted_mean(dist_impact, res_w)
        impact_cov_6 = cov(dist_impact, 6)
        road_noise_exposure_4 = cov(dist_road, 4)
        green_share = float(plan[IDX_GREEN].sum() / total_mass)
        impervious_share = float(plan[[0,5,6,11,15,16,17]].sum() / total_mass) if C > 17 else float(plan.sum() / total_mass)
        urban_heat_risk = float(np.clip(0.65 * impervious_share + 0.35 * (1.0 - green_share), 0.0, 1.0))
        residential_noise_exposure = float(np.clip(0.55 * impact_cov_6 + 0.45 * road_noise_exposure_4, 0.0, 1.0))
        buffer_conflict_r3 = float(np.clip(cov(dist_impact, 3), 0.0, 1.0))

        # Green connectivity / distribution
        green_components = _label_components(green_mask)
        green_largest_share = 0.0
        if green_mask.any():
            from collections import deque
            H2, W2 = green_mask.shape
            vis = np.zeros_like(green_mask, dtype=np.uint8)
            sizes = []
            for i in range(H2):
                for j in range(W2):
                    if green_mask[i, j] and not vis[i, j]:
                        q = deque([(i, j)])
                        vis[i, j] = 1
                        sz = 0
                        while q:
                            x, y = q.popleft(); sz += 1
                            for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
                                nx, ny = x + dx, y + dy
                                if 0 <= nx < H2 and 0 <= ny < W2 and green_mask[nx, ny] and not vis[nx, ny]:
                                    vis[nx, ny] = 1; q.append((nx, ny))
                        sizes.append(sz)
            if sizes:
                green_largest_share = float(max(sizes) / max(1, sum(sizes)))
        green_connectivity_v5 = float(np.clip(green_largest_share * min(1.0, green_components / 3.0) if green_components > 0 else 0.0, 0.0, 1.0))

        # service center balance via coarse blocks
        essential_field = plan[IDX_ESSENTIAL].sum(axis=0)
        jobs_field = plan[IDX_JOBS].sum(axis=0)
        green_field = plan[IDX_GREEN].sum(axis=0)
        service_field = plan[IDX_SERVICE].sum(axis=0)
        essential_blocks = _coarse_block_sum(essential_field, k=10).ravel()
        jobs_blocks = _coarse_block_sum(jobs_field, k=10).ravel()
        green_blocks = _coarse_block_sum(green_field, k=10).ravel()
        service_blocks = _coarse_block_sum(service_field, k=10).ravel()
        essential_block_entropy = _normalized_entropy(essential_blocks)
        jobs_block_entropy = _normalized_entropy(jobs_blocks)
        green_block_entropy = _normalized_entropy(green_blocks)
        service_block_entropy = _normalized_entropy(service_blocks)
        essential_gini = _gini(essential_blocks)
        jobs_gini = _gini(jobs_blocks)
        service_gini = _gini(service_blocks)

        land_use_compatibility = float(np.clip(
            0.35 * (1.0 - buffer_conflict_r3) +
            0.25 * (1.0 - residential_noise_exposure) +
            0.20 * min(1.0, service_cov_10 + school_cov_15) / 1.5 +
            0.20 * min(1.0, green_cov_15 + transit_cov_12) / 1.5,
            0.0, 1.0
        ))

        # selected-6 aligned subdimension proxies for the LLM evidence
        service_proximity_proxy = float(np.clip(
            0.34 * service_cov_10 +
            0.20 * essential_cov_12 +
            0.14 * school_cov_15 +
            0.12 * medical_cov_15 +
            0.10 * transit_cov_12 +
            0.10 * max(0.0, 1.0 - p75_service_dist / max(1.0, np.hypot(H, W))),
            0.0, 1.0
        ))
        amenity_diversity_proxy = float(np.clip(0.40 * entropy + 0.35 * simpson + 0.25 * mixing, 0.0, 1.0))
        public_space_access_proxy = float(np.clip(0.45 * green_cov_15 + 0.30 * green_connectivity_v5 + 0.25 * green_share / max(green_share, 0.25), 0.0, 1.0))
        resident_wellbeing_proxy = float(np.clip(
            0.30 * service_proximity_proxy +
            0.25 * public_space_access_proxy +
            0.20 * (1.0 - residential_noise_exposure) +
            0.15 * (1.0 - urban_heat_risk) +
            0.10 * (1.0 - buffer_conflict_r3),
            0.0, 1.0
        ))
        public_service_readiness_proxy = float(np.clip(
            0.32 * essential_cov_12 + 0.22 * medical_cov_15 + 0.18 * school_cov_15 +
            0.14 * transit_cov_12 + 0.14 * (1.0 - mean_essential_dist / max(1.0, np.hypot(H, W))),
            0.0, 1.0
        ))
        balanced_service_distribution_proxy = float(np.clip(
            0.45 * (1.0 - service_gini) + 0.30 * service_block_entropy + 0.25 * (1.0 - max(0.0, 1.0 - essential_cov_12)),
            0.0, 1.0
        ))
        activity_center_strength_proxy = float(np.clip(
            0.35 * jobs_block_entropy + 0.30 * service_block_entropy + 0.20 * (1.0 - jobs_gini) + 0.15 * (1.0 - abs(jobs_housing_ratio - 1.0) / 2.0),
            0.0, 1.0
        ))
        environmental_resilience_proxy = float(np.clip(
            0.38 * green_connectivity_v5 + 0.22 * green_cov_15 + 0.20 * (1.0 - urban_heat_risk) + 0.20 * (1.0 - green_largest_share if green_components == 1 else green_largest_share),
            0.0, 1.0
        ))
        safety_preparedness_proxy = float(np.clip(
            0.35 * medical_cov_15 + 0.25 * (1.0 - buffer_conflict_r3) + 0.20 * (1.0 - residential_noise_exposure) + 0.20 * (1.0 - impact_cov_6),
            0.0, 1.0
        ))

        evidence_by_dimension = {
            "spatial_coherence": {
                "aggregation_index": aggregation,
                "patch_density": patch,
                "compactness_ratio": compactness,
                "edge_density": edge,
                "dominant_residential_share": float(np.mean(dom == 11)),
                "dominant_green_share": float(np.mean(np.isin(dom, IDX_GREEN))),
            },
            "development_compactness": {
                "compactness_ratio": compactness,
                "patch_density": patch,
                "developed_share": float(developed.mean()),
                "aggregation_index": aggregation,
                "edge_density": edge,
            },
            "healthy_environment": {
                "urban_heat_risk": urban_heat_risk,
                "residential_noise_exposure": residential_noise_exposure,
                "buffer_conflict_r3": buffer_conflict_r3,
                "green_connectivity_v5": green_connectivity_v5,
                "land_use_compatibility": land_use_compatibility,
                "green_share": green_share,
                "green_access_coverage_15": green_cov_15,
            },
            "land_use_balance": {
                "land_mix_entropy": entropy,
                "simpson_diversity": simpson,
                "local_mixing_index": mixing,
                "jobs_housing_ratio": jobs_housing_ratio,
                "service_share": float(plan[IDX_SERVICE].sum() / total_mass),
                "green_share": green_share,
            },
            "community_convenience": {
                "service_proximity_proxy": service_proximity_proxy,
                "resident_wellbeing_proxy": resident_wellbeing_proxy,
                "amenity_diversity_proxy": amenity_diversity_proxy,
                "public_space_access_proxy": public_space_access_proxy,
                "service_coverage_10": service_cov_10,
                "essential_coverage_12": essential_cov_12,
                "school_coverage_15": school_cov_15,
                "medical_coverage_15": medical_cov_15,
                "transit_coverage_12": transit_cov_12,
                "mean_service_distance": mean_service_dist,
                "p75_service_distance": p75_service_dist,
                "mean_jobs_distance": mean_jobs_dist,
            },
            "urban_resilience": {
                "environmental_resilience_proxy": environmental_resilience_proxy,
                "safety_preparedness_proxy": safety_preparedness_proxy,
                "public_service_readiness_proxy": public_service_readiness_proxy,
                "balanced_service_distribution_proxy": balanced_service_distribution_proxy,
                "activity_center_strength_proxy": activity_center_strength_proxy,
                "green_access_proxy": public_space_access_proxy,
                "green_component_count": float(green_components),
                "green_largest_component_share": green_largest_share,
                "essential_block_entropy": essential_block_entropy,
                "jobs_block_entropy": jobs_block_entropy,
                "service_block_entropy": service_block_entropy,
                "essential_block_gini": essential_gini,
                "jobs_block_gini": jobs_gini,
                "service_block_gini": service_gini,
                "p90_essential_distance": p90_essential_dist,
                "mean_essential_distance": mean_essential_dist,
            },
        }

        presence_counts = {
            "residential_cells": int(_presence_mask(plan, IDX_RES, mode="argmax").sum()),
            "service_cells": int(service_mask.sum()),
            "essential_cells": int(essential_mask.sum()),
            "school_cells": int(school_mask.sum()),
            "medical_cells": int(medical_mask.sum()),
            "transit_cells": int(transit_mask.sum()),
            "green_cells": int(green_mask.sum()),
            "job_cells": int(jobs_mask.sum()),
        }

        return {
            "poi_totals": poi_totals,
            "poi_shares": poi_shares,
            "presence_counts": presence_counts,
            "evidence_by_dimension": evidence_by_dimension,
            "pattern_observations": describe_spatial_patterns(plan),
        }

    def _create_evaluation_prompt(self, characteristics: Dict[str, Any], plan_id: int, run_number: int) -> str:
        lens = RUN_LENSES[(run_number - 1) % len(RUN_LENSES)]
        evidence_by_dimension = characteristics["evidence_by_dimension"]
        # shuffle metric order slightly to reduce anchoring and encourage non-identical runs
        shuffled = {}
        rng = random.Random(plan_id * 1000 + run_number)
        for dim, block in evidence_by_dimension.items():
            items = list(block.items())
            rng.shuffle(items)
            shuffled[dim] = {k: v for k, v in items}

        metrics_block = json.dumps(shuffled, indent=2, sort_keys=False)
        poi_totals_block = json.dumps(characteristics["poi_totals"], indent=2, sort_keys=True)
        patterns = characteristics.get("pattern_observations", [])
        pattern_block = "\n".join([f"- {x}" for x in patterns[:14]]) if patterns else "- (none)"
        poi_shares_block = json.dumps(characteristics.get("poi_shares", {}), indent=2, sort_keys=True)
        presence_block = json.dumps(characteristics.get("presence_counts", {}), indent=2, sort_keys=True)

        prompt = f"""You are an expert urban planner. Judge ONLY the single sampled generated plan #{plan_id}.

This is run #{run_number}. Use the following secondary evaluation lens for this run:
{lens}

IMPORTANT:
- You are NOT allowed to score using any hidden deterministic formula.
- You MUST judge the six dimensions directly as an LLM urban-planning reviewer using the evidence below.
- Small score differences across runs are acceptable if your weighting or interpretation changes slightly, but stay grounded in the evidence.
- Evaluate ONLY this sampled plan. Do not infer model-level quality from any larger batch.
- Scores should usually not all be identical; reflect uncertainty where appropriate.

POI MAPPING (20 channels):
0 road; 1 car service; 2 car repair; 3 motorbike service; 4 food service; 5 shopping; 6 daily life service; 7 recreation service; 8 medical service; 9 lodging; 10 tourist attraction; 11 real estate (residential); 12 government place; 13 education; 14 transportation; 15 finance; 16 company; 17 road furniture; 18 specific address; 19 public service.

DIMENSION DEFINITIONS:
1. spatial_coherence: whether the plan forms a coherent, continuous, non-chaotic spatial structure.
2. development_compactness: whether development is spatially efficient and avoids fragmented sprawl.
3. healthy_environment: whether housing/residents are protected from incompatible or harmful exposures and have green support.
4. land_use_balance: whether uses are reasonably balanced and mixed without becoming chaotic or overly single-purpose.
5. community_convenience: whether daily life needs and key services are convenient for residents.
6. urban_resilience: whether the plan has redundancy, distributed support systems, and robustness rather than fragile over-concentration.

SCORING GUIDANCE:
- Score range: 0.0 to 1.0
- Confidence range: 0.0 to 1.0
- 0.2 = poor; 0.5 = mixed/acceptable; 0.8 = strong
- Lower confidence when evidence is ambiguous or conflicting.
- Give short, concrete justifications tied to the evidence.

POI TOTALS (raw intensity sums):
{poi_totals_block}

POI SHARES (fraction of total intensity):
{poi_shares_block}

DETECTED PRESENCE COUNTS (robust masks, not raw >0):
{presence_block}

EVIDENCE BY DIMENSION:
{metrics_block}

PATTERN OBSERVATIONS:
{pattern_block}

Return ONLY valid JSON with this exact schema:
{{
  "dimensions": {{
    "spatial_coherence": {{"score": <0..1>, "justification": "<text>", "confidence": <0..1>}},
    "development_compactness": {{"score": <0..1>, "justification": "<text>", "confidence": <0..1>}},
    "healthy_environment": {{"score": <0..1>, "justification": "<text>", "confidence": <0..1>}},
    "land_use_balance": {{"score": <0..1>, "justification": "<text>", "confidence": <0..1>}},
    "community_convenience": {{"score": <0..1>, "justification": "<text>", "confidence": <0..1>}},
    "urban_resilience": {{"score": <0..1>, "justification": "<text>", "confidence": <0..1>}}
  }},
  "overall_assessment": {{
    "strengths": ["<bullet>", "<bullet>", "<bullet>"],
    "weaknesses": ["<bullet>", "<bullet>", "<bullet>"],
    "improvements": ["<bullet>", "<bullet>", "<bullet>"],
    "insights": ["<bullet>", "<bullet>"]
  }}
}}
"""
        return prompt

    def _parse_llm_response(self, response: str, plan_id: int) -> Dict[str, Any]:
        response_clean = response.strip()
        candidate = response_clean
        if "```" in candidate:
            m = re.search(r"```(?:json)?\s*(.*?)```", candidate, re.DOTALL | re.IGNORECASE)
            if m:
                candidate = m.group(1).strip()
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = candidate[start:end+1]

        parsed = None
        last_err = None
        for parser in (
            lambda s: json.loads(s),
            (lambda s: json5.loads(s)) if json5 else None,
        ):
            if parser is None:
                continue
            try:
                parsed = parser(candidate)
                break
            except Exception as e:
                last_err = e
        if parsed is None:
            try:
                py_like = re.sub(r"\btrue\b", "True", candidate, flags=re.IGNORECASE)
                py_like = re.sub(r"\bfalse\b", "False", py_like, flags=re.IGNORECASE)
                py_like = re.sub(r"\bnull\b", "None", py_like, flags=re.IGNORECASE)
                parsed = ast.literal_eval(py_like)
            except Exception as e:
                last_err = e

        if parsed is None or not isinstance(parsed, dict):
            if self.allow_mock:
                return self._create_mock_result(plan_id, reason="parse error", error=str(last_err))
            raise ValueError(f"LLM response parsing failed for plan {plan_id}: {last_err}")

        result: Dict[str, Any] = {
            "plan_id": plan_id,
            "dimensions": {},
            "overall_assessment": parsed.get("overall_assessment", {}) if isinstance(parsed.get("overall_assessment", {}), dict) else {},
        }
        dims_block = parsed.get("dimensions", {})
        if not isinstance(dims_block, dict):
            dims_block = {}
        for dim in SELECTED_DIMENSIONS:
            dim_data = dims_block.get(dim, {})
            if not isinstance(dim_data, dict):
                continue
            try:
                score = float(dim_data.get("score", 0.5))
            except Exception:
                score = 0.5
            try:
                conf = float(dim_data.get("confidence", 0.5))
            except Exception:
                conf = 0.5
            score = float(np.clip(score, 0.0, 1.0))
            conf = float(np.clip(conf, 0.0, 1.0))
            just = dim_data.get("justification", "No justification provided")
            if not isinstance(just, str):
                just = str(just)
            result["dimensions"][dim] = {"score": score, "justification": just, "confidence": conf}
        if len(result["dimensions"]) == 0:
            if self.allow_mock:
                return self._create_mock_result(plan_id, reason="no dimensions")
            raise ValueError(f"Parsed JSON but found no usable dimensions block for plan {plan_id}")
        return result

    def _create_mock_result(self, plan_id: int, reason: str = "api error", error: Optional[str] = None) -> Dict[str, Any]:
        dims = {d: {"score": 0.5, "justification": f"Mock result - {reason}", "confidence": 0.3} for d in SELECTED_DIMENSIONS}
        return {
            "plan_id": plan_id,
            "is_mock": True,
            "mock_reason": reason,
            "mock_error": error,
            "dimensions": dims,
            "overall_assessment": {
                "strengths": [reason],
                "weaknesses": ["Using mock data"],
                "improvements": ["Check API connection / JSON parsing"],
                "insights": ["N/A"],
            },
        }

    def _aggregate_with_uncertainty(self, individual_results: List[Dict[str, Any]], plan_id: int, n_runs: int) -> Dict[str, Any]:
        aggregated = {
            "plan_id": plan_id,
            "n_runs": n_runs,
            "provider": self.llm.name,
            "dimensions": {},
        }
        for dim in SELECTED_DIMENSIONS:
            scores = np.array([r["dimensions"][dim]["score"] for r in individual_results if dim in r.get("dimensions", {})], dtype=float)
            confs = np.array([r["dimensions"][dim]["confidence"] for r in individual_results if dim in r.get("dimensions", {})], dtype=float)
            justs = [r["dimensions"][dim]["justification"] for r in individual_results if dim in r.get("dimensions", {})]
            if scores.size == 0:
                continue
            mean_score = float(np.mean(scores))
            std_score = float(np.std(scores))
            se = std_score / np.sqrt(max(1, len(scores)))
            ci_95 = [float(np.clip(mean_score - 1.96 * se, 0.0, 1.0)), float(np.clip(mean_score + 1.96 * se, 0.0, 1.0))]
            aggregated["dimensions"][dim] = {
                "scores": [float(x) for x in scores],
                "mean": mean_score,
                "std": std_score,
                "min": float(np.min(scores)),
                "max": float(np.max(scores)),
                "ci_95": ci_95,
                "justifications": justs,
                "confidences": [float(x) for x in confs],
                "confidence_mean": float(np.mean(confs)) if confs.size else 0.0,
            }
        all_stds = [aggregated["dimensions"][d]["std"] for d in aggregated["dimensions"]]
        aggregated["consistency"] = {
            "score_variance_avg": float(np.mean(all_stds)) if all_stds else 0.0,
            "stable": (float(np.mean(all_stds)) < 0.1) if all_stds else True,
            "note": "Lower variance means more consistent LLM judging across runs; nonzero std reflects uncertainty rather than a deterministic formula.",
        }
        strengths, weaknesses, improvements, insights = [], [], [], []
        for r in individual_results:
            oa = r.get("overall_assessment", {})
            strengths.extend(oa.get("strengths", []))
            weaknesses.extend(oa.get("weaknesses", []))
            improvements.extend(oa.get("improvements", []))
            insights.extend(oa.get("insights", []))
        aggregated["overall_analysis"] = {
            "strengths": strengths[:4],
            "weaknesses": weaknesses[:4],
            "improvements": improvements[:4],
            "insights": insights[:4],
        }
        return aggregated

    def evaluate_plan_with_uncertainty(self, plan: np.ndarray, plan_id: int, n_runs: int = 5, verbose: bool = False) -> Dict[str, Any]:
        if verbose:
            print(f"  Running {n_runs} LLM-judge evaluations for sampled plan {plan_id}...")
        characteristics = self._extract_plan_characteristics(plan)
        individual_results: List[Dict[str, Any]] = []
        failed_runs = 0
        for run in range(n_runs):
            if verbose:
                print(f"    Run {run+1}/{n_runs}...", end="", flush=True)
            prompt = self._create_evaluation_prompt(characteristics, plan_id, run + 1)
            try:
                response = self.llm.call_api(prompt)
                result = self._parse_llm_response(response, plan_id)
                result["pattern_observations"] = characteristics.get("pattern_observations", [])
                individual_results.append(result)
                if verbose:
                    print(" ✓")
            except Exception as e:
                failed_runs += 1
                if verbose:
                    print(f" ✗ ({e})")
                if self.allow_mock:
                    individual_results.append(self._create_mock_result(plan_id, reason="run failure", error=str(e)))
                elif failed_runs > n_runs // 2:
                    raise RuntimeError(f"Too many failed runs for plan {plan_id}: {failed_runs}/{n_runs}")
        if len(individual_results) == 0:
            raise RuntimeError(f"No successful runs for plan {plan_id}")
        out = self._aggregate_with_uncertainty(individual_results, plan_id, n_runs)
        out["pattern_observations"] = characteristics.get("pattern_observations", [])
        out["evidence_by_dimension"] = characteristics.get("evidence_by_dimension", {})
        return out


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_plans(generated_dir: str, n_sample: int, seed: int = 42) -> Tuple[List[np.ndarray], List[int]]:
    base = Path(generated_dir)
    if not base.exists() or not base.is_dir():
        raise ValueError(f"Directory does not exist: {generated_dir}")
    files = sorted(list(base.rglob("*.npy")) + list(base.rglob("*.npz")))
    if not files:
        raise ValueError(f"No .npy or .npz files found under: {generated_dir}")
    rng = random.Random(seed)
    rng.shuffle(files)
    plans: List[np.ndarray] = []
    plan_ids: List[int] = []

    def _infer_base_id(path: Path, fallback: int) -> int:
        m = re.findall(r"\d+", path.stem)
        return int(m[-1]) if m else fallback

    def _load_array(path: Path) -> Optional[np.ndarray]:
        try:
            if path.suffix == ".npz":
                npz = np.load(path, allow_pickle=True)
                if "arr_0" in npz:
                    return npz["arr_0"]
                if len(npz.files) == 1:
                    return npz[npz.files[0]]
                return npz[npz.files[0]]
            return np.load(path, allow_pickle=True)
        except Exception:
            return None

    for file_idx, fp in enumerate(files):
        arr = _load_array(fp)
        if arr is None or not hasattr(arr, "ndim"):
            continue
        base_id = _infer_base_id(fp, file_idx)
        if arr.ndim == 3:
            if arr.shape[0] == 20:
                plans.append(np.nan_to_num(arr, nan=0.0).astype(float))
                plan_ids.append(base_id)
        elif arr.ndim == 4:
            # expect (N,C,H,W) or (N,H,W,C)
            if arr.shape[1] == 20:
                batch = arr
            elif arr.shape[-1] == 20:
                batch = np.transpose(arr, (0, 3, 1, 2))
            else:
                continue
            idxs = list(range(batch.shape[0]))
            rng.shuffle(idxs)
            for bi in idxs:
                plans.append(np.nan_to_num(batch[bi], nan=0.0).astype(float))
                plan_ids.append(base_id * 10000 + int(bi))
                if len(plans) >= n_sample:
                    return plans[:n_sample], plan_ids[:n_sample]
        if len(plans) >= n_sample:
            break
    if len(plans) == 0:
        raise ValueError("Could not load any valid plans")
    return plans[:n_sample], plan_ids[:n_sample]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="LLM judge for six urban-planning dimensions with uncertainty")
    parser.add_argument("--provider", type=str, default="gemini", choices=["gemini", "claude", "gpt", "openai", "anthropic"], help="LLM provider")
    parser.add_argument("--model", type=str, default=None, help="Provider-specific model name. Examples: gpt-4o, gpt-4.1-mini, gemini-2.5-flash, gemini-2.5-pro, claude-sonnet-4-20250514")
    parser.add_argument("--generated_dir", type=str, required=True, help="Directory containing generated .npy/.npz plans")
    parser.add_argument("--n_sample", type=int, default=5, help="Number of sampled plans to evaluate")
    parser.add_argument("--n_runs", type=int, default=5, help="Number of LLM judge runs per sampled plan")
    parser.add_argument("--temperature", type=float, default=0.7, help="LLM sampling temperature")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed")
    parser.add_argument("--output", type=str, default="robust_llm_selected6_uncertainty.json", help="Output JSON")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("--allow_mock", action="store_true", help="Allow mock results on parsing/API failure")
    args = parser.parse_args()

    provider_norm = args.provider.lower()
    if provider_norm == "openai":
        provider_norm = "gpt"
    if provider_norm == "anthropic":
        provider_norm = "claude"

    if provider_norm == "gemini" and not os.environ.get("GOOGLE_API_KEY"):
        print("ERROR: GOOGLE_API_KEY not set")
        sys.exit(1)
    if provider_norm == "claude" and not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set")
        sys.exit(1)
    if provider_norm == "gpt" and not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)

    plans, plan_ids = sample_plans(args.generated_dir, args.n_sample, args.seed)
    evaluator = RobustLLMEvaluator(provider=provider_norm, model=args.model, temperature=args.temperature, allow_mock=args.allow_mock)

    if args.verbose:
        print(f"Using provider={evaluator.llm.name} model={evaluator.llm.model}")

    all_results = []
    for i, (plan, pid) in enumerate(zip(plans, plan_ids), 1):
        print(f"\nPlan {i}/{len(plans)} (ID: {pid}, shape={plan.shape})")
        res = evaluator.evaluate_plan_with_uncertainty(plan, pid, n_runs=args.n_runs, verbose=args.verbose)
        all_results.append(res)
        if args.verbose:
            for dim in SELECTED_DIMENSIONS:
                if dim in res["dimensions"]:
                    d = res["dimensions"][dim]
                    print(f"  {dim:26s} mean={d['mean']:.3f} std={d['std']:.3f} ci95=[{d['ci_95'][0]:.3f},{d['ci_95'][1]:.3f}]")

    out = {
        "provider": evaluator.llm.name,
        "model": evaluator.llm.model,
        "n_sample": len(all_results),
        "n_runs": args.n_runs,
        "temperature": args.temperature,
        "selected_dimensions": SELECTED_DIMENSIONS,
        "results": all_results,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()

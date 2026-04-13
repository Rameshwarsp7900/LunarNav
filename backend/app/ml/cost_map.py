"""
app/ml/cost_map.py
==================
Cost-map generation from a terrain elevation patch.

Production path:  U-Net model (PyTorch) if checkpoint found.
Fallback path:    Physics-inspired heuristic (slope + roughness).
Crater overlay:   Robbins crater CSV burned directly into cost map.

Usage
-----
>>> from app.terrain.patch_extractor import extract_patch
>>> patch = extract_patch(...)
>>> cost_map = generate_cost_map(patch, crater_csv="data/lunar_crater_database_robbins_2018.csv")
>>> cost_map.shape   # same as patch.data.shape, dtype float32, values in [0,1]
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.ndimage import uniform_filter, gaussian_filter

from app.terrain.patch_extractor import TerrainPatch

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    logger.warning("PyTorch not available — cost map will use heuristic mode.")

try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False


# ── U-Net architecture ────────────────────────────────────────────────────────

class _DoubleConv(nn.Module):  # type: ignore[misc]
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class _UNet(nn.Module):  # type: ignore[misc]
    """
    Lightweight U-Net for traversal-cost regression.
    Input  : (B, 1, H, W) float32 normalised elevation patch.
    Output : (B, 1, H, W) float32 cost in [0, 1].
    """

    def __init__(self):
        super().__init__()
        self.enc1 = _DoubleConv(1, 16)
        self.enc2 = _DoubleConv(16, 32)
        self.pool  = nn.MaxPool2d(2)
        self.bot   = _DoubleConv(32, 64)
        self.up1   = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1  = _DoubleConv(64, 32)
        self.up2   = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec2  = _DoubleConv(32, 16)
        self.head  = nn.Sequential(nn.Conv2d(16, 1, 1), nn.Sigmoid())

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        b  = self.bot(self.pool(e2))
        d1 = self.dec1(torch.cat([self.up1(b), e2], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d1), e1], dim=1))
        return self.head(d2)


_model_cache: dict[str, "_UNet"] = {}


def _load_model(checkpoint: str) -> "_UNet":
    if checkpoint in _model_cache:
        return _model_cache[checkpoint]
    model = _UNet()
    ckpt_path = Path(checkpoint)
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state)
        logger.info("Loaded U-Net checkpoint: %s", checkpoint)
    else:
        logger.warning("Checkpoint %s not found — using random weights (dev mode).", checkpoint)
    model.eval()
    _model_cache[checkpoint] = model
    return model


# ── Crater overlay ────────────────────────────────────────────────────────────

def _burn_craters_into_cost(
    cost_map: np.ndarray,
    patch: TerrainPatch,
    crater_csv: str,
    cost_weight: float = 0.8,
    buffer_px: int = 2,
) -> np.ndarray:
    """
    Overlay Robbins crater database onto the cost map.

    Craters within the patch bounding box are rasterized as filled discs.
    The cost inside each crater disc is pushed to >= cost_weight.

    Parameters
    ----------
    cost_map    : (H, W) float32 cost array in [0, 1].
    patch       : TerrainPatch for coordinate reference.
    crater_csv  : Path to Robbins 2018 CSV.
    cost_weight : Minimum cost assigned to crater cells.
    buffer_px   : Extra pixel radius beyond the crater edge.

    Returns
    -------
    Updated cost_map with craters burned in.
    """
    if not _PANDAS_AVAILABLE:
        logger.warning("pandas not available — skipping crater overlay.")
        return cost_map

    csv_path = Path(crater_csv)
    if not csv_path.exists():
        logger.warning("Crater CSV not found: %s — skipping overlay.", crater_csv)
        return cost_map

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        logger.warning("Failed to read crater CSV: %s", exc)
        return cost_map

    # Detect column names (Robbins CSV uses LAT_CIRC_IMG, LON_CIRC_IMG, DIAM_CIRC_IMG)
    lat_col  = next((c for c in df.columns if "LAT" in c.upper()), None)
    lon_col  = next((c for c in df.columns if "LON" in c.upper()), None)
    diam_col = next((c for c in df.columns if "DIAM" in c.upper()), None)

    if not all([lat_col, lon_col, diam_col]):
        logger.warning("Crater CSV columns not detected — skipping overlay.")
        return cost_map

    # Patch geographic bounds
    rows, cols_count = cost_map.shape
    transform = patch.transform
    res_deg = patch.resolution_deg

    # Compute patch bounds
    origin_lon = transform.c
    origin_lat = transform.f
    end_lon = origin_lon + res_deg * cols_count
    end_lat = origin_lat - res_deg * rows   # latitude decreases downward

    lat_min = min(origin_lat, end_lat)
    lat_max = max(origin_lat, end_lat)
    lon_min = min(origin_lon, end_lon)
    lon_max = max(origin_lon, end_lon)

    # Filter to craters intersecting the patch
    mask = (
        (df[lat_col] >= lat_min - 1) & (df[lat_col] <= lat_max + 1) &
        (df[lon_col] >= lon_min - 1) & (df[lon_col] <= lon_max + 1)
    )
    local = df[mask].copy()

    if local.empty:
        return cost_map

    from app.terrain.patch_extractor import latlon_to_pixel

    for _, row in local.iterrows():
        lat  = float(row[lat_col])
        lon  = float(row[lon_col])
        diam = float(row[diam_col])   # km
        if np.isnan(lat) or np.isnan(lon) or np.isnan(diam):
            continue

        # Radius in pixels (1 degree ≈ 512 pixels at 512ppd)
        radius_deg = (diam / 2.0) / 111.1  # km → degrees (approx)
        radius_px  = int(radius_deg / res_deg) + buffer_px

        if radius_px < 1:
            continue

        pr, pc = latlon_to_pixel(patch, lat, lon)

        # Draw filled disc
        r_min = max(0, pr - radius_px)
        r_max = min(rows - 1, pr + radius_px)
        c_min = max(0, pc - radius_px)
        c_max = min(cols_count - 1, pc + radius_px)

        for ri in range(r_min, r_max + 1):
            for ci in range(c_min, c_max + 1):
                if (ri - pr)**2 + (ci - pc)**2 <= radius_px**2:
                    cost_map[ri, ci] = max(cost_map[ri, ci], cost_weight)

    logger.info("Burned %d craters into cost map.", len(local))
    return cost_map


# ── Public API ────────────────────────────────────────────────────────────────

def generate_cost_map(
    patch_data: np.ndarray,
    patch: Optional[TerrainPatch] = None,
    crater_csv: Optional[str] = None,
    checkpoint: str = "checkpoints/unet_cost.pt",
    use_heuristic: Optional[bool] = None,
) -> np.ndarray:
    """
    Generate a normalised traversal cost map from an elevation patch.

    Parameters
    ----------
    patch_data    : 2-D float32 NumPy array (elevation values).
    patch         : TerrainPatch (needed for crater overlay).
    crater_csv    : Path to Robbins crater CSV for overlay (optional).
    checkpoint    : U-Net checkpoint path.
    use_heuristic : Force heuristic mode if True; auto-detect if None.

    Returns
    -------
    cost_map : np.ndarray, same shape as patch_data, float32, values in [0,1].
        0 = easy terrain  |  values near 1 = impassable / high risk
    """
    from app.config import get_settings
    settings = get_settings()

    if use_heuristic is None:
        use_heuristic = settings.USE_HEURISTIC_COST or not _TORCH_AVAILABLE

    if use_heuristic:
        cost_map = _heuristic_cost_map(patch_data)
    else:
        cost_map = _unet_cost_map(patch_data, checkpoint)

    # Optionally burn crater database into cost map
    if patch is not None and crater_csv:
        cost_map = _burn_craters_into_cost(
            cost_map, patch, crater_csv,
            cost_weight=settings.CRATER_COST_WEIGHT,
            buffer_px=settings.CRATER_BUFFER_RADIUS_PX,
        )

    return cost_map


# ── Heuristic cost map ────────────────────────────────────────────────────────

def _heuristic_cost_map(patch: np.ndarray) -> np.ndarray:
    """
    Physics-inspired heuristic:
      - Slope (gradient magnitude) = primary hazard proxy.
      - Roughness (local std of elevation) = secondary proxy.
      - NaN cells (crater interiors / no-data) = maximum cost.
    """
    elev = patch.copy().astype(np.float64)

    nan_mask = np.isnan(elev)
    if nan_mask.any():
        from scipy.ndimage import generic_filter
        fill = generic_filter(
            np.where(nan_mask, np.nan, elev),
            lambda v: np.nanmedian(v),
            size=5,
            mode="nearest",
        )
        elev = np.where(nan_mask, fill, elev)

    dy, dx = np.gradient(elev)
    slope = np.sqrt(dx**2 + dy**2)

    roughness = uniform_filter(elev**2, size=5) - uniform_filter(elev, size=5)**2
    roughness = np.clip(roughness, 0, None)

    cost = 0.7 * slope + 0.3 * np.sqrt(roughness)

    if nan_mask.any():
        cost[nan_mask] = cost.max() if cost.max() > 0 else 1.0

    cost = gaussian_filter(cost, sigma=1.5)
    cmin, cmax = cost.min(), cost.max()
    if cmax > cmin:
        cost = (cost - cmin) / (cmax - cmin)
    else:
        cost = np.zeros_like(cost)

    return cost.astype(np.float32)


def _unet_cost_map(patch: np.ndarray, checkpoint: str) -> np.ndarray:
    """Run the U-Net model on a patch and return a cost map."""
    model = _load_model(checkpoint)

    h, w = patch.shape
    elev = patch.copy().astype(np.float32)
    elev = np.nan_to_num(elev, nan=float(np.nanmean(elev)))

    e_min, e_max = elev.min(), elev.max()
    if e_max > e_min:
        elev = (elev - e_min) / (e_max - e_min)

    ph = ((h + 3) // 4) * 4
    pw = ((w + 3) // 4) * 4
    padded = np.zeros((ph, pw), dtype=np.float32)
    padded[:h, :w] = elev

    tensor = torch.from_numpy(padded).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        cost_tensor = model(tensor)

    cost = cost_tensor.squeeze().numpy()[:h, :w]
    return cost.astype(np.float32)

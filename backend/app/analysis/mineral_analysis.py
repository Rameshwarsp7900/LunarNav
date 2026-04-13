"""
app/analysis/mineral_analysis.py
==================================
Compute mineral exposure statistics along the rover path.

Datasets
--------
- hydrogenhd.dat   — Hydrogen abundance grid
- ironhd.dat       — Iron abundance grid
- thoriumhd.dat    — Thorium abundance grid

All three .dat files are assumed to be raw binary float32 arrays in the
same geographic coordinate system as the SLDEM JP2 (60°S–60°N, 0–360°E)
at a resolution defined by the file dimensions.

Usage
-----
>>> report = analyze_minerals(
...     path      = [(10, 20), (11, 21), ...],   # pixel coords in patch
...     patch     = terrain_patch,
...     hydrogen  = "data/hydrogenhd.dat",
...     iron      = "data/ironhd.dat",
...     thorium   = "data/thoriumhd.dat",
... )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from app.terrain.patch_extractor import TerrainPatch, pixel_to_latlon

logger = logging.getLogger(__name__)

Coord = tuple[int, int]

# Expected global grid dimensions for the 512 ppd dataset
_GRID_ROWS = 61_440   # 120° latitude × 512 ppd
_GRID_COLS = 184_320  # 360° longitude × 512 ppd
_LAT_MIN   = -60.0
_LAT_MAX   =  60.0
_LON_MIN   =   0.0
_LON_MAX   = 360.0


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class MineralReport:
    hydrogen_avg: float
    iron_avg: float
    thorium_avg: float
    hydrogen_exposure_pct: float   # % of path waypoints above detection threshold
    iron_exposure_pct: float
    thorium_exposure_pct: float
    n_samples: int


# ── Mineral grid loader ───────────────────────────────────────────────────────

def _load_mineral_grid(
    dat_file: str,
    rows: int = _GRID_ROWS,
    cols: int = _GRID_COLS,
) -> np.ndarray:
    """
    Load a raw binary float32 mineral abundance grid.

    Falls back to a zero-filled placeholder if the file is absent (dev mode).
    """
    path = Path(dat_file)
    if not path.exists():
        logger.warning(
            "Mineral file %s not found — using zero placeholder.", dat_file
        )
        return np.zeros((rows, cols), dtype=np.float32)

    data = np.fromfile(str(path), dtype=np.float32)
    expected = rows * cols
    if data.size != expected:
        # Attempt to infer grid dimensions from file size
        actual_rows = data.size // cols if data.size >= cols else 1
        logger.warning(
            "Expected %d elements, got %d — reshaping to (%d, %d).",
            expected, data.size, actual_rows, cols,
        )
        data = data[: actual_rows * cols].reshape(actual_rows, cols)
    else:
        data = data.reshape(rows, cols)

    return data


def _latlon_to_mineral_idx(
    lat: float, lon: float,
    rows: int = _GRID_ROWS,
    cols: int = _GRID_COLS,
) -> tuple[int, int]:
    """Map (lat, lon) → (row, col) in the global mineral grid."""
    # Clamp
    lat = np.clip(lat, _LAT_MIN, _LAT_MAX)
    lon = lon % 360.0   # normalise to [0, 360)

    row_f = (lat - _LAT_MIN) / (_LAT_MAX - _LAT_MIN) * rows
    col_f = (lon - _LON_MIN) / (_LON_MAX - _LON_MIN) * cols

    row = int(np.clip(round(rows - 1 - row_f), 0, rows - 1))   # flip: lat increases upward
    col = int(np.clip(round(col_f), 0, cols - 1))
    return row, col


# ── Public API ────────────────────────────────────────────────────────────────

def analyze_minerals(
    path: list[Coord],
    patch: TerrainPatch,
    hydrogen: str,
    iron: str,
    thorium: str,
    h_threshold: float = 50.0,    # ppm detection threshold
    fe_threshold: float = 5.0,    # wt% detection threshold
    th_threshold: float = 1.0,    # ppm detection threshold
) -> MineralReport:
    """
    Compute mineral exposure statistics along the rover path.

    Parameters
    ----------
    path      : List of (row, col) pixel waypoints (patch coordinates).
    patch     : TerrainPatch — provides pixel→lat/lon back-conversion.
    hydrogen  : Path to hydrogen abundance .dat file.
    iron      : Path to iron abundance .dat file.
    thorium   : Path to thorium abundance .dat file.
    h_threshold  : Hydrogen detection threshold (ppm).
    fe_threshold : Iron detection threshold (wt%).
    th_threshold : Thorium detection threshold (ppm).

    Returns
    -------
    MineralReport with average values and exposure percentages.
    """
    # Lazy-load grids (cached by module-level dict for the process lifetime)
    h_grid  = _get_or_load("hydrogen", hydrogen)
    fe_grid = _get_or_load("iron", iron)
    th_grid = _get_or_load("thorium", thorium)

    h_vals:  list[float] = []
    fe_vals: list[float] = []
    th_vals: list[float] = []

    for r, c in path:
        lat, lon = pixel_to_latlon(patch, r, c)
        gi, gj = _latlon_to_mineral_idx(lat, lon, h_grid.shape[0], h_grid.shape[1])

        h_vals.append(float(h_grid[gi, gj]))
        fe_vals.append(float(fe_grid[gi, gj]))
        th_vals.append(float(th_grid[gi, gj]))

    n = len(path)
    if n == 0:
        return MineralReport(0, 0, 0, 0, 0, 0, 0)

    h_arr  = np.array(h_vals,  dtype=np.float32)
    fe_arr = np.array(fe_vals, dtype=np.float32)
    th_arr = np.array(th_vals, dtype=np.float32)

    return MineralReport(
        hydrogen_avg         = round(float(np.nanmean(h_arr)),  4),
        iron_avg             = round(float(np.nanmean(fe_arr)), 4),
        thorium_avg          = round(float(np.nanmean(th_arr)), 4),
        hydrogen_exposure_pct= round(float(np.mean(h_arr  >= h_threshold))  * 100, 2),
        iron_exposure_pct    = round(float(np.mean(fe_arr >= fe_threshold)) * 100, 2),
        thorium_exposure_pct = round(float(np.mean(th_arr >= th_threshold)) * 100, 2),
        n_samples=n,
    )


# ── Internal cache ────────────────────────────────────────────────────────────

_grid_cache: dict[str, np.ndarray] = {}


def _get_or_load(key: str, path: str) -> np.ndarray:
    if key not in _grid_cache:
        _grid_cache[key] = _load_mineral_grid(path)
    return _grid_cache[key]

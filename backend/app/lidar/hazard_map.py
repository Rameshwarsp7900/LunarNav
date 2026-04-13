"""
app/lidar/hazard_map.py
=======================
Convert raw LiDAR obstacle JSON into a binary hazard grid.

Each obstacle is represented as a filled circle on the grid, inflated by
a configurable safety radius so the rover maintains a buffer distance.

Usage
-----
>>> lidar_data = {
...     "obstacles": [
...         {"x": 120, "y": 340, "radius": 5},
...         {"x": 200, "y": 100, "radius": 3},
...     ]
... }
>>> hazard = generate_local_hazard_map(lidar_data, patch_shape=(512, 512))
>>> hazard.shape    # (512, 512)
>>> hazard.dtype    # bool
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure

from app.config import get_settings

settings = get_settings()


# ── Public API ────────────────────────────────────────────────────────────────

def generate_local_hazard_map(
    lidar_data: dict[str, Any],
    patch_shape: tuple[int, int],
    inflation_radius: int | None = None,
) -> np.ndarray:
    """
    Build a binary hazard grid from LiDAR obstacle data.

    Parameters
    ----------
    lidar_data : dict with key ``"obstacles"``, each entry having
                 ``x`` (col), ``y`` (row), and ``radius`` (pixels).
    patch_shape : (rows, cols) matching the terrain patch dimensions.
    inflation_radius : Extra pixel margin beyond the reported obstacle
                       radius.  Defaults to settings.HAZARD_INFLATION_RADIUS.

    Returns
    -------
    hazard_map : np.ndarray, dtype bool, shape == patch_shape.
        True  = blocked / unsafe
        False = traversable

    Example input
    -------------
    {
        "obstacles": [
            {"x": 120, "y": 340, "radius": 5},
            {"x": 200, "y": 100, "radius": 3}
        ]
    }
    """
    if inflation_radius is None:
        inflation_radius = settings.HAZARD_INFLATION_RADIUS

    rows, cols = patch_shape
    hazard = np.zeros((rows, cols), dtype=bool)

    obstacles = lidar_data.get("obstacles", [])
    if not obstacles:
        return hazard

    for obs in obstacles:
        x   = int(obs["x"])        # column
        y   = int(obs["y"])        # row
        r   = int(obs.get("radius", 1))
        eff = r + inflation_radius  # inflated radius

        _draw_circle(hazard, center_row=y, center_col=x, radius=eff)

    return hazard


def obstacles_to_blocked_cells(
    lidar_data: dict[str, Any],
    patch_shape: tuple[int, int],
    inflation_radius: int | None = None,
) -> list[tuple[int, int]]:
    """
    Return a flat list of (row, col) cells that are blocked by obstacles.
    Useful for passing directly into D* Lite's update_obstacles().
    """
    hazard = generate_local_hazard_map(lidar_data, patch_shape, inflation_radius)
    rows, cols = np.where(hazard)
    return list(zip(rows.tolist(), cols.tolist()))


# ── Internal helpers ──────────────────────────────────────────────────────────

def _draw_circle(
    grid: np.ndarray,
    center_row: int,
    center_col: int,
    radius: int,
) -> None:
    """
    Fill a disc of given radius centred at (center_row, center_col)
    in the boolean grid (in-place).
    """
    h, w = grid.shape
    r_min = max(0, center_row - radius)
    r_max = min(h - 1, center_row + radius)
    c_min = max(0, center_col - radius)
    c_max = min(w - 1, center_col + radius)

    for row in range(r_min, r_max + 1):
        for col in range(c_min, c_max + 1):
            if (row - center_row) ** 2 + (col - center_col) ** 2 <= radius ** 2:
                grid[row, col] = True


def dilate_hazard_map(
    hazard_map: np.ndarray, extra_radius: int = 2
) -> np.ndarray:
    """
    Apply binary dilation to expand existing hazards by *extra_radius* pixels.
    Useful for adding a global safety margin after initial obstacle marking.
    """
    if extra_radius <= 0:
        return hazard_map.copy()
    struct = generate_binary_structure(2, 2)
    dilated = binary_dilation(hazard_map, structure=struct, iterations=extra_radius)
    return dilated.astype(bool)

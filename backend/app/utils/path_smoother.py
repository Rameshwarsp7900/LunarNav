"""
app/utils/path_smoother.py
===========================
Smooth a piecewise-linear rover path using Savitzky-Golay filtering.

Smoothed paths reduce wear on rover actuators and produce more
realistic traversal simulations.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    from scipy.signal import savgol_filter
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

Coord = tuple[int, int]


def smooth_path(
    path: list[Coord],
    window: int = 5,
    polyorder: int = 2,
) -> list[Coord]:
    """
    Smooth a list of (row, col) waypoints using Savitzky-Golay filtering.

    Parameters
    ----------
    path      : List of (row, col) waypoints.
    window    : Window size for S-G filter (must be odd, >= polyorder + 2).
    polyorder : Polynomial order for S-G filter.

    Returns
    -------
    Smoothed list of (row, col) waypoints, same length as input.
    """
    if len(path) < 4 or not _SCIPY_AVAILABLE:
        return path

    rows = np.array([p[0] for p in path], dtype=float)
    cols = np.array([p[1] for p in path], dtype=float)

    # Ensure window is valid
    win = min(window, len(path))
    if win % 2 == 0:
        win -= 1
    if win < polyorder + 2:
        return path

    smooth_rows = savgol_filter(rows, win, polyorder)
    smooth_cols = savgol_filter(cols, win, polyorder)

    # Force endpoints to match
    smooth_rows[0],  smooth_rows[-1]  = rows[0],  rows[-1]
    smooth_cols[0],  smooth_cols[-1]  = cols[0],  cols[-1]

    return [(int(round(r)), int(round(c)))
            for r, c in zip(smooth_rows, smooth_cols)]


def downsample_path(
    path: list[Coord], target_points: int = 200
) -> list[Coord]:
    """
    Reduce path density while preserving start/end and key turning points.

    Parameters
    ----------
    path          : Input path.
    target_points : Maximum number of waypoints to keep.

    Returns
    -------
    Downsampled path.
    """
    if len(path) <= target_points:
        return path

    indices = np.round(np.linspace(0, len(path) - 1, target_points)).astype(int)
    return [path[i] for i in indices]


def compute_path_length_m(path: list[Coord], pixel_size_m: float) -> float:
    """Return total Euclidean path length in metres."""
    if len(path) < 2:
        return 0.0
    total = 0.0
    for i in range(len(path) - 1):
        dr = path[i+1][0] - path[i][0]
        dc = path[i+1][1] - path[i][1]
        total += (dr**2 + dc**2) ** 0.5
    return total * pixel_size_m

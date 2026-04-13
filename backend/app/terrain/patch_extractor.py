"""
app/terrain/patch_extractor.py
===============================
Extract a rectangular elevation patch from the SLDEM2015 JP2 raster.

The returned patch is a 2-D NumPy array (float32) containing elevation
values in metres. The associated Rasterio affine transform is returned so
that downstream modules can convert array indices back to geo coordinates.

Usage
-----
>>> patch, transform = extract_patch(
...     jp2_file  = "data/SLDEM2015_512_60S_60N_000_360.JP2",
...     start_lat = -12.5, start_lon = 45.3,
...     goal_lat  = -12.1, goal_lon  = 45.8,
... )
>>> patch.shape   # (rows, cols)
>>> type(transform)  # affine.Affine
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import rasterio
from affine import Affine
from rasterio.windows import from_bounds

# ── Return type ───────────────────────────────────────────────────────────────

@dataclass
class TerrainPatch:
    """Holds the extracted elevation array and its geospatial metadata."""

    data: np.ndarray          # shape (rows, cols), dtype float32
    transform: Affine         # affine transform of this sub-window
    nodata: float | None      # nodata sentinel (may be None)
    origin_lat: float         # top-left latitude
    origin_lon: float         # top-left longitude
    resolution_deg: float     # approximate pixel size in degrees

    # convenience
    @property
    def shape(self) -> Tuple[int, int]:
        return self.data.shape  # type: ignore[return-value]


# ── Core function ─────────────────────────────────────────────────────────────

def extract_patch(
    jp2_file: str,
    start_lat: float,
    start_lon: float,
    goal_lat: float,
    goal_lon: float,
    padding_deg: float = 0.05,
) -> TerrainPatch:
    """
    Extract a bounding-box region from a JP2 elevation raster.

    Parameters
    ----------
    jp2_file     : Path to SLDEM2015 .JP2 file.
    start_lat    : Mission start latitude.
    start_lon    : Mission start longitude.
    goal_lat     : Mission goal latitude.
    goal_lon     : Mission goal longitude.
    padding_deg  : Extra geographic margin around the bounding box (degrees).

    Returns
    -------
    TerrainPatch
        .data      — NumPy float32 array, shape (rows, cols).
        .transform — Affine transform for the extracted window.
        .nodata    — NoData value from the source dataset (or None).

    Raises
    ------
    FileNotFoundError  : jp2_file does not exist.
    ValueError         : Requested bounds fall entirely outside the raster.
    """
    jp2_path = Path(jp2_file)
    if not jp2_path.exists():
        raise FileNotFoundError(f"JP2 file not found: {jp2_file}")

    # ── Build bounding box (with padding) ─────────────────────────────────────
    min_lat = min(start_lat, goal_lat) - padding_deg
    max_lat = max(start_lat, goal_lat) + padding_deg
    min_lon = min(start_lon, goal_lon) - padding_deg
    max_lon = max(start_lon, goal_lon) + padding_deg

    with rasterio.open(jp2_path) as ds:
        _validate_bounds(ds, min_lat, max_lat, min_lon, max_lon)

        # Rasterio window from geographic bounds
        window = from_bounds(
            left   = min_lon,
            bottom = min_lat,
            right  = max_lon,
            top    = max_lat,
            transform = ds.transform,
        )
        # Clamp to valid raster extent
        window = window.intersection(
            rasterio.windows.Window(0, 0, ds.width, ds.height)
        )

        patch_data: np.ndarray = ds.read(1, window=window).astype(np.float32)
        window_transform: Affine = ds.window_transform(window)
        nodata = ds.nodata

    # Replace nodata with NaN for downstream numerics
    if nodata is not None:
        patch_data[patch_data == nodata] = np.nan

    # Resolution from the window transform (absolute value of pixel size)
    res_deg = abs(window_transform.a)

    return TerrainPatch(
        data          = patch_data,
        transform     = window_transform,
        nodata        = nodata,
        origin_lat    = window_transform.f,
        origin_lon    = window_transform.c,
        resolution_deg= res_deg,
    )


# ── Coordinate helpers ────────────────────────────────────────────────────────

def latlon_to_pixel(
    patch: TerrainPatch, lat: float, lon: float
) -> Tuple[int, int]:
    """
    Convert a (lat, lon) pair to the nearest (row, col) index within a patch.

    Parameters
    ----------
    patch : TerrainPatch  returned by extract_patch.
    lat   : latitude  in decimal degrees.
    lon   : longitude in decimal degrees.

    Returns
    -------
    (row, col) : tuple[int, int]   0-based indices into patch.data.
    """
    inv = ~patch.transform
    col_f, row_f = inv * (lon, lat)
    row = int(np.clip(round(row_f), 0, patch.data.shape[0] - 1))
    col = int(np.clip(round(col_f), 0, patch.data.shape[1] - 1))
    return row, col


def pixel_to_latlon(
    patch: TerrainPatch, row: int, col: int
) -> Tuple[float, float]:
    """
    Convert a (row, col) index in the patch back to (lat, lon).

    Returns
    -------
    (lat, lon) : tuple[float, float]
    """
    geo_x, geo_y = patch.transform * (col + 0.5, row + 0.5)
    return geo_y, geo_x   # lat, lon


# ── Private helpers ───────────────────────────────────────────────────────────

def _validate_bounds(
    ds: rasterio.DatasetReader,
    min_lat: float, max_lat: float,
    min_lon: float, max_lon: float,
) -> None:
    raster_bounds = ds.bounds  # BoundingBox(left, bottom, right, top)
    if (
        max_lon < raster_bounds.left
        or min_lon > raster_bounds.right
        or max_lat < raster_bounds.bottom
        or min_lat > raster_bounds.top
    ):
        raise ValueError(
            f"Requested bounds (lat [{min_lat},{max_lat}], "
            f"lon [{min_lon},{max_lon}]) are entirely outside "
            f"the raster extent {raster_bounds}."
        )

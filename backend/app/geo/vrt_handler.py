"""
app/geo/vrt_handler.py
======================
Handles GDAL Virtual Raster (.vrt) files for pixel ↔ geo coordinate conversion.

Features
--------
- Lazy-loaded, LRU-cached transform objects (one per VRT path).
- Forward conversion : pixel (col, row) → (lat, lon).
- Inverse conversion : (lat, lon) → pixel (col, row).
- Pixel-bounds validation.
- Thread-safe (Rasterio datasets are opened read-only).

Usage
-----
>>> handler = VRTHandler("data/lunar_map.vrt")
>>> lat, lon = handler.pixel_to_geo(120, 340)
>>> col, row = handler.geo_to_pixel(lat, lon)
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Tuple

import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS

# ── Public type aliases ───────────────────────────────────────────────────────
GeoCoord  = Tuple[float, float]   # (latitude, longitude)
PixelCoord = Tuple[int, int]       # (col/x, row/y)


# ── Internal cache entry ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class _VRTMeta:
    """Immutable metadata extracted from a VRT dataset."""
    transform: Affine
    inv_transform: Affine
    crs: CRS
    width: int
    height: int
    origin_x: float   # top-left corner X (longitude or easting)
    origin_y: float   # top-left corner Y (latitude or northing)


# ── Module-level cache ────────────────────────────────────────────────────────

_cache_lock = threading.Lock()
_vrt_cache: dict[str, _VRTMeta] = {}


def _load_vrt_meta(vrt_path: str) -> _VRTMeta:
    """
    Open a VRT file with Rasterio and extract the affine transform + CRS.
    Results are cached by canonical path so repeated calls are free.
    """
    canonical = str(Path(vrt_path).resolve())

    with _cache_lock:
        if canonical in _vrt_cache:
            return _vrt_cache[canonical]

        if not Path(canonical).exists():
            raise FileNotFoundError(f"VRT file not found: {canonical}")

        with rasterio.open(canonical) as ds:
            transform: Affine = ds.transform
            crs: CRS = ds.crs
            width: int = ds.width
            height: int = ds.height

        meta = _VRTMeta(
            transform=transform,
            inv_transform=~transform,
            crs=crs,
            width=width,
            height=height,
            origin_x=transform.c,
            origin_y=transform.f,
        )
        _vrt_cache[canonical] = meta
        return meta


def clear_vrt_cache() -> None:
    """Evict all cached VRT transforms (useful in tests)."""
    with _cache_lock:
        _vrt_cache.clear()


# ── Main class ────────────────────────────────────────────────────────────────

class VRTHandler:
    """
    Coordinate conversion helper for a single VRT file.

    Parameters
    ----------
    vrt_file : str
        Path to the GDAL Virtual Raster (.vrt) file.

    Examples
    --------
    >>> h = VRTHandler("data/lunar_map.vrt")
    >>> lat, lon = h.pixel_to_geo(x=512, y=256)
    >>> col, row = h.geo_to_pixel(lat=-12.5, lon=45.3)
    """

    def __init__(self, vrt_file: str) -> None:
        self.vrt_file = vrt_file
        self._meta: _VRTMeta = _load_vrt_meta(vrt_file)

    # ── Public API ────────────────────────────────────────────────────────────

    def pixel_to_geo(self, x: int, y: int) -> GeoCoord:
        """
        Convert pixel coordinates to geographic coordinates.

        Parameters
        ----------
        x : int  Column index (0-based, left → right).
        y : int  Row index (0-based, top → bottom).

        Returns
        -------
        (lat, lon) : tuple[float, float]
        """
        self._validate_pixel(x, y)
        # Rasterio convention: (col + 0.5, row + 0.5) → centre of pixel
        geo_x, geo_y = self._meta.transform * (x + 0.5, y + 0.5)
        lat, lon = self._xy_to_latlon(geo_x, geo_y)
        return lat, lon

    def geo_to_pixel(self, lat: float, lon: float) -> PixelCoord:
        """
        Convert geographic coordinates to the nearest pixel.

        Parameters
        ----------
        lat : float  Latitude in decimal degrees.
        lon : float  Longitude in decimal degrees.

        Returns
        -------
        (col, row) : tuple[int, int]
        """
        geo_x, geo_y = self._latlon_to_xy(lat, lon)
        col_f, row_f = self._meta.inv_transform * (geo_x, geo_y)
        col, row = int(col_f), int(row_f)
        self._validate_pixel(col, row)
        return col, row

    def pixel_batch_to_geo(
        self, pixels: list[PixelCoord]
    ) -> list[GeoCoord]:
        """Vectorised conversion for a list of pixel coordinates."""
        xs = np.array([p[0] for p in pixels], dtype=np.float64) + 0.5
        ys = np.array([p[1] for p in pixels], dtype=np.float64) + 0.5
        t = self._meta.transform
        geo_xs = t.a * xs + t.b * ys + t.c
        geo_ys = t.d * xs + t.e * ys + t.f
        return [
            self._xy_to_latlon(gx, gy) for gx, gy in zip(geo_xs, geo_ys)
        ]

    @property
    def bounds(self) -> dict[str, int]:
        """Return pixel bounds of the raster."""
        return {"width": self._meta.width, "height": self._meta.height}

    @property
    def crs_wkt(self) -> str:
        """Well-Known Text of the coordinate reference system."""
        return self._meta.crs.to_wkt()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _validate_pixel(self, x: int, y: int) -> None:
        m = self._meta
        if not (0 <= x < m.width and 0 <= y < m.height):
            raise ValueError(
                f"Pixel ({x}, {y}) is out of bounds "
                f"(width={m.width}, height={m.height})."
            )

    @staticmethod
    def _xy_to_latlon(geo_x: float, geo_y: float) -> GeoCoord:
        """
        Map projected X/Y to (lat, lon).
        For geographic CRS the mapping is direct:  X → lon, Y → lat.
        For projected CRS a proper reprojection would be needed; here we
        assume the VRT is in a geographic CRS (EPSG:4326-like) as is standard
        for lunar datasets.
        """
        return geo_y, geo_x   # (lat, lon)

    @staticmethod
    def _latlon_to_xy(lat: float, lon: float) -> tuple[float, float]:
        """Inverse of _xy_to_latlon — returns (geo_x, geo_y)."""
        return lon, lat


# ── Module-level convenience functions (functional API) ───────────────────────

def load_vrt(vrt_file: str) -> VRTHandler:
    """
    Create (or return a cached) VRTHandler for *vrt_file*.

    Example
    -------
    >>> handler = load_vrt("data/lunar_map.vrt")
    """
    return VRTHandler(vrt_file)


def pixel_to_geo(vrt_file: str, x: int, y: int) -> GeoCoord:
    """
    One-shot pixel → geo conversion.

    Example
    -------
    >>> lat, lon = pixel_to_geo("data/lunar_map.vrt", 512, 256)
    """
    return VRTHandler(vrt_file).pixel_to_geo(x, y)


def geo_to_pixel(vrt_file: str, lat: float, lon: float) -> PixelCoord:
    """
    One-shot geo → pixel conversion.

    Example
    -------
    >>> col, row = geo_to_pixel("data/lunar_map.vrt", -12.5, 45.3)
    """
    return VRTHandler(vrt_file).geo_to_pixel(lat, lon)

"""
app/utils/visualization.py
===========================
Generate a PNG image of the planned path overlaid on the terrain cost map.
Returns raw PNG bytes suitable for serving via FastAPI's Response.
"""

from __future__ import annotations

import io
from typing import Optional

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

Coord = tuple[int, int]


def path_to_png(
    cost_map: np.ndarray,
    path: list[Coord],
    start: Coord,
    goal: Coord,
    hazard_map: Optional[np.ndarray] = None,
    replan_path: Optional[list[Coord]] = None,
    scale: int = 2,
) -> bytes:
    """
    Render cost map + path as a PNG image.

    Parameters
    ----------
    cost_map     : (H, W) float32 cost array in [0, 1].
    path         : Global A* path waypoints.
    start        : Start (row, col).
    goal         : Goal (row, col).
    hazard_map   : Optional binary obstacle mask.
    replan_path  : Optional D* Lite replanned path (shown in orange).
    scale        : Upscale factor for visibility.

    Returns
    -------
    PNG image bytes.
    """
    if not _PIL_AVAILABLE:
        return _fallback_png()

    h, w = cost_map.shape

    # Base terrain: grayscale inverted cost (low cost = bright)
    terrain = (255 * (1.0 - cost_map)).clip(0, 255).astype(np.uint8)
    rgb = np.stack([terrain] * 3, axis=-1)

    # Overlay hazard map in red
    if hazard_map is not None:
        rgb[hazard_map > 0] = [200, 20, 20]

    # Draw global path in green
    for i in range(len(path) - 1):
        r1, c1 = path[i]
        r2, c2 = path[i + 1]
        _draw_line_on_rgb(rgb, r1, c1, r2, c2, color=(0, 220, 80))

    # Draw replanned path in orange
    if replan_path:
        for i in range(len(replan_path) - 1):
            r1, c1 = replan_path[i]
            r2, c2 = replan_path[i + 1]
            _draw_line_on_rgb(rgb, r1, c1, r2, c2, color=(255, 160, 0))

    # Draw start (cyan circle) and goal (yellow circle)
    _draw_circle_on_rgb(rgb, start[0], start[1], radius=4, color=(0, 200, 255))
    _draw_circle_on_rgb(rgb, goal[0], goal[1], radius=4, color=(255, 220, 0))

    img = Image.fromarray(rgb, "RGB")

    if scale > 1:
        img = img.resize((w * scale, h * scale), Image.NEAREST)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _draw_line_on_rgb(
    rgb: np.ndarray, r1: int, c1: int, r2: int, c2: int,
    color: tuple[int, int, int],
) -> None:
    """Bresenham line draw on RGB array."""
    rows, cols = rgb.shape[:2]
    dr = abs(r2 - r1); dc = abs(c2 - c1)
    sr = 1 if r2 >= r1 else -1
    sc = 1 if c2 >= c1 else -1
    r, c = r1, c1
    err = dr - dc
    for _ in range(max(dr, dc) + 1):
        if 0 <= r < rows and 0 <= c < cols:
            rgb[r, c] = color
        if r == r2 and c == c2:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc; r += sr
        if e2 < dr:
            err += dr; c += sc


def _draw_circle_on_rgb(
    rgb: np.ndarray, row: int, col: int, radius: int,
    color: tuple[int, int, int],
) -> None:
    rows, cols = rgb.shape[:2]
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            if dr**2 + dc**2 <= radius**2:
                nr, nc = row + dr, col + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    rgb[nr, nc] = color


def _fallback_png() -> bytes:
    """Return a 1x1 transparent PNG when PIL is unavailable."""
    # Minimal valid 1×1 PNG bytes
    return (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
        b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00"
        b"\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18"
        b"\xd4n\x00\x00\x00\x00IEND\xaeB`\x82"
    )

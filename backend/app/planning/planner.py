"""
app/planning/planner.py
=======================
High-level planning orchestrator.

Execution order
---------------
1. A*          — fast, grid-based
2. Hybrid A*   — kinematic-aware, smoother
3. RRT*        — sampling-based fallback (always finds a path if one exists)

Returns the first successful result plus the algorithm name.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from app.planning.astar import astar
from app.planning.hybrid_astar import hybrid_astar
from app.planning.rrt_star import rrt_star
from app.terrain.patch_extractor import TerrainPatch, pixel_to_latlon

logger = logging.getLogger(__name__)

Coord = tuple[int, int]


# ── Path result dataclass ─────────────────────────────────────────────────────

from dataclasses import dataclass


@dataclass
class PlanResult:
    algorithm: str
    pixel_path: list[Coord]
    geo_path: list[tuple[float, float]]   # (lat, lon) per waypoint


# ── Main orchestrator ─────────────────────────────────────────────────────────

def plan_path(
    cost_map: np.ndarray,
    start: Coord,
    goal: Coord,
    patch: TerrainPatch,
    hazard_map: Optional[np.ndarray] = None,
    max_iter: int = 100_000,
) -> Optional[PlanResult]:
    """
    Try A* → Hybrid A* → RRT* until one succeeds.

    Parameters
    ----------
    cost_map   : (H, W) normalised traversal cost.
    start      : (row, col) start pixel.
    goal       : (row, col) goal pixel.
    patch      : TerrainPatch used for pixel→geo back-conversion.
    hazard_map : Optional binary blocked mask (from LiDAR).
    max_iter   : Per-algorithm node expansion budget.

    Returns
    -------
    PlanResult or None if all planners fail.
    """
    planners = [
        ("A*",         lambda: astar(cost_map, start, goal,
                                     max_iter=max_iter, hazard_map=hazard_map)),
        ("Hybrid A*",  lambda: hybrid_astar(cost_map, start, goal,
                                            max_iter=max_iter, hazard_map=hazard_map)),
        ("RRT*",       lambda: rrt_star(cost_map, start, goal,
                                        max_iter=max_iter // 2, hazard_map=hazard_map)),
    ]

    for name, planner_fn in planners:
        logger.info("Attempting path planning with %s …", name)
        try:
            path = planner_fn()
        except Exception as exc:
            logger.warning("%s raised an exception: %s", name, exc)
            path = None

        if path is not None:
            logger.info("%s succeeded — %d waypoints.", name, len(path))
            geo = [pixel_to_latlon(patch, r, c) for r, c in path]
            return PlanResult(algorithm=name, pixel_path=path, geo_path=geo)

        logger.info("%s found no path.", name)

    logger.error("All planners failed — no path from %s to %s.", start, goal)
    return None

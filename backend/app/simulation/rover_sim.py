"""
app/simulation/rover_sim.py
============================
Simulate rover traversal along a path.

For each consecutive waypoint pair the simulation computes:
- Euclidean pixel distance → converted to metres.
- Elevation change (from the terrain patch).
- Energy consumed (base + slope penalty).
- Local risk score (from the cost map).

Returns per-step telemetry and aggregate mission metrics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from app.config import get_settings
from app.terrain.patch_extractor import TerrainPatch

settings = get_settings()

Coord = tuple[int, int]


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class TelemetryStep:
    step:          int
    from_pixel:    Coord
    to_pixel:      Coord
    distance_m:    float
    elevation_change_m: float
    energy_wh:     float
    cost:          float         # local cost-map value
    cumulative_distance_m: float
    cumulative_energy_wh:  float


@dataclass
class SimulationResult:
    telemetry:            list[TelemetryStep]
    total_distance_m:     float
    total_energy_wh:      float
    max_slope_deg:        float
    risk_score:           float          # weighted avg cost [0,1]
    pixel_per_metre:      float


# ── Public API ────────────────────────────────────────────────────────────────

def simulate_traversal(
    path: list[Coord],
    patch: TerrainPatch,
    cost_map: np.ndarray,
    pixel_size_m: float | None = None,
) -> SimulationResult:
    """
    Simulate rover traversal along *path* on the given terrain.

    Parameters
    ----------
    path         : Ordered list of (row, col) waypoints.
    patch        : TerrainPatch providing elevation data and resolution.
    cost_map     : (H, W) normalised traversal cost.
    pixel_size_m : Ground sampling distance (metres per pixel).  If None,
                   derived from patch.resolution_deg assuming lunar radius.

    Returns
    -------
    SimulationResult with per-step telemetry and aggregate metrics.
    """
    if len(path) < 2:
        return SimulationResult(
            telemetry=[], total_distance_m=0.0,
            total_energy_wh=0.0, max_slope_deg=0.0,
            risk_score=0.0, pixel_per_metre=1.0,
        )

    if pixel_size_m is None:
        pixel_size_m = _deg_to_metres_lunar(patch.resolution_deg)

    telemetry: list[TelemetryStep] = []
    cum_dist = 0.0
    cum_energy = 0.0
    slopes: list[float] = []
    costs: list[float] = []

    for i in range(len(path) - 1):
        r0, c0 = path[i]
        r1, c1 = path[i + 1]

        pixel_dist = math.hypot(r1 - r0, c1 - c0)
        dist_m = pixel_dist * pixel_size_m

        elev0 = _safe_elev(patch.data, r0, c0)
        elev1 = _safe_elev(patch.data, r1, c1)
        delta_elev = elev1 - elev0

        slope_deg = math.degrees(math.atan2(abs(delta_elev), max(dist_m, 1e-6)))
        slopes.append(slope_deg)

        local_cost = float(cost_map[r1, c1]) if (
            0 <= r1 < cost_map.shape[0] and 0 <= c1 < cost_map.shape[1]
        ) else 0.5
        costs.append(local_cost)

        # Energy model: base consumption + slope penalty
        slope_factor = 1.0 + 0.5 * math.sin(math.radians(slope_deg))
        energy = dist_m * settings.ENERGY_PER_METER * slope_factor * (1.0 + local_cost)

        cum_dist += dist_m
        cum_energy += energy

        telemetry.append(TelemetryStep(
            step=i,
            from_pixel=path[i],
            to_pixel=path[i + 1],
            distance_m=round(dist_m, 3),
            elevation_change_m=round(delta_elev, 2),
            energy_wh=round(energy, 4),
            cost=round(local_cost, 4),
            cumulative_distance_m=round(cum_dist, 3),
            cumulative_energy_wh=round(cum_energy, 4),
        ))

    risk_score = float(np.mean(costs)) if costs else 0.0
    max_slope = max(slopes) if slopes else 0.0

    return SimulationResult(
        telemetry=telemetry,
        total_distance_m=round(cum_dist, 3),
        total_energy_wh=round(cum_energy, 4),
        max_slope_deg=round(max_slope, 2),
        risk_score=round(risk_score, 4),
        pixel_per_metre=round(1.0 / pixel_size_m, 4),
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_elev(data: np.ndarray, r: int, c: int) -> float:
    """Return elevation value or 0 if NaN / out-of-bounds."""
    h, w = data.shape
    if not (0 <= r < h and 0 <= c < w):
        return 0.0
    val = data[r, c]
    return float(val) if not math.isnan(val) else 0.0


def _deg_to_metres_lunar(deg: float) -> float:
    """
    Convert angular resolution (degrees) to metres on the lunar surface.
    Lunar mean radius = 1737.4 km.
    """
    LUNAR_RADIUS_M = 1_737_400.0
    return math.radians(deg) * LUNAR_RADIUS_M

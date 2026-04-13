"""
app/routers/lidar.py
=====================
Endpoints for ingesting RPLiDAR A3 scan data and converting to obstacle maps.

The RPLiDAR A3 outputs in polar format:
    theta: 10.23  Dist: 1532.45  Q: 47

This router:
1. Accepts raw scan points (theta, distance_m, quality)
2. Filters by quality and range
3. Converts polar → Cartesian (rover-centric pixel space)
4. Returns obstacle pixel coordinates for use with D* Lite replan
"""

from __future__ import annotations

import math
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.mission import Mission
from app.schemas.mission import LidarScanPayload, LidarObstacle, LidarPayload
from app.config import get_settings

router = APIRouter()
logger = logging.getLogger(__name__)
settings = get_settings()


@router.post("/scan/process", summary="Process raw RPLiDAR A3 scan into obstacle map")
async def process_lidar_scan(
    body: LidarScanPayload,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """
    Convert raw RPLiDAR A3 polar scan into pixel-space obstacles.

    The RPLiDAR A3 output format:
        theta: 10.23  Dist: 1532.45  Q: 47

    Input JSON example:
    ```json
    {
        "mission_id": 1,
        "rover_row": 45,
        "rover_col": 32,
        "pixels_per_metre": 0.2,
        "points": [
            {"theta": 10.23, "distance_m": 15.32, "quality": 47},
            {"theta": 11.01, "distance_m": 14.98, "quality": 48}
        ]
    }
    ```

    Returns list of obstacle pixel coordinates filtered by quality and range.
    """
    mission = await db.get(Mission, body.mission_id)
    if not mission:
        raise HTTPException(404, f"Mission {body.mission_id} not found.")

    obstacles: list[dict[str, Any]] = []

    for pt in body.points:
        # RPLiDAR A3 quality filter (< 15 = unreliable)
        if pt.quality < settings.LIDAR_MIN_QUALITY:
            continue
        # Range filter (> 100m exceeds A3 spec)
        if pt.distance_m > settings.LIDAR_MAX_RANGE_M or pt.distance_m <= 0.1:
            continue

        # Polar → Cartesian (rover-centric, metres)
        theta_rad = math.radians(pt.theta)
        x_m = pt.distance_m * math.cos(theta_rad)
        y_m = pt.distance_m * math.sin(theta_rad)

        # Metres → pixels
        ppm = body.pixels_per_metre  # pixels per metre
        dx_px = int(round(x_m * ppm))
        dy_px = int(round(y_m * ppm))

        if body.rover_row is not None and body.rover_col is not None:
            obs_row = body.rover_row + dy_px
            obs_col = body.rover_col + dx_px
        else:
            obs_row = dy_px
            obs_col = dx_px

        obstacles.append({
            "x":          obs_col,
            "y":          obs_row,
            "radius":     2,          # 2px default radius
            "distance_m": round(pt.distance_m, 2),
            "theta_deg":  round(pt.theta, 2),
            "quality":    pt.quality,
        })

    return {
        "mission_id":      body.mission_id,
        "points_received": len(body.points),
        "obstacles_found": len(obstacles),
        "obstacles":       obstacles,
    }


@router.post("/scan/inject", summary="Inject processed obstacles and trigger D* Lite replan")
async def inject_obstacles(
    body: LidarScanPayload,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """
    Process LiDAR scan and immediately trigger D* Lite replanning.

    This is a convenience endpoint that combines scan processing and replanning.
    Wraps the processed obstacles into the format expected by /mission/{id}/replan.
    """
    from fastapi import BackgroundTasks
    from app.models.mission import MissionStatus
    from app.routers.mission import _run_replan_pipeline

    mission = await db.get(Mission, body.mission_id)
    if not mission:
        raise HTTPException(404, f"Mission {body.mission_id} not found.")

    if mission.status not in (MissionStatus.COMPLETED, MissionStatus.REPLANNING):
        raise HTTPException(
            409,
            f"Mission {body.mission_id} is not in a replannable state "
            f"(status={mission.status})."
        )

    # Process scan → obstacles
    scan_result = await process_lidar_scan(body, db)

    if not scan_result["obstacles"]:
        return {
            "mission_id": body.mission_id,
            "status":     "no_obstacles",
            "message":    "No valid obstacles detected in scan.",
        }

    # Trigger replan
    lidar_data = {"obstacles": scan_result["obstacles"]}
    mission.status = MissionStatus.REPLANNING
    await db.commit()

    import asyncio
    asyncio.create_task(_run_replan_pipeline(body.mission_id, lidar_data))

    return {
        "mission_id":      body.mission_id,
        "status":          "replanning",
        "obstacles_found": scan_result["obstacles_found"],
        "message":         "D* Lite replan triggered.",
    }

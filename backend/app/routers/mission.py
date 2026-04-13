"""
app/routers/mission.py
======================
FastAPI router for all mission-related endpoints.

Endpoints
---------
POST   /mission/create                — Create and run full mission pipeline
GET    /mission/                      — List all missions
GET    /mission/{id}/status           — Poll mission status + results
GET    /mission/{id}/summary          — Full mission summary
GET    /mission/{id}/path-image       — PNG visualisation of planned path
GET    /mission/{id}/telemetry        — Full per-step telemetry
POST   /mission/{id}/replan           — Inject LiDAR data → D* Lite replan
DELETE /mission/{id}                  — Delete a mission record
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict
from typing import Any

import numpy as np
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.mission import Mission, MissionStatus
from app.schemas.mission import (
    MissionCreateRequest, MissionCreateResponse,
    MissionStatusResponse, ReplanRequest,
    MissionListItem,
)
from app.config import get_settings

from app.geo.vrt_handler import VRTHandler
from app.terrain.patch_extractor import extract_patch, latlon_to_pixel, pixel_to_latlon
from app.ml.cost_map import generate_cost_map
from app.planning.planner import plan_path
from app.simulation.rover_sim import simulate_traversal
from app.lidar.hazard_map import generate_local_hazard_map
from app.planning.dstar_lite import replan_with_dstar
from app.analysis.mineral_analysis import analyze_minerals
from app.utils.path_smoother import smooth_path, downsample_path
from app.utils.visualization import path_to_png
from app.routers.ws import (
    emit_status, emit_path_ready, emit_telemetry, emit_completed, emit_error,
)

router = APIRouter()
logger = logging.getLogger(__name__)
settings = get_settings()


# ── Shared telemetry store (in-memory, cleared on mission delete) ─────────────
_telemetry_store: dict[int, list[dict]] = {}


# ── Helper ────────────────────────────────────────────────────────────────────

async def _get_mission(mission_id: int, db: AsyncSession) -> Mission:
    mission = await db.get(Mission, mission_id)
    if not mission:
        raise HTTPException(status_code=404, detail=f"Mission {mission_id} not found.")
    return mission


# ── POST /mission/create ──────────────────────────────────────────────────────

@router.post("/create", response_model=MissionCreateResponse, status_code=202)
async def create_mission(
    body: MissionCreateRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
) -> MissionCreateResponse:
    """
    Create a new rover mission and kick off the full planning pipeline.

    **Mode 1 — Lat/Lon:**
    ```json
    {"start": {"lat": -12.5, "lon": 45.3}, "goal": {"lat": -12.1, "lon": 45.8}}
    ```

    **Mode 2 — Image pixel (requires VRT file):**
    ```json
    {"start_pixel": {"x": 120, "y": 340}, "goal_pixel": {"x": 800, "y": 600},
     "vrt_file": "data/lunar_map.vrt"}
    ```
    """
    if body.input_mode == "pixel":
        handler = VRTHandler(body.vrt_file)
        start_lat, start_lon = handler.pixel_to_geo(body.start_pixel.x, body.start_pixel.y)
        goal_lat,  goal_lon  = handler.pixel_to_geo(body.goal_pixel.x,  body.goal_pixel.y)
    else:
        start_lat, start_lon = body.start.lat, body.start.lon
        goal_lat,  goal_lon  = body.goal.lat,  body.goal.lon

    mission = Mission(
        status     = MissionStatus.PENDING,
        start_lat  = start_lat,  start_lon = start_lon,
        goal_lat   = goal_lat,   goal_lon  = goal_lon,
        input_mode = body.input_mode,
        vrt_file   = body.vrt_file,
    )
    db.add(mission)
    await db.flush()
    mission_id = mission.id
    await db.commit()

    background_tasks.add_task(
        _run_mission_pipeline,
        mission_id, start_lat, start_lon, goal_lat, goal_lon,
        use_heuristic=body.use_heuristic,
        smooth=body.smooth_path,
    )

    return MissionCreateResponse(
        mission_id = mission_id,
        status     = MissionStatus.PENDING,
        start_lat  = start_lat,  start_lon = start_lon,
        goal_lat   = goal_lat,   goal_lon  = goal_lon,
        input_mode = body.input_mode,
    )


# ── GET /mission/ ─────────────────────────────────────────────────────────────

@router.get("/", response_model=list[MissionListItem])
async def list_missions(
    db: AsyncSession = Depends(get_db),
    limit: int = 50,
    offset: int = 0,
) -> list[MissionListItem]:
    """Return a paginated list of all missions."""
    result = await db.execute(
        select(Mission).order_by(Mission.id.desc()).limit(limit).offset(offset)
    )
    missions = result.scalars().all()
    return [
        MissionListItem(
            mission_id = m.id,
            status     = m.status,
            start_lat  = m.start_lat,  start_lon = m.start_lon,
            goal_lat   = m.goal_lat,   goal_lon  = m.goal_lon,
            created_at = str(m.created_at) if m.created_at else None,
        )
        for m in missions
    ]


# ── GET /mission/{id}/status ──────────────────────────────────────────────────

@router.get("/{mission_id}/status", response_model=MissionStatusResponse)
async def get_mission_status(
    mission_id: int,
    db: AsyncSession = Depends(get_db),
) -> MissionStatusResponse:
    """Poll mission status; results populated when COMPLETED."""
    mission = await _get_mission(mission_id, db)
    return MissionStatusResponse(
        mission_id     = mission.id,
        status         = mission.status,
        summary        = mission.mission_summary,
        mineral_report = mission.mineral_report,
    )


# ── GET /mission/{id}/summary ─────────────────────────────────────────────────

@router.get("/{mission_id}/summary")
async def get_mission_summary(
    mission_id: int,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Return full mission summary, paths, and mineral analysis."""
    mission = await _get_mission(mission_id, db)
    if mission.status != MissionStatus.COMPLETED:
        raise HTTPException(409, "Mission not yet completed.")
    return {
        "mission_id":     mission.id,
        "status":         mission.status,
        "summary":        mission.mission_summary,
        "mineral_report": mission.mineral_report,
        "global_path":    mission.global_path,
        "local_path":     mission.local_path,
    }


# ── GET /mission/{id}/path-image ──────────────────────────────────────────────

@router.get("/{mission_id}/path-image", response_class=Response)
async def get_path_image(
    mission_id: int,
    db: AsyncSession = Depends(get_db),
) -> Response:
    """
    Return a PNG image of the planned path overlaid on the terrain cost map.
    Suitable for embedding in <img> tags or saving to disk.
    """
    mission = await _get_mission(mission_id, db)
    if not mission.global_path:
        raise HTTPException(409, "No path available yet.")

    try:
        patch = await asyncio.get_event_loop().run_in_executor(
            None, extract_patch,
            settings.JP2_PATH,
            mission.start_lat, mission.start_lon,
            mission.goal_lat,  mission.goal_lon,
        )
        cost_map = await asyncio.get_event_loop().run_in_executor(
            None, generate_cost_map, patch.data, patch, settings.CRATER_CSV,
        )
        global_px  = [tuple(p) for p in mission.global_path["pixels"]]
        replan_px  = (
            [tuple(p) for p in mission.local_path["pixels"]]
            if mission.local_path else None
        )
        start_px = latlon_to_pixel(patch, mission.start_lat, mission.start_lon)
        goal_px  = latlon_to_pixel(patch, mission.goal_lat,  mission.goal_lon)

        png_bytes = await asyncio.get_event_loop().run_in_executor(
            None, path_to_png,
            cost_map, global_px, start_px, goal_px, None, replan_px, 2,
        )
        return Response(content=png_bytes, media_type="image/png")
    except Exception as exc:
        logger.exception("path-image failed: %s", exc)
        raise HTTPException(500, str(exc))


# ── GET /mission/{id}/telemetry ───────────────────────────────────────────────

@router.get("/{mission_id}/telemetry")
async def get_mission_telemetry(
    mission_id: int,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Return full per-step telemetry for a completed mission."""
    await _get_mission(mission_id, db)
    telemetry = _telemetry_store.get(mission_id, [])
    return {"mission_id": mission_id, "steps": len(telemetry), "telemetry": telemetry}


# ── POST /mission/{id}/replan ─────────────────────────────────────────────────

@router.post("/{mission_id}/replan")
async def replan_mission(
    mission_id: int,
    body: ReplanRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """
    Inject LiDAR obstacle data and trigger D* Lite local replanning.

    ```json
    {
        "mission_id": 1,
        "lidar_data": {
            "obstacles": [{"x": 120, "y": 340, "radius": 5}]
        }
    }
    ```
    """
    mission = await _get_mission(mission_id, db)
    if mission.status not in (MissionStatus.COMPLETED, MissionStatus.REPLANNING):
        raise HTTPException(
            409,
            f"Mission {mission_id} cannot be replanned (status={mission.status})."
        )
    if not mission.global_path:
        raise HTTPException(409, "No global path available to replan from.")

    mission.status = MissionStatus.REPLANNING
    await db.commit()

    background_tasks.add_task(
        _run_replan_pipeline, mission_id, body.lidar_data.model_dump()
    )
    return {"mission_id": mission_id, "status": MissionStatus.REPLANNING}


# ── DELETE /mission/{id} ──────────────────────────────────────────────────────

@router.delete("/{mission_id}")
async def delete_mission(
    mission_id: int,
    db: AsyncSession = Depends(get_db),
) -> dict[str, str]:
    """Delete a mission record and its cached telemetry."""
    mission = await _get_mission(mission_id, db)
    await db.delete(mission)
    await db.commit()
    _telemetry_store.pop(mission_id, None)
    return {"message": f"Mission {mission_id} deleted."}


# ═══════════════════════════════════════════════════════════════════════════════
# Background pipeline
# ═══════════════════════════════════════════════════════════════════════════════

async def _run_mission_pipeline(
    mission_id: int,
    start_lat: float, start_lon: float,
    goal_lat: float,  goal_lon: float,
    use_heuristic: bool = True,
    smooth: bool = True,
) -> None:
    """Full planning pipeline: terrain → cost → plan → simulate → minerals."""
    from app.database import AsyncSessionFactory

    async with AsyncSessionFactory() as db:
        mission = await db.get(Mission, mission_id)
        try:
            mission.status = MissionStatus.PLANNING
            await db.commit()
            await emit_status(mission_id, "PLANNING", "Extracting terrain patch…")

            # 1. Terrain patch ──────────────────────────────────────────────────
            patch = await asyncio.get_event_loop().run_in_executor(
                None, extract_patch,
                settings.JP2_PATH,
                start_lat, start_lon, goal_lat, goal_lon,
            )

            # 2. Cost map ───────────────────────────────────────────────────────
            await emit_status(mission_id, "PLANNING", "Generating cost map…")
            cost_map = await asyncio.get_event_loop().run_in_executor(
                None, generate_cost_map,
                patch.data, patch, settings.CRATER_CSV,
                settings.MODEL_CHECKPOINT, use_heuristic,
            )

            # 3. Convert lat/lon → patch pixels ────────────────────────────────
            start_px = latlon_to_pixel(patch, start_lat, start_lon)
            goal_px  = latlon_to_pixel(patch, goal_lat,  goal_lon)

            # 4. Path planning ──────────────────────────────────────────────────
            await emit_status(mission_id, "PLANNING", "Running path planner…")
            plan = await asyncio.get_event_loop().run_in_executor(
                None, plan_path, cost_map, start_px, goal_px, patch,
                None, settings.PLANNING_MAX_ITER,
            )

            if plan is None:
                mission.status = MissionStatus.FAILED
                await db.commit()
                await emit_error(mission_id, "All planners failed — no feasible path.")
                return

            # Optional smoothing
            pixel_path = plan.pixel_path
            if smooth and len(pixel_path) > 4:
                pixel_path = await asyncio.get_event_loop().run_in_executor(
                    None, smooth_path, pixel_path, settings.SMOOTH_WINDOW, 2
                )

            # Compute geo path from (possibly smoothed) pixel path
            geo_path = [pixel_to_latlon(patch, r, c) for r, c in pixel_path]

            # Rough distance estimate
            dist_m = sum(
                ((pixel_path[i+1][0]-pixel_path[i][0])**2 +
                 (pixel_path[i+1][1]-pixel_path[i][1])**2)**0.5
                for i in range(len(pixel_path)-1)
            ) * abs(patch.transform.a) * 111_000  # deg→m approx

            await emit_path_ready(mission_id, plan.algorithm, len(pixel_path), dist_m)

            mission.global_path = {
                "algorithm": plan.algorithm,
                "pixels":    [list(p) for p in pixel_path],
                "geo":       [list(g) for g in geo_path],
            }

            # 5. Simulate traversal ─────────────────────────────────────────────
            mission.status = MissionStatus.SIMULATING
            await db.commit()
            await emit_status(mission_id, "SIMULATING", "Simulating traversal…")

            sim = await asyncio.get_event_loop().run_in_executor(
                None, simulate_traversal, pixel_path, patch, cost_map,
            )

            # Store and broadcast telemetry
            tele_records: list[dict] = []
            for step in sim.telemetry:
                r, c = step.to_pixel
                lat, lon = pixel_to_latlon(patch, r, c)
                record = {
                    "step":                  step.step,
                    "pixel":                 list(step.to_pixel),
                    "lat":                   round(lat, 6),
                    "lon":                   round(lon, 6),
                    "elevation_m":           round(step.elevation_change_m, 2),
                    "energy_wh":             round(step.energy_wh, 4),
                    "cumulative_distance_m": round(step.cumulative_distance_m, 3),
                    "cost":                  round(step.cost, 4),
                }
                tele_records.append(record)
                # Broadcast every 10th step to avoid flooding
                if step.step % 10 == 0:
                    await emit_telemetry(
                        mission_id, step.step, step.to_pixel,
                        lat, lon, step.elevation_change_m,
                        step.energy_wh, step.cumulative_distance_m, step.cost,
                    )

            _telemetry_store[mission_id] = tele_records

            # 6. Mineral analysis ───────────────────────────────────────────────
            await emit_status(mission_id, "SIMULATING", "Analysing minerals…")
            mineral = await asyncio.get_event_loop().run_in_executor(
                None, analyze_minerals,
                pixel_path, patch,
                settings.HYDROGEN_DAT, settings.IRON_DAT, settings.THORIUM_DAT,
            )

            # 7. Persist results ────────────────────────────────────────────────
            summary = {
                "total_distance_m": sim.total_distance_m,
                "total_energy_wh":  sim.total_energy_wh,
                "max_slope_deg":    sim.max_slope_deg,
                "risk_score":       sim.risk_score,
                "path_algorithm":   plan.algorithm,
                "waypoints":        len(pixel_path),
            }
            mission.mission_summary = summary
            mission.mineral_report  = asdict(mineral)
            mission.status = MissionStatus.COMPLETED
            await db.commit()

            await emit_completed(mission_id, summary)
            logger.info("[Mission %d] Completed successfully.", mission_id)

        except Exception as exc:
            logger.exception("[Mission %d] Pipeline failed: %s", mission_id, exc)
            try:
                mission.status = MissionStatus.FAILED
                await db.commit()
                await emit_error(mission_id, str(exc))
            except Exception:
                pass


async def _run_replan_pipeline(mission_id: int, lidar_data: dict) -> None:
    """D* Lite local replan pipeline triggered by LiDAR obstacle data."""
    from app.database import AsyncSessionFactory

    async with AsyncSessionFactory() as db:
        mission = await db.get(Mission, mission_id)
        try:
            await emit_status(mission_id, "REPLANNING", "D* Lite replan starting…")

            patch = await asyncio.get_event_loop().run_in_executor(
                None, extract_patch,
                settings.JP2_PATH,
                mission.start_lat, mission.start_lon,
                mission.goal_lat,  mission.goal_lon,
            )
            cost_map = await asyncio.get_event_loop().run_in_executor(
                None, generate_cost_map,
                patch.data, patch, settings.CRATER_CSV,
                settings.MODEL_CHECKPOINT, True,
            )

            global_pixels = [tuple(p) for p in mission.global_path["pixels"]]

            hazard_map = await asyncio.get_event_loop().run_in_executor(
                None, generate_local_hazard_map, lidar_data, patch.data.shape,
            )

            new_path = await asyncio.get_event_loop().run_in_executor(
                None, replan_with_dstar, global_pixels, hazard_map, cost_map,
            )

            if new_path is None:
                # Escalate to RRT*
                await emit_status(
                    mission_id, "REPLANNING",
                    "D* Lite failed — escalating to RRT*…"
                )
                from app.planning.rrt_star import rrt_star
                start_px = tuple(global_pixels[0])
                goal_px  = tuple(global_pixels[-1])
                new_path = await asyncio.get_event_loop().run_in_executor(
                    None, rrt_star,
                    cost_map, start_px, goal_px,
                    settings.RRT_STEP_SIZE, settings.RRT_MAX_ITER,
                    None, hazard_map,
                )
                algorithm = "RRT*"
            else:
                algorithm = "D* Lite"

            if new_path is None:
                mission.status = MissionStatus.FAILED
                await db.commit()
                await emit_error(mission_id, "Replan failed — no viable path found.")
                return

            geo_path = [pixel_to_latlon(patch, r, c) for r, c in new_path]

            mission.local_path = {
                "algorithm": algorithm,
                "pixels":    [list(p) for p in new_path],
                "geo":       [list(g) for g in geo_path],
            }

            sim = await asyncio.get_event_loop().run_in_executor(
                None, simulate_traversal, new_path, patch, cost_map,
            )
            mineral = await asyncio.get_event_loop().run_in_executor(
                None, analyze_minerals,
                new_path, patch,
                settings.HYDROGEN_DAT, settings.IRON_DAT, settings.THORIUM_DAT,
            )

            summary = {
                "total_distance_m": sim.total_distance_m,
                "total_energy_wh":  sim.total_energy_wh,
                "max_slope_deg":    sim.max_slope_deg,
                "risk_score":       sim.risk_score,
                "path_algorithm":   f"{algorithm} (replan)",
                "waypoints":        len(new_path),
            }
            mission.mission_summary = summary
            mission.mineral_report  = asdict(mineral)
            mission.status = MissionStatus.COMPLETED
            await db.commit()

            await emit_completed(mission_id, summary)

        except Exception as exc:
            logger.exception("[Mission %d] Replan failed: %s", mission_id, exc)
            try:
                mission.status = MissionStatus.FAILED
                await db.commit()
                await emit_error(mission_id, str(exc))
            except Exception:
                pass

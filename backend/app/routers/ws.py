"""
app/routers/ws.py
=================
WebSocket endpoint for real-time mission telemetry streaming.

Connect to:  ws://<host>/ws/mission/{id}/stream

Frame types
-----------
telemetry  — Per-step rover position + energy data
status     — Mission phase change (PLANNING, SIMULATING, etc.)
completed  — Final summary when mission finishes
error      — Pipeline failure
heartbeat  — Keep-alive every 30 s
pong       — Response to client "ping"
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()
logger = logging.getLogger(__name__)

# mission_id → set of WebSocket clients
_subscribers: dict[int, set[WebSocket]] = {}


def _register(mission_id: int, ws: WebSocket) -> None:
    _subscribers.setdefault(mission_id, set()).add(ws)


def _unregister(mission_id: int, ws: WebSocket) -> None:
    _subscribers.get(mission_id, set()).discard(ws)


async def broadcast(mission_id: int, payload: dict[str, Any]) -> None:
    """Broadcast a JSON payload to all subscribers of a mission."""
    subs = list(_subscribers.get(mission_id, []))
    dead: list[WebSocket] = []
    for ws in subs:
        try:
            await ws.send_json(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _unregister(mission_id, ws)


# ── WebSocket endpoint ────────────────────────────────────────────────────────

@router.websocket("/mission/{mission_id}/stream")
async def mission_telemetry_stream(
    websocket: WebSocket,
    mission_id: int,
) -> None:
    """
    Stream live telemetry for a running mission.

    Connect immediately after POST /mission/create.
    Send "ping" to get a "pong" keep-alive response.
    """
    await websocket.accept()
    _register(mission_id, websocket)
    logger.info("WS client connected — mission %d.", mission_id)

    # Send a welcome frame
    await websocket.send_json({
        "type":       "connected",
        "mission_id": mission_id,
        "message":    f"Subscribed to mission {mission_id} telemetry.",
    })

    try:
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                if data.strip().lower() == "ping":
                    await websocket.send_json({"type": "pong"})
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "heartbeat", "mission_id": mission_id})
    except WebSocketDisconnect:
        logger.info("WS client disconnected — mission %d.", mission_id)
    finally:
        _unregister(mission_id, websocket)


# ── Emit helpers (called from pipeline) ──────────────────────────────────────

async def emit_telemetry(
    mission_id: int,
    step: int,
    pixel: tuple[int, int],
    lat: float,
    lon: float,
    elevation_m: float,
    energy_wh: float,
    cumulative_distance_m: float,
    cost: float,
    slope_deg: float = 0.0,
) -> None:
    """Emit a single telemetry frame."""
    await broadcast(mission_id, {
        "type":                  "telemetry",
        "mission_id":            mission_id,
        "step":                  step,
        "pixel":                 list(pixel),
        "lat":                   round(lat, 6),
        "lon":                   round(lon, 6),
        "elevation_m":           round(elevation_m, 2),
        "energy_wh":             round(energy_wh, 4),
        "cumulative_distance_m": round(cumulative_distance_m, 3),
        "cost":                  round(cost, 4),
        "slope_deg":             round(slope_deg, 2),
    })


async def emit_status(mission_id: int, status: str, detail: str = "") -> None:
    """Emit a mission status-change frame."""
    await broadcast(mission_id, {
        "type":       "status",
        "mission_id": mission_id,
        "status":     status,
        "detail":     detail,
    })


async def emit_path_ready(
    mission_id: int,
    algorithm: str,
    waypoint_count: int,
    distance_m: float,
) -> None:
    """Emit when global path planning completes."""
    await broadcast(mission_id, {
        "type":           "path_ready",
        "mission_id":     mission_id,
        "algorithm":      algorithm,
        "waypoint_count": waypoint_count,
        "distance_m":     round(distance_m, 1),
    })


async def emit_completed(mission_id: int, summary: dict[str, Any]) -> None:
    """Emit mission-completed frame with final summary."""
    await broadcast(mission_id, {
        "type":       "completed",
        "mission_id": mission_id,
        "summary":    summary,
    })


async def emit_error(mission_id: int, message: str) -> None:
    """Emit an error frame."""
    await broadcast(mission_id, {
        "type":       "error",
        "mission_id": mission_id,
        "message":    message,
    })

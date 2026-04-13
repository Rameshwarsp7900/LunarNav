"""
app/schemas/mission.py — Pydantic v2 schemas for mission API I/O.
"""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, model_validator, Field


# ── Sub-schemas ───────────────────────────────────────────────────────────────

class LatLon(BaseModel):
    lat: float = Field(..., ge=-90,  le=90,  description="Latitude in decimal degrees")
    lon: float = Field(..., ge=-180, le=360, description="Longitude in decimal degrees")


class Pixel(BaseModel):
    x: int = Field(..., ge=0, description="Column pixel index")
    y: int = Field(..., ge=0, description="Row pixel index")


class LidarObstacle(BaseModel):
    x:      int             # column
    y:      int             # row
    radius: int = Field(default=5, ge=1, description="Obstacle radius in pixels")
    quality: Optional[int] = Field(default=None, description="RPLiDAR A3 quality score")
    distance_m: Optional[float] = Field(default=None, description="Distance in metres")


class LidarPayload(BaseModel):
    obstacles: list[LidarObstacle] = Field(..., description="List of detected obstacles")


# ── RPLiDAR A3 raw scan payload ───────────────────────────────────────────────

class LidarScanPoint(BaseModel):
    """Single point from RPLiDAR A3 output: theta, distance, quality."""
    theta:      float = Field(..., description="Angle in degrees [0, 360)")
    distance_m: float = Field(..., ge=0, le=100, description="Distance in metres")
    quality:    int   = Field(..., ge=0, le=63,  description="Measurement quality")


class LidarScanPayload(BaseModel):
    """Full 360° LiDAR scan from RPLiDAR A3."""
    mission_id: int
    points:     list[LidarScanPoint]
    rover_row:  Optional[int]   = None  # rover's current patch row
    rover_col:  Optional[int]   = None  # rover's current patch col
    pixels_per_metre: float = Field(default=1.0, description="Scale: pixels per metre")


# ── Mission creation ──────────────────────────────────────────────────────────

class MissionCreateRequest(BaseModel):
    """
    Supports two mutually exclusive input modes.

    Mode 1 — Direct lat/lon:
        {"start": {"lat": -12.5, "lon": 45.3}, "goal": {"lat": -12.1, "lon": 45.8}}

    Mode 2 — Image pixel:
        {
            "start_pixel": {"x": 120, "y": 340},
            "goal_pixel":  {"x": 800, "y": 600},
            "vrt_file":    "data/lunar_map.vrt"
        }
    """

    # Mode 1
    start: Optional[LatLon] = None
    goal:  Optional[LatLon] = None

    # Mode 2
    start_pixel: Optional[Pixel] = None
    goal_pixel:  Optional[Pixel] = None
    vrt_file:    Optional[str]   = None

    # Optional overrides
    use_heuristic: bool = True
    smooth_path:   bool = True
    notes:         Optional[str] = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def validate_input_mode(self) -> "MissionCreateRequest":
        mode1 = self.start is not None and self.goal is not None
        mode2 = (
            self.start_pixel is not None
            and self.goal_pixel is not None
            and self.vrt_file is not None
        )
        if not mode1 and not mode2:
            raise ValueError(
                "Provide either (start + goal) for lat/lon mode, "
                "or (start_pixel + goal_pixel + vrt_file) for image mode."
            )
        if mode1 and mode2:
            raise ValueError("Provide only one input mode, not both.")
        return self

    @property
    def input_mode(self) -> str:
        return "pixel" if self.start_pixel is not None else "latlon"


class MissionCreateResponse(BaseModel):
    mission_id: int
    status:     str
    start_lat:  float
    start_lon:  float
    goal_lat:   float
    goal_lon:   float
    input_mode: str
    message:    str = "Mission queued — pipeline running in background."


# ── Path planning result ──────────────────────────────────────────────────────

class Waypoint(BaseModel):
    pixel_row: int
    pixel_col: int
    lat:       float
    lon:       float


class PathResult(BaseModel):
    algorithm_used: str
    waypoints:      list[Waypoint]
    total_distance_m: float


# ── Replanning ────────────────────────────────────────────────────────────────

class ReplanRequest(BaseModel):
    mission_id: int
    lidar_data: LidarPayload


# ── Mission status ────────────────────────────────────────────────────────────

class MissionStatusResponse(BaseModel):
    mission_id:     int
    status:         str
    summary:        Optional[dict[str, Any]] = None
    mineral_report: Optional[dict[str, Any]] = None


# ── Mission list ──────────────────────────────────────────────────────────────

class MissionListItem(BaseModel):
    mission_id: int
    status:     str
    start_lat:  float
    start_lon:  float
    goal_lat:   float
    goal_lon:   float
    created_at: Optional[str] = None


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status:   str
    service:  str
    version:  str
    datasets: dict[str, bool]

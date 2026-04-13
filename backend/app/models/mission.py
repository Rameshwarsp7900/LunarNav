"""
app/models/mission.py — SQLAlchemy ORM model for rover missions.
"""

import enum
from datetime import datetime, timezone

from sqlalchemy import Column, Integer, Float, String, DateTime, JSON, Enum
from sqlalchemy.sql import func

from app.database import Base


class MissionStatus(str, enum.Enum):
    PENDING = "PENDING"
    PLANNING = "PLANNING"
    SIMULATING = "SIMULATING"
    REPLANNING = "REPLANNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class Mission(Base):
    __tablename__ = "missions"

    id = Column(Integer, primary_key=True, index=True)
    status = Column(Enum(MissionStatus), default=MissionStatus.PENDING, nullable=False)

    # ── Coordinates ───────────────────────────────────────────────────────────
    start_lat = Column(Float, nullable=False)
    start_lon = Column(Float, nullable=False)
    goal_lat = Column(Float, nullable=False)
    goal_lon = Column(Float, nullable=False)

    # ── Input mode bookkeeping ────────────────────────────────────────────────
    input_mode = Column(String(16), default="latlon")   # "latlon" | "pixel"
    vrt_file = Column(String(512), nullable=True)

    # ── Results (JSON blobs) ──────────────────────────────────────────────────
    global_path = Column(JSON, nullable=True)        # pixel + latlon waypoints
    local_path = Column(JSON, nullable=True)         # after D* Lite
    mission_summary = Column(JSON, nullable=True)    # distance / energy / risk
    mineral_report = Column(JSON, nullable=True)

    # ── Audit ─────────────────────────────────────────────────────────────────
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    def __repr__(self) -> str:
        return (
            f"<Mission id={self.id} status={self.status} "
            f"({self.start_lat},{self.start_lon}) → ({self.goal_lat},{self.goal_lon})>"
        )

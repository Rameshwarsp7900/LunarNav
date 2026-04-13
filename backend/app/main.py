"""
app/main.py — Lunar Rover Navigation Backend — FastAPI Entry Point
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.database import engine, Base
from app.routers import mission, ws, lidar
from app.config import get_settings

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: create all DB tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    # Shutdown: close DB connections
    await engine.dispose()


app = FastAPI(
    title=settings.API_TITLE,
    description=(
        "Production-grade backend for AI-powered lunar rover path planning. "
        "Supports dual input modes (lat/lon + image pixel), U-Net cost maps, "
        "Hybrid A*/D*/RRT* planning, RPLiDAR A3 integration, and mineral analysis."
    ),
    version=settings.API_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(mission.router, prefix="/mission", tags=["Mission"])
app.include_router(lidar.router,   prefix="/lidar",   tags=["LiDAR"])
app.include_router(ws.router,      prefix="/ws",      tags=["WebSocket"])


@app.get("/health", tags=["Health"])
async def health_check() -> dict:
    """Liveness probe — also checks dataset file availability."""
    datasets = {
        "sldem_jp2":   Path(settings.JP2_PATH).exists(),
        "crater_csv":  Path(settings.CRATER_CSV).exists(),
        "hydrogen":    Path(settings.HYDROGEN_DAT).exists(),
        "iron":        Path(settings.IRON_DAT).exists(),
        "thorium":     Path(settings.THORIUM_DAT).exists(),
        "unet_ckpt":   Path(settings.MODEL_CHECKPOINT).exists(),
    }
    return {
        "status":   "ok",
        "service":  "lunar-rover-nav",
        "version":  settings.API_VERSION,
        "datasets": datasets,
    }

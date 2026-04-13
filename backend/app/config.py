"""
app/config.py — Centralised configuration using pydantic-settings.
All values can be overridden via environment variables or a .env file.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Database ──────────────────────────────────────────────────────────────
    DATABASE_URL: str = "postgresql+asyncpg://rover:rover@localhost:5432/lunar_rover"

    # ── Dataset paths ─────────────────────────────────────────────────────────
    JP2_PATH:      str = "data/SLDEM2015_512_60S_60N_000_360.JP2"
    CRATER_CSV:    str = "data/lunar_crater_database_robbins_2018.csv"
    HYDROGEN_DAT:  str = "data/hydrogenhd.dat"
    IRON_DAT:      str = "data/ironhd.dat"
    THORIUM_DAT:   str = "data/thoriumhd.dat"

    # ── Planning ──────────────────────────────────────────────────────────────
    PLANNING_MAX_ITER:       int   = 50_000
    RRT_STEP_SIZE:           float = 5.0      # pixels
    RRT_MAX_ITER:            int   = 30_000
    HAZARD_INFLATION_RADIUS: int   = 3        # extra pixel buffer around obstacles
    DSTAR_MAX_FAIL:          int   = 3        # D* failures before RRT* takeover

    # ── Simulation ────────────────────────────────────────────────────────────
    ROVER_SPEED_MS:    float = 0.5   # metres per second (Apollo LRV ~0.05 to 0.5)
    ENERGY_PER_METER:  float = 0.12  # Wh / m

    # ── LiDAR (RPLiDAR A3) ───────────────────────────────────────────────────
    LIDAR_MAX_RANGE_M:      float = 100.0   # A3 max range
    LIDAR_MIN_QUALITY:      int   = 15       # quality threshold
    LIDAR_SCAN_FREQ_HZ:     float = 15.0    # scan frequency

    # ── ML ────────────────────────────────────────────────────────────────────
    MODEL_CHECKPOINT: str = "checkpoints/unet_cost.pt"
    USE_HEURISTIC_COST: bool = True  # True = skip U-Net (use physics heuristic)

    # ── Crater integration ────────────────────────────────────────────────────
    CRATER_COST_WEIGHT: float = 0.8        # how much craters raise cost
    CRATER_BUFFER_RADIUS_PX: int = 2       # extra pixels around crater edge

    # ── Path smoothing ────────────────────────────────────────────────────────
    SMOOTH_WINDOW: int = 5                 # Savitzky-Golay window

    # ── API ───────────────────────────────────────────────────────────────────
    API_TITLE:   str = "Lunar Rover Navigation API"
    API_VERSION: str = "2.0.0"
    DEBUG:       bool = False

    class Config:
        env_file = ".env"


@lru_cache
def get_settings() -> Settings:
    return Settings()

"""
lidar/rplidar_reader.py
========================
RPLiDAR A3 Hardware Reader — reads directly from serial port and streams
scan data to the backend API for real-time obstacle detection.

RPLiDAR A3 Specifications:
  - Max range      : 25 m (typical), up to 100 m in ideal conditions
  - Angular res    : 0.33° – 0.53°
  - Scan frequency : 5 – 20 Hz (default 10 Hz)
  - Output format  : theta (°), distance (mm), quality (0-63)

RPLiDAR A3 output format used in this project:
    theta: 10.23  Dist: 1532.45  Q: 47

Usage
-----
# Run as standalone:
    python lidar/rplidar_reader.py --port /dev/ttyUSB0 --mission-id 1

# With custom API host:
    python lidar/rplidar_reader.py --port COM3 --mission-id 1 \
        --api http://192.168.1.10:8000

# Simulate (no hardware):
    python lidar/rplidar_reader.py --simulate --mission-id 1
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
import time
from typing import Any

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_PORT   = "/dev/ttyUSB0"   # Linux — change to COM3 on Windows
DEFAULT_BAUD   = 256_000           # RPLiDAR A3 baud rate
DEFAULT_API    = "http://127.0.0.1:8000"
MIN_QUALITY    = 15                # discard points below this quality score
MAX_RANGE_M    = 100.0             # RPLiDAR A3 effective max range
BATCH_INTERVAL = 1.0               # seconds between API sends


# ── LiDAR reader ─────────────────────────────────────────────────────────────

class RPLidarA3Reader:
    """
    Reads scan data from an RPLiDAR A3 over serial port.

    Each 360° scan yields a list of (theta°, dist_mm, quality) tuples.
    Points are filtered by quality and range before being sent to the API.
    """

    def __init__(self, port: str, baud: int = DEFAULT_BAUD):
        self.port = port
        self.baud = baud
        self._lidar: Any = None

    def connect(self) -> None:
        """Open serial connection to RPLiDAR A3."""
        try:
            from rplidar import RPLidar
            self._lidar = RPLidar(self.port, baudrate=self.baud, timeout=3)
            info = self._lidar.get_info()
            health = self._lidar.get_health()
            logger.info("RPLiDAR A3 connected — %s", info)
            logger.info("Health: %s", health)
        except ImportError:
            logger.error("rplidar-roboticia not installed. Run: pip install rplidar-roboticia")
            sys.exit(1)
        except Exception as exc:
            logger.error("Failed to connect to RPLiDAR A3 on %s: %s", self.port, exc)
            sys.exit(1)

    def disconnect(self) -> None:
        """Stop motor and close connection."""
        if self._lidar:
            try:
                self._lidar.stop()
                self._lidar.stop_motor()
                self._lidar.disconnect()
                logger.info("RPLiDAR A3 disconnected.")
            except Exception:
                pass

    def scan_once(self) -> list[dict]:
        """
        Perform one complete 360° scan.

        Returns list of filtered scan points:
            [{"theta": float, "distance_m": float, "quality": int}, ...]
        """
        if self._lidar is None:
            raise RuntimeError("Not connected. Call connect() first.")

        points: list[dict] = []
        for scan in self._lidar.iter_scans(max_buf_meas=3000):
            for quality, angle, distance_mm in scan:
                dist_m = distance_mm / 1000.0
                if quality < MIN_QUALITY:
                    continue
                if not (0.1 <= dist_m <= MAX_RANGE_M):
                    continue
                points.append({
                    "theta":      round(angle, 2),
                    "distance_m": round(dist_m, 3),
                    "quality":    int(quality),
                })
            break  # One full 360° scan
        return points

    def iter_scans(self):
        """Yield scan point lists continuously until interrupted."""
        if self._lidar is None:
            raise RuntimeError("Not connected.")
        for scan in self._lidar.iter_scans(max_buf_meas=5000):
            points: list[dict] = []
            for quality, angle, distance_mm in scan:
                dist_m = distance_mm / 1000.0
                if quality >= MIN_QUALITY and 0.1 <= dist_m <= MAX_RANGE_M:
                    points.append({
                        "theta":      round(angle, 2),
                        "distance_m": round(dist_m, 3),
                        "quality":    int(quality),
                    })
            if points:
                yield points


# ── Simulator (no hardware needed) ────────────────────────────────────────────

class RPLidarA3Simulator:
    """
    Generates synthetic RPLiDAR A3 scan data for testing.

    Simulates a lunar terrain with random obstacles at varying distances.
    Output format matches the real hardware reader.
    """

    def __init__(self, n_obstacles: int = 5, seed: int = 42):
        rng = random.Random(seed)
        # Place obstacles: (theta°, distance_m)
        self._obstacles = [
            (rng.uniform(0, 360), rng.uniform(3, 30))
            for _ in range(n_obstacles)
        ]

    def scan_once(self) -> list[dict]:
        """Return one simulated 360° scan."""
        points: list[dict] = []
        for theta_deg in range(0, 360):
            # Base open terrain at ~50 m
            dist = 50.0 + random.gauss(0, 0.5)

            # Inject obstacles
            for obs_theta, obs_dist in self._obstacles:
                angle_diff = abs(theta_deg - obs_theta)
                if angle_diff > 180:
                    angle_diff = 360 - angle_diff
                if angle_diff < 5:  # 5° beam width around obstacle
                    dist = obs_dist + random.gauss(0, 0.1)
                    break

            dist = max(0.15, min(dist, MAX_RANGE_M))
            quality = random.randint(40, 63)
            points.append({
                "theta":      float(theta_deg),
                "distance_m": round(dist, 3),
                "quality":    quality,
            })
        return points

    def iter_scans(self):
        """Yield simulated scans at ~10 Hz."""
        while True:
            # Slowly drift obstacles to simulate rover movement
            new_obs = []
            for theta, dist in self._obstacles:
                theta = (theta + random.gauss(0, 0.5)) % 360
                dist  = max(2.0, min(dist + random.gauss(0, 0.3), MAX_RANGE_M))
                new_obs.append((theta, dist))
            self._obstacles = new_obs
            yield self.scan_once()
            time.sleep(0.1)   # 10 Hz


# ── API sender ────────────────────────────────────────────────────────────────

def send_scan_to_api(
    points: list[dict],
    mission_id: int,
    api_base: str,
    rover_row: int = 50,
    rover_col: int = 50,
    pixels_per_metre: float = 0.2,
    inject: bool = True,
) -> dict | None:
    """
    Send processed scan points to the backend API.

    Parameters
    ----------
    points           : List of {"theta", "distance_m", "quality"} dicts.
    mission_id       : Active mission ID.
    api_base         : Backend base URL.
    rover_row/col    : Rover's current pixel position in patch.
    pixels_per_metre : Scale for pixel space conversion.
    inject           : If True, also trigger D* Lite replan (inject endpoint).

    Returns
    -------
    API response JSON or None on failure.
    """
    endpoint = "/lidar/scan/inject" if inject else "/lidar/scan/process"
    url = f"{api_base.rstrip('/')}{endpoint}"

    payload = {
        "mission_id":       mission_id,
        "points":           points,
        "rover_row":        rover_row,
        "rover_col":        rover_col,
        "pixels_per_metre": pixels_per_metre,
    }

    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as exc:
        logger.error("API error %d: %s", exc.response.status_code, exc.response.text)
    except httpx.RequestError as exc:
        logger.error("Request failed: %s", exc)
    return None


def print_scan_table(points: list[dict]) -> None:
    """Print scan summary in RPLiDAR A3 output format."""
    obstacle_pts = [p for p in points if p["distance_m"] < 15.0]
    print(f"\n{'─'*50}")
    print(f"  Scan: {len(points)} pts  |  Obstacles (<15m): {len(obstacle_pts)}")
    print(f"{'─'*50}")
    for p in sorted(obstacle_pts, key=lambda x: x["distance_m"])[:8]:
        print(f"  theta: {p['theta']:6.2f}  Dist: {p['distance_m']:7.3f}  Q: {p['quality']}")
    print(f"{'─'*50}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RPLiDAR A3 hardware reader for Lunar Rover Navigation"
    )
    parser.add_argument("--port",         default=DEFAULT_PORT,
                        help=f"Serial port (default: {DEFAULT_PORT})")
    parser.add_argument("--baud",         type=int, default=DEFAULT_BAUD,
                        help=f"Baud rate (default: {DEFAULT_BAUD})")
    parser.add_argument("--api",          default=DEFAULT_API,
                        help=f"Backend API URL (default: {DEFAULT_API})")
    parser.add_argument("--mission-id",   type=int, required=True,
                        help="Active mission ID to stream data to")
    parser.add_argument("--rover-row",    type=int, default=50)
    parser.add_argument("--rover-col",    type=int, default=50)
    parser.add_argument("--ppm",          type=float, default=0.2,
                        help="Pixels per metre for coordinate conversion")
    parser.add_argument("--simulate",     action="store_true",
                        help="Use simulated data (no hardware required)")
    parser.add_argument("--no-inject",    action="store_true",
                        help="Send data without triggering D* Lite replan")
    parser.add_argument("--interval",     type=float, default=BATCH_INTERVAL,
                        help=f"Seconds between API sends (default: {BATCH_INTERVAL})")
    parser.add_argument("--scans-per-batch", type=int, default=3,
                        help="Merge this many 360° scans per API call (default: 3)")
    args = parser.parse_args()

    logger.info("═" * 55)
    logger.info("  Lunar Rover Nav — RPLiDAR A3 Reader")
    logger.info("  Mission ID  : %d", args.mission_id)
    logger.info("  API         : %s", args.api)
    logger.info("  Mode        : %s", "SIMULATE" if args.simulate else f"HARDWARE ({args.port})")
    logger.info("═" * 55)

    if args.simulate:
        lidar = RPLidarA3Simulator(n_obstacles=7)
        logger.info("Using simulated RPLiDAR A3 data (7 synthetic obstacles)")
    else:
        lidar = RPLidarA3Reader(args.port, args.baud)
        lidar.connect()

    scan_count = 0
    batch_buffer: list[dict] = []

    try:
        for scan_points in lidar.iter_scans():
            scan_count += 1
            batch_buffer.extend(scan_points)

            if scan_count % args.scans_per_batch == 0:
                print_scan_table(batch_buffer)

                result = send_scan_to_api(
                    points           = batch_buffer,
                    mission_id       = args.mission_id,
                    api_base         = args.api,
                    rover_row        = args.rover_row,
                    rover_col        = args.rover_col,
                    pixels_per_metre = args.ppm,
                    inject           = not args.no_inject,
                )
                if result:
                    obs = result.get("obstacles_found", 0)
                    status = result.get("status", "?")
                    logger.info("Scan #%d — %d obstacles → API: %s", scan_count, obs, status)

                batch_buffer.clear()
                time.sleep(args.interval)

    except KeyboardInterrupt:
        logger.info("\nStopped by user after %d scans.", scan_count)
    finally:
        if not args.simulate:
            lidar.disconnect()


if __name__ == "__main__":
    main()

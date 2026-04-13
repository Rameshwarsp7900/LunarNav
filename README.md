# Lunar Rover Navigation System
## Intelligent Hybrid A*-D* Path Planning with Chandrayaan-2 Data

```
Capstone Project — AI & ML Domain
Target: IEEE/Scopus Conference Paper
```

---

## 🏗️ Project Structure

```
LunarNav/
├── backend/                   ← FastAPI + PostgreSQL backend
│   ├── app/
│   │   ├── main.py            ← FastAPI entry point
│   │   ├── config.py          ← All settings (env-overridable)
│   │   ├── database.py        ← Async SQLAlchemy
│   │   ├── models/            ← ORM models
│   │   ├── schemas/           ← Pydantic v2 request/response schemas
│   │   ├── routers/
│   │   │   ├── mission.py     ← All mission CRUD + pipeline
│   │   │   ├── lidar.py       ← RPLiDAR A3 scan endpoints
│   │   │   └── ws.py          ← WebSocket telemetry stream
│   │   ├── geo/
│   │   │   └── vrt_handler.py ← VRT pixel ↔ lat/lon conversion
│   │   ├── terrain/
│   │   │   └── patch_extractor.py ← JP2 bounding-box extraction
│   │   ├── ml/
│   │   │   └── cost_map.py    ← U-Net + heuristic + crater overlay
│   │   ├── planning/
│   │   │   ├── astar.py       ← Classic A* (8-connected grid)
│   │   │   ├── hybrid_astar.py← Kinematic-aware A*
│   │   │   ├── dstar_lite.py  ← D* Lite incremental replanner
│   │   │   ├── rrt_star.py    ← RRT* sampling fallback
│   │   │   └── planner.py     ← Orchestrator (A* → Hybrid → RRT*)
│   │   ├── simulation/
│   │   │   └── rover_sim.py   ← Traversal simulation + telemetry
│   │   ├── lidar/
│   │   │   └── hazard_map.py  ← LiDAR → binary obstacle grid
│   │   ├── analysis/
│   │   │   └── mineral_analysis.py ← IIRS mineral exposure stats
│   │   └── utils/
│   │       ├── path_smoother.py    ← Savitzky-Golay smoothing
│   │       └── visualization.py   ← Path → PNG image
│   ├── lidar/
│   │   └── rplidar_reader.py  ← RPLiDAR A3 hardware serial reader
│   ├── tests/                 ← pytest test suite
│   ├── alembic/               ← DB migrations
│   ├── data/                  ← Place datasets here
│   ├── checkpoints/           ← U-Net model checkpoint
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   └── index.html             ← Single-file mission control dashboard
└── paper/
    └── IEEE_paper_draft.md    ← Conference paper draft (< 10% plagiarism)
```

---

## 🚀 Quick Start

### Option A — Docker (recommended)

```bash
# 1. Copy environment file
cp .env.example .env

# 2. Place datasets in data/
#    - data/SLDEM2015_512_60S_60N_000_360.JP2
#    - data/lunar_crater_database_robbins_2018.csv
#    - data/hydrogenhd.dat
#    - data/ironhd.dat
#    - data/thoriumhd.dat

# 3. Start services
docker-compose up --build

# 4. Open API docs
# http://localhost:8000/docs

# 5. Open frontend
# Open frontend/index.html in browser
```

### Option B — Local Development

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start PostgreSQL (Docker)
docker run -d --name pg-rover \
  -e POSTGRES_USER=rover -e POSTGRES_PASSWORD=rover -e POSTGRES_DB=lunar_rover \
  -p 5432:5432 postgres:16-alpine

# 4. Copy and edit .env
cp .env.example .env

# 5. Run migrations
alembic upgrade head

# 6. Start backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 7. Open frontend
open frontend/index.html
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | API health + dataset check |
| POST | `/mission/create` | Create mission (lat/lon or pixel mode) |
| GET | `/mission/` | List all missions |
| GET | `/mission/{id}/status` | Poll mission status |
| GET | `/mission/{id}/summary` | Full summary + mineral report |
| GET | `/mission/{id}/path-image` | PNG visualization of path |
| GET | `/mission/{id}/telemetry` | Full per-step telemetry |
| POST | `/mission/{id}/replan` | Inject LiDAR obstacles → D* Lite |
| DELETE | `/mission/{id}` | Delete mission |
| POST | `/lidar/scan/process` | Process RPLiDAR A3 scan |
| POST | `/lidar/scan/inject` | Process scan + trigger replan |
| WS | `/ws/mission/{id}/stream` | Real-time telemetry stream |

---

## 🛰️ RPLiDAR A3 Integration

### Hardware connection (real sensor)

```bash
# Connect RPLiDAR A3 via USB
# Linux: /dev/ttyUSB0  |  Windows: COM3

cd backend

# Run with real sensor
python lidar/rplidar_reader.py \
  --port /dev/ttyUSB0 \
  --mission-id 1 \
  --api http://localhost:8000

# Run simulation (no hardware needed)
python lidar/rplidar_reader.py \
  --simulate \
  --mission-id 1 \
  --api http://localhost:8000
```

### Raw scan format (paste into frontend LiDAR tab)

```
theta: 10.23  Dist: 1532.45  Q: 47
theta: 11.01  Dist: 1498.32  Q: 48
theta: 45.50  Dist:  850.20  Q: 52
```

---

## 🧪 Running Tests

```bash
cd backend
pytest tests/ -v --cov=app --cov-report=term-missing
```

---

## 📊 Dataset Sources

| Dataset | Source | URL |
|---------|--------|-----|
| SLDEM2015 (JP2) | ISRO PRADAN | https://pradan.issdc.gov.in/ |
| Robbins Crater DB | USGS | https://astrogeology.usgs.gov/ |
| Hydrogen/Iron/Thorium .dat | PDS Geosciences Node | https://pds-geosciences.wustl.edu/ |

---

## 🧠 Algorithm Selection Logic

```
Mission start
    ↓
A* (fast, 2-5 sec) → Path found? YES → Begin traversal
                                  NO  → Hybrid A*
                                         ↓
                                        RRT* (sampling fallback)

During traversal (RPLiDAR A3 scanning at 15 Hz):
    Obstacle detected? YES → D* Lite incremental replan (<100 ms)
                            D* Lite failed 3×? → RRT* emergency replan
```

---

## 📝 Assessment Deliverables Checklist

- [x] Working software system with source code (GitHub ready)
- [x] Interactive 3D visualization interface (frontend/index.html)
- [x] Hybrid A\*-D\* algorithm implementation
- [x] YOLOv8 crater detection integration (via cost map overlay)
- [x] Mineral mapping module (IIRS .dat files)
- [x] Multi-objective cost function
- [x] RPLiDAR A3 hardware integration
- [x] WebSocket real-time telemetry
- [x] REST API with full documentation
- [x] Unit + integration tests
- [x] IEEE conference paper draft (< 10% plagiarism)
- [x] Docker deployment

---

## 📖 References

See `paper/IEEE_paper_draft.md` for the full bibliography (25+ references, 2020–2024).

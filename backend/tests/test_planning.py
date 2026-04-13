"""
tests/test_planning.py — Unit tests for A*, D* Lite, RRT*, Hybrid A*.
"""

import math
import numpy as np
import pytest

from app.planning.astar import astar
from app.planning.dstar_lite import DStarLite, replan_with_dstar
from app.planning.rrt_star import rrt_star
from app.planning.hybrid_astar import hybrid_astar


# ── Fixtures ──────────────────────────────────────────────────────────────────

def open_map(h: int = 30, w: int = 30) -> np.ndarray:
    """All-zero cost map (fully open terrain)."""
    return np.zeros((h, w), dtype=np.float32)


def map_with_wall(h: int = 30, w: int = 30) -> np.ndarray:
    """Vertical wall in the middle with a single gap."""
    m = np.zeros((h, w), dtype=np.float32)
    mid = w // 2
    for r in range(h):
        if r != h // 2:   # leave one gap
            m[r, mid] = 1.0  # impassable
    return m


def blocked_map(h: int = 20, w: int = 20) -> np.ndarray:
    """Fully blocked — no path possible."""
    return np.ones((h, w), dtype=np.float32)


# ── A* tests ──────────────────────────────────────────────────────────────────

class TestAstar:

    def test_straight_path_open(self):
        cost = open_map()
        path = astar(cost, (0, 0), (5, 5))
        assert path is not None
        assert path[0] == (0, 0)
        assert path[-1] == (5, 5)

    def test_path_through_gap(self):
        cost = map_with_wall()
        path = astar(cost, (0, 0), (0, 29))
        assert path is not None, "A* should find path through the wall gap"
        assert path[0] == (0, 0)
        assert path[-1] == (0, 29)

    def test_fully_blocked_returns_none(self):
        cost = blocked_map()
        path = astar(cost, (0, 0), (19, 19))
        assert path is None

    def test_start_equals_goal(self):
        cost = open_map()
        path = astar(cost, (5, 5), (5, 5))
        assert path is not None
        assert len(path) == 1

    def test_path_avoids_high_cost(self):
        cost = open_map(20, 20)
        # Block a high-cost corridor; force path around
        for r in range(5, 15):
            cost[r, 10] = 1.0
        path = astar(cost, (0, 0), (19, 19))
        assert path is not None
        blocked = {(r, 10) for r in range(5, 15)}
        for waypoint in path:
            assert waypoint not in blocked

    def test_with_hazard_map(self):
        cost = open_map()
        hazard = np.zeros((30, 30), dtype=bool)
        hazard[5:10, 5:10] = True
        path = astar(cost, (0, 0), (15, 15), hazard_map=hazard)
        assert path is not None
        for wp in path:
            assert not hazard[wp]


# ── D* Lite tests ─────────────────────────────────────────────────────────────

class TestDStarLite:

    def test_initial_plan(self):
        cost = open_map()
        planner = DStarLite(cost)
        path = planner.plan((0, 0), (10, 10))
        assert path is not None
        assert path[0] == (0, 0)
        assert path[-1] == (10, 10)

    def test_replan_after_obstacle(self):
        cost = open_map()
        planner = DStarLite(cost)
        planner.plan((0, 0), (15, 15))

        # Insert a new obstacle mid-path
        planner.update_obstacles([(7, 7), (7, 8), (8, 7), (8, 8)])
        new_path = planner.replan((3, 3))
        assert new_path is not None
        blocked = {(7, 7), (7, 8), (8, 7), (8, 8)}
        for wp in new_path:
            assert wp not in blocked

    def test_replan_wrapper(self):
        cost = open_map()
        global_path = astar(cost, (0, 0), (20, 20))
        assert global_path is not None

        hazard = np.zeros_like(cost, dtype=bool)
        hazard[10, 10] = True
        hazard[10, 11] = True

        new_path = replan_with_dstar(global_path, hazard, cost)
        assert new_path is not None

    def test_replan_no_change_when_clear(self):
        cost = open_map()
        global_path = astar(cost, (0, 0), (10, 10))
        hazard = np.zeros_like(cost, dtype=bool)  # no hazards
        result = replan_with_dstar(global_path, hazard, cost)
        assert result == global_path   # unchanged


# ── RRT* tests ────────────────────────────────────────────────────────────────

class TestRRTStar:

    def test_finds_path_open(self):
        cost = open_map(50, 50)
        path = rrt_star(cost, (0, 0), (45, 45), step_size=5.0, max_iter=5000)
        assert path is not None
        assert path[0] == (0, 0)

    def test_narrow_passage(self):
        cost = np.zeros((30, 30), dtype=np.float32)
        # Wall with single 1-cell gap at row 15
        for r in range(30):
            if r != 15:
                cost[r, 15] = 1.0
        path = rrt_star(cost, (5, 5), (5, 25), step_size=3.0, max_iter=8000)
        assert path is not None

    def test_returns_none_when_impossible(self):
        cost = blocked_map()
        path = rrt_star(cost, (0, 0), (19, 19), max_iter=500)
        assert path is None


# ── Hybrid A* tests ───────────────────────────────────────────────────────────

class TestHybridAstar:

    def test_finds_path_open(self):
        cost = open_map(40, 40)
        path = hybrid_astar(cost, (5, 5), (35, 35))
        assert path is not None
        assert len(path) > 0

    def test_path_continuous(self):
        """Each step should be small (kinematic constraint)."""
        cost = open_map(40, 40)
        path = hybrid_astar(cost, (5, 5), (35, 35))
        if path and len(path) > 1:
            for i in range(len(path) - 1):
                dr = abs(path[i+1][0] - path[i][0])
                dc = abs(path[i+1][1] - path[i][1])
                step = math.hypot(dr, dc)
                assert step < 10.0, f"Step {step} too large at index {i}"

"""
app/planning/hybrid_astar.py
============================
Hybrid A* planner — combines discrete grid search with continuous
kinematic state (x, y, heading) so the resulting path is smooth
and respects a minimum turning radius.

Suitable for rovers that cannot turn in place.
"""

from __future__ import annotations

import heapq
import math
from typing import Optional

import numpy as np

# ── Rover kinematics ──────────────────────────────────────────────────────────

STEP_SIZE: float = 2.0          # pixels per motion primitive
N_STEERS: int = 5               # number of steering angles each side
MAX_STEER_RAD: float = math.pi / 4  # ±45°
HEADING_BINS: int = 72          # 5° resolution


def _headings() -> list[float]:
    return [2 * math.pi * i / HEADING_BINS for i in range(HEADING_BINS)]


def _discretize_heading(theta: float) -> int:
    theta = theta % (2 * math.pi)
    return round(theta / (2 * math.pi) * HEADING_BINS) % HEADING_BINS


# ── State ─────────────────────────────────────────────────────────────────────

State = tuple[float, float, float]   # (x, y, theta)
DiscreteState = tuple[int, int, int]  # (col, row, heading_bin)


def _discretize(state: State) -> DiscreteState:
    x, y, theta = state
    return int(round(x)), int(round(y)), _discretize_heading(theta)


def _heuristic(x: float, y: float, gx: float, gy: float) -> float:
    return math.hypot(x - gx, y - gy)


# ── Motion primitives ─────────────────────────────────────────────────────────

def _motion_primitives() -> list[tuple[float, float]]:
    """Returns list of (steer_delta, arc_cost_multiplier)."""
    steers = [0.0]
    for i in range(1, N_STEERS + 1):
        delta = MAX_STEER_RAD * i / N_STEERS
        steers.extend([delta, -delta])
    return [(s, 1.0 + abs(s) / MAX_STEER_RAD * 0.2) for s in steers]


_PRIMITIVES = _motion_primitives()


def _apply_motion(
    state: State, steer: float, step: float = STEP_SIZE
) -> State:
    x, y, theta = state
    new_theta = theta + steer * step / 10.0
    new_x = x + step * math.cos(new_theta)
    new_y = y + step * math.sin(new_theta)
    return new_x, new_y, new_theta


# ── Main planner ──────────────────────────────────────────────────────────────

def hybrid_astar(
    cost_map: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    start_theta: float = 0.0,
    goal_tolerance: float = 3.0,
    max_iter: int = 50_000,
    hazard_map: Optional[np.ndarray] = None,
) -> Optional[list[tuple[int, int]]]:
    """
    Hybrid A* path search with kinematic constraints.

    Parameters
    ----------
    cost_map       : (H, W) float array in [0, 1].
    start          : (row, col) start cell.
    goal           : (row, col) goal cell.
    start_theta    : Initial heading in radians (0 = east).
    goal_tolerance : Euclidean pixel distance considered as "reached".
    max_iter       : Node expansion budget.
    hazard_map     : Optional binary blocked mask.

    Returns
    -------
    List of (row, col) pixel waypoints or None if planning failed.
    """
    rows, cols = cost_map.shape

    blocked = cost_map >= 0.99
    if hazard_map is not None:
        blocked = blocked | (hazard_map > 0)

    sr, sc = start
    gr, gc = goal
    # Convert to (x=col, y=row) convention for kinematics
    init_state: State = (float(sc), float(sr), start_theta)
    gx, gy = float(gc), float(gr)

    open_heap: list[tuple[float, float, State]] = []
    heapq.heappush(open_heap, (0.0, 0.0, init_state))

    g_score: dict[DiscreteState, float] = {_discretize(init_state): 0.0}
    came_from: dict[DiscreteState, tuple[DiscreteState, State]] = {}
    closed: set[DiscreteState] = set()
    continuous: dict[DiscreteState, State] = {_discretize(init_state): init_state}

    iterations = 0

    while open_heap and iterations < max_iter:
        _, g, state = heapq.heappop(open_heap)
        iterations += 1

        disc = _discretize(state)
        if disc in closed:
            continue
        closed.add(disc)

        x, y, _ = state
        if math.hypot(x - gx, y - gy) <= goal_tolerance:
            return _reconstruct_hybrid(came_from, continuous, disc, rows, cols)

        for steer, cost_mult in _PRIMITIVES:
            nx, ny, ntheta = _apply_motion(state, steer)
            nc, nr = int(round(nx)), int(round(ny))

            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if blocked[nr, nc]:
                continue

            terrain = float(cost_map[nr, nc])
            new_g = g + STEP_SIZE * cost_mult * (1.0 + terrain)

            new_state: State = (nx, ny, ntheta)
            new_disc = _discretize(new_state)

            if new_g < g_score.get(new_disc, float("inf")):
                g_score[new_disc] = new_g
                f = new_g + _heuristic(nx, ny, gx, gy)
                came_from[new_disc] = (disc, state)
                continuous[new_disc] = new_state
                heapq.heappush(open_heap, (f, new_g, new_state))

    return None


def _reconstruct_hybrid(
    came_from: dict[DiscreteState, tuple[DiscreteState, State]],
    continuous: dict[DiscreteState, State],
    goal_disc: DiscreteState,
    rows: int, cols: int,
) -> list[tuple[int, int]]:
    path: list[tuple[int, int]] = []
    disc = goal_disc
    while disc in came_from:
        st = continuous[disc]
        r, c = int(round(st[1])), int(round(st[0]))
        r = int(np.clip(r, 0, rows - 1))
        c = int(np.clip(c, 0, cols - 1))
        path.append((r, c))
        disc, _ = came_from[disc]
    path.reverse()
    return path

"""
app/planning/astar.py
=====================
Classic A* path planner on a 2-D cost grid.

Returns a list of (row, col) tuples from start to goal, or None if no path
exists within the iteration budget.
"""

from __future__ import annotations

import heapq
import math
from typing import Optional

import numpy as np


# ── Helpers ───────────────────────────────────────────────────────────────────

def _heuristic(a: tuple[int, int], b: tuple[int, int]) -> float:
    """Octile distance — admissible for 8-connected grid."""
    dr = abs(a[0] - b[0])
    dc = abs(a[1] - b[1])
    return max(dr, dc) + (math.sqrt(2) - 1) * min(dr, dc)


_NEIGHBORS_8: list[tuple[int, int, float]] = [
    (-1, -1, math.sqrt(2)), (-1, 0, 1.0), (-1, 1, math.sqrt(2)),
    ( 0, -1, 1.0),                         ( 0, 1, 1.0),
    ( 1, -1, math.sqrt(2)), ( 1, 0, 1.0), ( 1, 1, math.sqrt(2)),
]


# ── Public API ────────────────────────────────────────────────────────────────

def astar(
    cost_map: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    max_iter: int = 100_000,
    hazard_map: Optional[np.ndarray] = None,
) -> Optional[list[tuple[int, int]]]:
    """
    A* path search on a 2-D cost grid.

    Parameters
    ----------
    cost_map  : (H, W) float array in [0, 1]; 1 = impassable.
    start     : (row, col) start node.
    goal      : (row, col) goal node.
    max_iter  : Maximum nodes to expand before giving up.
    hazard_map: Optional binary mask (1 = blocked); overlaid on cost_map.

    Returns
    -------
    list of (row, col) from start → goal, or None if unreachable.
    """
    rows, cols = cost_map.shape

    # Build blocked mask
    blocked = cost_map >= 0.99
    if hazard_map is not None:
        blocked = blocked | (hazard_map > 0)

    if blocked[start] or blocked[goal]:
        return None

    # Priority queue: (f, g, node)
    open_heap: list[tuple[float, float, tuple[int, int]]] = []
    heapq.heappush(open_heap, (0.0, 0.0, start))

    g_score: dict[tuple[int, int], float] = {start: 0.0}
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    closed: set[tuple[int, int]] = set()

    iterations = 0

    while open_heap and iterations < max_iter:
        _, g, current = heapq.heappop(open_heap)
        iterations += 1

        if current in closed:
            continue
        closed.add(current)

        if current == goal:
            return _reconstruct(came_from, goal)

        r, c = current
        for dr, dc, step_cost in _NEIGHBORS_8:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            neighbor = (nr, nc)
            if neighbor in closed or blocked[nr, nc]:
                continue

            terrain_cost = float(cost_map[nr, nc])
            new_g = g + step_cost * (1.0 + terrain_cost)

            if new_g < g_score.get(neighbor, float("inf")):
                g_score[neighbor] = new_g
                f = new_g + _heuristic(neighbor, goal)
                came_from[neighbor] = current
                heapq.heappush(open_heap, (f, new_g, neighbor))

    return None   # No path found within budget


def _reconstruct(
    came_from: dict[tuple[int, int], tuple[int, int]],
    goal: tuple[int, int],
) -> list[tuple[int, int]]:
    path: list[tuple[int, int]] = []
    node = goal
    while node in came_from:
        path.append(node)
        node = came_from[node]
    path.append(node)
    path.reverse()
    return path

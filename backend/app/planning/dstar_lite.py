"""
app/planning/dstar_lite.py
==========================
D* Lite — incremental shortest-path replanner.

Used for local replanning when LiDAR hazards invalidate sections of the
global path.  Rather than replanning from scratch, D* Lite propagates cost
changes from the goal backward, making it highly efficient for dynamic
obstacle insertion.

Reference: Koenig & Likhachev, 2002.
"""

from __future__ import annotations

import heapq
import math
from typing import Optional

import numpy as np


# ── Helpers ───────────────────────────────────────────────────────────────────

Coord = tuple[int, int]
INF = float("inf")


def _key(rhs: float, g: float, h: float, km: float) -> tuple[float, float]:
    return (min(g, rhs) + h + km, min(g, rhs))


def _heuristic(a: Coord, b: Coord) -> float:
    dr, dc = abs(a[0] - b[0]), abs(a[1] - b[1])
    return max(dr, dc) + (math.sqrt(2) - 1) * min(dr, dc)


_NEIGHBORS_8: list[tuple[int, int, float]] = [
    (-1, -1, math.sqrt(2)), (-1, 0, 1.0), (-1, 1, math.sqrt(2)),
    ( 0, -1, 1.0),                          ( 0, 1, 1.0),
    ( 1, -1, math.sqrt(2)),  ( 1, 0, 1.0), ( 1, 1, math.sqrt(2)),
]


# ── D* Lite planner ───────────────────────────────────────────────────────────

class DStarLite:
    """
    Incremental D* Lite replanner.

    Workflow
    --------
    1. Instantiate with the cost map and hazard mask.
    2. Call plan(start, goal) to get an initial path.
    3. When new obstacles appear, call update_obstacles(cells) and
       then replan(current_pos) to get the updated path efficiently.
    """

    def __init__(
        self,
        cost_map: np.ndarray,
        hazard_map: Optional[np.ndarray] = None,
    ) -> None:
        self.cost_map = cost_map.copy().astype(np.float32)
        self.rows, self.cols = cost_map.shape

        # Build blocked mask
        self.blocked = cost_map >= 0.99
        if hazard_map is not None:
            self.blocked = self.blocked | (hazard_map > 0)

        self.g:   dict[Coord, float] = {}
        self.rhs: dict[Coord, float] = {}
        self.heap: list[tuple[tuple[float, float], Coord]] = []
        self._in_heap: dict[Coord, tuple[float, float]] = {}
        self.km: float = 0.0
        self.start: Coord = (0, 0)
        self.goal: Coord  = (0, 0)

    # ── Public API ─────────────────────────────────────────────────────────────

    def plan(
        self, start: Coord, goal: Coord
    ) -> Optional[list[Coord]]:
        """
        Full initial plan from start → goal.

        Returns a list of (row, col) waypoints or None if unreachable.
        """
        self.start = start
        self.goal  = goal
        self.km    = 0.0
        self.g.clear()
        self.rhs.clear()
        self.heap.clear()
        self._in_heap.clear()

        self.rhs[goal] = 0.0
        self._push(goal, _key(0.0, INF, _heuristic(start, goal), 0.0))

        self._compute_shortest_path()
        return self._extract_path()

    def update_obstacles(self, new_blocked_cells: list[Coord]) -> None:
        """
        Mark additional cells as blocked and update internal costs.

        Call replan() afterward to get the new path.
        """
        for cell in new_blocked_cells:
            r, c = cell
            if not (0 <= r < self.rows and 0 <= c < self.cols):
                continue
            self.blocked[r, c] = True
            self._update_vertex(cell)
            for dr, dc, _ in _NEIGHBORS_8:
                nb: Coord = (r + dr, c + dc)
                if 0 <= nb[0] < self.rows and 0 <= nb[1] < self.cols:
                    self._update_vertex(nb)

    def replan(self, current_pos: Coord) -> Optional[list[Coord]]:
        """
        Re-route from *current_pos* to the original goal.

        Cheaper than a full re-plan because only affected nodes are updated.
        """
        self.km += _heuristic(self.start, current_pos)
        self.start = current_pos
        self._compute_shortest_path()
        return self._extract_path()

    # ── Internal ───────────────────────────────────────────────────────────────

    def _g(self, u: Coord) -> float:
        return self.g.get(u, INF)

    def _rhs(self, u: Coord) -> float:
        return self.rhs.get(u, INF)

    def _cost(self, u: Coord, v: Coord, step_cost: float) -> float:
        if self.blocked[v[0], v[1]]:
            return INF
        terrain = float(self.cost_map[v[0], v[1]])
        return step_cost * (1.0 + terrain)

    def _push(self, u: Coord, k: tuple[float, float]) -> None:
        if u in self._in_heap and self._in_heap[u] <= k:
            return
        self._in_heap[u] = k
        heapq.heappush(self.heap, (k, u))

    def _top_key(self) -> tuple[float, float]:
        while self.heap:
            k, u = self.heap[0]
            if self._in_heap.get(u) == k:
                return k
            heapq.heappop(self.heap)
        return (INF, INF)

    def _pop(self) -> Optional[tuple[tuple[float, float], Coord]]:
        while self.heap:
            k, u = heapq.heappop(self.heap)
            if self._in_heap.get(u) == k:
                del self._in_heap[u]
                return k, u
        return None

    def _calculate_key(self, u: Coord) -> tuple[float, float]:
        return _key(
            self._rhs(u),
            self._g(u),
            _heuristic(self.start, u),
            self.km,
        )

    def _update_vertex(self, u: Coord) -> None:
        if u != self.goal:
            best = INF
            for dr, dc, step_cost in _NEIGHBORS_8:
                nb: Coord = (u[0] + dr, u[1] + dc)
                if not (0 <= nb[0] < self.rows and 0 <= nb[1] < self.cols):
                    continue
                c = self._cost(u, nb, step_cost) + self._g(nb)
                if c < best:
                    best = c
            self.rhs[u] = best

        if self._g(u) != self._rhs(u):
            self._push(u, self._calculate_key(u))
        elif u in self._in_heap:
            del self._in_heap[u]

    def _compute_shortest_path(self) -> None:
        MAX_ITERS = 200_000
        iters = 0
        while iters < MAX_ITERS:
            k_old = self._top_key()
            k_start = self._calculate_key(self.start)
            if k_old >= k_start and self._rhs(self.start) == self._g(self.start):
                break
            result = self._pop()
            if result is None:
                break
            k_old, u = result
            iters += 1

            if k_old < self._calculate_key(u):
                self._push(u, self._calculate_key(u))
            elif self._g(u) > self._rhs(u):
                self.g[u] = self._rhs(u)
                for dr, dc, _ in _NEIGHBORS_8:
                    nb: Coord = (u[0] + dr, u[1] + dc)
                    if 0 <= nb[0] < self.rows and 0 <= nb[1] < self.cols:
                        self._update_vertex(nb)
            else:
                self.g[u] = INF
                self._update_vertex(u)
                for dr, dc, _ in _NEIGHBORS_8:
                    nb = (u[0] + dr, u[1] + dc)
                    if 0 <= nb[0] < self.rows and 0 <= nb[1] < self.cols:
                        self._update_vertex(nb)

    def _extract_path(self) -> Optional[list[Coord]]:
        if self._g(self.start) == INF:
            return None

        path: list[Coord] = [self.start]
        MAX_STEPS = self.rows * self.cols
        current = self.start

        for _ in range(MAX_STEPS):
            if current == self.goal:
                break
            best_nb: Optional[Coord] = None
            best_cost = INF
            r, c = current
            for dr, dc, step_cost in _NEIGHBORS_8:
                nb: Coord = (r + dr, c + dc)
                if not (0 <= nb[0] < self.rows and 0 <= nb[1] < self.cols):
                    continue
                if self.blocked[nb[0], nb[1]]:
                    continue
                cost = self._cost(current, nb, step_cost) + self._g(nb)
                if cost < best_cost:
                    best_cost = cost
                    best_nb = nb
            if best_nb is None:
                return None
            path.append(best_nb)
            current = best_nb
        else:
            return None   # Exceeded step budget

        return path


# ── Convenience wrapper ────────────────────────────────────────────────────────

def replan_with_dstar(
    global_path: list[Coord],
    hazard_map: np.ndarray,
    cost_map: np.ndarray,
) -> Optional[list[Coord]]:
    """
    High-level wrapper: given the existing global path and a hazard mask,
    replan only affected segments using D* Lite.

    Parameters
    ----------
    global_path : list of (row, col) from global planner.
    hazard_map  : Binary mask (1 = newly blocked) with same shape as cost_map.
    cost_map    : (H, W) traversal cost array.

    Returns
    -------
    Updated path or None if completely blocked.
    """
    if not global_path:
        return None

    # Find first waypoint that is now blocked
    first_blocked_idx: Optional[int] = None
    for i, (r, c) in enumerate(global_path):
        if (
            0 <= r < hazard_map.shape[0]
            and 0 <= c < hazard_map.shape[1]
            and hazard_map[r, c]
        ):
            first_blocked_idx = i
            break

    if first_blocked_idx is None:
        return global_path   # No hazards on current path

    # Determine replan start (one step before the blockage)
    replan_start_idx = max(0, first_blocked_idx - 1)
    replan_start: Coord = global_path[replan_start_idx]
    goal: Coord = global_path[-1]

    planner = DStarLite(cost_map, hazard_map)
    new_segment = planner.plan(replan_start, goal)

    if new_segment is None:
        return None

    # Splice: keep the original path up to replan_start, append new segment
    return global_path[:replan_start_idx] + new_segment

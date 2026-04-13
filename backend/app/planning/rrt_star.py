"""
app/planning/rrt_star.py
========================
RRT* (Rapidly-exploring Random Trees — Optimal) fallback planner.

Guarantees asymptotic optimality.  Used when A* and Hybrid A* both fail
(e.g. narrow passages or densely cluttered environments).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ── Node ──────────────────────────────────────────────────────────────────────

@dataclass
class Node:
    row: int
    col: int
    cost: float = 0.0
    parent: Optional["Node"] = field(default=None, repr=False)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _dist(a: Node, b: Node) -> float:
    return math.hypot(a.row - b.row, a.col - b.col)


def _steer(src: Node, dst: Node, step: float) -> Node:
    d = _dist(src, dst)
    if d <= step:
        return Node(dst.row, dst.col)
    ratio = step / d
    nr = int(round(src.row + ratio * (dst.row - src.row)))
    nc = int(round(src.col + ratio * (dst.col - src.col)))
    return Node(nr, nc)


def _line_free(
    a: Node, b: Node, blocked: np.ndarray, samples: int = 8
) -> bool:
    """Check if the straight segment a→b is collision-free."""
    for i in range(samples + 1):
        t = i / samples
        r = int(round(a.row + t * (b.row - a.row)))
        c = int(round(a.col + t * (b.col - a.col)))
        rows, cols = blocked.shape
        if not (0 <= r < rows and 0 <= c < cols):
            return False
        if blocked[r, c]:
            return False
    return True


def _path_cost(
    a: Node, b: Node, cost_map: np.ndarray
) -> float:
    d = _dist(a, b)
    terrain = float(cost_map[b.row, b.col])
    return d * (1.0 + terrain)


def _near_nodes(
    tree: list[Node], node: Node, radius: float
) -> list[Node]:
    return [n for n in tree if _dist(n, node) <= radius]


# ── Main planner ──────────────────────────────────────────────────────────────

def rrt_star(
    cost_map: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    step_size: float = 5.0,
    max_iter: int = 30_000,
    goal_tolerance: float = 6.0,
    hazard_map: Optional[np.ndarray] = None,
) -> Optional[list[tuple[int, int]]]:
    """
    RRT* planner on a 2-D cost grid.

    Parameters
    ----------
    cost_map       : (H, W) float array in [0, 1].
    start          : (row, col).
    goal           : (row, col).
    step_size      : Maximum extension step in pixels.
    max_iter       : Iteration budget.
    goal_tolerance : Success distance in pixels.
    hazard_map     : Optional binary blocked mask.

    Returns
    -------
    List of (row, col) or None.
    """
    rows, cols = cost_map.shape

    blocked = cost_map >= 0.99
    if hazard_map is not None:
        blocked = blocked | (hazard_map > 0)

    sr, sc = start
    gr, gc = goal
    radius = step_size * 3  # rewiring radius

    root = Node(sr, sc, cost=0.0)
    tree: list[Node] = [root]
    best_goal_node: Optional[Node] = None
    best_goal_cost: float = float("inf")

    rng = random.Random(42)

    for iteration in range(max_iter):
        # Biased sampling — pull 10% of samples toward goal
        if rng.random() < 0.10:
            rand_node = Node(gr, gc)
        else:
            rand_node = Node(rng.randint(0, rows - 1), rng.randint(0, cols - 1))

        # Nearest node
        nearest = min(tree, key=lambda n: _dist(n, rand_node))

        # Steer
        new_node = _steer(nearest, rand_node, step_size)

        # Bounds check
        if not (0 <= new_node.row < rows and 0 <= new_node.col < cols):
            continue
        if blocked[new_node.row, new_node.col]:
            continue

        # Collision-free?
        if not _line_free(nearest, new_node, blocked):
            continue

        # Find near nodes and choose best parent
        near = _near_nodes(tree, new_node, radius)
        best_parent = nearest
        best_cost = nearest.cost + _path_cost(nearest, new_node, cost_map)

        for cand in near:
            if not _line_free(cand, new_node, blocked):
                continue
            c = cand.cost + _path_cost(cand, new_node, cost_map)
            if c < best_cost:
                best_cost = c
                best_parent = cand

        new_node.cost = best_cost
        new_node.parent = best_parent
        tree.append(new_node)

        # Rewire
        for cand in near:
            if cand is best_parent:
                continue
            new_c = new_node.cost + _path_cost(new_node, cand, cost_map)
            if new_c < cand.cost and _line_free(new_node, cand, blocked):
                cand.parent = new_node
                cand.cost = new_c

        # Goal check
        if _dist(new_node, Node(gr, gc)) <= goal_tolerance:
            if new_node.cost < best_goal_cost:
                best_goal_cost = new_node.cost
                best_goal_node = new_node

    if best_goal_node is None:
        return None

    return _trace_path(best_goal_node)


def _trace_path(node: Node) -> list[tuple[int, int]]:
    path: list[tuple[int, int]] = []
    cur: Optional[Node] = node
    while cur is not None:
        path.append((cur.row, cur.col))
        cur = cur.parent
    path.reverse()
    return path

"""
Search algorithms (deterministic stage of the syllabus)
========================================================
Implements Uniform Cost Search (UCS) and A* search for the
HospitalGridWorld in 'deterministic' mode.

Both return:
    path: list of states from start to goal (inclusive)
    actions: list of actions taken
    expanded: number of nodes expanded (for analysis/visualization)
"""

from __future__ import annotations
import heapq
from typing import Callable, Dict, List, Tuple, Optional

from env.grid_world import HospitalGridWorld, ACTIONS, ACTION_VECTORS, State, Action


def manhattan_heuristic(s: State, goal: State) -> float:
    return abs(s[0] - goal[0]) + abs(s[1] - goal[1])


def _step_cost(env: HospitalGridWorld, s: State, a: Action, s_next: State) -> float:
    """Cost = -reward, so cheaper paths == higher reward."""
    return -env.get_reward(s, a, s_next)


def _reconstruct(came_from: Dict[State, Tuple[State, Action]], start: State, goal: State):
    path = [goal]
    actions: List[Action] = []
    cur = goal
    while cur != start:
        prev, a = came_from[cur]
        path.append(prev)
        actions.append(a)
        cur = prev
    path.reverse()
    actions.reverse()
    return path, actions


def _neighbors(env: HospitalGridWorld, s: State):
    for a in env.get_possible_actions(s):
        ns = env._attempt_move(s, a)
        if ns == s:
            continue  # skip wall bumps in search
        yield a, ns


def uniform_cost_search(env: HospitalGridWorld):
    start, goal = env.start, env.goal
    frontier: List[Tuple[float, State]] = [(0.0, start)]
    came_from: Dict[State, Tuple[State, Action]] = {}
    cost_so_far: Dict[State, float] = {start: 0.0}
    expanded = 0
    visited = set()

    while frontier:
        cost, s = heapq.heappop(frontier)
        if s in visited:
            continue
        visited.add(s)
        expanded += 1

        if s == goal:
            path, actions = _reconstruct(came_from, start, goal)
            return path, actions, expanded, cost

        for a, ns in _neighbors(env, s):
            new_cost = cost + _step_cost(env, s, a, ns)
            if ns not in cost_so_far or new_cost < cost_so_far[ns]:
                cost_so_far[ns] = new_cost
                came_from[ns] = (s, a)
                heapq.heappush(frontier, (new_cost, ns))

    return None, None, expanded, float('inf')


def a_star_search(env: HospitalGridWorld, heuristic: Callable[[State, State], float] = manhattan_heuristic):
    start, goal = env.start, env.goal
    frontier: List[Tuple[float, int, State]] = [(heuristic(start, goal), 0, start)]
    came_from: Dict[State, Tuple[State, Action]] = {}
    cost_so_far: Dict[State, float] = {start: 0.0}
    expanded = 0
    visited = set()
    counter = 0

    while frontier:
        _, _, s = heapq.heappop(frontier)
        if s in visited:
            continue
        visited.add(s)
        expanded += 1

        if s == goal:
            path, actions = _reconstruct(came_from, start, goal)
            return path, actions, expanded, cost_so_far[goal]

        for a, ns in _neighbors(env, s):
            new_cost = cost_so_far[s] + _step_cost(env, s, a, ns)
            if ns not in cost_so_far or new_cost < cost_so_far[ns]:
                cost_so_far[ns] = new_cost
                came_from[ns] = (s, a)
                counter += 1
                priority = new_cost + heuristic(ns, goal)
                heapq.heappush(frontier, (priority, counter, ns))

    return None, None, expanded, float('inf')

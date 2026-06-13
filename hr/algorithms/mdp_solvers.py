"""
MDP solvers: Value Iteration and Policy Iteration
==================================================
Operate on a HospitalGridWorld in 'stochastic' mode, where the
transition model T(s,a,s') and reward function R(s,a,s') are
fully known and queried via:

    env.get_states()
    env.get_possible_actions(s)
    env.get_transition_states_and_probs(s, a)
    env.get_reward(s, a, s_next)
"""

from __future__ import annotations
from typing import Dict, List, Tuple

from env.grid_world import HospitalGridWorld, State, Action


def q_value(env: HospitalGridWorld, V: Dict[State, float], s: State, a: Action, gamma: float) -> float:
    total = 0.0
    for s_next, p in env.get_transition_states_and_probs(s, a):
        r = env.get_reward(s, a, s_next)
        total += p * (r + gamma * V.get(s_next, 0.0))
    return total


def value_iteration(env: HospitalGridWorld, gamma: float = 0.9, theta: float = 1e-4,
                     max_iterations: int = 1000):
    """
    Returns:
        V: dict state -> value
        policy: dict state -> best action
        history: list of V snapshots (for visualizing convergence)
    """
    states = env.get_states()
    V: Dict[State, float] = {s: 0.0 for s in states}
    history: List[Dict[State, float]] = [dict(V)]

    for it in range(max_iterations):
        new_V: Dict[State, float] = {}
        delta = 0.0
        for s in states:
            actions = env.get_possible_actions(s)
            if not actions:
                new_V[s] = 0.0
                continue
            best = max(q_value(env, V, s, a, gamma) for a in actions)
            new_V[s] = best
            delta = max(delta, abs(best - V[s]))
        V = new_V
        history.append(dict(V))
        if delta < theta:
            break

    policy = extract_policy(env, V, gamma)
    return V, policy, history


def extract_policy(env: HospitalGridWorld, V: Dict[State, float], gamma: float) -> Dict[State, Action]:
    policy: Dict[State, Action] = {}
    for s in env.get_states():
        actions = env.get_possible_actions(s)
        if not actions:
            continue
        best_a = max(actions, key=lambda a: q_value(env, V, s, a, gamma))
        policy[s] = best_a
    return policy


def policy_evaluation(env: HospitalGridWorld, policy: Dict[State, Action], gamma: float,
                       theta: float = 1e-4, max_iterations: int = 1000) -> Dict[State, float]:
    states = env.get_states()
    V: Dict[State, float] = {s: 0.0 for s in states}
    for _ in range(max_iterations):
        delta = 0.0
        for s in states:
            if s not in policy:
                continue
            new_v = q_value(env, V, s, policy[s], gamma)
            delta = max(delta, abs(new_v - V[s]))
            V[s] = new_v
        if delta < theta:
            break
    return V


def policy_iteration(env: HospitalGridWorld, gamma: float = 0.9, max_iterations: int = 100):
    states = env.get_states()
    # initialize with an arbitrary valid policy
    policy: Dict[State, Action] = {}
    for s in states:
        actions = env.get_possible_actions(s)
        if actions:
            policy[s] = actions[0]

    history: List[Dict[State, float]] = []
    V: Dict[State, float] = {s: 0.0 for s in states}

    for _ in range(max_iterations):
        V = policy_evaluation(env, policy, gamma)
        history.append(dict(V))
        stable = True
        for s in states:
            actions = env.get_possible_actions(s)
            if not actions:
                continue
            best_a = max(actions, key=lambda a: q_value(env, V, s, a, gamma))
            if best_a != policy.get(s):
                policy[s] = best_a
                stable = False
        if stable:
            break

    return V, policy, history

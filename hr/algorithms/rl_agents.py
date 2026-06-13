"""
Model-free RL: Q-learning and SARSA
=====================================
Operate on a HospitalGridWorld in 'unknown' mode, where the agent
only interacts through:

    env.reset() -> state
    env.step(action) -> (next_state, reward, done, info)

and does NOT have access to the transition model.
"""

from __future__ import annotations
import random
from typing import Dict, List, Tuple

from env.grid_world import HospitalGridWorld, ACTIONS, State, Action


def _epsilon_greedy(Q: Dict[Tuple[State, Action], float], s: State, actions: List[Action],
                    epsilon: float, rng: random.Random) -> Action:
    if rng.random() < epsilon:
        return rng.choice(actions)
    qvals = [Q.get((s, a), 0.0) for a in actions]
    best = max(qvals)
    best_actions = [a for a, q in zip(actions, qvals) if q == best]
    return rng.choice(best_actions)


def q_learning(env: HospitalGridWorld, episodes: int = 500, alpha: float = 0.5,
                gamma: float = 0.9, epsilon: float = 0.2, epsilon_decay: float = 0.999,
                max_steps: int = 200, seed: int = 0):
    """
    Returns:
        Q: dict (state, action) -> value
        policy: dict state -> best action
        episode_returns: list of total return per episode (for learning curve)
    """
    rng = random.Random(seed)
    Q: Dict[Tuple[State, Action], float] = {}
    episode_returns: List[float] = []
    eps = epsilon

    for ep in range(episodes):
        s = env.reset()
        total_r = 0.0
        for _ in range(max_steps):
            actions = env.get_possible_actions(s) or ACTIONS
            a = _epsilon_greedy(Q, s, actions, eps, rng)
            s_next, r, done, _ = env.step(a)
            total_r += r

            next_actions = env.get_possible_actions(s_next) or ACTIONS
            best_next = max((Q.get((s_next, a2), 0.0) for a2 in next_actions), default=0.0)
            td_target = r + (0.0 if done else gamma * best_next)
            Q[(s, a)] = Q.get((s, a), 0.0) + alpha * (td_target - Q.get((s, a), 0.0))

            s = s_next
            if done:
                break
        episode_returns.append(total_r)
        eps = max(0.01, eps * epsilon_decay)

    policy = _extract_policy_from_Q(env, Q)
    return Q, policy, episode_returns


def sarsa(env: HospitalGridWorld, episodes: int = 500, alpha: float = 0.5,
          gamma: float = 0.9, epsilon: float = 0.2, epsilon_decay: float = 0.999,
          max_steps: int = 200, seed: int = 0):
    rng = random.Random(seed)
    Q: Dict[Tuple[State, Action], float] = {}
    episode_returns: List[float] = []
    eps = epsilon

    for ep in range(episodes):
        s = env.reset()
        actions = env.get_possible_actions(s) or ACTIONS
        a = _epsilon_greedy(Q, s, actions, eps, rng)
        total_r = 0.0

        for _ in range(max_steps):
            s_next, r, done, _ = env.step(a)
            total_r += r

            next_actions = env.get_possible_actions(s_next) or ACTIONS
            a_next = _epsilon_greedy(Q, s_next, next_actions, eps, rng)

            td_target = r + (0.0 if done else gamma * Q.get((s_next, a_next), 0.0))
            Q[(s, a)] = Q.get((s, a), 0.0) + alpha * (td_target - Q.get((s, a), 0.0))

            s, a = s_next, a_next
            if done:
                break
        episode_returns.append(total_r)
        eps = max(0.01, eps * epsilon_decay)

    policy = _extract_policy_from_Q(env, Q)
    return Q, policy, episode_returns


def _extract_policy_from_Q(env: HospitalGridWorld, Q: Dict[Tuple[State, Action], float]) -> Dict[State, Action]:
    policy: Dict[State, Action] = {}
    for s in env.get_states():
        actions = env.get_possible_actions(s)
        if not actions:
            continue
        qvals = {a: Q.get((s, a), 0.0) for a in actions}
        policy[s] = max(qvals, key=qvals.get)
    return policy

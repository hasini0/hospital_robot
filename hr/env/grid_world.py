"""
Hospital Robot GridWorld
=========================
A configurable grid-world environment representing a hospital floor.

Cell types:
    '.'  - free corridor cell
    '#'  - wall / obstacle (impassable)
    'S'  - start cell (robot's charging dock)
    'G'  - goal cell (patient room / delivery target)
    'H'  - hazard cell (e.g. wet floor, biohazard spill) -> big negative reward
    'W'  - wind zone (stochastic drift, used in MDP/RL stages)

The environment supports three "modes" that map directly onto the
search -> MDP -> RL syllabus progression:

    mode="deterministic"  - actions always succeed exactly as intended.
                             Used for A* / UCS search.
    mode="stochastic"     - actions succeed with probability `success_prob`;
                             otherwise the robot drifts to a random
                             perpendicular neighbour (simulating wind),
                             and the battery can randomly fail
                             (battery_fail_prob) ending the episode with
                             a penalty. Used for Value/Policy Iteration
                             (transition model IS known and exposed).
    mode="unknown"        - same dynamics as "stochastic", but the agent
                             is NOT given access to the transition model
                             T(s,a,s'); it can only sample
                             (next_state, reward, done) via step().
                             Used for Q-learning / SARSA.
"""

from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

Action = str  # 'N', 'S', 'E', 'W', 'STAY'
State = Tuple[int, int]

ACTIONS: List[Action] = ['N', 'S', 'E', 'W']
ACTION_VECTORS: Dict[Action, Tuple[int, int]] = {
    'N': (-1, 0),
    'S': (1, 0),
    'E': (0, 1),
    'W': (0, -1),
    'STAY': (0, 0),
}

# Perpendicular actions, used for stochastic "drift"
PERPENDICULAR: Dict[Action, List[Action]] = {
    'N': ['E', 'W'],
    'S': ['E', 'W'],
    'E': ['N', 'S'],
    'W': ['N', 'S'],
}

DEFAULT_REWARDS = {
    'step': -1.0,        # cost of moving / time penalty
    'goal': +50.0,       # reached the patient room
    'hazard': -25.0,     # ran into a hazard (spill, biohazard)
    'wall_bump': -2.0,   # tried to move into a wall (stays in place)
    'battery_fail': -10.0,  # battery died mid-route, episode ends
}


# A library of preset hospital layouts -----------------------------------

GRIDS = {
    "small": [
        "S....",
        ".###.",
        ".#W..",
        ".#.#.",
        "...#G",
    ],
    "ward": [
        "S....#......",
        ".####.####..",
        ".#..H.#..W#..",
        ".#.##.#.##.#.",
        ".#.#...#....#",
        ".#.#.###.##.#",
        ".#.......#..#",
        ".#######.#.#.",
        ".........#.G.",
    ],
    "long_hall": [
        "S..........",
        ".#########.",
        ".W.........",
        ".#########.",
        "...........",
        ".#########.",
        ".H.........",
        ".#########G",
    ],
}


@dataclass
class HospitalGridWorld:
    grid: List[str]
    mode: str = "deterministic"   # 'deterministic' | 'stochastic' | 'unknown'
    success_prob: float = 0.8     # P(intended action succeeds) when stochastic
    battery_fail_prob: float = 0.02  # per-step chance of battery failure
    rewards: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_REWARDS))
    seed: Optional[int] = None

    def __post_init__(self):
        self.n_rows = len(self.grid)
        self.n_cols = len(self.grid[0])
        self.start: State = self._find('S')
        self.goal: State = self._find('G')
        self.hazards = self._find_all('H')
        self.walls = self._find_all('#')
        self.wind_zones = self._find_all('W')
        self.rng = random.Random(self.seed)
        self.agent_pos: State = self.start
        self.battery_dead = False

    # -- grid parsing -----------------------------------------------------
    def _find(self, ch: str) -> Optional[State]:
        for r, row in enumerate(self.grid):
            for c, val in enumerate(row):
                if val == ch:
                    return (r, c)
        return None

    def _find_all(self, ch: str) -> set:
        out = set()
        for r, row in enumerate(self.grid):
            for c, val in enumerate(row):
                if val == ch:
                    out.add((r, c))
        return out

    # -- public API ---------------------------------------------------------
    def in_bounds(self, s: State) -> bool:
        r, c = s
        return 0 <= r < self.n_rows and 0 <= c < self.n_cols

    def is_wall(self, s: State) -> bool:
        return s in self.walls

    def is_terminal(self, s: State) -> bool:
        return s == self.goal or s in self.hazards

    def get_states(self) -> List[State]:
        """All non-wall cells, for MDP solvers (value/policy iteration)."""
        return [
            (r, c)
            for r in range(self.n_rows)
            for c in range(self.n_cols)
            if (r, c) not in self.walls
        ]

    def get_possible_actions(self, s: State) -> List[Action]:
        if self.is_terminal(s):
            return []
        return list(ACTIONS)

    # -- reward model ---------------------------------------------------
    def get_reward(self, s: State, a: Action, s_next: State) -> float:
        if s_next in self.hazards:
            return self.rewards['hazard']
        if s_next == self.goal:
            return self.rewards['goal']
        if s_next == s:  # bumped a wall / boundary, didn't move
            return self.rewards['wall_bump']
        return self.rewards['step']

    # -- deterministic move primitive -------------------------------------
    def _attempt_move(self, s: State, a: Action) -> State:
        dr, dc = ACTION_VECTORS[a]
        ns = (s[0] + dr, s[1] + dc)
        if not self.in_bounds(ns) or self.is_wall(ns):
            return s  # bump -> stay
        return ns

    # -- full transition model T(s,a,s') -> probability ---------------------
    def get_transition_states_and_probs(self, s: State, a: Action) -> List[Tuple[State, float]]:
        """
        Returns a list of (next_state, probability) pairs.
        Exposed only for 'deterministic' and 'stochastic' modes
        (Value/Policy Iteration need the model).
        """
        if self.is_terminal(s):
            return [(s, 1.0)]

        intended = self._attempt_move(s, a)

        if self.mode == "deterministic":
            return [(intended, 1.0)]

        # stochastic / unknown dynamics: with success_prob go intended,
        # otherwise drift to one of the two perpendicular directions
        # (extra drift probability if standing in / moving through a wind zone)
        drift_bonus = 0.15 if (s in self.wind_zones or intended in self.wind_zones) else 0.0
        p_success = max(0.05, self.success_prob - drift_bonus)
        p_drift_each = (1.0 - p_success) / 2.0

        outcomes: Dict[State, float] = {}
        outcomes[intended] = outcomes.get(intended, 0.0) + p_success
        for pa in PERPENDICULAR[a]:
            drift_state = self._attempt_move(s, pa)
            outcomes[drift_state] = outcomes.get(drift_state, 0.0) + p_drift_each

        return list(outcomes.items())

    # -- sampling step (for RL: Q-learning / SARSA) ------------------------
    def reset(self) -> State:
        self.agent_pos = self.start
        self.battery_dead = False
        return self.agent_pos

    def step(self, a: Action) -> Tuple[State, float, bool, dict]:
        """
        Sample one transition. Returns (next_state, reward, done, info).
        Does NOT expose the transition model -> used for model-free RL.
        """
        s = self.agent_pos
        if self.is_terminal(s) or self.battery_dead:
            return s, 0.0, True, {"battery_dead": self.battery_dead}

        # battery failure check
        if self.mode != "deterministic" and self.rng.random() < self.battery_fail_prob:
            self.battery_dead = True
            return s, self.rewards['battery_fail'], True, {"battery_dead": True}

        outcomes = self.get_transition_states_and_probs(s, a)
        r = self.rng.random()
        cum = 0.0
        s_next = outcomes[-1][0]
        for ns, p in outcomes:
            cum += p
            if r <= cum:
                s_next = ns
                break

        reward = self.get_reward(s, a, s_next)
        self.agent_pos = s_next
        done = self.is_terminal(s_next)
        return s_next, reward, done, {"battery_dead": False}

    # -- convenience for visualization -----------------------------------
    def cell_type(self, s: State) -> str:
        r, c = s
        return self.grid[r][c]

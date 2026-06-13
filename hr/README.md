# Hospital Robot — Decision-Making Under Uncertainty

An original grid-world project covering the full **search → MDP → reinforcement
learning** pipeline, applied to a hospital delivery robot that must navigate
from its charging dock (`S`) to a patient room (`G`), avoiding hazards (`H`)
like spills/biohazards and contending with wind zones (`W`) and random battery
failures.

Inspired by the structure of Berkeley CS188 Pac-Man Project 3 (Value
Iteration / Q-learning), but built from scratch with an original environment,
reward structure, and an interactive Streamlit dashboard — no course code
reused.

## Project structure

```
hospital_robot/
├── env/
│   └── grid_world.py       # HospitalGridWorld environment (3 modes)
├── algorithms/
│   ├── search.py            # UCS, A* (deterministic stage)
│   ├── mdp_solvers.py        # Value Iteration, Policy Iteration
│   └── rl_agents.py          # Q-learning, SARSA
├── app/
│   ├── main.py               # Streamlit dashboard
│   └── render.py             # Matplotlib grid/policy visualization
├── tests/
│   └── smoke_test.py
└── requirements.txt
```

## The three stages

1. **Deterministic search (A\* / UCS)** — the robot's motors are perfectly
   reliable. Find the lowest-cost path to the patient room. Demonstrates
   classical graph search with an admissible Manhattan-distance heuristic.

2. **MDP planning (Value/Policy Iteration)** — wind zones can blow the robot
   sideways, and the battery can fail randomly each step. The full
   transition model `T(s,a,s')` and reward function are known, so we solve
   the Bellman equations exactly to get the optimal value function and
   policy.

3. **Model-free RL (Q-learning / SARSA)** — same stochastic dynamics, but the
   transition model is hidden. The agent learns purely from
   `(state, action, reward, next_state)` experience, using epsilon-greedy
   exploration.

All three stages share one environment (`env/grid_world.py`), so you can
directly compare the optimal MDP policy against what Q-learning/SARSA
converge to.

## Running the dashboard

```bash
pip install -r requirements.txt
streamlit run app/main.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

## Running the smoke tests

```bash
python tests/smoke_test.py
```

## Possible extensions

- Add new hospital layouts to `env/grid_world.py:GRIDS`
- Add a "patient priority" reward shaping (multiple goals with different values)
- Add function-approximation Q-learning for larger grids
- Add eligibility traces (SARSA(λ))
- Export learned policies as GIF animations of the robot's route

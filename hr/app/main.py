"""
Hospital Robot — Decision-Making Dashboard
============================================
A Streamlit app demonstrating the full search -> MDP -> RL pipeline
on a custom hospital grid-world environment.

Run with:
    streamlit run app/main.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import matplotlib.pyplot as plt

from env import HospitalGridWorld, GRIDS
from algorithms import (
    uniform_cost_search, a_star_search,
    value_iteration, policy_iteration,
    q_learning, sarsa,
)
from app.render import render_grid

st.set_page_config(page_title="Hospital Robot Dashboard", layout="wide")

st.title("🏥 Hospital Robot — Decision-Making Dashboard")
st.caption(
    "A from-scratch grid-world covering the full AI decision-making syllabus: "
    "deterministic search (A*/UCS) → MDPs (Value/Policy Iteration) → "
    "model-free RL (Q-learning/SARSA)."
)

# ------------------------------------------------------------------ sidebar
st.sidebar.header("Environment")
grid_name = st.sidebar.selectbox("Hospital layout", list(GRIDS.keys()), index=1)
grid = GRIDS[grid_name]

st.sidebar.markdown(
    "**Legend:** `S` start/dock · `G` patient room · `H` hazard "
    "(spill/biohazard) · `W` wind zone · `#` wall"
)

stage = st.sidebar.radio(
    "Stage",
    ["1. Deterministic Search (A* / UCS)",
     "2. MDP — Value & Policy Iteration",
     "3. Model-Free RL — Q-learning & SARSA"],
)

st.sidebar.divider()

# ============================================================== STAGE 1
if stage.startswith("1"):
    st.header("Stage 1 — Deterministic Search")
    st.markdown(
        "The robot's wheels and battery work perfectly: every action has a "
        "guaranteed outcome. We find the lowest-cost route from the charging "
        "dock to the patient room using **Uniform Cost Search** and **A\\*** "
        "(Manhattan distance heuristic)."
    )

    env = HospitalGridWorld(grid, mode="deterministic")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Run UCS"):
            path, actions, expanded, cost = uniform_cost_search(env)
            st.session_state['ucs'] = (path, actions, expanded, cost)
    with col2:
        if st.button("Run A*"):
            path, actions, expanded, cost = a_star_search(env)
            st.session_state['astar'] = (path, actions, expanded, cost)

    fig_col, info_col = st.columns([2, 1])

    show_path = None
    info_lines = []
    if 'ucs' in st.session_state:
        path, actions, expanded, cost = st.session_state['ucs']
        show_path = path
        info_lines.append(f"**UCS:** path length {len(path)}, cost {cost:.1f}, nodes expanded {expanded}")
    if 'astar' in st.session_state:
        path, actions, expanded, cost = st.session_state['astar']
        show_path = path
        info_lines.append(f"**A\\*:** path length {len(path)}, cost {cost:.1f}, nodes expanded {expanded}")

    with fig_col:
        fig = render_grid(env, path=show_path, title=f"{grid_name} — deterministic")
        st.pyplot(fig)
        plt.close(fig)

    with info_col:
        st.subheader("Results")
        if info_lines:
            for line in info_lines:
                st.markdown(line)
            st.markdown(
                "A\\* typically expands fewer nodes than UCS because the "
                "Manhattan heuristic guides the search toward the goal, "
                "while both find the same optimal-cost path."
            )
        else:
            st.info("Click a button to run a search algorithm.")

# ============================================================== STAGE 2
elif stage.startswith("2"):
    st.header("Stage 2 — MDPs with Stochastic Wind & Battery Failure")
    st.markdown(
        "Now the floor has **wind zones** (`W`) that can blow the robot off "
        "course, and there's a small per-step **battery failure** chance. "
        "The transition model is fully known, so we can solve for the "
        "optimal policy with **Value Iteration** or **Policy Iteration**."
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        gamma = st.slider("Discount γ", 0.5, 0.99, 0.9, 0.01)
    with c2:
        success_prob = st.slider("P(intended action succeeds)", 0.5, 1.0, 0.8, 0.05)
    with c3:
        battery_fail = st.slider("Battery failure prob/step", 0.0, 0.1, 0.02, 0.01)
    with c4:
        algo = st.selectbox("Algorithm", ["Value Iteration", "Policy Iteration"])

    env = HospitalGridWorld(grid, mode="stochastic",
                             success_prob=success_prob,
                             battery_fail_prob=battery_fail, seed=42)

    if st.button("Solve MDP"):
        if algo == "Value Iteration":
            V, policy, history = value_iteration(env, gamma=gamma)
        else:
            V, policy, history = policy_iteration(env, gamma=gamma)
        st.session_state['mdp_result'] = (V, policy, history, algo)

    if 'mdp_result' in st.session_state:
        V, policy, history, algo_used = st.session_state['mdp_result']

        fig_col, info_col = st.columns([2, 1])
        with fig_col:
            fig = render_grid(env, V=V, policy=policy, title=f"{grid_name} — {algo_used}")
            st.pyplot(fig)
            plt.close(fig)

        with info_col:
            st.subheader("Results")
            st.metric("Value at start (S)", f"{V.get(env.start, 0.0):.2f}")
            st.metric("Iterations to converge", len(history))
            st.markdown(
                "Color shows state values (green = high, red = low). "
                "Arrows show the greedy optimal policy: which way the "
                "robot should move from each cell."
            )

        st.subheader("Convergence of V(start)")
        start_values = [h.get(env.start, 0.0) for h in history]
        st.line_chart(start_values)
    else:
        fig = render_grid(env, title=f"{grid_name} — layout")
        st.pyplot(fig)
        plt.close(fig)

# ============================================================== STAGE 3
else:
    st.header("Stage 3 — Model-Free RL (Unknown Dynamics)")
    st.markdown(
        "Same wind + battery dynamics as Stage 2, but now the robot **does "
        "not know** the transition model — it must learn purely from "
        "experience via trial-and-error using **Q-learning** (off-policy) "
        "or **SARSA** (on-policy)."
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        gamma = st.slider("Discount γ", 0.5, 0.99, 0.9, 0.01, key="rl_gamma")
    with c2:
        alpha = st.slider("Learning rate α", 0.01, 1.0, 0.5, 0.01)
    with c3:
        epsilon = st.slider("Initial ε (exploration)", 0.0, 1.0, 0.3, 0.05)
    with c4:
        episodes = st.slider("Training episodes", 50, 2000, 500, 50)
    with c5:
        algo = st.selectbox("Algorithm", ["Q-learning", "SARSA"], key="rl_algo")

    success_prob = st.slider("P(intended action succeeds)", 0.5, 1.0, 0.8, 0.05, key="rl_success")
    battery_fail = st.slider("Battery failure prob/step", 0.0, 0.1, 0.02, 0.01, key="rl_battery")

    env = HospitalGridWorld(grid, mode="unknown",
                             success_prob=success_prob,
                             battery_fail_prob=battery_fail, seed=7)

    if st.button("Train Agent"):
        if algo == "Q-learning":
            Q, policy, returns = q_learning(env, episodes=episodes, alpha=alpha,
                                              gamma=gamma, epsilon=epsilon)
        else:
            Q, policy, returns = sarsa(env, episodes=episodes, alpha=alpha,
                                        gamma=gamma, epsilon=epsilon)

        V = {}
        for s in env.get_states():
            actions = env.get_possible_actions(s)
            if actions:
                V[s] = max(Q.get((s, a), 0.0) for a in actions)
            else:
                V[s] = 0.0

        st.session_state['rl_result'] = (V, policy, returns, algo)

    if 'rl_result' in st.session_state:
        V, policy, returns, algo_used = st.session_state['rl_result']

        fig_col, info_col = st.columns([2, 1])
        with fig_col:
            fig = render_grid(env, V=V, policy=policy, title=f"{grid_name} — {algo_used} (learned Q)")
            st.pyplot(fig)
            plt.close(fig)

        with info_col:
            st.subheader("Results")
            st.metric("Final episode return (avg of last 20)",
                      f"{sum(returns[-20:]) / min(20, len(returns)):.2f}")
            st.markdown(
                "Color/arrows show the policy implied by the learned "
                "Q-values: `max_a Q(s,a)` and `argmax_a Q(s,a)`."
            )

        st.subheader("Learning curve (episode return)")
        st.line_chart(returns)
    else:
        fig = render_grid(env, title=f"{grid_name} — layout")
        st.pyplot(fig)
        plt.close(fig)

st.divider()
st.caption(
    "Hospital Robot project — covers search (A*/UCS), MDP planning "
    "(Value & Policy Iteration), and model-free RL (Q-learning, SARSA) "
    "on a single shared grid-world environment, inspired by but "
    "independent from CS188 Pac-Man Project 3."
)

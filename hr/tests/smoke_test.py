import sys
sys.path.insert(0, '.')
from env import HospitalGridWorld, GRIDS
from algorithms import uniform_cost_search, a_star_search, value_iteration, policy_iteration, q_learning, sarsa

print("=== Deterministic search (small grid) ===")
env = HospitalGridWorld(GRIDS["small"], mode="deterministic")
path, actions, expanded, cost = uniform_cost_search(env)
print("UCS path:", path, "cost:", cost, "expanded:", expanded)
path, actions, expanded, cost = a_star_search(env)
print("A* path:", path, "cost:", cost, "expanded:", expanded)

print("\n=== Value Iteration (ward, stochastic) ===")
env2 = HospitalGridWorld(GRIDS["ward"], mode="stochastic", success_prob=0.8, seed=1)
V, policy, history = value_iteration(env2, gamma=0.9)
print("Start value:", V[env2.start], "iterations:", len(history))
print("Policy at start:", policy[env2.start])

print("\n=== Policy Iteration (ward, stochastic) ===")
V2, policy2, hist2 = policy_iteration(env2, gamma=0.9)
print("Start value:", V2[env2.start], "iterations:", len(hist2))
print("Policy matches VI?", policy == policy2)

print("\n=== Q-learning (long_hall, unknown) ===")
env3 = HospitalGridWorld(GRIDS["long_hall"], mode="unknown", success_prob=0.8, seed=2)
Q, qpolicy, returns = q_learning(env3, episodes=300)
print("Last 10 returns:", returns[-10:])
print("Policy at start:", qpolicy.get(env3.start))

print("\n=== SARSA (long_hall, unknown) ===")
Q2, spolicy, sreturns = sarsa(env3, episodes=300)
print("Last 10 returns:", sreturns[-10:])
print("Policy at start:", spolicy.get(env3.start))

print("\nAll smoke tests ran successfully.")

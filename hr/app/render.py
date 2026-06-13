"""
Rendering helpers: turn grid + values/policy into matplotlib figures.
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, Optional, List, Tuple

from env.grid_world import HospitalGridWorld, ACTION_VECTORS, State, Action

CELL_COLORS = {
    '#': '#3b3b3b',   # wall
    '.': '#f5f5f5',   # corridor
    'S': '#cfe8ff',   # start
    'G': '#c8f7c5',   # goal
    'H': '#ffd2d2',   # hazard
    'W': '#e8e0ff',   # wind zone
}

ARROW = {
    'N': (0, 0.32),
    'S': (0, -0.32),
    'E': (0.32, 0),
    'W': (-0.32, 0),
}


def render_grid(env: HospitalGridWorld,
                 V: Optional[Dict[State, float]] = None,
                 policy: Optional[Dict[State, Action]] = None,
                 path: Optional[List[State]] = None,
                 agent_pos: Optional[State] = None,
                 title: str = ""):
    n_rows, n_cols = env.n_rows, env.n_cols
    fig, ax = plt.subplots(figsize=(max(4, n_cols * 0.7), max(4, n_rows * 0.7)))

    vmin = vmax = None
    if V:
        vals = list(V.values())
        vmin, vmax = min(vals), max(vals)

    for r in range(n_rows):
        for c in range(n_cols):
            s = (r, c)
            ch = env.cell_type(s)
            color = CELL_COLORS.get(ch, '#ffffff')

            if V is not None and ch not in ('#',) and vmax is not None and vmax > vmin:
                v = V.get(s, 0.0)
                norm = (v - vmin) / (vmax - vmin)
                color = plt.cm.RdYlGn(norm)

            rect = patches.Rectangle((c, n_rows - r - 1), 1, 1,
                                      facecolor=color, edgecolor='gray', linewidth=0.5)
            ax.add_patch(rect)

            label = ""
            if ch == 'S':
                label = "S"
            elif ch == 'G':
                label = "G"
            elif ch == 'H':
                label = "H"
            elif ch == 'W':
                label = "~"

            if label:
                ax.text(c + 0.5, n_rows - r - 1 + 0.5, label,
                        ha='center', va='center', fontsize=12, fontweight='bold', color='black')

            if V is not None and ch != '#':
                ax.text(c + 0.5, n_rows - r - 1 + 0.18, f"{V.get(s, 0.0):.1f}",
                        ha='center', va='center', fontsize=7, color='black')

            if policy is not None and s in policy and ch not in ('G', 'H'):
                a = policy[s]
                dx, dy = ARROW.get(a, (0, 0))
                ax.arrow(c + 0.5, n_rows - r - 1 + 0.5, dx, dy,
                         head_width=0.12, head_length=0.1, fc='black', ec='black', length_includes_head=True)

    if path:
        xs = [c + 0.5 for (r, c) in path]
        ys = [n_rows - r - 1 + 0.5 for (r, c) in path]
        ax.plot(xs, ys, color='blue', linewidth=2.5, alpha=0.6, marker='o', markersize=4)

    if agent_pos is not None:
        r, c = agent_pos
        ax.add_patch(patches.Circle((c + 0.5, n_rows - r - 1 + 0.5), 0.25,
                                     facecolor='royalblue', edgecolor='black', zorder=5))

    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig

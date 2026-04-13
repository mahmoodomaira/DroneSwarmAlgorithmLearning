"""
animations/consensus_animation.py
-----------------------------------
Two-panel animation for the consensus protocol:
  Left  — agent graph with node colors encoding current value
  Right — convergence lines (value vs. time for each agent)

Run:
  python -m animations.consensus_animation
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from consensus.network import ConsensusNetwork


def make_consensus_animation(
    n_agents: int = 8,
    topology: str = "random",        # try "fully_connected" vs "ring"
    epsilon: float = 0.1,
    n_frames: int = 60,
    interval: int = 150,
) -> animation.FuncAnimation:
    """
    Build and return a FuncAnimation showing consensus convergence.

    Left panel — the communication graph:
      - Lay out agents in a circle (fixed positions).
      - Draw edges from self.graph (networkx is NOT required — just
        use ax.plot() for each edge and ax.scatter() for nodes).
      - Color nodes by current value using a colormap (e.g. 'coolwarm').
      - Update node colors each frame.

    Right panel — value trajectories:
      - x-axis: time step
      - y-axis: value
      - One line per agent, updated each frame.
      - Add a horizontal dashed line at the true mean (np.mean of initial
        values) so you can see what the agents are converging toward.

    Hint for circular layout of n agents:
      angles = np.linspace(0, 2*np.pi, n_agents, endpoint=False)
      positions = np.column_stack([np.cos(angles), np.sin(angles)])

    Steps to implement:
      1. Build ConsensusNetwork(n_agents, topology, epsilon).
      2. Run ALL steps up front with net.run(n_frames) to get history
         shape (n_frames+1, n_agents). This is simpler than stepping
         inside the animation callback.
      3. Set up fig with two subplots.
      4. Draw static graph edges in the left panel.
      5. Define update(frame) that:
           a. Updates node scatter colors from history[frame].
           b. Extends the trajectory lines up to frame.
           c. Returns all updated artists.
      6. Return FuncAnimation(..., blit=True).

    Extra challenge (optional):
      Add a slider widget (matplotlib.widgets.Slider) for epsilon so you
      can see how step size affects convergence speed interactively.
    """
    # ── 1. Build network and run all steps up front ──────────────────────
    net = ConsensusNetwork(n_agents=n_agents, topology=topology, epsilon=epsilon)
    history = net.run(n_frames)          # shape: (n_frames+1, n_agents)
    true_mean = history[0].mean()        # what everyone converges toward

    # ── 2. Circular layout: fixed (x, y) position for each agent node ────
    angles = np.linspace(0, 2 * np.pi, n_agents, endpoint=False)
    node_pos = np.column_stack([np.cos(angles), np.sin(angles)])  # (n, 2)

    # ── 3. Figure: two side-by-side panels ───────────────────────────────
    fig, (ax_graph, ax_conv) = plt.subplots(
        1, 2, figsize=(12, 5),
        facecolor="#0f0f0f",
    )
    fig.suptitle(
        f"Consensus protocol  —  topology: {topology}  |  ε = {epsilon}",
        color="white", fontsize=13,
    )

    # ── Left panel: communication graph ──────────────────────────────────
    ax_graph.set_facecolor("#0f0f0f")
    ax_graph.set_xlim(-1.5, 1.5)
    ax_graph.set_ylim(-1.5, 1.5)
    ax_graph.set_aspect("equal")
    ax_graph.axis("off")
    ax_graph.set_title("Agent graph  (color = value)", color="white", fontsize=11)

    # Draw static edges once
    for i, neighbours in net.graph.items():
        for j in neighbours:
            if j > i:   # draw each edge once
                xs = [node_pos[i, 0], node_pos[j, 0]]
                ys = [node_pos[i, 1], node_pos[j, 1]]
                ax_graph.plot(xs, ys, color="#333333", linewidth=1, zorder=1)

    # Colormap: maps value → color.  vmin/vmax fixed to initial value range
    cmap = plt.cm.coolwarm
    vmin, vmax = history[0].min(), history[0].max()

    # Initial node scatter (we will update facecolors each frame)
    initial_colors = cmap((history[0] - vmin) / (vmax - vmin))
    scatter = ax_graph.scatter(
        node_pos[:, 0], node_pos[:, 1],
        c=history[0], cmap=cmap, vmin=vmin, vmax=vmax,
        s=260, zorder=2, edgecolors="white", linewidths=0.8,
    )

    # Agent ID labels inside each node
    for i, (x, y) in enumerate(node_pos):
        ax_graph.text(
            x, y, str(i),
            ha="center", va="center",
            fontsize=8, color="white", fontweight="bold", zorder=3,
        )

    # Value label just outside each node
    value_texts = []
    for i, (x, y) in enumerate(node_pos):
        # push the label outward from center
        lx, ly = x * 1.28, y * 1.28
        t = ax_graph.text(
            lx, ly, f"{history[0][i]:.1f}",
            ha="center", va="center",
            fontsize=7.5, color="#cccccc", zorder=3,
        )
        value_texts.append(t)

    # Frame counter text
    frame_text = ax_graph.text(
        0, -1.45, "t = 0",
        ha="center", va="center",
        fontsize=10, color="#aaaaaa",
    )

    # ── Right panel: convergence trajectories ────────────────────────────
    ax_conv.set_facecolor("#0f0f0f")
    ax_conv.set_xlim(0, n_frames)
    ax_conv.set_ylim(vmin - 5, vmax + 5)
    ax_conv.set_xlabel("time step", color="#aaaaaa")
    ax_conv.set_ylabel("value", color="#aaaaaa")
    ax_conv.set_title("Value convergence", color="white", fontsize=11)
    ax_conv.tick_params(colors="#aaaaaa")
    for spine in ax_conv.spines.values():
        spine.set_edgecolor("#333333")

    # Dashed horizontal line at the true mean
    ax_conv.axhline(
        true_mean, color="#ffffff", linewidth=1,
        linestyle="--", alpha=0.4, label=f"mean = {true_mean:.1f}",
    )
    ax_conv.legend(fontsize=8, facecolor="#1a1a1a", labelcolor="white",
                   edgecolor="#333333")

    # One line per agent, colored by initial value
    agent_colors = cmap((history[0] - vmin) / (vmax - vmin))
    traj_lines = []
    for i in range(n_agents):
        line, = ax_conv.plot(
            [], [],
            color=agent_colors[i], linewidth=1.4, alpha=0.85,
        )
        traj_lines.append(line)

    # ── 4. Animation update function ─────────────────────────────────────
    def update(frame):
        # a) Update node colors in the graph panel
        scatter.set_array(history[frame])

        # b) Update value labels around each node
        for i, t in enumerate(value_texts):
            t.set_text(f"{history[frame][i]:.1f}")

        # c) Update frame counter
        frame_text.set_text(f"t = {frame}")

        # d) Extend each trajectory line up to current frame
        time_axis = np.arange(frame + 1)
        for i, line in enumerate(traj_lines):
            line.set_data(time_axis, history[:frame + 1, i])

        return [scatter, frame_text, *value_texts, *traj_lines]

    # ── 5. Build and return the animation ────────────────────────────────
    anim = animation.FuncAnimation(
        fig, update,
        frames=n_frames + 1,
        interval=interval,
        blit=True,
    )
    return anim


if __name__ == "__main__":
    anim = make_consensus_animation()
    plt.tight_layout()
    plt.show()
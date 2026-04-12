# drone_swarm

Home project: implement swarm intelligence algorithms from scratch.

## Goal

Understand how **local rules → global behavior** in distributed multi-agent systems.

---

## Algorithms

### 1. Boids (Craig Reynolds, 1986)

Each drone follows three rules based only on its nearby neighbors:

| Rule | What it does |
|---|---|
| **Separation** | Steer away from neighbors that are too close |
| **Alignment**  | Match the average heading of neighbors |
| **Cohesion**   | Steer toward the average position of neighbors |

No drone knows the flock's global state. Emergent flocking arises purely from local interaction.

### 2. Average Consensus

Each agent holds a scalar value. It repeatedly nudges toward its neighbors' values:

```
x_i(t+1) = x_i(t) + ε * Σ_j∈N(i) [ x_j(t) - x_i(t) ]
```

All agents converge to the global mean — without any agent ever seeing it.

---

## Project structure

```
drone_swarm/
├── shared/
│   └── vector.py             ← 2D vector math (normalize, limit, distance)
│
├── boids/
│   ├── boid.py               ← Single agent: separation, alignment, cohesion
│   └── simulation.py         ← Flock manager, step()
│
├── consensus/
│   ├── agent.py              ← Single agent: compute_update / apply_update
│   └── network.py            ← Graph topology, two-phase step(), history
│
└── animations/
    ├── boids_animation.py    ← matplotlib quiver animation
    └── consensus_animation.py← two-panel graph + trajectory animation
```

---

## Implementation order

Work in this order — each step builds on the previous:

1. `shared/vector.py`          — `normalize`, `limit`, `distance`
2. `boids/boid.py`             — `get_neighbours`, then the three forces, then `update`
3. `boids/simulation.py`       — `_init_boids`, `step`
4. `animations/boids_animation.py` — get a flock moving first
5. `consensus/agent.py`        — `compute_update`
6. `consensus/network.py`      — `_build_graph`, `step`, `run`
7. `animations/consensus_animation.py` — two-panel convergence view

---

## Running

```bash
# Boids flock animation
python -m animations.boids_animation

# Consensus convergence animation
python -m animations.consensus_animation
```

---

## Experiments to try after implementing

### Boids
- Increase `separation` weight → sparser flock, harder to cohere
- Set `alignment = 0` → chaotic swirling mass with no direction
- Set `cohesion = 0` → aligned streams that drift apart
- Reduce `perception_radius` → multiple disconnected sub-flocks

### Consensus
- Compare `ring` vs `fully_connected` topology — which converges faster and why?
- Increase `epsilon` beyond `1 / max_degree` → watch it diverge/oscillate
- Disconnect an agent (remove all its edges) → it never converges

---

## Key insight

Both algorithms demonstrate the same principle:

> Simple local rules + repeated interaction = complex coordinated global behavior

No central controller. No global knowledge. Just neighbors talking to neighbors.

This is the foundation of real drone swarm coordination protocols.
"""
boids/simulation.py
-------------------
Manages the flock and advances the simulation each frame.
"""

import numpy as np
from boids.boid import Boid
from shared.vector import normalize


class BoidsSimulation:
    def __init__(
        self,
        n_boids: int = 80,
        world_size: tuple[int, int] = (800, 600),
        weights: dict[str, float] | None = None,
    ):
        self.world_size = world_size
        self.weights = weights or {"sep": 1.5, "ali": 1.0, "coh": 1.0}
        self.boids = self._init_boids(n_boids)

    def _init_boids(self, n: int) -> list[Boid]:
        """
        Spawn n boids at random positions with random velocities.

        Hint:
          - positions: np.random.uniform over world_size
          - velocities: np.random.uniform(-1, 1, size=(2,)) for each boid
          - Then normalize and scale to a small speed (e.g. 1.5)
        """
        boids = []
        for _ in range(n):
            position = np.random.uniform([0, 0], self.world_size)
            velocity = np.random.uniform(-1, 1, size=(2,))
            velocity = normalize(velocity) * 1.5
            boids.append(Boid(position, velocity))
        return boids

    def step(self) -> None:
        """
        Advance simulation by one frame.

        Important: compute ALL new states before applying any of them.
        If you update boid 0 first, its new position will corrupt the
        neighbour calculations of boids 1..n.

        Pattern (two-pass update):
          1. Compute acceleration for every boid (read-only phase).
          2. Apply all updates at once.

        Simplest approach: just call boid.update() on each boid — this
        works because each boid re-reads positions at call time and the
        list reference doesn't change between calls in a single-threaded
        loop. (For a more rigorous version, snapshot positions first.)
        """
        for boid in self.boids:
            boid.update(self.boids, self.world_size, self.weights)

    @property
    def positions(self) -> np.ndarray:
        """Return (n, 2) array of all boid positions. Useful for plotting."""
        return np.array([b.position for b in self.boids])

    @property
    def velocities(self) -> np.ndarray:
        """Return (n, 2) array of all boid velocities. Useful for quiver plot."""
        return np.array([b.velocity for b in self.boids])
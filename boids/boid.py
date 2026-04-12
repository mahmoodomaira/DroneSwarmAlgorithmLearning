"""
boids/boid.py
-------------
A single Boid (drone agent).

State:
  position  np.ndarray shape (2,)   — x, y in world space
  velocity  np.ndarray shape (2,)   — current velocity vector

Each update step:
  1. Collect nearby neighbours (within perception_radius).
  2. Compute three steering forces.
  3. Sum them (with weights), clamp to max_force.
  4. Apply to velocity, clamp to max_speed.
  5. Update position (wrap at world boundaries).
"""

import numpy as np
from shared.vector import normalize, limit, distance


class Boid:
    def __init__(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        max_speed: float = 3.0,
        max_force: float = 0.1,
        perception_radius: float = 50.0,
        separation_radius: float = 25.0,
    ):
        self.position = position.astype(float)
        self.velocity = velocity.astype(float)
        self.max_speed = max_speed
        self.max_force = max_force
        self.perception_radius = perception_radius
        self.separation_radius = separation_radius

    # ------------------------------------------------------------------
    # Neighbour query
    # ------------------------------------------------------------------

    def get_neighbours(self, all_boids: list["Boid"]) -> list["Boid"]:
        """
        Return all boids (excluding self) within self.perception_radius.

        Hint:
          - iterate all_boids
          - skip self (use `is not self`)
          - use distance() from shared.vector
        """
        neighbors = []
        for boid in all_boids:
            if boid is not self and distance(self.position, boid.position) < self.perception_radius:
                neighbors.append(boid)
        return neighbors

    # ------------------------------------------------------------------
    # The three steering rules
    # ------------------------------------------------------------------

    def separation(self, neighbours: list["Boid"]) -> np.ndarray:
        """
        Rule 1 — Separation.
        Steer away from neighbours that are closer than self.separation_radius.

        Algorithm:
          1. For each close neighbour, compute a vector pointing AWAY:
               diff = self.position - neighbour.position
          2. Weight the diff by 1/distance so closer neighbours repel more.
          3. Average the weighted diffs.
          4. If the average is non-zero:
               - Set its magnitude to max_speed  (desired velocity)
               - Subtract current velocity       (steering = desired - current)
               - Clamp to max_force
          5. Return a zero vector if no close neighbours.
        """
        steering = np.zeros(2)
        total = 0
        for neighbor in neighbours:
            dist = distance(self.position, neighbor.position)
            if dist < self.separation_radius and dist > 0:
                diff = self.position - neighbor.position
                steering += diff / dist
                total += 1
        if total > 0:
            steering /= total
            steering = normalize(steering) * self.max_speed
            steering -= self.velocity
            steering = limit(steering, self.max_force)
        return steering

    def alignment(self, neighbours: list["Boid"]) -> np.ndarray:
        """
        Rule 2 — Alignment.
        Steer toward the average velocity of neighbours.

        Algorithm:
          1. Average the velocity vectors of all neighbours.
          2. Normalize and scale to max_speed  (desired velocity).
          3. Subtract current velocity          (steering = desired - current).
          4. Clamp to max_force.
          5. Return zero vector if no neighbours.
        """
        steering = np.zeros(2)
        total = 0
        for neighbor in neighbours:
            steering += neighbor.velocity
            total += 1
        if total > 0:
            steering /= total
            steering = normalize(steering) * self.max_speed
            steering -= self.velocity
            steering = limit(steering, self.max_force)
        return steering

    def cohesion(self, neighbours: list["Boid"]) -> np.ndarray:
        """
        Rule 3 — Cohesion.
        Steer toward the center of mass of neighbours.

        Algorithm:
          1. Average the positions of all neighbours  → center_of_mass.
          2. Compute desired velocity:
               desired = center_of_mass - self.position
          3. Normalize and scale to max_speed.
          4. Subtract current velocity (steering = desired - current).
          5. Clamp to max_force.
          6. Return zero vector if no neighbours.
        """
        steering = np.zeros(2)
        total = 0
        for neighbor in neighbours:
            steering += neighbor.position
            total += 1
        if total > 0:
            steering /= total
            steering -= self.position
            steering = normalize(steering) * self.max_speed
            steering -= self.velocity
            steering = limit(steering, self.max_force)
        return steering

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(
        self,
        all_boids: list["Boid"],
        world_size: tuple[int, int],
        weights: dict[str, float] | None = None,
    ) -> None:
        """
        One simulation step for this boid.

        Steps:
          1. Get neighbours.
          2. Compute all three forces.
          3. Combine:
               acceleration = w_sep * sep + w_ali * ali + w_coh * coh
             Default weights: sep=1.5, ali=1.0, coh=1.0
          4. velocity += acceleration
          5. velocity = limit(velocity, max_speed)
          6. position += velocity
          7. Wrap position at world boundaries (toroidal world):
               x = x % world_size[0]
               y = y % world_size[1]

        Hint on weights argument: accept a dict like
          {"sep": 1.5, "ali": 1.0, "coh": 1.0}
        and fall back to defaults when None.
        """
        neighbours = self.get_neighbours(all_boids)
        sep = self.separation(neighbours)
        ali = self.alignment(neighbours)
        coh = self.cohesion(neighbours)
        w_sep = weights["sep"] if weights and "sep" in weights else 1.5
        w_ali = weights["ali"] if weights and "ali" in weights else 1.0
        w_coh = weights["coh"] if weights and "coh" in weights else 1.0
        acceleration = w_sep * sep + w_ali * ali + w_coh * coh
        self.velocity += acceleration
        self.velocity = limit(self.velocity, self.max_speed)
        self.position += self.velocity
        self.position[0] = self.position[0] % world_size[0]
        self.position[1] = self.position[1] % world_size[1]
        
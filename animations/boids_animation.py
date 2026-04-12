"""
animations/boids_animation.py
------------------------------
Animate the boids simulation using matplotlib FuncAnimation.

Run:
  python -m animations.boids_animation
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from boids.simulation import BoidsSimulation


def make_boids_animation(
    n_boids: int = 80,
    n_frames: int = 400,
    interval: int = 30,           # ms between frames
    world_size: tuple = (800, 600),
    weights: dict | None = None,
) -> animation.FuncAnimation:
    """
    Build and return a FuncAnimation for the boids flock.

    Visualization tips:
      - Use ax.quiver() to draw arrows showing velocity direction.
      - Or ax.scatter() for dots — simpler but less informative.
      - quiver(x, y, u, v) where u,v = velocity components.
      - Call quiv.set_UVC(u, v) and quiv.set_offsets(xy) in the update fn
        to avoid clearing the axes every frame (much faster).

    Steps to implement:
      1. Create BoidsSimulation(n_boids, world_size, weights).
      2. Set up fig, ax with world_size limits.
      3. Initial quiver/scatter plot.
      4. Define update(frame) function:
           a. sim.step()
           b. Read sim.positions and sim.velocities
           c. Update the plot artist
           d. Return the artist in a list (required by FuncAnimation)
      5. Return FuncAnimation(fig, update, frames=n_frames, interval=interval,
                              blit=True)

    Hint on quiver colors:
      Color each arrow by speed magnitude for visual richness:
        speeds = np.linalg.norm(sim.velocities, axis=1)
      Pass colors=speeds to quiver (with a colormap like 'plasma').
      Call quiv.set_array(speeds) in the update function.
    """
    sim = BoidsSimulation(n_boids=n_boids, world_size=world_size, weights=weights)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, world_size[0])
    ax.set_ylim(0, world_size[1])
    quiv = ax.quiver(
        sim.positions[:, 0],
        sim.positions[:, 1],
        sim.velocities[:, 0],
        sim.velocities[:, 1],
        angles="xy",
        scale_units="xy",
        scale=1,
        color=plt.cm.plasma(np.linalg.norm(sim.velocities, axis=1) / sim.boids[0].max_speed)
    )

    def update(frame):
        sim.step()
        positions = sim.positions
        velocities = sim.velocities

        quiv.set_offsets(positions)
        quiv.set_UVC(velocities[:, 0], velocities[:, 1])

        return [quiv]

    return animation.FuncAnimation(fig, update, frames=n_frames, interval=interval, blit=True)




if __name__ == "__main__":
    anim = make_boids_animation()
    plt.show()
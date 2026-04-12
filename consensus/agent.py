"""
consensus/agent.py
------------------
A single agent in a consensus network.

Each agent holds a scalar value (e.g. a desired altitude or heading).
On each step it nudges its value toward its neighbours' values.

The update rule (average consensus):
  x_i(t+1) = x_i(t) + epsilon * sum_j [ x_j(t) - x_i(t) ]

where j ranges over neighbours of i in the communication graph.
"""

import numpy as np


class ConsensusAgent:
    def __init__(self, agent_id: int, initial_value: float, epsilon: float = 0.1):
        """
        Parameters
        ----------
        agent_id      : unique int, used to identify self in neighbour lists
        initial_value : the scalar this agent starts with (e.g. random altitude)
        epsilon       : step size (learning rate). Too large → oscillates.
                        Stable range: 0 < epsilon < 1 / max_degree
        """
        self.id = agent_id
        self.value = float(initial_value)
        self.epsilon = epsilon

    def compute_update(self, neighbour_values: list[float]) -> float:
        """
        Compute the NEW value after one consensus step.
        Does NOT modify self.value — caller applies the update.

        Algorithm:
          delta = epsilon * sum( v_j - self.value for v_j in neighbour_values )
          return self.value + delta

        Hint: why return instead of assign directly?
          Because all agents must read the OLD values before any agent writes.
          The simulation does a two-phase update: collect → apply.
        """
        delta = self.epsilon * sum(v_j - self.value for v_j in neighbour_values)
        return self.value + delta

    def apply_update(self, new_value: float) -> None:
        """Commit a previously computed update."""
        self.value = new_value
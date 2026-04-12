"""
consensus/network.py
--------------------
Manages a graph of ConsensusAgent nodes and advances the protocol.

The communication graph defines who talks to whom.
We represent it as an adjacency list: dict[int, list[int]].

Example (ring topology for 4 agents):
  { 0: [1, 3],
    1: [0, 2],
    2: [1, 3],
    3: [2, 0] }
"""

import numpy as np
from consensus.agent import ConsensusAgent


class ConsensusNetwork:

    TOPOLOGIES = ["fully_connected", "ring", "random"]

    def __init__(
        self,
        n_agents: int = 6,
        topology: str = "ring",       # "fully_connected" | "ring" | "random"
        epsilon: float = 0.1,
        value_range: tuple[float, float] = (0.0, 100.0),
        random_edge_prob: float = 0.4,  # used only for "random" topology
    ):
        assert topology in self.TOPOLOGIES
        self.n_agents = n_agents
        self.epsilon = epsilon

        # Spawn agents with random initial values
        initial_values = np.random.uniform(*value_range, size=n_agents)
        self.agents = [
            ConsensusAgent(i, initial_values[i], epsilon)
            for i in range(n_agents)
        ]

        # Build adjacency list
        self.graph = self._build_graph(topology, random_edge_prob)

        # History: list of value snapshots per step, for plotting
        self.history: list[np.ndarray] = [self.values.copy()]

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(
        self, topology: str, random_edge_prob: float
    ) -> dict[int, list[int]]:
        """
        Build and return an adjacency list for the chosen topology.

        Topologies to implement:

        "fully_connected":
          Every agent is a neighbour of every other agent.
          adjacency[i] = [j for j in range(n) if j != i]

        "ring":
          Each agent i talks to (i-1) % n and (i+1) % n.
          adjacency[i] = [(i-1) % n, (i+1) % n]

        "random":
          Add each undirected edge (i, j) with probability random_edge_prob.
          Make sure the graph is connected (simplest: always add ring edges
          first, then add random additional edges on top).

        Return a dict mapping agent_id -> list of neighbour ids.
        """
        # TODO: implement
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Simulation step
    # ------------------------------------------------------------------

    def step(self) -> None:
        """
        One round of the consensus protocol.

        Two-phase update (critical — do NOT merge into one loop):

        Phase 1 — collect:
          For each agent i, gather the current values of its neighbours.
          Call agent.compute_update(neighbour_values) but do NOT apply yet.
          Store results in a temporary list.

        Phase 2 — apply:
          For each agent i, call agent.apply_update(new_values[i]).

        After applying, append the current values snapshot to self.history.

        Why two phases?
          If you update agent 0 and then agent 1 reads agent 0's new value,
          the protocol is no longer symmetric — agents run at different
          logical times. The two-phase pattern ensures every agent reads
          the same generation of values.
        """
        # TODO: implement
        raise NotImplementedError

    def run(self, n_steps: int) -> np.ndarray:
        """
        Run n_steps of consensus and return the full history as a
        (n_steps+1, n_agents) array.

        Hint: just call self.step() in a loop.
        """
        # TODO: implement
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def values(self) -> np.ndarray:
        """Current values of all agents as a 1D array."""
        return np.array([a.value for a in self.agents])

    @property
    def has_converged(self, tol: float = 0.5) -> bool:
        """
        Return True when all agents are within `tol` of each other.
        Hint: np.max(values) - np.min(values) < tol
        """
        # TODO: implement
        raise NotImplementedError
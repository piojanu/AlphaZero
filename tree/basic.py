import numpy as np

from abc import ABCMeta, abstractmethod


class Node(object):
    """Represents state in MCTS search tree.

    Args:
        state (object): The environment state corresponding to this node in the search tree.

    Note:
        Node object is immutable. Node is left without exit edges (empty dict) when it's terminal.
    """

    def __init__(self, state):
        self._state = state
        self._edges = None

    @property
    def state(self):
        """object: The environment state corresponding to this node in the search tree."""
        return self._state

    @property
    def edges(self):
        """list of Edges: Mapping from this node's possible actions to corresponding edges."""
        return self._edges

    def expand(self, edges):
        """Initialize Node object with edges.

        Args:
            edges (dict of Edges): Mapping from this node's possible actions to corresponding edges.
        """

        self._edges = edges

    def select_edge(self, c=1.):
        """Choose next action (edge) according to UCB formula.

        Args:
            c (float): The parameter c >= 0 controls the trade-off between choosing lucrative nodes
                       (low c) and exploring nodes with low visit counts (high c). (Default: 1)

        Returns:
            int: Action chosen with UCB formula.
            Edge: Edge which represents proper action chosen with UCB formula.

            or

            None: If it is terminal node and has no exit edges.
        """

        assert self.edges is not None, "This node hasn't been expanded yet!"

        if len(self.edges) == 0:
            return None

        state_visits = 0
        scores = {}

        # Initialize every edge's score to its Q-value and count current state visits
        for action, edge in self.edges.items():
            state_visits += edge.num_visits
            scores[(action, edge)] = edge.qvalue

        # Add exploration term to every edge's score
        for action, edge in self.edges.items():
            scores[(action, edge)] += c * edge.prior * \
                np.sqrt(state_visits) / (1 + edge.num_visits)

        # Choose next action and edge with highest score
        action_edge = max(scores, key=scores.get)
        return action_edge


class Edge(object):
    """Represents state-actions pair in MCTS search tree.

    Args:
        prior (float): Action probability from prior policy. (Default: 1.)
    """

    def __init__(self, prior=1.):
        self._prior = prior
        self._next_node = None
        self._reward = 0
        self._qvalue = 0
        self._num_visits = 0

    def expand(self, next_node, reward):
        """Explore this edge.

        Args:
            next_node (Node): Node that this edge points to.
            reward (float): Reward of transition represented by this edge.
        """

        self._next_node = next_node
        self._reward = reward

    def update(self, return_t):
        """Update edge with data from child.

        Args:
            return_t (float): (Un)discounted return from timestep 't' (this edge).
        """

        self._num_visits += 1

        # This is formula for iteratively calculating average
        # NOTE: You can check that first arbitrary value will be forgotten after fist update
        self._qvalue += (return_t - self._qvalue) / self.num_visits

    @property
    def next_node(self):
        """next_node (Node): Node that this edge points to."""
        return self._next_node

    @property
    def reward(self):
        """float: Reward of transition represented by this edge."""
        return self._reward

    @property
    def qvalue(self):
        """float: Quality value of this edge state-action pair."""
        return self._qvalue

    @property
    def prior(self):
        """float: Action probability from prior policy."""
        return self._prior

    @property
    def num_visits(self):
        """int: Number of times this state-action pair was visited."""
        return self._num_visits

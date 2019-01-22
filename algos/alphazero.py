import time

import logging as log
import numpy as np

from humblerl import Callback, Mind
from tree.basic import Edge, Node


class Planner(Callback, Mind):
    """AlphaZero search operations and planning logic.

    Args:
        model (humblerl.MDP): Game's MDP.
        nn (NeuralNet): Artificial neural mind used to evaluate leaf states.
        params (dict): MCTS search hyper-parameters. Available:
          * 'c' (float)           : UCT exploration-exploitation trade-off param (Default: 1.)
          * 'dirichlet_noise'     : Dirichlet noise added to root prior (Default: 0.03)
          * 'noise_ratio'          : Noise contribution to prior probabilities (Default: 0.25)
          * 'gamma' (float)       : Discounting factor for value. (Default: 1./no discounting)
          * 'n_simulations' (int) : Number of simulations to perform before choosing action.
                                    (Default: 25)

    Note:
        Add it as callback to humblerl loop to clear tree between episodes in train mode.
    """

    def __init__(self, model, nn, params):
        self.model = model
        self.nn = nn

        self.c = params['c']
        self.dirichlet_noise = params['dirichlet_noise']
        self.gamma = params['gamma']
        self.n_simulations = float(params['n_simulations'])
        self.noise_ratio = params['noise_ratio']
        self.timeout = float(params['timeout'])

        if self.n_simulations == self.timeout == float("inf"):
            raise Exception(
                "n_simulations and timeout cannot be set to inf simultaneously in config!")

        self._tree = {}

    def simulate(self, start_node):
        """Search through tree from start node to leaf.

        Args:
            start_node (Node): Where to start the search.

        Returns:
            (Node): Leaf node.
            (list): List of edges that make path from start node to leaf node.
        """

        current_node = start_node
        path = []

        while True:
            action_edge = current_node.select_edge(self.c)
            if action_edge is None:
                # This is leaf node, return now
                return current_node, path
            action, edge = action_edge

            path.append(edge)
            next_node = edge.next_node

            if next_node is None:
                # This edge wasn't explored yet, create leaf node and return
                next_state, reward = self.model.transition(current_node.state, action)
                leaf_node = self.expand(next_state)
                # Link edge with new node and set it's reward to R(S, A, S')
                # NOTE: Board games are deterministic: R(S, A) = R(S, A, S')
                edge.expand(leaf_node, reward)

                return leaf_node, path

            current_node = next_node

    def evaluate(self, leaf_node, train_mode, is_root=False):
        """Expand and evaluate leaf node.

        Args:
            leaf_node (object): Leaf node to expand and evaluate.
            train_mode (bool): Informs whether add additional Dirichlet noise for exploration.
            is_root (bool): Whether this is tree root. (Default: False)

        Returns:
            (float): Node (state) value.
        """

        # Evaluate state
        pi, value = self.nn.predict(np.expand_dims(leaf_node.state.raw, axis=0))

        # Take first element in batch
        pi = pi[0]
        value = value[0][0]

        # Create edges of this node
        edges = {}

        is_terminal = self.model.is_terminal_state(leaf_node.state)
        if is_terminal:  # Change value to zero and don't expand possible moves
            # Terminal state has no value
            # NOTE: Transition to terminal state has value (reward) = R(S, A, S')
            value = 0
        else:  # Calculate prior probability of possible actions from leaf node
            # Add Dirichlet noise to root node prior
            if is_root and train_mode:
                pi = (1 - self.noise_ratio) * pi + self.noise_ratio * \
                    np.random.dirichlet([self.dirichlet_noise, ] * len(pi))

            # Get valid moves probabilities
            valid_moves = self.model.get_valid_actions(leaf_node.state)

            # Renormalize valid actions probabilities, but only if there are any valid actions
            if len(valid_moves) != 0:
                probs = np.zeros_like(pi)
                probs[valid_moves] = pi[valid_moves]
                sum_probs = np.sum(probs)
                if sum_probs <= 0:
                    # If all valid moves were masked make all valid moves equally probable
                    log.warning("All valid moves were masked, do workaround!")
                    probs[valid_moves] = 1
                # Normalize probabilities
                probs = probs / sum_probs

                # Fill this node edges with priors
                for m in valid_moves:
                    edges[m] = Edge(prior=probs[m])

        # Expand node with edges
        leaf_node.expand(edges)

        return value

    def backup(self, path, value):
        """Backup value to ancestry nodes.

        Args:
            path (list): List of edges that make path from start node to leaf node.
            value (float): Value to backup to all the edges on path.
        """

        # NOTE: Node higher in tree is opponent node in zero-sum game.
        #       Gamma should be < 0 to flip return sign.
        if self.gamma >= 0:
            log.info("In two players zero-sum games like board games opponents returns are mutual"
                     "negations, gamma should be < 0.")

        # For leaf state return is approximated with value function: G = R(S, A, S') + gamma * V[S']
        return_t = value
        for edge in reversed(path):
            return_t = edge.reward + self.gamma * return_t
            edge.update(return_t)

    def expand(self, state):
        """Add new node to search tree.

        Args:
            state (np.ndarray): Canonical board game (from perspective of current player).

        Return:
            Node: Node in search tree representing given state.

        Note:
            For now just store mapping in 'tree' dict from state.tostring() to Node. arrays are so
            small, that it'll have good performance:
            https://stackoverflow.com/questions/16589791/most-efficient-property-to-hash-for-numpy-array

            If you'll deal with bigger arrays in the future e.g. from Atari games, consider wrapping
            it with own class with __hash__ and __eq__ implementation and in __hash_ convert
            to string only smaller part of original array. Allow python to deal with collisions.
        """

        node = Node(state)
        self._tree[state] = node

        return node

    def clear_tree(self):
        """Empty search tree."""

        self._tree.clear()

    def query_tree(self, state):
        """Get node of given state from search tree.

        Args:
            state (np.ndarray): Canonical board game (from perspective of current player).

        Returns:
            Node: Node in search tree representing given state.
        """

        return self._tree.get(state, None)

    def plan(self, state, train_mode, debug_mode):
        """Conduct planning on state.

        Args:
            state (np.ndarray): Canonical board game (from perspective of current player).
            train_mode (bool): Informs planner whether it's in training mode and should enable
                               additional exploration.
            debug_mode (bool): Informs planner whether it's in debug mode or not.

        Returns:
            np.ndarray: Planning result, unnormalized action probabilities.
            dict: Planning metrics.
        """

        # Get/create root node
        root = self.query_tree(state)
        if root is None:
            root = self.expand(state)
            _ = self.evaluate(root, train_mode, is_root=True)

        # Perform simulations
        max_depth = 0
        max_simulations = self.n_simulations
        simulations = 0
        start_time = time.time()
        while time.time() < start_time + self.timeout and simulations < max_simulations:
            # Simulate
            simulations += 1
            leaf, path = self.simulate(root)

            # Keep max search depth
            max_depth = max(len(path), max_depth)

            # Expand and evaluate
            value = self.evaluate(leaf, train_mode)

            # Backup value
            self.backup(path, value)

        metrics = {"max_depth": max_depth, "simulations": simulations,
                   "simulation_time": time.time() - start_time}
        if debug_mode:
            self._debug_log(root, metrics)

        # Get actions' visit counts
        actions = np.zeros(self.model.action_space.num)
        for action, edge in root.edges.items():
            actions[action] = edge.num_visits

        return actions, metrics

    def on_episode_start(self, episode, train_mode):
        """Empty search tree between episodes if in train mode."""

        if train_mode:
            self.clear_tree()

    def _debug_log(self, root, metrics):
        # Evaluate root state
        pi, value = self.nn.predict(np.expand_dims(root.state.raw, axis=0))

        # Log root state
        log.debug("Max search depth: %d", metrics['max_depth'])
        log.debug("Performed simulations: %d", metrics['simulations'])
        log.debug("Time: %d", metrics['simulation_time'])

        # Log MCTS root value and NN predicted value
        state_visits = 0
        state_value = 0
        for action, edge in root.edges.items():
            state_visits += edge.num_visits
            state_value += edge.qvalue * edge.num_visits

        log.debug("MCTS root value: %.5f", state_value / state_visits)
        log.debug("NN root value: %.5f\n", value[0])

        # Action size must be multiplication of board width
        BOARD_WIDTH = root.state.raw.shape[1]
        action_size = self.model.action_space.num
        if action_size % BOARD_WIDTH == 1:
            # There is extra 'null' action, ignore it
            # NOTE: For this WA to work 'null' action has to have last idx in the environment!
            action_size -= 1

        # Log MCTS actions scores and qvalues and NN prior probs
        visits = np.zeros(action_size)
        qvalues = np.zeros_like(visits)
        scores = np.zeros_like(visits)
        for action, edge in root.edges.items():
            visits[action] = edge.num_visits
            qvalues[action] = edge.qvalue
            scores[action] = edge.qvalue

        ucts = np.zeros_like(visits)
        for action, edge in root.edges.items():
            ucts[action] = self.c * edge.prior * \
                np.sqrt(1 + np.sum(visits)) / (1 + edge.num_visits)
            scores[action] += ucts[action]

        log.debug("Prior probabilities:\n%s\n", np.array2string(
            pi[0][:action_size].reshape([-1, BOARD_WIDTH]), formatter={'float_kind': lambda x: "%.5f" % x}))
        log.debug("Exploration bonuses:\n%s\n", np.array2string(
            ucts.reshape([-1, BOARD_WIDTH]), formatter={'float_kind': lambda x: "%.5f" % x}))
        log.debug("Actions qvalues:\n%s\n", np.array2string(
            qvalues.reshape([-1, BOARD_WIDTH]), formatter={'float_kind': lambda x: "%.5f" % x}))
        log.debug("Actions scores:\n%s\n", np.array2string(
            scores.reshape([-1, BOARD_WIDTH]), formatter={'float_kind': lambda x: "%.5f" % x}))
        log.debug("Actions visits:\n%s\n", visits.reshape([-1, BOARD_WIDTH]))

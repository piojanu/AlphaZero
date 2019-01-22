from abc import ABCMeta, abstractmethod

import numpy as np
from humblerl import Callback, Environment, MDP
from humblerl.environments import Discrete


class GameState(metaclass=ABCMeta):
    """Game state interface.

    Args:
        state (object): Game state, type depending on implementation.
    """

    def __init__(self, state):
        self.raw = state

    @abstractmethod
    def __hash__(self):
        pass

    @abstractmethod
    def __eq__(self, other):
        pass


class BoardState(GameState):
    """Board games state.

    Args:
        state (np.ndarray): Board state.
    """

    def __init__(self, state):
        super(BoardState, self).__init__(state)

    def __hash__(self):
        return hash(self.raw.tostring())

    def __eq__(self, other):
        return np.all(self.raw == other.raw)


class BoardGameMDP(MDP):
    """Define board game MDP.

    Args:
        game (Game): Board game object.
    """

    def __init__(self, game):
        self._game = game
        self._first_player = 1
        self._action_space = Discrete(num=game.getActionSize())

    def transition(self, state, action):
        """Perform `action` in `state`. Return outcome.

        Args:
            state (BoardState): Canonical board game (from perspective of current player).
            action (int): Board game action.

        Returns:
            BoardState: Next canonical board game state (from perspective of next player).
            float: 1 if current player won, -1 if current player lost, 0 for draw (it it's
                   terminal state).
        """

        # In whole MDP we operate only on canonical board representations.
        # Canonical means, that it's from perspective of current player.
        # From perspective of some player means that he is 1 on the board.
        next_state = self._game.getNextState(state.raw, 1, action)
        # Draw has some small value, truncate it and leave only:
        # -1 (lose), 0 (draw), 1 (win)
        reward = float(int(self._game.getGameEnded(next_state[0], 1)))
        canonical_state = self._game.getCanonicalForm(*next_state)

        return BoardState(canonical_state), reward

    def get_init_state(self):
        """Prepare and return initial state.

        Returns:
            BoardState: Initial state.
        """

        # We need to represent init state from perspective of starting player.
        # Otherwise different first players could have different starting conditions e.g in Othello.
        init_state = self._game.getCanonicalForm(self._game.getInitBoard(), self.first_player)
        return BoardState(init_state)

    def get_valid_actions(self, state):
        """Get available actions in `state`.

        Args:
            state (BoardState): Canonical board game (from perspective of current player).

        Returns:
            np.ndarray: Array with available moves numbers in given state.
        """

        valid_moves_map = self._game.getValidMoves(state.raw, 1).astype(bool)
        return np.arange(valid_moves_map.shape[0])[valid_moves_map]

    def is_terminal_state(self, state):
        """Check if `state` is terminal.

        Args:
            state (BoardState): MDP's state.

        Returns:
            bool: Whether state is terminal or not.
        """

        return self._game.getGameEnded(state.raw, 1) != 0

    @property
    def action_space(self):
        """Discrete: Discrete action space."""

        return self._action_space

    @property
    def state_space(self):
        """tuple: A tuple of board dimensions."""

        return self._game.getBoardSize()

    @property
    def first_player(self):
        """Access first player in initial state."""

        return self._first_player

    @first_player.setter
    def first_player(self, value):
        """value (int): Set first player in initial state."""

        assert value == 1 or value == -1, "First player can be only 1 or -1!"
        self._first_player = value


class BoardGameEnv(Callback, Environment):
    """Environment for board games from https://github.com/suragnair/alpha-zero-general

    Args:
        game (Game): Board game object.

    Note:
        step(...) returns reward from perspective of player one!
        Also to alternate starting player between episodes add this object to loop as callback too.
    """

    def __init__(self, game):
        self._game = game
        self._first_player = 1
        self._last_action = -1
        self._last_player = -1
        self._action_space = Discrete(num=game.getActionSize())

    def step(self, action):
        next_state = self._game.getNextState(*self.current_state, action)

        # Current player took action, get reward from perspective of player one
        end = self._game.getGameEnded(next_state[0], 1)
        # Draw has some small value, truncate it and leave only:
        # -1 (lose), 0 (draw), 1 (win)
        reward = float(int(end))

        self._last_action = action
        self._last_player = self.current_state[1]

        self._current_state = next_state
        return next_state, reward, end != 0, None

    def reset(self, train_mode=True):
        self.train_mode = train_mode
        # We need to represent init state from perspective of starting player.
        # Otherwise different first players could have different starting conditions e.g in Othello.
        self._current_state = (self._game.getCanonicalForm(self._game.getInitBoard(), self._first_player),
                               self._first_player)

        return self.current_state

    def render(self, fancy=False):
        """Display board when environment is in test mode.

        Args:
            fancy (bool): Display a fancy 2D board.
        """

        print("Player {}, Action {}".format(
            self._last_player, self._last_action))
        if fancy and self.current_state[0].ndim == 2:
            self.render_fancy_board()
        else:
            print(self.current_state[0])

    def render_fancy_board(self):
        def line_sep(length):
            print(" ", end="")
            for _ in range(length):
                print("=", end="")
            print("")

        state = self.current_state[0].astype(int)
        m, n = state.shape
        line_sep(3 * n + 1)
        legend = {1: "X", -1: "O"}
        for i in range(m):
            print("|", end=" ")
            for j in range(n):
                s = legend.get(state[i][j], "-")
                if (i * m + j) == self._last_action:
                    print("\033[1m{:2}\033[0m".format(s), end=" ")
                else:
                    print("{:2}".format(s), end=" ")
            print("|")
        line_sep(3 * n + 1)

    def on_episode_end(self, episode, train_mode):
        """Event after environment was reset.

        Args:
            episode (int): Episode number.
            train_mode (bool): Informs whether episode is in training or evaluation mode.

        Note:
            You can assume, that this event occurs after step to terminal state.
        """

        # Alternate starting player between episodes
        self._first_player *= -1

    @property
    def action_space(self):
        """Discrete: Discrete action space."""

        return self._action_space

    @property
    def state_space(self):
        """tuple: A tuple of board dimensions."""

        return self._game.getBoardSize()

    @property
    def current_state(self):
        """object: Current state."""

        return self._current_state

    @property
    def valid_actions(self):
        """np.ndarray: A binary vector of length self.action_space(), 1 for moves that are
               valid from the current state, 0 for invalid moves."""

        valid_moves_map = self._game.getValidMoves(*self.current_state).astype(bool)
        return np.arange(valid_moves_map.shape[0])[valid_moves_map]

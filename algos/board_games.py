import pandas as pd

from common_utils import Storage
from env import BoardState
from humblerl import Callback, Mind, Vision


class AdversarialMinds(Mind, Callback):
    """Wraps two minds and dispatch work to appropriate one based on player id in state.

    Args:
        one (Mind): Mind which will plan for player "1".
        two (Mind): Mind which will plan for player "-1".
    """

    def __init__(self, one, two):
        # Index '1' for player one, index '-1' for player two
        self.players = [None, one, two]

    def plan(self, state, train_mode, debug_mode):
        """Conduct planning on state.

        Args:
            state tuple(numpy.ndarray, int): State of game to plan on and current player id.
            train_mode (bool): Informs planner whether it's in training mode and should enable
                               additional exploration.
            debug_mode (bool): Informs planner whether it's in debug mode or not.

        Returns:
            np.ndarray: Planning result, unnormalized action probabilities.
            dict: Planning metrics.
        """

        board, player = state
        return self.players[player].plan(BoardState(board), train_mode, debug_mode)

    def clear_tree(self):
        """Clear search tree of players."""

        self.players[1].clear_tree()
        self.players[-1].clear_tree()

    def on_episode_start(self, episode, train_mode):
        """Empty search tree between episodes if in train mode."""

        if train_mode:
            self.clear_tree()


class BoardVision(Vision):
    """Transforms board game state and reward to canonical one.

    Args:
        game (Game): Board game object.
    """

    def __init__(self, game):
        self.game = game

    def __call__(self, state, reward=0.):
        """Transform board game state and reward:

        Args:
            state (tuple): Board and player packed in tuple.
            reward (float): Transition reward. (Default: 0.)

        Returns:
            state (np.ndarray): Canonical board game (from perspective of current player).
            reward (float): Canonical transition reward (from perspective of current player).
        """

        board, player = state
        cannonical_state = self.game.getCanonicalForm(board, player)
        # WARNING! SHIT CODE... Please refactor if you have better idea.
        # Reward from env is from player one perspective, so we multiply reward by player
        # id which is 1 for player one or -1 player two. We also multiply by -1 because this is
        # id of "next player", and we want to represent reward from perspective of current player.
        cannonical_reward = reward * player * -1

        return (cannonical_state, player), cannonical_reward


class BoardRender(Callback):
    def __init__(self, env, render, fancy=False):
        self.env = env
        self.render = render
        self.fancy = fancy

    def on_step_taken(self, step, transition, info):
        self.do_render()

    def do_render(self):
        if self.render:
            self.env.render(self.fancy)


class BoardStorage(Storage):
    """Wraps Storage callback to unpack state from board game.

    Args:
        config (Config): Configuration object with parameters from .json file.
    """

    def __init__(self, config):
        super(BoardStorage, self).__init__(
            out_path=config.storage["save_data_path"],
            exp_replay_size=config.storage["exp_replay_size"],
            gamma=config.planner["gamma"]
        )

    def _create_small_package(self, transition):
        return (transition.state[0], transition.reward, self._recent_action_probs)


class ELOScoreboard(object):
    """Calculates and keeps players ELO statistics.

    Args:
        players_ids (list): List of players ids (weights file name will be fine).
        init_elo (float): Initial ELO of each player. (Default: 1000)

    Look at https://github.com/rshk/elo for reference

    Note:
        This is supposed to be used with board games.
    """

    def __init__(self, players_ids, init_elo=1000):
        self.scores = pd.DataFrame(index=players_ids, columns=['elo'])
        self.scores.loc[:] = init_elo

    @staticmethod
    def _get_expected_score(A, B):
        """Calculate expected score of A in a match against B.

        Args:
            A (int): Elo rating for player A.
            B (int): Elo rating for player B.
        """

        return 1 / (1 + 10 ** ((B - A) / 400))

    @staticmethod
    def _get_updated_elo(old, exp, score, k=32):
        """Calculate the new Elo rating for a player.

        Args:
            old (int): The previous Elo rating.
            exp (float): The expected score for this match.
            score (float): The actual score for this match.
            k (int): The k-factor for Elo (default: 32).
        """

        return old + k * (score - exp)

    @staticmethod
    def calculate_update(p1_elo, p2_elo, p1_wins, p2_wins, draws):
        """Update ELO rating of two players after their matches.

        Args:
            p1_elo (float): Player one ELO.
            p2_elo (float): Player two ELO.
            p1_wins (int): Number of player one wins.
            p2_wins (int): Number of player two wins.
            draws (int): Number of draws between players.

        Return:
            float: Player one updated ELO rating.
            float: Player two updated ELO rating.
        """

        n_games = p1_wins + p2_wins + draws

        p1_score = p1_wins + .5 * draws
        p1_expected = ELOScoreboard._get_expected_score(
            p1_elo, p2_elo) * n_games

        p2_score = p2_wins + .5 * draws
        p2_expected = ELOScoreboard._get_expected_score(
            p2_elo, p1_elo) * n_games

        p1_updated = ELOScoreboard._get_updated_elo(
            p1_elo, p1_expected, p1_score)
        p2_updated = ELOScoreboard._get_updated_elo(
            p2_elo, p2_expected, p2_score)

        return p1_updated, p2_updated

    @staticmethod
    def load_csv(path):
        """Loads ELO scoreboard from .csv file.

        Args:
            path (str): Path to .csv file with data.

        Returns:
            ELOScoreboard: ELO scoreboard object with loaded data.
        """

        df = pd.read_csv(path, index_col=0, header=None)
        return ELOScoreboard(df.index, df.values)

    def save_csv(self, path):
        """Saves ELO scoreboard to .csv file.

        Args:
            path (str): Path to destination .csv file.
        """

        self.scores.to_csv(path, header=False)

    def update_player(self, player_id, opponents_elo, wins, draws, n_games=2):
        """Update ELO rating of player after matches with opponents.

        Args:
            player_id (str): Player identifier.
            opponents_elo (list of int): ELO ratings of opponent(s).
            wins (int): Number of player wins.
            draws (int): Number of draws between players.
            n_games (int): Number of games played between each pair. (Default: 2)
        """

        if not hasattr(opponents_elo, "__iter__"):
            opponents_elo = [opponents_elo, ]

        player_score = wins + .5 * draws
        player_elo = self.scores.loc[player_id, 'elo']

        expected_score = 0
        for opponent_elo in opponents_elo:
            expected_score += self._get_expected_score(
                player_elo, opponent_elo) * n_games

        updated_elo = self._get_updated_elo(
            player_elo, expected_score, player_score)
        self.scores.loc[player_id, 'elo'] = updated_elo

    def update_players(self, p1_id, p2_id, p1_wins, p2_wins, draws):
        """Update ELO rating of two players after their matches.

        Args:
            p1_id (str): Player one identifier.
            p2_id (str): Player two identifier.
            p1_wins (int): Number of player one wins.
            p2_wins (int): Number of player two wins.
            draws (int): Number of draws between players.
        """

        p1_elo = self.scores.loc[p1_id, 'elo']
        p2_elo = self.scores.loc[p2_id, 'elo']

        p1_updated, p2_updated = \
            self.calculate_update(p1_elo, p2_elo, p1_wins, p2_wins, draws)

        self.scores.loc[p1_id, 'elo'] = p1_updated
        self.scores.loc[p2_id, 'elo'] = p2_updated

    def plot(self, ax=None):
        """Plot players ELO ratings.

        Args:
            ax (matplotlib.axes.Axes): Axis to plot in. If None, then plot on global axis.
                                       (Default: None)
        """

        import matplotlib.pyplot as plt

        self.scores.plot(ax=ax)
        plt.show()

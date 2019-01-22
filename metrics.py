from abc import ABCMeta, abstractmethod
from humblerl import Callback

from algos.board_games import ELOScoreboard


class Tournament(Callback):
    """Calculates winning rates of player one (wannabe) and player two (best) and draws.

    Args:
        update_threshold (float): If current player win count divided by number of games (draws
            doesn't count) is greater then this threshold, then current player is better then
            opponent. (Default: 0.5)

    Note:
        This is supposed to be used with board games.
    """

    def __init__(self, update_threshold=.5):
        self.threshold = update_threshold

    def on_loop_start(self):
        self.reset()

    def on_step_taken(self, step, transition, info):
        if transition.is_terminal:
            # NOTE: Because players have fixed player id, and reward is returned from perspective
            #       of current player, we transform it into perspective of player one and check
            #       who wins.
            player = transition.state[1]
            reward = player * transition.reward
            if reward == 0:
                self.draws += 1
            elif reward > 0:
                self.wannabe += 1
            else:
                self.best += 1

    def reset(self):
        self.wannabe, self.best, self.draws = 0, 0, 0

    @property
    def metrics(self):
        return {"wannabe": self.wannabe, "best": self.best, "draws": self.draws}

    @property
    def results(self):
        return self.wannabe, self.best, self.draws

    def compare(self, other_score):
        """Compare two agents, one that you are and the other one.

        Args:
            other_score (float): Other agent score.

        Return:
            float: Current agent score.
            bool: If current agent is better then other agent.
        """

        wins, losses, draws = self.results

        # Update ELO rating, use best player ELO as current player ELO
        # NOTE: We update it this way as we don't need exact ELO values, we just need to see
        #       how much if at all has current player improved.
        #       Decision based on: https://github.com/gcp/leela-zero/issues/354
        current_score, _ = ELOScoreboard.calculate_update(
            other_score, other_score, wins, losses, draws)

        return current_score, wins > 0 and float(wins) / (wins + losses) > self.threshold

import logging as log
import os.path
import humblerl as hrl
import utils

from abc import ABCMeta, abstractmethod
from keras.callbacks import TensorBoard

from algos.alphazero import Planner
from algos.board_games import AdversarialMinds, BoardStorage, BoardVision
from nn import build_keras_trainer
from metrics import Tournament
from humblerl.callbacks import BasicStats, CSVSaverWrapper


class Coach(object):
    """AlphaZero coach, all the operations to train and evaluate algorithm.

    Args:
        config (Config): Configuration loaded from .json file.
        best_ckpt (string): Path to best agent nn weights. If None, then newest ckpt file in
            ckpt dir from config is taken. (Default: None)
        current_ckpt (string): Path to current agent (trained one) nn weights. If None, then the
            weights are copied from best nn. (Default: None)

    Attributes:
        cfg (Config): Configuration loaded from .json file.
        env (hrl.Environment): Environment to play in.
        vision (hrl.Vision): state and reward preprocessing.
        best_nn (NeuralNet): Value and Policy network of best agent.
        current_nn (NeuralNet): Value and Policy network of current agent.
        best_mind (hrl.Mind): Best agent's mind.
        current_mind (hrl.Mind): Current agent's mind.
        storage (Storage): Experience replay buffer.
        scoreboard (hrl.Callback): Callback that measure agent's score.
        play_callbacks (list): Play phase callbacks for `hrl.loop(...)`.
        train_callbacks (list): Train phase callbacks for `NeuralNet.train(...)`.
        eval_callbacks (list): Evaluation phase callbacks for `hrl.loop(...)`.
        global_epoch (int): Current epoch of training (play->train->evaluate iteration).
        best_score (float): Current best agent score.
    """

    def __init__(self, config, best_ckpt=None, current_ckpt=None):
        self.cfg = config
        self.env = self.cfg.env

        self.vision = BoardVision(self.cfg.game)

        self.best_nn = build_keras_trainer(self.cfg.game, self.cfg)
        self.current_nn = build_keras_trainer(self.cfg.game, self.cfg)

        best_player = Planner(self.cfg.mdp, self.best_nn.model, self.cfg.planner)
        current_player = Planner(self.cfg.mdp, self.current_nn.model, self.cfg.planner)
        self.best_mind = AdversarialMinds(best_player, best_player)
        self.current_mind = AdversarialMinds(current_player, best_player)

        self.storage = BoardStorage(self.cfg)
        # Load storage date from disk (path in config)
        self.storage.load()

        self.scoreboard = CSVSaverWrapper(
            Tournament(self.cfg.self_play["update_threshold"]),
            self.cfg.logging['save_tournament_log_path'], True)

        self.play_callbacks = [
            CSVSaverWrapper(BasicStats(), self.cfg.logging['save_self_play_log_path'])]
        self.train_callbacks = [
            TensorBoard(log_dir=utils.create_tensorboard_log_dir(
                self.cfg.logging['tensorboard_log_folder'], 'self_play'))]
        self.eval_callbacks = [self.cfg.env]  # env added to alternate starting player

        # Load best nn checkpoint if available
        try:
            if best_ckpt:
                ckpt_path = best_ckpt
            else:
                ckpt_dir = self.cfg.logging['save_checkpoint_folder']
                ckpt_path = os.path.join(ckpt_dir, utils.get_newest_ckpt_fname(ckpt_dir))

            self.best_nn.load_checkpoint(ckpt_path)
            self.global_epoch = utils.get_checkpoints_epoch(ckpt_path)
            self.best_score = utils.get_checkpoints_elo(ckpt_path)

            log.info("Best mind has loaded latest checkpoint: %s", ckpt_path)
        except BaseException:
            log.info("No initial checkpoint, starting tabula rasa.")
            self.global_epoch = 0
            self.best_score = 1000

        if current_ckpt:
            # Load current nn checkpoint
            self.current_nn.load_checkpoint(current_ckpt)
        else:
            # Copy best nn weights to current nn that will be trained
            self.current_nn.model.set_weights(self.best_nn.model.get_weights())

    def play(self, desc="Play"):
        """Self-play phase, gather data using best nn and save to storage.

        Args:
            desc (str): Progress bar description.
        """

        hrl.loop(self.env, self.best_mind, self.vision, policy='proportional', trian_mode=True,
                 warmup=self.cfg.self_play['policy_warmup'],
                 debug_mode=self.cfg.debug, n_episodes=self.cfg.self_play['n_self_plays'],
                 name=desc, verbose=1,
                 callbacks=[self.best_mind, self.storage, *self.play_callbacks])

        # Store gathered data
        self.storage.store()

    def train(self):
        """Training phase, improve neural net."""

        self.global_epoch = self.current_nn.train(self.storage.big_bag,
                                                  initial_epoch=self.global_epoch,
                                                  callbacks=self.train_callbacks)

    def evaluate(self, desc="Evaluation", tournament_mode=False, render_mode=False, n_games=None):
        """Evaluation phase, check how good current mind is.

        Args:
            desc (str): Progress bar description.
            tournament_mode (bool): If current agent should be compared to best too or only evaluated.
            render_mode (bool): Enable rendering game. (Default: False)
            n_games (int): How many games to play. If None, then value is taken from config.
                (Default: None)

        Note:
            `self.scoreboard` should measure and keep performance of mind
            from last call to `hrl.loop`.
        """
        # Clear current agent tree and evaluate it
        self.current_mind.clear_tree()
        hrl.loop(self.env, self.current_mind, self.vision, policy='deterministic', train_mode=False,
                 debug_mode=self.cfg.debug, render_mode=render_mode,
                 n_episodes=n_games if n_games else self.cfg.self_play['n_tournaments'], name=desc,
                 verbose=2, callbacks=[self.scoreboard, *self.eval_callbacks])

        if tournament_mode:
            current_score, is_better = self.scoreboard.unwrapped.compare(self.best_score)
            if is_better:
                self._update_best(current_score)

    def _update_best(self, best_score):
        """Updates best score and saves current nn as new best.

        Args:
            best_score (float): New best agent score.
        """

        self.best_score = best_score

        # Create checkpoint file name and log it
        best_fname = utils.create_checkpoint_file_name('self_play',
                                                       self.cfg.self_play["game"],
                                                       self.global_epoch,
                                                       self.best_score)
        log.info("New best player: %s", best_fname)

        # Save best and exchange weights
        self.current_nn.save_checkpoint(self.cfg.logging['save_checkpoint_folder'], best_fname)
        self.best_nn.model.set_weights(self.current_nn.model.get_weights())

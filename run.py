#!/usr/bin/env python3
import click
import humblerl as hrl
import logging as log
import numpy as np
import utils

from algos.alphazero import Planner
from algos.board_games import AdversarialMinds, BoardRender, BoardStorage, BoardVision, ELOScoreboard
from algos.human import HumanPlayer
from coach import Coach
from common_utils import TensorBoardLogger, mute_tf_logs_if_needed
from keras.callbacks import EarlyStopping, TensorBoard
from metrics import Tournament
from nn import build_keras_trainer
from tabulate import tabulate
from utils import Config


@click.group()
@click.pass_context
@click.option('-c', '--config', type=click.Path(exists=False),
              help="Path to configuration file (Default: config.json)", default="config.json")
@click.option('--debug/--no-debug', help="Enable debug logging (Default: False)", default=False)
def cli(ctx, config, debug):
    # Get and set up logger level and formatter
    mute_tf_logs_if_needed()
    log.basicConfig(level=log.DEBUG if debug else log.INFO,
                    format="[%(levelname)s]: %(message)s")

    # Create context with config
    ctx.obj = Config(config, debug)


@cli.command()
@click.pass_context
def self_play(ctx):
    """Train by self-play, retraining from self-played frames and changing best player when
    new trained player beats currently best player.

    Args:
        ctx (click.core.Context): context object.
            Parameters for training:
                * 'game' (string)                     : game name (Default: tictactoe)
                * 'max_iter' (int)                    : number of train process iterations
                                                        (Default: -1)
                * 'min_examples' (int)                : minimum number of examples to start training
                                                        nn, if -1 then no threshold. (Default: -1)
                * 'policy_warmup' (int)               : how many stochastic warmup steps should take
                                                        deterministic policy (Default: 12)
                * 'n_self_plays' (int)                : number of self played episodes
                                                        (Default: 100)
                * 'n_tournaments' (int)               : number of tournament episodes (Default: 20)
                * 'save_checkpoint_folder' (string)   : folder to save best models
                                                        (Default: "checkpoints")
                * 'save_checkpoint_filename' (string) : filename of best model (Default: "best")
                * 'save_self_play_log_path' (string)  : where to save self-play logs.
                                                        (Default: "./logs/self-play.log")
                * 'save_tournament_log_path' (string) : where to save tournament logs.
                                                        (Default: "./logs/tournament.log")
                * 'update_threshold' (float):         : required threshold to be new best player
                                                        (Default: 0.55)
    """

    cfg = ctx.obj
    coach = Coach(cfg)

    # Create TensorBoard logger
    tb_logger = TensorBoardLogger(utils.create_tensorboard_log_dir(
        cfg.logging['tensorboard_log_folder'], 'score'))

    iteration = coach.global_epoch // cfg.training['epochs']
    while cfg.self_play["max_iter"] == -1 or iteration < cfg.self_play["max_iter"]:
        iter_counter_str = "{:03d}/{:03d}".format(iteration + 1, cfg.self_play["max_iter"]) \
            if cfg.self_play["max_iter"] > 0 else "{:03d}/inf".format(iteration + 1)

        coach.play("Self-play  " + iter_counter_str)

        # Proceed to training only if threshold is fulfilled
        if len(coach.storage.big_bag) <= cfg.self_play["min_examples"]:
            log.warning(
                "Skip training, gather minimum %d training examples!",
                cfg.self_play["min_examples"]
            )
            continue

        coach.train()
        coach.evaluate("Tournament " + iter_counter_str, tournament_mode=True)

        # Log current player's score
        tb_logger.log_scalar("Best score", coach.best_score, iteration)

        # Increment iterator
        iteration += 1


@cli.command()
@click.pass_context
@click.option('-ckpt', '--checkpoint', help="Path to NN checkpoint, if None then start fresh (Default: None)", type=click.Path(), default=None)
@click.option('-save', '--save-dir', help="Dir where to save NN checkpoint, if None then don't save (Default: None)", type=click.Path(), default=None)
@click.option('--tensorboard/--no-tensorboard', help="Enable tensorboard logging (Default: False)", default=False)
def train(ctx, checkpoint, save_dir, tensorboard):
    """Train NN from passed configuration."""

    cfg = ctx.obj
    coach = Coach(cfg, checkpoint)

    # Create TensorBoard logging callback if enabled
    if tensorboard:
        coach.train_callbacks = [
            TensorBoard(log_dir=utils.create_tensorboard_log_dir(
                cfg.logging['tensorboard_log_folder'], 'train'))]
    else:
        coach.train_callbacks = []

    coach.train()

    # Save model checkpoint if path passed
    if save_dir:
        save_fname = utils.create_checkpoint_file_name(
            'train', cfg.self_play["game"], coach.global_epoch, coach.best_score)
        coach.current_nn.save_checkpoint(save_dir, save_fname)


@cli.command()
@click.pass_context
@click.option('-n', '--n-steps', help="Number of optimization steps (Default: 100)", default=100)
def hopt(ctx, n_steps):
    """Hyper-parameter optimization.
       All hyperparameters (except loss function) passed to config as list are optimized.
    """

    import os
    from skopt import gp_minimize
    from skopt.plots import plot_convergence
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    import matplotlib.pyplot as plt

    cfg = ctx.obj

    # Create storage and load data
    storage = BoardStorage(cfg)
    storage.load()

    # Prepare training data
    trained_data = storage.big_bag
    boards_input, target_pis, target_values = list(zip(*trained_data))

    data = np.array(boards_input)
    targets = [np.array(target_pis), np.array(target_values)]

    # Prepare search space
    space = []
    num_parameters_to_optimize = 0
    for k, v in cfg.nn.items():
        # Ignore loss in hyper-param tuning
        if isinstance(v, list) and k != "loss":
            num_parameters_to_optimize += 1
            if isinstance(v[0], float):
                space.append(Real(v[0], v[1], name=k))
            elif isinstance(v[0], int):
                space.append(Integer(v[0], v[1], name=k))
            else:
                space.append(Categorical(v, name=k))

    assert num_parameters_to_optimize > 0

    @use_named_args(space)
    def objective(**params):
        # Prepare neural net parameters
        for k, v in params.items():
            cfg.nn[k] = v

        # Build Keras neural net model
        model = build_keras_trainer(cfg.game, cfg).model

        # Fit model
        history = model.fit(data, targets,
                            batch_size=cfg.training["batch_size"],
                            epochs=cfg.training['epochs'],
                            validation_split=0.2,
                            callbacks=[EarlyStopping(patience=7)],
                            verbose=0)

        return history.history['val_loss'][-1]

    # Perform hyper-parameter bayesian optimization
    model_gp = gp_minimize(objective, space, n_calls=n_steps, verbose=True)

    # Print results
    print("Best score: {}".format(model_gp.fun))
    print("Best parameters:")
    for i, dim in enumerate(space):
        print("\t{} = {}".format(dim.name, model_gp.x[i]))

    # Plot convergence
    if "DISPLAY" in os.environ:
        _ = plot_convergence(model_gp)
        plt.savefig("hopt_convergence.png")


@cli.command()
@click.pass_context
@click.argument('first-model-path', nargs=1, type=click.Path(exists=True))
@click.argument('second-model-path', nargs=1, type=click.Path(exists=True))
@click.option('--render/--no-render', help="Enable rendering game (Default: True)", default=True)
@click.option('-n', '--n-games', help="Number of games (Default: 2)", default=2)
def clash(ctx, first_model_path, second_model_path, render, n_games):
    """Test two models. Play `n_games` between themselves.

        Args:
            first_model_path: (string): Path to player one model.
            second_model_path (string): Path to player two model.
    """

    cfg = ctx.obj
    coach = Coach(cfg, current_ckpt=first_model_path, best_ckpt=second_model_path)

    coach.scoreboard = Tournament()
    coach.evaluate(
        desc="Test models: {} vs {}".format(
            first_model_path.split("/")[-1], second_model_path.split("/")[-1]),
        render_mode=render,
        n_games=n_games
    )

    log.info("%s vs %s results: %s",
             first_model_path.split("/")[-1],
             second_model_path.split("/")[-1],
             coach.scoreboard.results)


@cli.command()
@click.pass_context
@click.argument('model-path', nargs=1, type=click.Path(exists=True))
@click.option('-n', '--n-games', help="Number of games (Default: 2)", default=2)
def human_play(ctx, model_path, n_games):
    """Play `n_games` with trained model.

        Args:
            model_path: (string): Path to trained model.
    """

    cfg = ctx.obj
    coach = Coach(cfg, model_path)

    coach.current_mind.players[1] = HumanPlayer(cfg.mdp)
    coach.eval_callbacks.append(BoardRender(cfg.env, render=True, fancy=True))
    coach.scoreboard = Tournament()

    coach.evaluate(
        desc="Test models: Human vs. {}".format(model_path.split("/")[-1]),
        n_games=n_games
    )

    log.info("Human vs. %s results: %s",
             model_path.split("/")[-1],
             coach.scoreboard.results)


@cli.command()
@click.pass_context
@click.option('-d', '--checkpoints-dir', type=click.Path(exists=True), default=None,
              help="Path to checkpoints. If None then take from config (Default: None)")
@click.option('-g', '--gap', help="Gap between versions of best model (Default: 2)", default=2)
@click.option('-sc', '--second-config', type=click.File('r'),
              help="Path to second configuration file", default=None)
def cross_play(ctx, checkpoints_dir, gap, second_config):
    """Validate trained models. Best networks play with each other."""

    cfg = ctx.obj
    second_cfg = Config(second_config) if second_config is not None else cfg

    # Create board games vision
    vision = BoardVision(cfg.game)

    # Set checkpoints_dir if not passed
    if checkpoints_dir is None:
        checkpoints_dir = cfg.logging['save_checkpoint_folder']

    # Create players and their minds
    first_player_trainer = build_keras_trainer(cfg.game, cfg)
    second_player_trainer = build_keras_trainer(second_cfg.game, second_cfg)
    first_player = Planner(cfg.mdp, first_player_trainer.model, cfg.planner)
    second_player = Planner(second_cfg.mdp, second_player_trainer.model, second_cfg.planner)
    players = AdversarialMinds(first_player, second_player)

    # Create callbacks
    tournament = Tournament()

    # Get checkpoints paths
    all_checkpoints_paths = utils.get_checkpoints_for_game(
        checkpoints_dir, cfg.self_play["game"])

    # Reduce gap to play at least one game when there is more than one checkpoint
    if gap >= len(all_checkpoints_paths):
        gap = len(all_checkpoints_paths) - 1
        log.info("Gap is too big. Reduced to %d", gap)

    # Gather players ids and checkpoints paths for cross-play
    players_ids = []
    checkpoints_paths = []
    for idx in range(0, len(all_checkpoints_paths), gap):
        players_ids.append(idx)
        checkpoints_paths.append(all_checkpoints_paths[idx])

    # Create table for results, extra column for player id
    results = np.zeros(
        (len(checkpoints_paths), len(checkpoints_paths)), dtype=int)

    # Create ELO scoreboard
    elo = ELOScoreboard(players_ids)

    for i, (first_player_id, first_checkpoint_path) in enumerate(zip(players_ids, checkpoints_paths)):
        first_player_trainer.load_checkpoint(first_checkpoint_path)

        tournament_wins = tournament_draws = 0
        opponents_elo = []
        for j in range(i + 1, len(players_ids)):
            second_player_id, second_checkpoint_path = players_ids[j], checkpoints_paths[j]
            second_player_trainer.load_checkpoint(second_checkpoint_path)

            # Clear players tree
            first_player.clear_tree()
            second_player.clear_tree()

            hrl.loop(cfg.env, players, vision, policy='deterministic', n_episodes=2,
                     train_mode=False, name="{} vs {}".format(first_player_id, second_player_id),
                     callbacks=[tournament, cfg.env])

            wins, losses, draws = tournament.results

            # Book keeping
            tournament_wins += wins
            tournament_draws += draws

            results[i][j] = wins - losses
            results[j][i] = losses - wins

            opponents_elo.append(elo.scores.loc[second_player_id, 'elo'])

            # Update ELO rating of second player
            elo.update_player(second_player_id, elo.scores.loc[first_player_id, 'elo'],
                              losses, draws)

        # Update ELO rating of first player
        elo.update_player(first_player_id, opponents_elo,
                          tournament_wins, tournament_draws)

    # Save elo to csv
    elo.save_csv(cfg.logging['save_elo_scoreboard_path'])

    scoreboard = np.concatenate(
        (np.array(players_ids).reshape(-1, 1), results,
         np.sum(results, axis=1).reshape(-1, 1),
         elo.scores.elo.values.reshape(-1, 1).astype(np.int)),
        axis=1
    )

    tab = tabulate(scoreboard, headers=players_ids + ["sum", "elo"], tablefmt="fancy_grid")
    log.info("Results:\n%s", tab)
    for player_id, player_elo, checkpoint_path in zip(players_ids, elo.scores['elo'], checkpoints_paths):
        log.info("ITER: %3d, ELO: %4d, PATH: %s", player_id, int(player_elo), checkpoint_path)


if __name__ == "__main__":
    cli()

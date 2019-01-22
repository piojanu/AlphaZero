[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# AlphaZero
Implementation of [AlphaZero paper](https://arxiv.org/abs/1712.01815) using [HumbleRL](https://github.com/piojanu/humblerl)
framework. Allows to easily train an AlphaZero model to play board games, such as Tic-Tac-Toe, Othello, Connect 4
or Go Bang (Gomoku).

## Getting started
Clone this repository and install requirements: `pip install -r requirements.txt`

## Basic usage
`python run.py [OPTIONS] COMMAND ARGS`

To see script-level help, run: `python run.py --help`.
Note: By default, logs from TensorFlow are hidden by setting `TF_CPP_MIN_LOG_LEVEL` environment variable to value of 3. To change it, define environment variable `TF_CPP_MIN_LOG_LEVEL`.

### Commands
Commands correspond to different actions you can perform with `run.py`.

To see available commands, run: `python run.py --help`.

To see command-level help, run: `python run.py COMMAND --help`.

Commands are described below, in context of training an AlphaZero model from scratch.

## Config
Parameters used in training and evaluation are stored in JSON config.

By default, `config.json.dist` will be used. To customize configuration you can provide your own config using option `-c PATH`.
Default custom config filename is `config.json`.

If you don't specify some parameter in .json config, then default value from `config.json.dist` is used.

### Neural Network
AlphaZero is based on Monte Carlo Tree Search (MCTS) that uses a neural network for policy (prior probability, pi)
and value predictions.

Neural network is structured like this:

`Input Convolutional layer -> Residual blocks -> final feature extractor -> pi/value heads`

Config parameters:
```
"neural_net": {
    "conv_filters"             : 256,       -- Input Convolutional layer: number of filters
    "conv_kernel"              : 3,         -- Input Convolutional layer: kernel size (filter height/width)
    "conv_stride"              : 1,         -- Input Convolutional layer: stride
    "residual_bottleneck"      : 128,       -- Residual layer: number of bottleneck filters (should be lower than residual_filters)
    "residual_filters"         : 256,       -- Residual layer: number of filters
    "residual_kernel"          : 3,         -- Residual layer: kernel size
    "residual_num"             : 3,         -- Residual layer: number of residual blocks
    "feature_extractor"        : "agz",     -- Final feature extractor: agz/avgpool/flatten
    "dense_size"               : 256,       -- Size of last (dense) layer
    "loss"                     : ["categorical_crossentropy", "mean_squared_error"],    -- Loss for pi and value heads
    "l2_regularizer"           : 0.0001,    -- L2 regularizer strength
    "lr"                       : 0.02,      -- Learning rate
    "momentum"                 : 0.9        -- SGD's momentum
},
"training": {
    "batch_size"               : 256,                   -- Batch size used when training the NN
    "epochs"                   : 25,                    -- Number of epochs for training the NN
    "save_training_log_path"   : "./logs/training.log"  -- Path to training logs (values of training metrics)
},
```

### MCTS Planner
Actions are chosen by performing simulations with MCTS, which uses Upper Confidence Tree (UCT) score to select nodes.

Config parameters:
```
"planner": {
    "c"                        : 1.0,       -- UCT exploration-exploitation trade-off param
    "dirichlet_noise"          : 0.3,       -- Dirichlet noise added to root prior
    "noise_ratio"              : 0.25,      -- Noise contribution to prior probabilities (Default: 0.25)
    "gamma"                    : -1.0,      -- Discounting factor for value. In 2-player zero-sum games (like board games)
                                            -- opponents' returns are mutual negations, so gamma should be < 0.
    "n_simulations"            : 50,        -- Number of simulations to perform before choosing action, can be "inf" if timeout is set.
    "timeout"                  : "inf"      -- Timeout value for performing simulations, can be "inf" if n_simulations is set.
}
```

### Logging
A variety of logging information is stored throught the training and evaluation process.

Config parameters:
```
"logging": {
    "save_checkpoint_folder"   : "./checkpoints",           -- Path where model's checkpoints are stored
    "tensorboard_log_folder"   : "./logs/tensorboard",      -- Path where tensorboard logs are stored
    "save_self_play_log_path"  : "./logs/self-play.log",    -- Path to self-play logs
    "save_tournament_log_path" : "./logs/tournament.log",   -- Path to tournament logs
    "save_elo_scoreboard_path" : "./logs/scoreboard.csv"    -- Path to ELO scoreboard logs.
}
```

### Storage
For training the NN, we need to store some transitions.

Config parameters:
```
"storage": {
    "exp_replay_size"          : 200000,                        -- Maximum number of transitions that can be stored.
    "save_data_path"           : "./checkpoints/data.examples"  -- Path to where the storage is saved.
}
```


## Training

### Self-play
AlphaZero model is trained by playing against different versions of itself (self-play), updating the best model only when
the new model beats the previous best in tournament (number of games against each other) with given win-ratio.

To perform self-play, run:

`python run.py self_play`

Config parameters:
```
"self_play": {
    "game"                     : "othello",     -- Game name: tictactoe/othello/connect4/gobang
    "max_iter"                 : -1,            -- Number of self-play iterations, -1 means infinite
    "min_examples"             : 2000,          -- Minimum examples that must be gathered to train the NN
    "policy_warmup"            : 12,            -- Number of steps that policy is 'proportional' (weighted random)
                                                -- before switching to 'deterministic'
    "n_self_plays"             : 100,           -- Number of episodes played in each self-play loop
    "n_tournaments"            : 50,            -- Number of tournament games between current and previous best NN.
    "update_threshold"         : 0.55           -- Minimum win-ratio that must be achieved by current NN in tournament
                                                -- in order to replace previous best NN
 }
```

### Neural Network training outside of self-play
Neural network can be trained outside of self-play using stored examples gathered during previous self-play.

To perform NN training, run:

`python run.py train -save checkpoints/`

Command-line options:
```
-ckpt, --checkpoint            - Path to NN checkpoint, if None then start fresh (Default: None)
-save, --save-dir              - Dir where to save NN checkpoint, if None then don't save (Default: None)
--tensorboard/--no-tensorboard - Enable tensorboard logging (Default: False)
```

### Hyperparameter optimization
We can perform automatic hyperparameter optimization to find optimal values for our neural net hyperparameters.
To do this, set their values in config to lists. In list you need to pass two integers or floats. This is range in
which optimiser will search for optimal parameters. You can also pass list of strings. Then optimiser will search for
optimal parameter from those. Example:
```
"neural_net": {
     ...
     "residual_num"             : [1, 5],
     "feature_extractor"        : ["agz", "avgpool", "flatten"],
     "lr"                       : [0.00001, 0.1],
     ...
 }
```

"loss" parameter is not affected and can't be searched for.

To perform hyperparameter optimization, run:

`python run.py hopt`

Command-line options:
```
-n, --n-steps - Number of optimization steps (Default: 100)
```

## Evaluation
We can test our model in a few different ways.

### Clash
Test two model's checkpoints by playing a number of games between them.

To perform clash, run:

`python clash <first_ckpt_path> <second_ckpt_path>`

Command-line options:
```
first_ckpt_path      - Path to first model's checkpoint
second_ckpt_path     - Path to first model's checkpoint

--render/--no-render - Enable rendering game (Default: True)
-n, --n-games        - Number of games (Default: 2)
```

### Cross-play
Perform a tournament between the model's checkpoints, which play a number of games between each other.
At the end, a table with game results and ELO is displayed.

To perform cross-play, run:

`python cross-play`

Command-line options:
```
-d, --checkpoints-dir - Path to checkpoints. If None then take from config (Default: None)
-g, --gap             - Gap between versions of best model (Default: 2)
```

### Play against the model yourself (human-play)
Play a number of games against a trained model.

To play against a model, run:

`python human_play <model_ckpt_path>`

Command-line options:
```
model_ckpt_path - Path to trained model's checkpoint

-n, --n-games        - Number of games (Default: 2)
```

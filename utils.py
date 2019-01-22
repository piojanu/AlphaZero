import datetime as dt
import glob
import os

from common_utils import get_configs
from env import BoardGameEnv, BoardGameMDP
from games import *  # This allows to create every game from games


class Config(object):
    """Loads custom configuration, unspecified parameters are taken from default configuration.

    Args:
        config_path (str): Path to .json file with custom configuration
        debug (boolean): Specify to enable debugging features
    """

    def __init__(self, config_path, debug=False):
        default_config, custom_config = get_configs(config_path)

        # Merging default and custom configs, for repeating keys, key-value pairs from second dict are taken
        self.nn = {**default_config["neural_net"], **custom_config.get("neural_net", {})}
        self.training = {**default_config["training"], **custom_config.get("training", {})}
        self.self_play = {**default_config["self_play"], **custom_config.get("self_play", {})}
        self.logging = {**default_config["logging"], **custom_config.get("logging", {})}
        self.storage = {**default_config["storage"], **custom_config.get("storage", {})}
        self.planner = {**default_config["planner"], **custom_config.get("planner", {})}

        self.game = eval(self.self_play["game"])()
        self.env = BoardGameEnv(self.game)
        self.mdp = BoardGameMDP(self.game)
        self.debug = debug


def create_tensorboard_log_dir(logdir, prefix):
    return os.path.join(logdir, prefix, dt.datetime.now().strftime("%d-%mT%H:%M"))


def create_checkpoint_file_name(prefix, game_name, epoch, score):
    return "_".join([prefix, game_name, '{0:05d}'.format(epoch), str(int(score))]) + ".ckpt"


def get_checkpoints_epoch(filename):
    """Get checkpoint epoch from its filename"""

    return int(filename.replace('_', '.').split('.')[-3])


def get_checkpoints_elo(filename):
    """Get checkpoint epoch from its filename"""

    return int(filename.replace('_', '.').split('.')[-2])


def get_newest_ckpt_fname(dirname):
    """Looks for newest file with '.ckpt' extension in dirname."""
    list_of_files = glob.glob(os.path.join(dirname, '*.ckpt'))
    latest_file = max(list_of_files, key=get_checkpoints_epoch)

    return os.path.basename(latest_file)


def get_checkpoints_for_game(dirname, game_name):
    """Looks for files with game_name in filename and '.ckpt' extension in dirname."""
    files = list(filter(os.path.isfile,
                        glob.glob(os.path.join(dirname, '*' + game_name + '*.ckpt'))))
    files.sort(key=lambda x: get_checkpoints_epoch(x))

    return files

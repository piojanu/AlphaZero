from abc import ABCMeta, abstractmethod
import logging as log
import os

from keras.optimizers import SGD
from keras.callbacks import CSVLogger
from keras.backend import image_data_format
from keras.layers import Activation, add, BatchNormalization, Conv2D, Dense, Flatten
from keras.layers import GlobalAveragePooling2D, Input, Reshape
from keras.models import Model
from keras.regularizers import l2
import numpy as np


class ModelTrainer(metaclass=ABCMeta):
    """Model trainer.

    Args:
        model (object): Model, implementation specific.
        params (dict): Train/inference hyper-parameters.
    """

    @abstractmethod
    def __init__(self, model, params):
        pass

    @abstractmethod
    def train(self, dataset, initial_epoch, callbacks=None):
        """Perform training according.

        Args:
            dataset (object): Dataset, type depends on implementation.
            initial_epoch (int): Epoch at which to start training. (Default: 0)
            callbacks (list): Extra callbacks to pass to keras model fit method. (Default: None)
         """

        pass

    @abstractmethod
    def save_checkpoint(self, path, filename=None):
        """Saves the current neural network (with its parameters) in folder/filename.

        Args:
            path (str): Directory for saving checkpoints to or full path to file
                        if filename is None.
            filename (str): File name of saved nn checkpoint. (Default: None)
        """

        pass

    @abstractmethod
    def load_checkpoint(self, path, filename=None):
        """Loads parameters of the neural network from folder/filename.

        Args:
            path (str): Directory for loading checkpoints from or full path to file
                        if filename is None.
            filename (str): File name of saved nn checkpoint. (Default: None)
        """

        pass


class KerasTrainer(ModelTrainer):
    """Artificial neural mind of planning.

    Args:
        model (keras.Model): Neural network model.
        params (dict): Train/inference hyper-parameters.
    """

    def __init__(self, model, params):
        self.model = model
        self.batch_size = params['batch_size']
        self.epochs = params['epochs']

        # Initialize callbacks list with CSVLogger
        self.callbacks = [
            CSVLogger(params.get('save_training_log_path', './logs/training.log'), append=True)]

    def train(self, dataset, initial_epoch=0, callbacks=None):
        """Perform training according to passed parameters in `build` call.

        Args:
            dataset (object): Dataset, type depends on implementation.
            initial_epoch (int): Epoch at which to start training. (Default: 0)
            callbacks (list): Extra callbacks to pass to keras model fit method. (Default: None)

        Return:
            int: Number of training epochs to this moment.
        """

        epochs = self.epochs + initial_epoch
        callbacks = callbacks if callbacks else []

        boards_input, target_pis, target_values = list(zip(*dataset))
        self.model.fit(np.array(boards_input),
                       [np.array(target_pis), np.array(target_values)],
                       batch_size=self.batch_size,
                       epochs=epochs,
                       initial_epoch=initial_epoch,
                       callbacks=self.callbacks + callbacks)

        return epochs

    def save_checkpoint(self, path, filename=None):
        """Saves the current neural network (with its parameters) in folder/filename.

        Args:
            path (str): Directory for saving checkpoints to or full path to file
                        if filename is None.
            filename (str): File name of saved nn checkpoint. (Default: None)
        """

        if filename is None:
            filepath = path
            dirpath = os.path.dirname(path)
        else:
            filepath = os.path.join(path, filename)
            dirpath = path

        if not os.path.exists(dirpath):
            log.warning("Checkpoint directory does not exist! Creating directory %s", dirpath)
            os.mkdir(dirpath)

        self.model.save_weights(filepath)

    def load_checkpoint(self, path, filename=None):
        """Loads parameters of the neural network from folder/filename.

        Args:
            path (str): Directory for loading checkpoints from or full path to file
                        if filename is None.
            filename (str): File name of saved nn checkpoint. (Default: None)
        """

        if filename is None:
            filepath = path
        else:
            filepath = os.path.join(path, filename)

        if not os.path.exists(filepath):
            raise("No model in path {}".format(filepath))
        self.model.load_weights(filepath)


def build_keras_trainer(game, config):
    """Build neural network model in Keras.

    Args:
        game (Game): Perfect information dynamics/game. Used to get information
                     like action/state space sizes etc.
        config (Config): Configuration loaded json .from file.

    Returns:
        KerasTrainer: Keras Sequential model wrapped in trainer object.
    """

    conv_filters = config.nn["conv_filters"]
    conv_kernel = config.nn["conv_kernel"]
    conv_stride = config.nn["conv_stride"]
    residual_bottleneck = config.nn["residual_bottleneck"]
    residual_filters = config.nn["residual_filters"]
    residual_kernel = config.nn["residual_kernel"]
    residual_num = config.nn["residual_num"]
    feature_extractor = config.nn["feature_extractor"]
    dense_size = config.nn["dense_size"]

    loss = config.nn['loss']
    l2_reg = config.nn["l2_regularizer"]
    lr = config.nn['lr']
    momentum = config.nn['momentum']

    DATA_FORMAT = image_data_format()
    BOARD_HEIGHT, BOARD_WIDTH = game.getBoardSize()
    ACTION_SIZE = game.getActionSize()

    def conv2d_n_batchnorm(x, filters, kernel_size, strides=1, shortcut=None):
        conv = Conv2D(filters, kernel_size=kernel_size, strides=strides,
                      padding="same", kernel_regularizer=l2(l2_reg), data_format=DATA_FORMAT)(x)

        if DATA_FORMAT == 'channels_first':
            bn = BatchNormalization(axis=1)(conv)
        else:
            bn = BatchNormalization(axis=3)(conv)

        if shortcut is not None:
            out = add([bn, shortcut])
        else:
            out = bn

        return Activation(activation='relu')(out)

    def residual_block(x, filters, bottleneck, kernel_size):
        y = conv2d_n_batchnorm(x, bottleneck, kernel_size=1, strides=1)
        y = conv2d_n_batchnorm(y, bottleneck, kernel_size, strides=1)
        return conv2d_n_batchnorm(y, filters, kernel_size=1, strides=1, shortcut=x)

    # Add batch dimension to inputs
    boards_input = Input(shape=(BOARD_HEIGHT, BOARD_WIDTH))
    if DATA_FORMAT == 'channels_first':
        x = Reshape((1, BOARD_HEIGHT, BOARD_WIDTH))(boards_input)
    else:
        x = Reshape((BOARD_HEIGHT, BOARD_WIDTH, 1))(boards_input)

    # Input convolution
    if conv_filters > 0:
        x = conv2d_n_batchnorm(
            x, filters=conv_filters, kernel_size=conv_kernel, strides=conv_stride)

    # Tower of residual blocks
    if residual_filters > 0:
        if conv_filters != residual_filters:
            # Add additional layer to even out the number of filters between input CNN
            # and residual blocks, so that residual shortcut connection works properly
            x = conv2d_n_batchnorm(x, filters=residual_filters, kernel_size=residual_kernel,
                                   strides=1)
        for _ in range(residual_num):
            x = residual_block(x, residual_filters, residual_bottleneck, residual_kernel)

    # Final feature extractors
    if feature_extractor == "agz":
        pi = Flatten()(conv2d_n_batchnorm(x, filters=2, kernel_size=1, strides=1))
        value = Flatten()(conv2d_n_batchnorm(x, filters=1, kernel_size=1, strides=1))
        value = Dense(dense_size, activation='relu',
                      kernel_regularizer=l2(l2_reg))(value)
    elif feature_extractor == "avgpool":
        x = GlobalAveragePooling2D(data_format=DATA_FORMAT)(x)
        pi = value = Dense(dense_size, activation='relu',
                           kernel_regularizer=l2(l2_reg))(x)
    elif feature_extractor == "flatten":
        x = Flatten()(x)
        pi = value = Dense(dense_size, activation='relu',
                           kernel_regularizer=l2(l2_reg))(x)
    else:
        raise ValueError("Unknown feature extractor! Possible values: 'agz', 'avgpool', 'flatten'")

    # Heads
    pi = Dense(ACTION_SIZE, activation='softmax',
               kernel_regularizer=l2(l2_reg), name='pi')(pi)
    value = Dense(1, activation='tanh', kernel_regularizer=l2(
        l2_reg), name='value')(value)

    # Create model
    model = Model(inputs=boards_input, outputs=[pi, value])

    # Compile model
    model.compile(loss=loss,
                  optimizer=SGD(lr=lr,
                                momentum=momentum,
                                nesterov=True),
                  metrics=['accuracy'])

    # Log model architecture
    model.summary(print_fn=lambda x: log.debug("%s", x))
    return KerasTrainer(model, config.training)

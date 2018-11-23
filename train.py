import tensorflow as tf
import numpy as np
import gen
import discr
import util


def train(x):
    """
    Function that generates the network and trains it using the data
    :param x:
    :return:
    """
    # define the batch size
    batch_size = 32

    """
    Define placeholders to feed the data
    """
    real = tf.placeholder(dtype=tf.float32, shape=[batch_size, 1024, 1024, 3], name="input")
    label = tf.placeholder(dtype=tf.float32, shape=[batch_size, 1, 1, 1], name="label")

    """
    Create first blocks for both the generator and the discriminator networks and train
    """

    g = gen.make(batch_size)

    inp = tf.placeholder(dtype=tf.float32, shape=[batch_size, 1024, 1024, 3], name="D-input")
    d = discr.make(inp)

    """
    Create the loss function
    """
    loss = util.loss(logits=d, labels=label, name="Loss")

    """
    Create the optimizers that will minimize the loss function for D and maximize it for G
    """
    train_g = util.opt("G/Adam").minimize(-loss)
    train_d = util.opt("D/Adam").minimize(loss)


if __name__ == "__main__":
    train(np.arange(3))
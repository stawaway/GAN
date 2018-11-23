import tensorflow as tf
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
    real = tf.placeholder(dtype=tf.float32, shape=[], name="input")
    label = tf.placeholder(dtype=tf.float32, shape=[], name="label")

    """
    Create first blocks for both the generator and the discriminator networks and train
    """

    gen_1 = gen.first_block(x, batch_size)
    inp = gen_1

    discr_1 = discr.first_block(inp, batch_size)


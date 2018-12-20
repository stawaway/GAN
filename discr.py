import tensorflow as tf
import util


def first_block(inp):
    """
    Function that returns the first layer block
    :param inp: Batch of input images
    :return:
    """
    with tf.variable_scope("4x4"):
        # create first layer block
        lay = util.conv_lay(inp, filter_size=[1, 1], num_filters=4, activation="leaky_relu", scope="lay_0")
        lay = util.conv_lay(lay, filter_size=[3, 3], num_filters=4, activation="leaky_relu", scope="lay_1")
        lay = util.conv_lay(lay, filter_size=[3, 3], num_filters=8, activation="leaky_relu", scope="lay_2")

        # downsample
        lay = util.downsample(lay)

    return lay


def layer_block(inp, num_filters, name):
    """
    Function that creates the middle layer blocks
    :param inp: Input block
    :param num_filters: Number of feature maps in each layer of the block
    :param name: Name of the current block
    :return:
    """
    with tf.variable_scope(name):
        in_ch = inp.shape[-1].value

        # create sub-layer and feed the upscaled block
        lay = util.conv_lay(inp, filter_size=[3, 3], num_filters=in_ch,
                            activation="leaky_relu", scope="lay_0")

        # next sub-layer
        lay = util.conv_lay(lay, filter_size=[3, 3], num_filters=num_filters,
                            activation="leaky_relu", scope="lay_1")

        # downsample
        lay = util.downsample(lay)

    return lay


def final_block(inp):
    """
    Function that returns the final block of the discriminator
    :param inp: Input block
    :return:
    """
    with tf.variable_scope("128x128"):
        # create sub-layer and feed the upscaled block
        lay = util.conv_lay(inp, filter_size=[3, 3], num_filters=128, activation="leaky_relu", scope="lay_0")

        # next sub-layer
        lay = util.conv_lay(lay, filter_size=[4, 4], num_filters=128, activation="leaky_relu", scope="lay_1")

        # fully-connected layer
        lay = util.dense_lay(lay, "dense")

    return lay


def make(inp, reuse):
    """
    Function that returns a discriminator network
    :param inp: Placeholder for input images of shape [batch_size, h, w, in_ch]
    :param reuse: If the network reuses the same variables as previously created (used for the fake the part)
    :return: The constructed discriminator network
    """
    with tf.variable_scope("D", reuse):
        # Create the first block for the network
        block = first_block(inp)

        # Create the other blocks for the discriminator
        for i in reversed(range(1, 5)):
            num_filters = 128
            block = layer_block(block, num_filters, "{k}x{k}".format(k=4 * 2**i))

        # Create the final block for the generator
        block = final_block(block)

    return block

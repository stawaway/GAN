import tensorflow as tf
import util


def first_block(z):
    """
    Function that returns the first layer block
    :param z: Latent vector placeholder of size [batch_size, 1, 1, 128]
    :return:
    """
    with tf.variable_scope("G/4x4"):
        # create first layer block
        lay = tf.image.resize_images(z, size=[4, 4], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        lay = util.conv_lay(lay, filter_size=[4, 4], num_filters=128, activation="leaky_relu", scope="lay_0")
        lay = util.conv_lay(lay, filter_size=[3, 3], num_filters=128, activation="leaky_relu", scope="lay_1")

        # normalization
        lay = util.pixel_normalization(lay)

    return lay


def layer_block(inp, num_filters, name):
    """
    Function that creates the middle layer blocks
    :param inp: Input block
    :param smooth: scaling factor
    :param num_filters: Number of feature maps in each layer of the block
    :param name: Name of the current block
    :return:
    """
    with tf.variable_scope(name):
        # upsample
        lay_ = util.upsample(inp)

        # create sub-layer and feed the upscaled block
        lay = util.conv_lay(lay_, filter_size=[3, 3], num_filters=num_filters,
                            activation="leaky_relu", scope="lay_0")

        # normalization
        lay = util.pixel_normalization(lay)

        # next sub-layer
        lay = util.conv_lay(lay, filter_size=[3, 3], num_filters=num_filters,
                            activation="leaky_relu", scope="lay_1")

        # normalization
        lay = util.pixel_normalization(lay)

    return lay


def final_block(inp):
    """
    Function that returns the final block of the generator
    :param inp: Input block
    :return:
    """
    with tf.variable_scope("G/128x128"):
        # upsample
        lay = util.upsample(inp)

        # create sub-layer and feed the upscaled block
        lay = util.conv_lay(lay, filter_size=[3, 3], num_filters=4,
                            activation="leaky_relu", scope="lay_0")

        # normalization
        lay = util.pixel_normalization(lay)

        # next sub-layer
        lay = util.conv_lay(lay, filter_size=[3, 3], num_filters=4,
                            activation="leaky_relu", scope="lay_1")

        # normalization
        lay = util.pixel_normalization(lay)

        # next sub-layer
        lay = util.conv_lay(lay, filter_size=[1, 1], num_filters=3,
                            activation="leaky_relu", scope="lay_2")

    return lay


def make(z):
    """
    Function that returns the generator network
    :param z: Placeholder for laten vector of shape [batch_size, 1, 1, 128]
    :return: The constructed generator network
    """
    # Create the first block for the network
    block = first_block(z)

    # Create the other blocks for the generator
    for i in range(1, 5):
        num_filters = 128 if i < 4 else 64
        block = layer_block(block, num_filters, "G/{k}x{k}".format(k=4 * 2**i))

    # Create the final block for the generator
    block = final_block(block)

    return block

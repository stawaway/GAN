import tensorflow as tf
import util


def first_block(z):
    """
    Function that returns the first layer block
    :param z: Latent vector placeholder of size [batch_size, 1, 1, 128]
    :return:
    """
    # create first layer block
    lay = tf.image.resize_images(z, size=[4, 4], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    lay = util.conv_lay(lay, filter_size=[4, 4], num_filters=128, scope="G/4x4/lay_0", collection="GEN_VAR")
    lay = util.conv_lay(lay, filter_size=[3, 3], num_filters=128, scope="G/4x4/lay_1", collection="GEN_VAR")

    return lay


def layer_block(inp, num_filters, name):
    """
    Function that creates the middle layer blocks
    :param inp: Input block
    :param num_filters: Number of feature maps in each layer of the block
    :param name: Name of the current block
    :return:
    """
    # upsample
    lay = util.upsample(inp)

    # create sub-layer and feed the upscaled block
    lay = util.conv_lay(lay, filter_size=[3, 3], num_filters=num_filters, scope=name+"/lay_0", collection="GEN_VAR")

    # activation function
    lay = tf.nn.relu(lay)

    # next sub-layer
    lay = util.conv_lay(lay, filter_size=[3, 3], num_filters=num_filters, scope=name+"/lay_1", collection="GEN_VAR")

    # activation function
    lay = tf.nn.relu(lay)

    return lay


def final_block(inp):
    """
    Function that returns the final block of the generator
    :param inp: Input block
    :return:
    """
    # upsample
    lay = util.upsample(inp)

    # create sub-layer and feed the upscaled block
    lay = util.conv_lay(lay, filter_size=[3, 3], num_filters=4, scope="G/128x128/lay_0", collection="GEN_VAR")

    # activation function
    lay = tf.nn.relu(lay)

    # next sub-layer
    lay = util.conv_lay(lay, filter_size=[3, 3], num_filters=4, scope="G/128x128/lay_1", collection="GEN_VAR")

    # activation function
    lay = tf.nn.relu(lay)

    # next sub-layer
    lay = util.conv_lay(lay, filter_size=[1, 1], num_filters=3, scope="G/128x128/lay_2", collection="GEN_VAR")

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

    # put the output in the [-1, 1] range
    min_, max_ = tf.reduce_min(block), tf.reduce_max(block)
    block = 2. * (block - min_) / (max_ - min_) - 1.

    return block

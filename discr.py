import tensorflow as tf
import util


def first_block(inp):
    """
    Function that returns the first layer block
    :param inp: Batch of input images
    :return:
    """
    # create first layer block
    lay = util.conv_lay(inp, filter_size=[1, 1], num_filters=4, scope="D/1024x1024/lay_0", collection="DISCR_VAR")
    lay = util.conv_lay(lay, filter_size=[3, 3], num_filters=8, scope="D/1024x1024/lay_1", collection="DISCR_VAR")
    lay = util.conv_lay(lay, filter_size=[3, 3], num_filters=8, scope="D/1024x1024/lay_2", collection="DISCR_VAR")
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
    in_ch = inp.shape[-1].value

    # create sub-layer and feed the upscaled block
    lay = util.conv_lay(inp, filter_size=[3, 3], num_filters=in_ch, scope=name+"/lay_0", collection="DISCR_VAR")

    # activation function
    lay = tf.nn.relu(lay)

    # next sub-layer
    lay = util.conv_lay(lay, filter_size=[3, 3], num_filters=num_filters, scope=name+"/lay_1", collection="DISCR_VAR")

    # activation function
    lay = tf.nn.relu(lay)

    # downsample
    lay = util.downsample(lay)

    return lay


def final_block(inp):
    """
    Function that returns the final block of the discriminator
    :param inp: Input block
    :return:
    """
    # create sub-layer and feed the upscaled block
    lay = util.conv_lay(inp, filter_size=[3, 3], num_filters=128, scope="D/4x4/lay_0", collection="DISCR_VAR")

    # activation function
    lay = tf.nn.relu(lay)

    # next sub-layer
    lay = util.conv_lay(lay, filter_size=[4, 4], num_filters=128, scope="D/4x4/lay_1", collection="DISCR_VAR")

    # activation function
    lay = tf.nn.relu(lay)

    # fully-connected layer
    lay = util.dense_lay(lay, "D/4x4/dense", collection="DISCR_VAR")

    return lay


def make(inp):
    """
    Function that returns a discriminator network
    :param inp: Placeholder for input images of shape [batch_size, h, w, in_ch]
    :return: The constructed discriminator network
    """
    # Create the first block for the network
    block = first_block(inp)

    # Create the other blocks for the discriminator
    for i in reversed(range(7)):
        num_filters = 128 if i < 4 else 2 * 2**(8 - i)
        block = layer_block(block, num_filters, "D/{k}x{k}".format(k=4 * 2**(i+1)))

    # Create the final block for the generator
    block = final_block(block)

    return block

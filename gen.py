import tensorflow as tf
import util


def first_block(batch_size):
    """
    Function that returns the first layer block
    :param batch_size: Size of the mini-batches
    :return:
    """
    # create first layer block
    lay = tf.random_normal(shape=[batch_size, 1, 1, 512], name="latent", dtype=tf.float32)
    lay = tf.image.resize_images(lay, size=[4, 4], method=tf.image.ResizeMethod.NEAREST_NEIGHBOUR)

    lay = util.conv_lay(lay, filter_size=[4, 4], num_filters=128, scope="G/4x4/lay:0")
    lay = util.conv_lay(lay, filter_size=[3, 3], num_filters=128, scope="G/4x4/lay:1")

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
    lay = inp + util.upsample(inp)

    # create sub-layer and feed the upscaled block
    lay = util.conv_lay(lay, filter_size=[3, 3], num_filters=num_filters, scope=name+"/lay:0")

    # activation function
    lay = tf.nn.relu(lay)

    # next sub-layer
    lay = util.conv_lay(lay, filter_size=[3, 3], num_filters=num_filters, scope=name+"/lay:1")

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
    lay = inp + util.upsample(inp)

    # create sub-layer and feed the upscaled block
    lay = util.conv_lay(lay, filter_size=[3, 3], num_filters=1024, scope="G/1024x1024/lay:0")

    # activation function
    lay = tf.nn.relu(lay)

    # next sub-layer
    lay = util.conv_lay(lay, filter_size=[3, 3], num_filters=1024, scope="G/1024x1024/lay:1")

    # activation function
    lay = tf.nn.relu(lay)

    # next sub-layer
    lay = util.conv_lay(lay, filter_size=[1, 1], num_filters=3, scope="G/1024x1024/lay:2")

    return lay


def make(batch_size):
    """
    Function that returns the generator network
    :param batch_size: Size of the mini-batches
    :return: The constructed generator network
    """
    # Create the first block for the network
    block = first_block(batch_size)

    # Create the other blocks for the generator
    for i in range(7):
        num_filters = 128 if i < 3 else 2 * 2**(8 - i)
        block = layer_block(block, num_filters, "G/{k}x{k}".format(k=4 * 2**(i+1)))

    # Create the final block for the generator
    block = final_block(block)

    return block

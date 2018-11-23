import tensorflow as tf


def upsample(x):
    """
    Function that upscales an image to double its size
    :param x: The current block to upsample. Of shape [batch, h, w, ch]
    :return: The upsampled block of shape [batch, 2*h, 2*w, ch]
    """
    h, w = tf.shape(x)[:2]

    up = tf.image.resize_images(x, size=[2*h, 2*w], method=tf.image.ResizeMethod.NEAREST_NEIGHBOUR)

    return up


def downsample(x):
    """
    Function that downsamples an image to half its size
    :param x: The current block of shape [batch, h, w, ch] to downsample
    :return: The downsampled block of shape [batch, 0.5*h, 0.5*w, ch]
    """
    h,w = tf.shape(x)[:2]

    down = tf.image.resize_images(x, size=[0.5*h, 0.5*w], method=tf.image.ResizeMethod.NEAREST_NEIGHBOUR)

    return down


def conv_lay(x, filter_size, num_filters, scope):
    """
    Functio that creates a convolutional layer that uses filters with the given size
    :param x: Input layer of shape [batch, h, w, in_ch]
    :param filter_size: The size of the filters to be applied in the convolution operation
    :param num_filters: The number of filter maps i.e. the number of output channels
    :param scope: Name that is given to the layer
    :return: The new layer that has been created
    """

    with tf.variable_scope(scope):
        h, w, in_ch = tf.shape(x)

        init = tf.random_normal_initializer()

        w = tf.get_variable(name="w", shape=filter_size+[in_ch, num_filters],
                            dtype=tf.float32, initializer=init, trainable=True)

        lay = tf.nn.convolution(input=x, filter=w, padding="SAME")

        b = tf.get_variable(name="bias", shape=[num_filters, ],
                            dtype=tf.float32, initializer=tf.zeros_initializer(dtype=tf.float32), trainable=True)

        lay += b

    return lay


def dense_lay(x):
    """
    Fully connected layer
    :param x:
    :return: The new dense layer that was created
    """



def loss(x, scope):
    """
    Function that returns the loss operation for the graph
    :param x:
    :param scope: Name of the scope to define the operation
    :return: The operation that computes the loss
    """
    with tf.variable_scope(scope):
        op = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0., beta2=0.99, epsilon=1e-8)

    return op

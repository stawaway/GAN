import tensorflow as tf


def upsample(x):
    """
    Function that upscales an image to double its size
    :param x: The current block to upsample. Of shape [batch, h, w, ch]
    :return: The upsampled block of shape [batch, 2*h, 2*w, ch]
    """
    h, w = x.shape[1:3]

    up = tf.image.resize_images(x, size=[2*h, 2*w], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return up


def downsample(x):
    """
    Function that downsamples an image to half its size
    :param x: The current block of shape [batch_size, h, w, ch] to downsample
    :return: The downsampled block of shape [batch, 0.5*h, 0.5*w, ch]
    """
    h,w = x.shape[1:3]
    h, w = int(0.5 * h.value), int(0.5 * w.value)

    down = tf.image.resize_images(x, size=[h, w], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return down


def conv_lay(x, filter_size, num_filters, scope, collection):
    """
    Functio that creates a convolutional layer that uses filters with the given size
    :param x: Input layer of shape [batch_size, h, w, in_ch]
    :param filter_size: The size of the filters to be applied in the convolution operation
    :param num_filters: The number of filter maps i.e. the number of output channels
    :param scope: Name that is given to the layer
    :param collection: Collection name to which we add the variables
    :return: The new layer that has been created
    """

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        h, w, in_ch = x.shape[1:]

        init = tf.random_normal_initializer()

        w = tf.get_variable(name="w", shape=filter_size+[in_ch, num_filters],
                            dtype=tf.float32, initializer=init, trainable=True)
        tf.add_to_collection(collection, w)

        lay = tf.nn.convolution(input=x, filter=w, padding="SAME")

        b = tf.get_variable(name="bias", shape=[num_filters, ],
                            dtype=tf.float32, initializer=tf.zeros_initializer(dtype=tf.float32), trainable=True)
        tf.add_to_collection(collection, b)

        lay += b

    return lay


def dense_lay(x, scope, collection):
    """
    Fully connected layer
    :param x: Input layer of shape [batch_size, h, w, in_ch]
    :param scope: Name of the scope
    :param collection: Collection name to which we add the variables
    :return: The new dense layer that was created of shape [batch_size, 1, 1, 1]
    """
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        h, w, in_ch = x.shape[1:]

        init = tf.random_normal_initializer()

        w = tf.get_variable(name="w", shape=[h,w, in_ch, 1],
                            dtype=tf.float32, initializer=init, trainable=True)
        tf.add_to_collection(collection, w)

        lay = tf.nn.convolution(input=x, filter=w, padding="VALID")

        b = tf.get_variable(name="bias", shape=[1, ],
                            dtype=tf.float32, initializer=tf.zeros_initializer(dtype=tf.float32), trainable=True)
        tf.add_to_collection(collection, b)

        lay += b

    return lay


def opt(scope):
    """
    Function that returns the optimizer
    :param scope: Name of the scope to define the operation
    :return: The optimizer the will minimize the loss
    """
    with tf.variable_scope(scope):
        op = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0., beta2=0.99, epsilon=1e-8)

    return op


def loss(logits, labels, name):
    """
    Function that returns the loss operation
    :param logits: Input layer from which we compute the loss
    :param labels: True labels of the data i.e. 0 for a false image and 1 for a true image
    :param name: Name of the operation
    :return: The loss given the input
    """
    with tf.variable_scope(name):
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
        loss = tf.reduce_mean(loss)

    return loss

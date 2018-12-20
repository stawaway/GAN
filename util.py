import tensorflow as tf
from PIL import Image
import numpy as np
import os


def load_img(path, size):
    """
    Function that loads all images found in path
    :param path:
    :param size: size of the input images as a tuple (h, w)
    :return: All the images as an array of size [n, h, w, 3]
    """
    # empty array that will contain all the images
    h, w = size
    img = np.empty(shape=[0, h, w, 3], dtype=np.float32)
    for file in os.listdir(path)[:100]:
        if file.split(".")[-1] == "jpg":
            image = Image.open(os.path.join(path, file))

            array = None

            if image.size != (w, h):
                face_width = face_height = 128
                j = (image.size[0] - face_width) // 2
                i = (image.size[1] - face_height) // 2
                image = image.crop([j, i, j + face_width, i + face_height])
                image = image.resize([w, h], Image.BILINEAR)
                # array = np.array(image.convert("RGB"))
                array = np.array(image)
                array = (array - 127.5) / 127.5
            else:
                array = np.array(image)
                array = (array - 127.5) / 127.5

            img = np.append(img, [array], axis=0)

    return img


def save_img(x, path):
    """
    Function that saves all images found in path
    :param: the images to save as an array of shape [n, h, w, 3]
    :param path: Path where to save the batch of images
    :return:
    """
    i = 0
    for el in x:
        im = Image.fromarray(el.astype("uint8"), "RGB")
        im.save(path+"/{}.jpg".format(i))
        i += 1


def upsample(x, scale=(2, 2)):
    """
    Function that upscales an image to double its size
    :param x: The current block to upsample. Of shape [batch, h, w, ch]
    :return: The upsampled block of shape [batch, 2*h, 2*w, ch]
    """
    h, w = x.shape[1:3]

    up = tf.image.resize_images(x, size=[h*scale[0], w*scale[1]], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return up


def downsample(x, scale=(0.5, 0.5)):
    """
    Function that downsamples an image to half its size
    :param x: The current block of shape [batch_size, h, w, ch] to downsample
    :return: The downsampled block of shape [batch, 0.5*h, 0.5*w, ch]
    """
    h,w = x.shape[1:3]
    h, w = int(h.value * scale[0]), int(w.value * scale[1])

    down = tf.image.resize_images(x, size=[h, w], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return down


def conv_lay(x, filter_size, num_filters, activation, scope):
    """
    Functio that creates a convolutional layer that uses filters with the given size
    :param x: Input layer of shape [batch_size, h, w, in_ch]
    :param filter_size: The size of the filters to be applied in the convolution operation
    :param num_filters: The number of filter maps i.e. the number of output channels
    :param activation: activation function to be used
    :param scope: Name that is given to the layer
    :return: The new layer that has been created
    """

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        h, w, in_ch = x.shape[1:]

        init = tf.random_normal_initializer(stddev=0.1)

        w = tf.get_variable(name="w", shape=filter_size+[in_ch, num_filters],
                            dtype=tf.float32, initializer=init, trainable=True)

        lay = tf.nn.convolution(input=x, filter=w, padding="SAME")

        b = tf.get_variable(name="bias", shape=[num_filters, ],
                            dtype=tf.float32, initializer=tf.zeros_initializer(dtype=tf.float32), trainable=True)

        lay += b

        if activation == "leaky_relu":
            return tf.nn.leaky_relu(lay)
        elif activation == "relu":
            return tf.nn.relu

    return lay


def dense_lay(x, scope):
    """
    Fully connected layer
    :param x: Input layer of shape [batch_size, h, w, in_ch]
    :param scope: Name of the scope
    :return: The new dense layer that was created of shape [batch_size, 1, 1, 1]
    """
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        h, w, in_ch = x.shape[1:]

        init = tf.random_normal_initializer()

        w = tf.get_variable(name="w", shape=[h,w, in_ch, 1],
                            dtype=tf.float32, initializer=init, trainable=True)

        lay = tf.nn.convolution(input=x, filter=w, padding="VALID")

        b = tf.get_variable(name="bias", shape=[1, ],
                            dtype=tf.float32, initializer=tf.zeros_initializer(dtype=tf.float32), trainable=True)

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


def pixel_normalization(x):
    """
    Perform pixel normalization on the input layer
    :param x: Input layer of shape [batch_size, h, w, in_ch]
    :return: Normalized layer of the same shape
    """
    num_filters = x.shape[-1]

    norm = tf.sqrt(tf.reduce_mean(tf.square(x), axis=3, keepdims=True) + 1e-8)

    return x / norm
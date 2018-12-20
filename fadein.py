import numpy as np
import tensorflow as tf
import util
from PIL import Image
import argparse
import sys

parser = argparse.ArgumentParser()

batch_size = 100
training = True
img_path = "/Users/williamst-arnaud/U de M/IFT6269/celeba-128"
eps = float(np.finfo(np.float32).tiny)
restore_path = None
epochs = 2000
alpha = 32


def schedule(step, epochs):
    """
    Function that determines the frame size on which to train given the number of steps
    :param step: The current training step
    :param epochs: Number of epochs to train on
    :return:
    """
    if step <= 100:
        return "4x4", step / 100.
    elif step <= 200:
        return "8x8", (step - 100) / 100.
    elif step <= 800:
        return "16x16", (step - 200) / 600
    else:
        return "32x32", (step - 800)


def add_g_layer(inp, filters, kernel_size):
    """
    Function that adds new convolution transpose layer to the generator
    :param inp: Input layer to the model
    :param filters: The number of filters
    :param kernel_size: The size of each filter map
    :return:
    """
    conv = tf.layers.conv2d_transpose(inp, filters, kernel_size, strides=2, padding="SAME")

    return conv


def add_d_layer(inp, filters, kernel_size):
    """
    Function that adds new convolution transpose to the model
    :param inp: Input layer to the model
    :param filters: The number of filters
    :param kernel_size: The size of each filter map
    :return:
    """
    conv = tf.layers.conv2d(inp, filters, kernel_size, stries=2, padding="SAME")

    return conv


def generator_layer(inp, frame_size, number):
    with tf.variable_scope("block_{}".format(number), reuse=tf.AUTO_REUSE):
        # Convolution on 4x4
        up_0 = util.upsample(inp)
        conv_1 = tf.layers.conv2d(up_0, 128, 4, padding="SAME", name="layer_0")
        conv_2 = tf.layers.conv2d(conv_1, 128, 3, padding="SAME", name="layer_1")
        relu_0 = tf.nn.relu(conv_2)
        relu_0.set_shape([None, frame_size, frame_size, 128])

    return relu_0


def generator_fadein(inp, relu, alpha):
    fade_1_l = tf.layers.conv2d(util.upsample(inp), 3, 1, padding="SAME", reuse=tf.AUTO_REUSE, name="to_rgb")
    fade_1_r = tf.layers.conv2d(relu, 3, 1, padding="SAME", reuse=True, name="to_rgb")
    out_1 = (1. - alpha) * fade_1_l + alpha * fade_1_r

    return out_1


def generator_last(inp, frame_size):
    with tf.variable_scope("output", reuse=tf.AUTO_REUSE):
        dense = tf.layers.conv2d(inp, 3, 1, padding="SAME", name="to_rgb")
        dense = tf.squeeze(dense)
        dense.set_shape([None, frame_size, frame_size, 3])

    return dense


def generator(inp, alpha):
    """
    Function that creates a generator network
    :param inp: Input to the network
    :param frame_size: Size of the images to generate/train on
    :return:
    """
    with tf.variable_scope("generator"):
        # define the first fully-connected layer
        lay = tf.layers.dense(inp, 16*1024, name="layer_0")
        lay = tf.reshape(lay, [-1, 4, 4, 1024])
        lay = tf.nn.relu(lay)

        # define the first convolution layer 8x8
        lay = util.upsample(lay)
        lay = tf.layers.conv2d(lay, 128, 4, name="layer_1", padding="SAME")
        lay = tf.layers.conv2d(lay, 128, 4, name="layer_2", padding="SAME")
        lay = tf.nn.relu(lay)

        # define the second layer 16x16
        lay = util.upsample(lay)
        lay = tf.layers.conv2d(lay, 128, 4, padding="SAME", name="layer_3")
        lay = tf.layers.conv2d(lay, 128, 4, padding="SAME", name="layer_4")
        lay = tf.nn.relu(lay)

        # define the second layer 32x32
        lay = util.upsample(lay)
        fade_l = tf.layers.conv2d(lay, 128, 1, padding="SAME", name="to_rgb_0")
        lay = tf.layers.conv2d(lay, 128, 4, padding="SAME", name="layer_5")
        lay = tf.layers.conv2d(lay, 128, 4, padding="SAME", name="layer_6")
        fade_r = tf.layers.conv2d(lay, 128, 1, padding="SAME", name="to_rgb_1")
        lay = (1. - alpha) * fade_l + alpha * fade_r
        lay = tf.layers.conv2d(lay, 3, 4, strides=1, padding="SAME", name="layer_7")

        lay = tf.nn.tanh(lay)

    return lay


def discriminator(inp, alpha, reuse):
    """
    Function that creates the discriminator network
    :param inp: Input to  the discriminator
    :param frame_size: Size of the images to train on
    :param reuse: If the network reuses previously created weights or not
    :return:
    """

    with tf.variable_scope("discriminator", reuse=reuse):
        # define the second layer 16x16
        fade_l = util.downsample(inp)
        fade_l = tf.layers.conv2d(fade_l, 128, 1, padding="SAME", name="from_rgb_0")
        fade_r = tf.layers.conv2d(fade_l, 128, 1, padding="SAME", name="from_rgb_1")
        lay = tf.nn.relu(fade_r)
        lay = tf.layers.conv2d(lay, 128, 4, padding="SAME", name="layer_0")
        lay = tf.nn.relu(lay)
        lay = tf.layers.conv2d(lay, 128, 4, padding="SAME", name="layer_1")
        lay = tf.nn.relu(lay)
        lay = (1. - alpha) * fade_l + alpha * lay

        # define the third layer 8x8
        lay = tf.layers.conv2d(lay, 128, 4, padding="SAME", name="layer_2")
        lay = tf.layers.conv2d(lay, 128, 4, padding="SAME", name="layer_3")
        lay = tf.nn.relu(lay)

        # define the third layer 4x4
        lay = tf.layers.conv2d(lay, 128, 4, padding="SAME", name="layer_4")
        lay = tf.layers.conv2d(lay, 128, 4, padding="SAME", name="layer_5")
        lay = tf.nn.relu(lay)

        # define the first fully-connected layer
        lay = tf.reshape(lay, [-1, 4 * 4 * 256])
        lay = tf.layers.dense(lay, 1, name="layer_6")
        lay = tf.squeeze(lay)

    return lay


def model(latent, real, alpha):
    """
    Function that creates the GAN by assembling the generator and the discriminator parts
    :param latent: Latent vector that serves as the input to the generator
    :param real: The real images
    :param frame_size: Placeholder of the frame size on which we are currently training
    :param alpha: Placeholder of the alpha value for fading in layers
    :return:
    """
    g = generator(latent, alpha)
    d_real = discriminator(real, reuse=False, alpha=alpha)
    d_fake = discriminator(g, reuse=True, alpha=alpha)

    # define the generator loss
    g_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_fake), logits=d_fake)
    g_loss = tf.reduce_mean(g_loss)

    # define the discriminator loss
    d_loss_real = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_real), logits=d_real)
    d_loss_real = tf.reduce_mean(d_loss_real)

    d_loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(d_fake), logits=d_fake)
    d_loss_fake = tf.reduce_mean(d_loss_fake)

    return g, g_loss, d_loss_real + d_loss_fake


def train(g_weights=None, d_weights=None):
    """
    Function that trains the network
    :param g_weights: Pre-trained weights for the network
    :param d_weights: Pre-trained weights for the network
    :return:
    """
    img = util.load_img(img_path, [32, 32])
    fake = tf.placeholder(dtype=tf.float32, shape=[None, 1, 1, 128], name="latent")
    real = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3], name="real")
    alpha_pl = tf.placeholder(dtype=tf.float32, shape=[], name="alpha")
    g, g_loss_op, d_loss_op = model(fake, real, alpha)

    # Optimizers for the generator and the discriminator
    adam = tf.train.AdamOptimizer(7e-5)
    train_g = adam.minimize(g_loss_op,
                            var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator"))
    train_d = adam.minimize(d_loss_op,
                            var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator"))

    # add summary scalars
    tf.summary.scalar("discriminator loss", d_loss_op)
    tf.summary.scalar("generator loss", g_loss_op)
    merged = tf.summary.merge_all()

    # savers to save the trained weights for the generator and the discriminator
    g_saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator"))
    d_saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator"))

    # train the network
    gen_loss = np.inf
    discr_loss = np.inf
    with tf.Session() as sess:
        # create writer for summaries
        writer = tf.summary.FileWriter("debug", sess.graph)

        # initialize the GAN and restore weights if they were pre-trained
        sess.run(tf.variables_initializer(tf.global_variables()))
        if g_weights is not None:
            g_saver.restore(sess, g_weights)
        if d_weights is not None:
            d_saver.restore(sess, d_weights)

        step = 1
        while True:
            g_batch_loss = np.empty([0, ])
            d_batch_loss = np.empty([0, ])
            for i in np.arange(np.ceil(img.shape[0] / batch_size), dtype=np.int32):
                min_ = batch_size * i
                max_ = np.minimum(min_ + batch_size, img.shape[0])

                batch = img[min_:max_, :, :, :]

                # generate images
                gen_img = np.random.normal(loc=0., scale=1., size=[max_ - min_, 1, 1, 128])

                # train the discriminator on the fake images
                _, discr_loss_ = sess.run([train_d, d_loss_op], feed_dict={real: batch, fake: gen_img,
                                                                           alpha_pl: 1.})

                # train the generator to fool discriminator
                _, gen_loss_ = sess.run([train_g, g_loss_op], feed_dict={fake: gen_img, alpha_pl: 1.})

                # output test images every 50 step
                """if step % 100 == 0:
                    samples = sess.run(g, feed_dict={fake: gen_img})
                    for image in samples[:1]:
                        image = np.uint8((127.5 * image) + 127.5)
                        Image.fromarray(image).show()"""

                g_batch_loss = np.append(g_batch_loss, gen_loss_)
                d_batch_loss = np.append(d_batch_loss, discr_loss_)

            print("Step ", step)
            print("Generator loss is: ", np.mean(g_batch_loss))
            print("Discriminator loss is: ", np.mean(d_batch_loss), "\n")

            # save checkpoint every 10 steps and print to terminal
            if step % 100 == 0. or step == 1.:
                summary = sess.run(merged,
                                   feed_dict={
                                       real: img,
                                       fake: np.random.normal(loc=0., scale=1., size=[img.shape[0], 1, 1, 128]),
                                       alpha_pl: 1.
                                    })
                writer.add_summary(summary, step)
                g_saver.save(sess, "model/model.ckpt", global_step=step - 1)

            if step > 2000:  # np.abs(gen_loss - np.mean(g_batch_loss)) < 0.0001:
                gen_loss, discr_loss = np.mean(g_batch_loss), np.mean(d_batch_loss)
                g_saver.save(sess, "model/g_weights.ckpt")
                d_saver.save(sess, "model/d_weights.ckpt")
                latent = np.random.normal(0., 1. , size=[batch_size, 1, 1, 128])
                images = sess.run(g ,feed_dict={fake: latent, alpha_pl: 1.})
                util.save_img((127.5 * images) + 127.5, save_path)

                break
            else:
                gen_loss, discr_loss = np.mean(g_batch_loss), np.mean(d_batch_loss)
                step += 1


def sample():
    """
    Function that samples the generator
    :return:
    """
    fake = tf.placeholder(dtype=tf.float32, shape=[batch_size, 1, 1, 128], name="latent")

    g = generator(fake)

    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator"))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # restore the trained weights
        saver.restore(sess, restore_path)

        latent = np.random.normal(loc=0., scale=1., size=[10, 1, 1, 128])

        samples = sess.run(g, feed_dict={fake: latent})

        for img in samples:
            img = np.uint8((127.5 * img) + 127.5)
            Image.fromarray(img).show()


if __name__ == "__main__":
    args = sys.argv
    _, img_path, save_path = args
    train()


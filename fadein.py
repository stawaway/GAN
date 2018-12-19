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


def schedule(step):
    """
    Function that determines the frame size on which to train given the number of steps
    :param step: The current training step
    :return:
    """
    if step <= 100:
        return 4
    elif step <= 200:
        return 8
    elif step <= 800:
        return 16
    else:
        return 32

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


def generator(inp, frame_size):
    """
    Function that creates a generator network
    :param inp: Input to the network
    :param frame_size: Size of the images to generate/train on
    :return:
    """
    with tf.variable_scope("generator", reuse=False):
        with tf.variable_scope("block_0"):
            # define the first fully-connected layer
            dense = tf.layers.dense(inp, 16 * 1024, name="layer_0")
            r_dense = tf.reshape(dense, [-1, 4, 4, 1024])
            batch_0 = tf.layers.batch_normalization(r_dense, training=True)
            relu_0 = tf.nn.relu(batch_0)
            drop_0 = tf.nn.dropout(relu_0, 0.5)

        with tf.variable_scope("block_1"):
            # define the first convolution layer 8x8
            conv_1 = tf.layers.conv2d_transpose(drop_0, 512, 4, strides=2, name="layer_1", padding="SAME")
            conv_2 = tf.layers.conv2d_transpose(conv_1, 512, 4, strides=1, name="layer_2", padding="SAME")
            batch_2 = tf.layers.batch_normalization(conv_2, training=True)
            relu_2 = tf.nn.relu(batch_2)
            drop_2 = tf.nn.dropout(relu_2, 0.5)

        with tf.variable_scope("block_2"):
            # define the second layer 16x16
            conv_3 = tf.layers.conv2d_transpose(drop_2, 256, 4, strides=2, padding="SAME", name="layer_3")
            conv_4 = tf.layers.conv2d_transpose(conv_3, 256, 4, strides=1, padding="SAME", name="layer_4")
            batch_4 = tf.layers.batch_normalization(conv_4, training=True)
            relu_4 = tf.nn.relu(batch_4)
            drop_4 = tf.nn.dropout(relu_4, 0.5)

        with tf.variable_scope("block_3"):
            # define the second layer 32x32
            conv_5 = tf.layers.conv2d_transpose(drop_4, 3, 4, strides=2, padding="SAME", name="layer_5")
            tanh = tf.nn.tanh(conv_5)

        out = tf.case([(tf.equal(frame_size, 4), lambda: tf.layers.conv2d_transpose(drop_0, 3, 4, padding="SAME")),
                       (tf.equal(frame_size, 8), lambda: tf.layers.conv2d_transpose(drop_2, 3, 4, padding="SAME")),
                      (tf.equal(frame_size, 16), lambda: tf.layers.conv2d_transpose(drop_4, 3, 4, padding="SAME")),
                       (tf.equal(frame_size, 32), lambda: drop_4)],
                      exclusive=True)

    return out


def discriminator(inp, frame_size, reuse):
    """
    Function that creates the discriminator network
    :param inp: Input to  the discriminator
    :param frame_size: Size of the images to train on
    :param reuse: If the network reuses previously created weights or not
    :return:
    """
    with tf.variable_scope("discriminator", reuse=reuse):
        with tf.variable_scope("block_0", reuse=reuse):
            # define the second layer 16x16
            conv_0 = tf.layers.conv2d(inp, 64, 4, strides=2, padding="SAME", name="layer_0")
            conv_1 = tf.layers.conv2d(conv_0, 64, 4, strides=1, padding="SAME", name="layer_1")
            batch_1 = tf.layers.batch_normalization(conv_1, training=True)
            relu_1 = tf.nn.relu(batch_1)
            drop_1 = tf.nn.dropout(relu_1, 0.5)

        inp_1 = tf.cond(tf.equal(frame_size, 16), lambda: inp, lambda: drop_1)

        with tf.variable_scope("block_1", reuse=reuse):
            # define the third layer 8x8
            conv_2 = tf.layers.conv2d(inp_1, 128, 4, strides=2, padding="SAME", name="layer_2")
            conv_3 = tf.layers.conv2d(conv_2, 128, 4, strides=1, padding="SAME", name="layer_3")
            batch_3 = tf.layers.batch_normalization(conv_3, training=True)
            relu_3 = tf.nn.relu(batch_3)
            drop_3 = tf.nn.dropout(relu_3, 0.5)

        inp_2 = tf.cond(tf.equal(frame_size, 8), lambda: inp, lambda: drop_3)

        with tf.variable_scope("block_2", reuse=reuse):
            # define the third layer 4x4
            conv_4 = tf.layers.conv2d(inp_2, 256, 4, strides=2, padding="SAME", name="layer_4")
            conv_5 = tf.layers.conv2d(conv_4, 256, 4, strides=1, padding="SAME", name="layer_5")
            batch_5 = tf.layers.batch_normalization(conv_5, training=True)
            relu_5 = tf.nn.relu(batch_5)
            drop_5 = tf.nn.dropout(relu_5, 0.5)

        inp_3 = tf.cond(tf.equal(frame_size, 4), lambda: inp, lambda: drop_5)

        with tf.variable_scope("block_3", reuse=reuse):
            # define the first fully-connected layer
            r_drop_5 = tf.reshape(inp_3, [batch_size, -1])
            dense = tf.layers.dense(r_drop_5, 1, name="layer_6")
            dense = tf.squeeze(dense)

    return dense


def model(latent, real, frame_size):
    """
    Function that creates the GAN by assembling the generator and the discriminator parts
    :param latent: Latent vector that serves as the input to the generator
    :param real: The real images
    :param frame_size: Placeholder of the frame size on which we are currently training
    :return:
    """
    g = generator(latent, frame_size)
    d_real = discriminator(real, frame_size, reuse=False)
    d_fake = discriminator(g, frame_size, reuse=True)

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
    fake = tf.placeholder(dtype=tf.float32, shape=[batch_size, 1, 1, 128], name="latent")
    real = tf.placeholder(dtype=tf.float32, shape=[batch_size, 32, 32, 3], name="real")
    frame_size = tf.placeholder(dtype=tf.int32, shape=[], name="frame_size")
    g, g_loss_op, d_loss_op = model(fake, real, frame_size)

    train_g_ops, train_d_ops = [], []
    for i in range(3):
        # Optimizers for the generator and the discriminator
        train_g = tf.train.AdamOptimizer(7e-5).minimize(
            g_loss_op,
            var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator/block_{}".format(i)))
        train_g_ops.append(train_g)

        train_d = tf.train.AdamOptimizer(7e-5).minimize(
            d_loss_op,
            var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator/block_{}".format(i)))
        train_d_ops.append(train_d)

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
                gen_img = np.random.normal(loc=0., scale=1., size=[batch_size, 1, 1, 128])

                # train the discriminator on the fake images
                _, discr_loss_ = sess.run([train_d_ops[schedule(step)], d_loss_op],
                                          feed_dict={real: batch, fake: gen_img, frame_size: schedule(step)})

                # train the generator to fool discriminator
                _, gen_loss_ = sess.run([train_g_ops[schedule(step)], g_loss_op],
                                        feed_dict={fake: gen_img, frame_size: schedule(step)})

                # output test images every 50 step
                if step % 100 == 0:
                    samples = sess.run(g, feed_dict={fake: gen_img, frame_size: schedule(step)})
                    for image in samples[:1]:
                        image = np.uint8((127.5 * image) + 127.5)
                        Image.fromarray(image).show()

                g_batch_loss = np.append(g_batch_loss, gen_loss_)
                d_batch_loss = np.append(d_batch_loss, discr_loss_)

            print("Step ", step)
            print("Generator loss is: ", np.mean(g_batch_loss))
            print("Discriminator loss is: ", np.mean(d_batch_loss), "\n")

            # save checkpoint every 10 steps and print to terminal
            if step % 100 or step == 1.:
                summary = sess.run(merged,
                                   feed_dict={
                                       real: img,
                                       fake: np.random.normal(loc=0., scale=1., size=[img.shape[0], 1, 1, 128])})
                writer.add_summary(summary, step)
                g_saver.save(sess, "model/model.ckpt", global_step=step - 1)

            if step > 2000:  # np.abs(gen_loss - np.mean(g_batch_loss)) < 0.0001:
                gen_loss, discr_loss = np.mean(g_batch_loss), np.mean(d_batch_loss)
                g_saver.save(sess, "model/g_weights.ckpt")
                d_saver.save(sess, "model/d_weights.ckpt")
                latent = np.random.normal(0., 1., size=[batch_size, 1, 1, 128])
                images = sess.run(g, feed_dict={fake: latent, frame_size: schedule(step)})
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
    frame_size = tf.placeholder(dtype=tf.int32, shape=[1, ], name="frame_size")

    g = generator(fake, frame_size)

    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator"))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # restore the trained weights
        saver.restore(sess, restore_path)

        latent = np.random.normal(loc=0., scale=1., size=[10, 1, 1, 128])

        samples = sess.run(g, feed_dict={fake: latent, frame_size: 32})

        for img in samples:
            img = np.uint8((127.5 * img) + 127.5)
            Image.fromarray(img).show()


if __name__ == "__main__":
    args = sys.argv
    _, img_path, save_path = args
    train()


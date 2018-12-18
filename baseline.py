import numpy as np
import tensorflow as tf
import util
from PIL import Image
import argparse
import sys

parser = argparse.ArgumentParser()
# parser.add_argument("--img_path")

batch_size = 100
training = True
img_path = "../celeba-128"
save_path = "baseline_imgs"
eps = float(np.finfo(np.float32).tiny)
restore_path = None


def generator(inp):
    """
    Function that creates a generator network
    :param inp: Input to the network
    :return:
    """
    with tf.variable_scope("generator"):
        # define the first fully-connected layer
        lay = tf.layers.dense(inp, 16*1024, name="layer_0")
        lay = tf.reshape(lay, [-1, 4, 4, 1024])
        lay = tf.layers.batch_normalization(lay, training=True)
        lay = tf.nn.relu(lay)
        lay = tf.nn.dropout(lay, 0.5)

        # define the first convolution layer 8x8
        lay = tf.layers.conv2d_transpose(lay, 512, 4, strides=2, name="layer_1", padding="SAME")
        lay = tf.layers.conv2d_transpose(lay, 512, 4, strides=1, name="layer_2", padding="SAME")
        lay = tf.layers.batch_normalization(lay, training=True)
        lay = tf.nn.relu(lay)
        lay = tf.nn.dropout(lay, 0.5)

        # define the second layer 16x16
        lay = tf.layers.conv2d_transpose(lay, 256, 4, strides=2, padding="SAME", name="layer_3")
        lay = tf.layers.conv2d_transpose(lay, 256, 4, strides=1, padding="SAME", name="layer_4")
        lay = tf.layers.batch_normalization(lay, training=True)
        lay = tf.nn.relu(lay)
        lay = tf.nn.dropout(lay, 0.5)

        # define the second layer 32x32
        lay = tf.layers.conv2d_transpose(lay, 3, 4, strides=2, padding="SAME", name="layer_5")
        lay = tf.nn.tanh(lay)

    return lay


def discriminator(inp, reuse):
    """
    Function that creates the discriminator network
    :param inp: Input to  the discriminator
    :param reuse: If the network reuses previously created weights or not
    :return:
    """
    with tf.variable_scope("discriminator", reuse=reuse):

        # define the second layer 16x16
        lay = tf.layers.conv2d(inp, 64, 4, strides=2, padding="SAME", name="layer_0")
        lay = tf.layers.conv2d(lay, 64, 4, strides=1, padding="SAME", name="layer_1")
        lay = tf.layers.batch_normalization(lay, training=True)
        lay = tf.nn.relu(lay)
        lay = tf.nn.dropout(lay, 0.5)

        # define the third layer 8x8
        lay = tf.layers.conv2d(lay, 128, 4, strides=2, padding="SAME", name="layer_2")
        lay = tf.layers.conv2d(lay, 128, 4, strides=1, padding="SAME", name="layer_3")
        lay = tf.layers.batch_normalization(lay, training=True)
        lay = tf.nn.relu(lay)
        lay = tf.nn.dropout(lay, 0.5)

        # define the third layer 4x4
        lay = tf.layers.conv2d(lay, 256, 4, strides=2, padding="SAME", name="layer_4")
        lay = tf.layers.conv2d(lay, 256, 4, strides=1, padding="SAME", name="layer_5")
        lay = tf.layers.batch_normalization(lay, training=True)
        lay = tf.nn.relu(lay)
        lay = tf.nn.dropout(lay, 0.5)

        # define the first fully-connected layer
        lay = tf.reshape(lay, [-1, 4*4*256])
        lay = tf.layers.dense(lay, 1, name="layer_6")
        lay = tf.squeeze(lay)

    return lay


def model(latent, real):
    """
    Function that creates the GAN by assembling the generator and the discriminator parts
    :param latent: Latent vector that serves as the input to the generator
    :param real: The real images
    :return:
    """
    g = generator(latent)
    d_real = discriminator(real, reuse=False)
    d_fake = discriminator(g, reuse=True)

    # define the generator loss
    g_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_fake), logits=d_fake)
    g_loss = tf.reduce_mean(g_loss)

    # define the discriminator loss
    d_loss_real = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_real), logits=d_real)
    d_loss_real = tf.reduce_mean(d_loss_real)

    d_loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(d_fake), logits=d_fake)
    d_loss_fake = tf.reduce_mean(d_loss_fake)
    #d_loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_fake), logits=d_fake)
    #d_loss_fake = tf.reduce_mean(-d_loss_fake)


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
    g, g_loss_op, d_loss_op = model(fake, real)

    # Optimizers for the generator and the discriminator
    train_g = tf.train.AdamOptimizer(7e-5).minimize(g_loss_op,
                                                    var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                                               scope="generator"))
    train_d = tf.train.AdamOptimizer(7e-5).minimize(d_loss_op,
                                                    var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                                               scope="discriminator"))

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
                _, discr_loss_ = sess.run([train_d, d_loss_op], feed_dict={real: batch, fake: gen_img})

                # train the generator to fool discriminator
                _, gen_loss_ = sess.run([train_g, g_loss_op], feed_dict={fake: gen_img})

                # output test images every 50 step
                if step % 100 == 0:
                    samples = sess.run(g, feed_dict={fake: gen_img})
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
                latent = np.random.normal(0., 1. , size=[batch_size, 1, 1, 128])
                images = sess.run(g ,feed_dict={fake: latent})
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


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
    inp4, inp8, inp16, inp32 = inp
    with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
        # 4x4 convolutions
        up_0 = util.upsample(inp4)
        relu_0 = generator_layer(up_0, 4, 0)
        out_0 = generator_last(relu_0, 4)
        for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator"):
            tf.add_to_collection("g4", var)

    with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
        # 8x8 convolutions
        up_0 = util.upsample(inp8)
        relu_0 = generator_layer(up_0, 4, 0)
        relu_1 = generator_layer(relu_0, 8, 1)
        out_1 = generator_fadein(relu_0, relu_1, alpha)
        for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator"):
            tf.add_to_collection("g8", var)

    with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
        # 16x16 convolutions
        up_0 = util.upsample(inp16)
        relu_0 = generator_layer(up_0, 4, 0)
        relu_1 = generator_layer(relu_0, 8, 1)
        relu_2 = generator_layer(relu_1, 16, 2)
        out_2 = generator_fadein(relu_1, relu_2, alpha)
        for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator"):
            tf.add_to_collection("g16", var)

    with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
        # 32x32 convolutions
        up_0 = util.upsample(inp32)
        relu_0 = generator_layer(up_0, 4, 0)
        relu_1 = generator_layer(relu_0, 8, 1)
        relu_2 = generator_layer(relu_1, 16, 2)
        relu_3 = generator_layer(relu_2, 32, 3)
        out_3 = generator_fadein(relu_2, relu_3, alpha)
        for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator"):
            tf.add_to_collection("g32", var)

    return [out_0, out_1, out_2, out_3]


def discriminator_layer(inp, frame_size, number):
    with tf.variable_scope("block_{}".format(number), reuse=tf.AUTO_REUSE):
        # Convolution on 32x32
        conv_1 = tf.layers.conv2d(inp, 128, 3, padding="SAME", name="layer_0")
        conv_2 = tf.layers.conv2d(conv_1, 128, 3, padding="SAME", name="layer_1")
        relu_0 = tf.nn.relu(conv_2)
        relu_0.set_shape([None, frame_size, frame_size, 128])
        down = util.downsample(relu_0)

    return down


def discriminator_fadein(inp, relu, alpha):
    fade_0_l = tf.layers.conv2d(util.downsample(inp), 128, 1, padding="SAME", reuse=True, name="from_rgb")
    out_0 = (1. - alpha) * fade_0_l + alpha * relu

    return out_0


def discriminator_dense(inp):
    with tf.variable_scope("Dense", reuse=tf.AUTO_REUSE):
        conv_1 = tf.layers.conv2d(inp, 128, 3, padding="SAME", name="layer_0")
        conv_2 = tf.layers.conv2d(conv_1, 128, 4, padding="VALID", name="layer_1")
        conv_2.set_shape([None, 1, 1, 128])
        conv_2 = tf.squeeze(conv_2, axis=[1, 2])
        dense = tf.layers.dense(conv_2, 1, name="dense")
        dense = tf.squeeze(dense)

    return dense


def discriminator(inp, reuse, alpha):
    """
    Function that creates the discriminator network
    :param inp: Input to  the discriminator
    :param frame_size: Size of the images to train on
    :param reuse: If the network reuses previously created weights or not
    :return:
    """
    inp4, inp8, inp16, inp32 = inp
    with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
        # 4x4 convolutions
        in_0 = tf.layers.conv2d(inp4, 128, 1, padding="SAME", name="from_rgb")
        dense_0 = discriminator_dense(in_0)
        for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator"):
            tf.add_to_collection("d4", var)

    with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
        # 8x8 convolutions
        in_1 = tf.layers.conv2d(inp8, 128, 1, padding="SAME", reuse=True, name="from_rgb")
        relu_0 = discriminator_layer(in_1, 8, 1)
        out_0 = discriminator_fadein(inp8, relu_0, alpha)
        dense_1 = discriminator_dense(out_0)
        for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator"):
            tf.add_to_collection("d8", var)

    with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
        # 16x16 Convolutions
        in_2 = tf.layers.conv2d(inp16, 128, 1, padding="SAME", reuse=True, name="from_rgb")
        relu_0 = discriminator_layer(in_2, 16, 1)
        fade_0 = discriminator_fadein(inp16, relu_0, alpha)
        relu_1 = discriminator_layer(fade_0, 8, 2)
        dense_2 = discriminator_dense(relu_1)
        for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator"):
            tf.add_to_collection("d16", var)

    with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
        # 32x32 Convolutions
        in_3 = tf.layers.conv2d(inp32, 128, 1, padding="SAME", reuse=True, name="from_rgb")
        relu_0 = discriminator_layer(in_3, 32, 0)
        fade_0 = discriminator_fadein(inp32, relu_0, alpha)
        relu_1 = discriminator_layer(fade_0, 16, 1)
        relu_2 = discriminator_layer(relu_1, 8, 2)
        dense_3 = discriminator_dense(relu_2)
        for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator"):
            tf.add_to_collection("d32", var)

    return dense_0, dense_1, dense_2, dense_3


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

    g_loss_ops, d_loss_ops = [], []
    for i in range(4):
        # define the generator loss
        g_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_fake[i]), logits=d_fake[i])
        g_loss = tf.reduce_mean(g_loss)
        g_loss_ops.append(g_loss)

        # define the discriminator loss
        d_loss_real = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_real[i]), logits=d_real[i])
        d_loss_real = tf.reduce_mean(d_loss_real)

        d_loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(d_fake[i]), logits=d_fake[i])
        d_loss_fake = tf.reduce_mean(d_loss_fake)

        d_loss_ops.append(d_loss_real + d_loss_fake)

    return g_loss_ops, d_loss_ops, {"4x4": g[0], "8x8": g[1], "16x16": g[2], "32x32": g[3]}


def train(g_weights=None, d_weights=None):
    """
    Function that trains the network
    :param g_weights: Pre-trained weights for the network
    :param d_weights: Pre-trained weights for the network
    :return:
    """
    img = {}
    for i in range(4):
        h = 4 * 2**i
        w = h
        img["{i}x{j}".format(i=h, j=w)] = util.load_img(img_path, [h, w])
    fake = [tf.placeholder_with_default(tf.zeros(shape=[1, 1, 1, 128], dtype=tf.float32),
                                        shape=[None, 1, 1, 128], name="real") for i in range(4)]
    real = [tf.placeholder_with_default(tf.zeros(shape=[1, 4*2**i, 4*2**i, 3], dtype=tf.float32),
                                        shape=[None, 4*2**i, 4*2**i, 3], name="real") for i in range(4)]
    alpha_pl = tf.placeholder(dtype=tf.float32, shape=[], name="alpha")
    g_loss_ops, d_loss_ops, gen_ops = model(fake, real, alpha_pl)

    train_g_ops, train_d_ops = {}, {}
    adam = tf.train.AdamOptimizer(7e-5)
    for pos, i in enumerate([4, 8, 16, 32]):
        # Optimizers for the generator and the discriminator
        train_g = adam.minimize(g_loss_ops[pos], var_list=tf.get_collection("g{}".format(i), scope="generator"))

        train_g_ops["{i}x{i}".format(i=i)] = train_g

        train_d = adam.minimize(d_loss_ops[pos], var_list=tf.get_collection("d{}".format(i), scope="discriminator"))

        train_d_ops["{i}x{i}".format(i=i)] = train_d


    # add summary scalars

    tf.summary.scalar("generator loss 4x4", g_loss_ops[0])
    tf.summary.scalar("generator loss 8x8", g_loss_ops[1])
    tf.summary.scalar("generator loss 16x416", g_loss_ops[2])
    tf.summary.scalar("generator loss 32x32", g_loss_ops[3])
    tf.summary.scalar("discriminator loss 4x4", d_loss_ops[0])
    tf.summary.scalar("discriminator loss 8x8", d_loss_ops[1])
    tf.summary.scalar("discriminator loss 16x16", d_loss_ops[2])
    tf.summary.scalar("discriminator loss 32x32", d_loss_ops[3])
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
            format, alpha = schedule(step, epochs)
            j = int(np.log2(float(format[0])))-2
            for i in np.arange(np.ceil(img[format].shape[0] / batch_size), dtype=np.int32):
                min_ = batch_size * i
                max_ = np.minimum(min_ + batch_size, img[format].shape[0])

                batch = img[format][min_:max_, :, :, :]

                # generate images
                noise = np.random.normal(loc=0., scale=1., size=[batch_size, 1, 1, 128])

                # train the discriminator on the fake images
                gen_img = sess.run(gen_ops[format], feed_dict={real[j]: batch, fake[j]: noise, alpha: alpha})
                _, discr_loss_ = sess.run([train_d_ops[format], d_loss_ops],
                                          feed_dict={real[j]: batch,
                                                     fake[j]: noise,
                                                     gen_ops[format]: gen_img,
                                                     alpha_pl: alpha
                                                     })

                # train the generator to fool discriminator
                _, gen_loss_ = sess.run([train_g_ops[format], g_loss_ops],
                                        feed_dict={fake[j]: noise, alpha_pl: alpha})

                # output test images every 50 step
                if step % 1000 == 0:
                    samples = gen_img
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
                                       real[j]: img[format],
                                       fake[j]: np.random.normal(loc=0., scale=1.,
                                                              size=[img[format].shape[0], 1, 1, 128]),
                                       alpha_pl: alpha})
                writer.add_summary(summary, step)
                g_saver.save(sess, "model/model.ckpt", global_step=step - 1)

            if step > epochs:  # np.abs(gen_loss - np.mean(g_batch_loss)) < 0.0001:
                gen_loss, discr_loss = np.mean(g_batch_loss), np.mean(d_batch_loss)
                g_saver.save(sess, "model/g_weights.ckpt")
                d_saver.save(sess, "model/d_weights.ckpt")
                latent = np.random.normal(0., 1., size=[batch_size, 1, 1, 128])
                images = sess.run(gen_ops[format], feed_dict={fake[j]: latent, alpha_pl: alpha})
                print(images.shape)
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
    frame_size = tf.placeholder(dtype=tf.int32, shape=[], name="frame_size")
    alpha_pl = tf.placeholder(dtype=tf.float32, shape=[], name="alpha")

    g = generator(fake, frame_size, alpha_pl)

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


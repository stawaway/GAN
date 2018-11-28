import tensorflow as tf
import numpy as np
import gen
import discr
import util
import sys


def train(img):
    """
    Function that generates the network and trains it using the data
    :param img: Dataset of images
    :return:
    """
    # define the batch size
    batch_size = 32

    """
    Define placeholders to feed the data
    """
    real = tf.placeholder(dtype=tf.float32, shape=[None, 128, 128, 3], name="data")
    fake = tf.placeholder(dtype=tf.float32, shape=[None, 1, 1, 128], name="latent")

    """
    Create first blocks for both the generator and the discriminator networks and train
    """
    g = gen.make(fake)
    d_fake = discr.make(g)
    d_real = discr.make(real)

    """
    Create the loss function
    """
    with tf.variable_scope("D_Loss"):
        d_loss_op = tf.reduce_mean(d_real) - tf.reduce_mean(d_fake)

    with tf.variable_scope("G_Loss"):
        g_loss_op = -tf.reduce_mean(d_fake)

    """
    Create the optimizers that will minimize the loss function for D and maximize it for G
    """
    with tf.variable_scope("G/RMS"):
        train_g = tf.train.RMSPropOptimizer(5e-5).minimize(g_loss_op)
    with tf.variable_scope("D/RMS"):
        train_d = tf.train.RMSPropOptimizer(5e-5).minimize(-d_loss_op)

    """
    Create the clip variables op
    """
    clip_d = [var.assign(tf.clip_by_value(var, -0.01, 0.01)) for var in tf.get_collection("DISCR_VAR")]

    """
    Create saver to store the trained weights
    """
    saver = tf.train.Saver(tf.get_collection("GEN_VAR"))

    """
    Train the network
    """
    gen_loss = np.inf
    discr_loss = np.inf
    with tf.Session() as sess:
        # initialize the GAN
        sess.run(tf.variables_initializer(tf.global_variables()))

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
                _, _, discr_loss_ = sess.run([train_d, clip_d, d_loss_op], feed_dict={real: batch, fake: gen_img})

                # train the generator to fool discriminator
                _, gen_loss_ = sess.run([train_g, g_loss_op], feed_dict={fake: gen_img})

                g_batch_loss = np.append(g_batch_loss, gen_loss_)
                d_batch_loss = np.append(d_batch_loss, discr_loss_)

            print("Step ", step)
            print("Generator loss is: ", np.mean(g_batch_loss))
            print("Discriminator loss is: ", np.mean(d_batch_loss), "\n")

            # save checkpoint every 10 steps and print to terminal
            if step % 10 == 0:
                saver.save(sess, "model/model.ckpt", global_step=step)

            if step > 200:# np.abs(gen_loss - np.mean(g_batch_loss)) < 0.0001:
                gen_loss, discr_loss = np.mean(g_batch_loss), np.mean(d_batch_loss)
                saver.save(sess, "model/trained_model.ckpt")
                break
            else:
                gen_loss, discr_loss = np.mean(g_batch_loss), np.mean(d_batch_loss)
                step += 1


if __name__ == "__main__":
    args = sys.argv
    # Set up input for training
    data = util.load_img(args[1], [128, 128])
    data = (data - 127.5) / 127.5
    train(data)
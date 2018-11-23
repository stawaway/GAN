import tensorflow as tf
import numpy as np
import gen
import discr
import util


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
    real = tf.placeholder(dtype=tf.float32, shape=[batch_size, 1024, 1024, 3], name="input")
    label = tf.placeholder(dtype=tf.float32, shape=[batch_size, 1, 1, 1], name="label")

    """
    Create first blocks for both the generator and the discriminator networks and train
    """

    g = gen.make(batch_size)

    inp = tf.placeholder(dtype=tf.float32, shape=[batch_size, 1024, 1024, 3], name="D-input")
    d = discr.make(inp)

    """
    Create the loss function
    """
    loss = util.loss(logits=d, labels=label, name="Loss")

    """
    Create the optimizers that will minimize the loss function for D and maximize it for G
    """
    train_g = util.opt("G/Adam").minimize(-loss)
    train_d = util.opt("D/Adam").minimize(loss)

    """
    Set up input for training
    """

    """
    Train the network
    """
    gen_loss = -np.inf
    discr_loss = np.inf
    with tf.Session() as sess:
        while True:
            for i in range(img.shape[0]):
                min = batch_size * i
                max = np.max(min + batch_size, img.shape[0])

                batch = img[min:max, :, :, :]

                # generate images
                gen_img = sess.run(g)

                # train the generator to fool discriminator
                gen_loss_ = sess.run(train_g, feed_dict={inp: gen_img})

                # combine the fake and real images
                discr_inp = np.concatenate((gen_img, batch), axis=0)

                # train the discriminator on the fake images
                discr_loss_ = sess.run(train_d, feed_dict={inp: discr_inp})

                if np.abs(gen_loss - gen_loss_) < 0.0001 or np.abs(discr_loss - discr_loss_) < 0.0001:
                    gen_loss, discr_loss = gen_loss_, discr_loss_
                    break
                else:
                    gen_loss, disct_loss = gen_loss_, discr_loss_


if __name__ == "__main__":
    train(np.arange(3))
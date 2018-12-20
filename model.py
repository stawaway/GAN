import numpy as np
import tensorflow as tf
import gen
import util
import sys


def main(path):
    """
    Function creates the GAN and loads the learned parameters and then runs on the data
    :param path: Path where to save the generated images
    :return:
    """
    # define the batch size
    batch_size = 32

    """
    Define placeholders to feed the data
    """
    fake = tf.placeholder(dtype=tf.float32, shape=[None, 1, 1, 128], name="latent")

    """
    Create first blocks for both the generator
    """
    g = gen.make(fake)

    """
    Create saver to store the trained weights
    """
    saver = tf.train.Saver()

    """
    Load the learned parameters and run the network to generate the fake images
    """
    with tf.Session() as sess:
        # load trained variables for model
        saver.restore(sess, "model/model.ckpt-100")

        # generate images
        latent = np.random.normal(loc=0., scale=1., size=[batch_size, 1, 1, 128])

        gen_img = sess.run(g, feed_dict={fake: latent})
        gen_img = (127.5 * gen_img) + 127.5

        util.save_img(gen_img, path)


if __name__ == "__main__":
    args = sys.argv

    main(args[1])

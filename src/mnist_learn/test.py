import os

import tensorflow as tf
from . import config

def test(mnist):

    saver = tf.train.import_meta_graph(os.path.join(config.store_path, "mnist.ckpt-9000.meta"))
    with tf.Session() as sess:
        saver.restore(sess, os.path.join(config.store_path, "mnist.ckpt-9000"))

        x = tf.get_default_graph().get_tensor_by_name("x-input:0")
        y = tf.get_default_graph().get_tensor_by_name("y-input:0")

        predict_y = tf.get_default_graph().get_tensor_by_name("add_1:0")

        correct_predict = tf.equal(tf.argmax(predict_y, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))
        test_feed = {x: mnist.test.images, y: mnist.test.labels}
        accuracy_ot = sess.run(accuracy, feed_dict=test_feed)
        print("test accuracy:%f" % accuracy_ot)
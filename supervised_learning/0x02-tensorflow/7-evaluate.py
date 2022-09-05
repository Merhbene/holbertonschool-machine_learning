#!/usr/bin/env python3


""" contains evaluate funct"""

import tensorflow.compat.v1 as tf
 
def evaluate(X, Y, save_path):
    """ return the y prediction, accuracy and the cost"""
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph('model.ckpt.meta')
        new_saver.restore(
            sess,
            save_path
        )
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        loss = tf.get_collection('loss')[0]
        accuracy = tf.get_collection('accuracy')[0]
        y_pred = tf.get_collection('y_pred')[0]
        y_pred, loss_value, accu_value = sess.run(
            (y_pred, loss, accuracy), feed_dict={x: X, y: Y})
        return y_pred, accu_value, loss_value

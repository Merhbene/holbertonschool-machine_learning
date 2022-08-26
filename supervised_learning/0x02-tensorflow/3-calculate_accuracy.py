#!/usr/bin/env python3


""" contains accuracy funct"""

import tensorflow.compat.v1 as tf


def calculate_accuracy(y, y_pred):
    """ accuracy count funct"""
    correct = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    return accuracy
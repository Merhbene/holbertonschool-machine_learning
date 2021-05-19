#!/usr/bin/env python3
"Adam upgraded"
import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    "creates the training operation for a neural network in tensorflow"
    optimizer = tf.train.AdamOptimizer(alpha, beta1, beta2, epsilon=epsilon)

    return optimizer.minimize(loss)

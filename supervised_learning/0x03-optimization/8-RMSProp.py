#!/usr/bin/env python3
"RMSProp upgraded"
import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    "create the training operation for a neural network in tensorflow"
    optimizer = tf.train.RMSPropOptimizer(alpha, beta2, epsilon=epsilon)

    return optimizer.minimize(loss)

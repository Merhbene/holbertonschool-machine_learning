
#!/usr/bin/env python3


""" contains train funct"""
import tensorflow.compat.v1 as tf


def create_train_op(loss, alpha):
    """ train count funct"""
    optimizer = tf.train.GradientDescentOptimizer(alpha)
    return optimizer.minimize(loss)
#!/usr/bin/env python3
"Learning Rate Decay Upgraded"
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation): 
    """that creates a batch normalization layer for a
     neural network in tensorflow"""

    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(n, kernel_initializer=init)(prev)

    
    gamma = tf.Variable(tf.constant(1.0, shape=[n]), name='gamma', trainable=True)
    beta = tf.Variable(tf.constant(0.0, shape=[n]), name='beta', trainable=True)

    m, v = tf.nn.moments(layer, axes=0)

    z_norm = tf.nn.batch_normalization(layer, m, v, beta, gamma, 1e-8)

    return activation(z_norm)

#!/usr/bin/env python3

""" contains func create_layer"""

import tensorflow.compat.v1 as tf


def create_layer(prev, n, activation):
    """ creates a layer"""
    return tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            mode='fan_avg'),
        name="layer")(prev)
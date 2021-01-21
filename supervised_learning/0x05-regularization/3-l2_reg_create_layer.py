#!/usr/bin/env python3
"l2 regularization function "
import numpy as np
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    "create a tensorflow layer that includes L2 regularization"
    k = tf.contrib.layers.l2_regularizer(lambtha)
    layer = tf.keras.layers.Dense(n, activation=activation, kernel_regularizer=k)
    return layer(prev)

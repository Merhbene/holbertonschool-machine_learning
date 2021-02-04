#!/usr/bin/env python3
"""
    Convolutional Neural Networks
"""
import tensorflow as tf


def lenet5(x, y):
    """ Builds modified LeNet-5 using tensorflow
    """
    init = tf.contrib.layers.variance_scaling_initializer()

    layer1 = tf.layers.Conv2D(filters=6, kernel_size=(5, 5),
                              padding='same', activation='relu',
                              kernel_initializer=init)(x)

    layer2 = tf.layers.MaxPooling2D((2, 2), strides=(2, 2))(layer1)

    layer3 = tf.layers.Conv2D(filters=16, kernel_size=(5, 5),
                              padding='valid', activation='relu',
                              kernel_initializer=init)(layer2)

    layer4 = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(layer3)

    flat = tf.layers.Flatten()(layer4)

    layer5 = tf.layers.Dense(120, activation='relu',
                             kernel_initializer=init)(flat)

    layer6 = tf.layers.Dense(84, activation='relu',
                             kernel_initializer=init)(layer5)

    layer7 = tf.layers.Dense(10, activation='softmax',
                             kernel_initializer=init)(layer6)
    loss = tf.losses.softmax_cross_entropy(y, y_pred)
    
    y_pred = tf.nn.softmax(layer7)

    grady = tf.train.AdamOptimizer()
    op = grady.minimize(loss)
    acc = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(acc, tf.float32), name="Mean")
    return y_pred, op, loss, accuracy


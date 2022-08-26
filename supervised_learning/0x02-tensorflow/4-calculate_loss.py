#!/usr/bin/env python3


""" contains loss funct"""

import tensorflow.compat.v1 as tf


def calculate_loss(y, y_pred):
    """ loss count funct"""
    return tf.losses.softmax_cross_entropy(onehot_labels=y, logits=y_pred)
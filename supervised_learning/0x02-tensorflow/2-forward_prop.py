#!/usr/bin/env python3


""" contains forward prop funct"""

import tensorflow.compat.v1 as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """ forward prop funct"""
    for i in range(len(layer_sizes)):
        x = create_layer(x, layer_sizes[i], activations[i])
    return x
#!/usr/bin/env python3
import numpy as np


def one_hot_encode(Y, classes):
    """Convert a numeric label vector into a one-hot matrix"""
    m = Y.shape[0]
    Y_one_hot = np.zeros(shape=(classes, m))
    for i in range(m) :
        Y_one_hot[Y[i]][i] = 1

    return Y_one_hot

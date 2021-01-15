#!/usr/bin/env python3
""" function one_hot_encode """
import numpy as np


def one_hot_encode(Y, classes):
    """Convert a numeric label vector into a one-hot matrix"""
    if Y is None or classes is None:
        return None
    try:
        m = Y.shape[0]
        "classes is the maximum number of classes found in Y"
        Y_one_hot = np.zeros(shape=(classes, m))
        for i in range(m):
            Y_one_hot[Y[i]][i] = 1

        return Y_one_hot
    except Exception:
        return None

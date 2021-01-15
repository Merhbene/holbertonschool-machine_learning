#!/usr/bin/env python3
"""one hor decode function"""
import numpy as np


def one_hot_decode(one_hot):
    "Convert a one-hot matrix into a vector of labels"
    try:
        classes, m = one_hot.shape
        "classes is the maximum number of classes"
        Y_decoded = np.zeros(shape=(m,))
        for i in range(m):
            for j in range(classes):
                if one_hot[j][i] == 1:
                    Y_decoded[i] = j
        return Y_decoded.astype(int)
    except Exception:
        return None

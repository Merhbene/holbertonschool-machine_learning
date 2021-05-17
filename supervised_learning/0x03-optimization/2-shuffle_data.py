#!/usr/bin/env python3
"Shuffle Data"
import numpy as np


def shuffle_data(X, Y):
    " shuffles the data points in two matrices the same way"
    n = np.random.permutation(len(X))
    return X[n], Y[n]

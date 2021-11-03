#!/usr/bin/env python3
"""variance"""
import numpy as np


def variance(X, C):
    """calculates the total intra-cluster variance for a data set"""
    if type(X) is not np.ndarray or X.ndim != 2:
        return None
    if type(C) is not np.ndarray or C.ndim != 2 or C.shape[1] != X.shape[1]:
        return None
    dist = np.square(X[:, None, :] - C[None, :, :]).sum(axis=-1)
    centroid_dist = np.min(dist, axis=1)    
    var = np.sum(centroid_dist)
    return var

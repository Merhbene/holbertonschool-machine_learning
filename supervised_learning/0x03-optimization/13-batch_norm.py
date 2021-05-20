#!/usr/bin/env python3
" Batch Normalization"
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    "normalize an unactivated output of a neural network using batch normalization"
    m = np.mean(Z, axis=0)
    v = np.var(Z, axis=0)
    Z_norm = (Z - m) / np.sqrt(v + epsilon)
    Z_norm = gamma * Z_norm  + beta
    return Z_norm

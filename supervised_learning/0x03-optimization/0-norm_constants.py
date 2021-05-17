#!/usr/bin/env python3
"Normalization Constants"
import numpy as np


def normalization_constants(X):
    """creates normalization constants"""
    m = np.mean(X, axis=0)
    s = np.std(X, axis=0)
    return m, s

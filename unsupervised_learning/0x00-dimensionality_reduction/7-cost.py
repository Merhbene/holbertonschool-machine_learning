#!/usr/bin/env python3
"""Cost"""
import numpy as np


def cost(P, Q):
    """calculates the cost of the transformation"""
    C = np.sum(P * np.log(np.maximum(P, 1e-12) / np.maximum(Q, 1e-12)))
    return C

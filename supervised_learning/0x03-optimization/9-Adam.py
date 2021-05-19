#!/usr/bin/env python3
"""Adam"""
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """update a variable in place using the Adam optimization algorithm"""
    v = beta1 * v + (1 - beta1) * grad
    s = beta2 * s + (1 - beta2) * (grad ** 2)

    v_t = v / (1 - beta1 ** t)
    s_t = s / (1 - beta2 ** t)

    var = var - ((alpha * v_t) / (np.sqrt(s_t) + epsilon))

    return var, v, s

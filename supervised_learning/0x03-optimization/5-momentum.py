#!/usr/bin/env python3
"Momentum"


def update_variables_momentum(alpha, beta1, var, grad, v):
    """update a variable using the gradient descent with momentum
    optimization algorithm"""
    v_t = beta1 * v + (1 - beta1) * grad
    var = var - alpha * v_t

    # Returns: the updated variable and the new moment, respectively
    return var, v_t

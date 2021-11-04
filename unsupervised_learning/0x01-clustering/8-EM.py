#!/usr/bin/env python3
"""EM"""
import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """performs the EM algorithm"""
    if type(X) is not np.ndarray or X.ndim != 2:
        return None, None, None, None, None
    if type(k) is not int or int(k) != k or k < 1:
        return None, None, None, None, None
    if type(iterations) is not int or int(iterations) != iterations or iterations < 1:
        return None, None, None, None, None
    if type(tol) is not float or tol < 0:
        return None, None, None, None, None
    if type(verbose) is not bool:
        return None, None, None, None, None
    pi, m, S = initialize(X, k)
    lo = None
    for i in range(iterations):
        g, l = expectation(X, pi, m, S)
        if lo is not None and np.abs(l - lo) <= tol:
            if verbose:
                print('Log Likelihood after {} iterations: {}'.format(i, l.round(5)))
            break
        if verbose and i % 10 == 0:
            print('Log Likelihood after {} iterations: {}'.format(i, l.round(5)))
        pi, m, S = maximization(X, g)
        lo = l
    else:
        g, l = expectation(X, pi, m, S)
        if verbose:
            print('Log Likelihood after {} iterations: {}'.format(iterations, l.round(5)))
    return pi, m, S, g, l

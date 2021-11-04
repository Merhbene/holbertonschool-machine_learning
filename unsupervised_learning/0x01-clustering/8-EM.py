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

import matplotlib.pyplot as plt
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization

if __name__ == '__main__':
    np.random.seed(11)
    a = np.random.multivariate_normal([30, 40], [[75, 5], [5, 75]], size=10000)
    b = np.random.multivariate_normal([5, 25], [[16, 10], [10, 16]], size=750)
    c = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], size=750)
    d = np.random.multivariate_normal([20, 70], [[35, 10], [10, 35]], size=1000)
    X = np.concatenate((a, b, c, d), axis=0)
    np.random.shuffle(X)
    k = 4
    pi, m, S, g, l = expectation_maximization(X, k, 150, verbose=True)
    clss = np.sum(g * np.arange(k).reshape(k, 1), axis=0)
    plt.scatter(X[:, 0], X[:, 1], s=20, c=clss)
    plt.scatter(m[:, 0], m[:, 1], s=50, c=np.arange(k), marker='*')
    plt.show()
    print(X.shape[0] * pi)
    print(m)
    print(S)
    print(l)

#!/usr/bin/env python3
import sklearn.mixture


def gmm(X, k):
    """calculates a GMM from a dataset"""
    g = sklearn.mixture.GaussianMixture(n_components=k)
    # Generate random observations with two modes centered on 0
    # and 10 to use for training.
    g.fit(X) 
    pi = g.weights_
    m = g.means_
    S = g.covariances_

    clss = g.predict(X)
    bic = g.bic(X)

    return pi, m, S, clss, bic


import matplotlib.pyplot as plt
import numpy as np
gmm = __import__('11-gmm').gmm

if __name__ == '__main__':
    np.random.seed(11)
    a = np.random.multivariate_normal([30, 40], [[75, 5], [5, 75]], size=10000)
    b = np.random.multivariate_normal([5, 25], [[16, 10], [10, 16]], size=750)
    c = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], size=750)
    d = np.random.multivariate_normal([20, 70], [[35, 10], [10, 35]], size=1000)
    X = np.concatenate((a, b, c, d), axis=0)
    np.random.shuffle(X)

    pi, m, S, clss, bic = gmm(X, 4)
    print(pi)
    print(m)
    print(S)
    print(bic)
    plt.scatter(X[:, 0], X[:, 1], s=10, c=clss)
    plt.scatter(m[:, 0], m[:, 1], s=50, marker='*', c=list(range(4)))
    plt.show()

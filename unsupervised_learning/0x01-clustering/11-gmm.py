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

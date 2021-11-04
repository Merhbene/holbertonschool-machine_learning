#!/usr/bin/env python3
""" Agglomerative """
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """ perform agglomerative clustering on a dataset """
    h = scipy.cluster.hierarchy
    Z = h.linkage(X, 'ward')
    ind = h.fcluster(Z, t=dist, criterion="distance")
    fig = plt.figure()
    dn = h.dendrogram(Z, color_threshold=dist)
    plt.show()
    return ind

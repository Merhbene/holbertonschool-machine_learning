#!/usr/bin/env python3
""" Kmeans using sklearn """
import sklearn.cluster


def kmeans(X, k):
    """performs K-means on a dataset"""
    kmeans = sklearn.cluster.KMeans(n_clusters=k, random_state=0).fit(X)
    C = kmeans.cluster_centers_
    clss = kmeans.labels_
    return C, clss

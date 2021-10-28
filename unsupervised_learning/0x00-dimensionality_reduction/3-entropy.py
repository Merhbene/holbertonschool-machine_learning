#!/usr/bin/env python3
"Entropy"
import numpy as np


def HP(Di, beta):
    """calculates the p affinities and entropy for a given datapoint"""
    Pi = np.exp(- Di * beta)
    Pi = Pi / np.sum(Pi)
    Hi = np.sum(-Pi * np.log2(Pi))
    return Hi, Pi


pca = __import__('1-pca').pca
P_init = __import__('2-P_init').P_init
HP = __import__('3-entropy').HP

X = np.loadtxt("mnist2500_X.txt")
X = pca(X, 50)
D, P, betas, _ = P_init(X, 30.0)
H0, P[0, 1:] = HP(D[0, 1:], betas[0])
print(H0)
print(P[0])

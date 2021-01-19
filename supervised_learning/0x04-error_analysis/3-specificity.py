#!/usr/bin/env python3
import numpy as np
""" new function"""


def specificity(confusion):
    classes = confusion.shape[0]
    specificity = np.zeros((classes,))
    for j in range(classes):
        X = np.delete(confusion, [j], axis=0)
        TN = np.sum(np.delete(X, [j], axis=1))
        N = np.sum(X)
        specificity[j] = TN / N

    return specificity

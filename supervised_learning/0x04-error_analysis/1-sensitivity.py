#!/usr/bin/env python3
import numpy as np
""" new function"""


def sensitivity(confusion):
    """"Sensitivity measures the proportion of 
    positives that are correctly identified
    """
    classes = confusion.shape[0]
    sensitivity = np.zeros((classes,))
    for j in range(classes):
        TP = confusion[j][j]
        P = np.sum(confusion, axis=1)[j]
        sensitivity[j] = TP / P

    return sensitivity

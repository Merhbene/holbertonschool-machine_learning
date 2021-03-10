#!/usr/bin/env python3
import numpy as np
""" new function"""


def precision(confusion):
    classes = confusion.shape[0]
    precision = np.zeros((classes,))
    for j in range(classes):
        TP = confusion[j][j]
        FP = np.sum(confusion, axis=0)[j]
        precision[j] = TP / FP

    return precision

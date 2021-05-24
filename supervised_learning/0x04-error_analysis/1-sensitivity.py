#!/usr/bin/env python3
import numpy as np
"""Sensitivity"""


def sensitivity(confusion):
    """"
    Sensitivity measures the proportion of
    positives that are correctly identified
    """
    s = []
    for i in range(len(confusion)):
        TP = confusion[i][i]
        P = np.sum(confusion[i])
        s.append( TP / P)

    return np.array(s)

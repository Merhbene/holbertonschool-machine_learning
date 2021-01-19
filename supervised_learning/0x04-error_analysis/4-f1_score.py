#!/usr/bin/env python3
import numpy as np
""" new function"""


def f1_score(confusion):
    classes = confusion.shape[0]
    f1_score = np.zeros((classes,))
    for j in range(classes):
        TP = confusion[j][j]
        FP = np.sum(np.delete(confusion, [j],axis=0), axis=0)[j]
        FN = np.sum(np.delete(confusion, [j],axis=1), axis=1)[j]
        f1_score[j] = TP / (TP + 0.5 * (FP + FN))

    return f1_score   

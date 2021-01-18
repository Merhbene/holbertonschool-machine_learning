#!/usr/bin/env python3
import numpy as np


def create_confusion_matrix(labels, logits):
    "create a confusion matrix"
    m, classes = labels.shape
    "classes is the number of classes"
    confusion_matrix = np.zeros((classes, classes))
    for c in range(classes):
        for line in range(m):
            "m is the number of data points"
            if logits[line][c] == 1:
                for col in range(classes):
                    confusion_matrix[col][c] += labels[line][col]

    return confusion_matrix

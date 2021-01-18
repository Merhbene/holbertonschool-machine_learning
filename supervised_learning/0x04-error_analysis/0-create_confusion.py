#!/usr/bin/env python3
import numpy as np


def create_confusion_matrix(labels, logits):
    """labels is a one-hot numpy.ndarray of shape (m, classes)
    containing the correct labels for each data point
    """

    m, classes = labels.shape
    """classes is the number of classes"""
    confusion_matrix = np.zeros((classes, classes))
    """m is the number of data points"""
    for c in range(classes):
       for line in range(m):
           if logits[line][c] == 1:
              for col in range(classes):
                  confusion_matrix[col][c] += labels[line][col]

    return confusion_matrix


#!/usr/bin/env python3
import numpy as np


def create_confusion_matrix(labels, logits):
     """labels is a one-hot numpy.ndarray of shape (m, classes)
     containing the correct labels for each data point
     """
     m, classes = labels.shape
     confusion_matrix = np.zeros((classes,classes))

     for c in range (classes):
        for line in range (m):
            if logits[line][c] == 1:
                for col in range (classes):
                    confusion_matrix[col][c] += labels[line][col]

     return confusion_matrix

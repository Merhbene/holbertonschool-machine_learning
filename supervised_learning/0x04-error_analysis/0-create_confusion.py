#!/usr/bin/env python3
""" Create Confusion"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """creates a confusion matrix from labels and logits"""
    m, classes = labels.shape
    confusion = np.zeros((classes, classes))
    labels = np.argmax(labels, axis=1)
    logits = np.argmax(logits, axis=1)
    for i in range(m):
        confusion[labels[i], logits[i]] += 1
    return confusion

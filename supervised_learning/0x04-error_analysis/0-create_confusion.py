#!/usr/bin/env python3
import numpy as np
""" Create Confusion"""

def create_confusion_matrix(labels, logits):
    """creates a confusion matrix from labels and logits"""
    np.matmul(labels.T, logits)
    return confusion

#!/usr/bin/env python3
"Entropy"
import numpy as np


def HP(Di, beta):
    """calculates the p affinities and entropy for a given datapoint"""
    Pi = np.exp(- Di * beta)
    Pi = Pi / np.sum(Pi)
    Hi = np.sum(-Pi * np.log2(Pi))
    return Hi, Pi


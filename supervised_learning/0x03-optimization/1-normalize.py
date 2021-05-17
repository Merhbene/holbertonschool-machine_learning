#!/usr/bin/env python3
"Normalize"
import numpy as np


def normalize(X, m, s):
    return (X-s) / s

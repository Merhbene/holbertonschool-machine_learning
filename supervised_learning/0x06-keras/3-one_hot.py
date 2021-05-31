#!/usr/bin/env python3
"One Hot"
import tensorflow.keras as k


def one_hot(labels, classes=None): 
    "converts a label vector into a one-hot matrix"
    return k.utils.to_categorical(labels, num_classes=classes)

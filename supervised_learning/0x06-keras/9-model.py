#!/usr/bin/env python3
"Save and Load Model"
import tensorflow.keras as K


def save_model(network, filename):
    "saves a model’s weights"
    network.save(filename)


def load_model(filename):
    "loads a model’s weights"
    return K.models.load_model(filename)

#!/usr/bin/env python3
"One Hot"
import tensorflow.keras as k


def train_model(network, data, labels, batch_size, epochs, verbose=True, shuffle=False):
    "trains a model using mini-batch gradient descent"
    return network.fit(data, labels, batch_size=batch_size, epochs=epochs,
                       verbose=verbose, shuffle=shuffle)

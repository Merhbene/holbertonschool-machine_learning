#!/usr/bin/env python3
"Valid"
import tensorflow.keras as k


def train_model(network, data, labels, batch_size, epochs, verbose=True, shuffle=False):
    "also analyze validaiton data"
    return network.fit(data, labels, batch_size=batch_size, epochs=epochs, validation_data=validation_data
                       verbose=verbose, shuffle=shuffle)

#!/usr/bin/env python3
"Sequential"
import tensorflow.keras as k


def build_model(nx, layers, activations, lambtha, keep_prob):
    "builds a neural network with the Keras library"
    reg = k.regularizers.l2(lambtha)
    model = k.Sequential()
    model.add(k.layers.Dense(layers[0], input_shape=(nx,),
                             activation=activations[0], kernel_regularizer=reg))
    for i in range(1, len(layers)):
        model.add(k.layers.Dropout(rate=1-keep_prob))# Fraction of the input units to drop
        model.add(k.layers.Dense(layers[i], activation=activations[i], kernel_regularizer=reg))

    return model

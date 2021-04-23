#!/usr/bin/env python3
""" creates an autoencoder """
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    "Encoder model"
    Input = keras.layers.Input(shape=(input_dims,))
    x = Input
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)
    Output = keras.layers.Dense(latent_dims, activation='relu',
                              activity_regularizer=keras.regularizers.l1(lambtha))(x)

    encoder = keras.Model(inputs=Input, outputs=Output)

    "Decoder model"
    Input1 = keras.layers.Input(shape=(latent_dims,))
    x = Input1
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)
    Output1 = keras.layers.Dense(input_dims, activation='sigmoid')(x)

    decoder = keras.Model(inputs=Input1, outputs=Output1)

    " Autoencoder "
    auto = keras.Model(inputs=Input, outputs=decoder(encoder(Input)))
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto

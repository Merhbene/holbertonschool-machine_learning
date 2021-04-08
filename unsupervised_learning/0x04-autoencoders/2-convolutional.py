#!/usr/bin/env python3

import tensorflow.keras as keras

def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    if hidden_layers is None:
        hidden_layers = []
    
    Ei = keras.layers.Input(shape=(input_dims,))
    X = Ei
    for l in hidden_layers:
        X = keras.layers.Dense(l, activation='relu')(X)
    Eo = keras.layers.Dense(latent_dims, activation='relu', activity_regularizer=keras.regularizers.l1(lambtha))(X)

    Di = keras.layers.Input(shape=(latent_dims,))
    X = Di
    for l in reversed(hidden_layers):
        X = keras.layers.Dense(l, activation='relu')(X)
    Do = keras.layers.Dense(input_dims, activation='sigmoid')(X)

    encoder = keras.Model(inputs=Ei, outputs=Eo)
    decoder = keras.Model(inputs=Di, outputs=Do)
    auto = keras.Model(inputs=Ei, outputs=decoder(encoder(Ei)))
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
  

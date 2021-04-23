#!/usr/bin/env python3
""" creates an autoencoder """
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    "Encoder"
    input = keras.layers.Input(shape=(input_dims,))
    x = input 
    for layer in hidden_layers:
        x = keras.layers.Dense(layer,activation='relu')(x)
    output = keras.layers.Dense(latent_dims,activation='relu', activity_regularizer=keras.regularizers.l1(lambtha))(x)

    encoder = keras.Model(input,output)
    
    "decoder"
    input2 = keras.layers.Input(shape=(latent_dims,))
    x = input2
    for layer in reversed(hidden_layers):
        x = keras.layers.Dense(layer, activation='relu')(x) 
    output2 = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    
    decoder = keras.Model(input2, output2)

    "Autoencoder"
    auto = keras.Model(input, decoder(encoder(input)))
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto

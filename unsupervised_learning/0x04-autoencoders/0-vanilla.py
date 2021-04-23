#!/usr/bin/env python3
""" creates an autoencoder """
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    "Encoder"
    input = tf.keras.layers.Input(shape=input_dims)
    x = input 
    for layer in hidden_layers:
        x = tf.keras.layers.Dense(layer,activation='relu')(x)
    output = tf.keras.layers.Dense(latent_dims,activation='relu')(x)

    encoder = tf.keras.Model(input,output)
    
    "decoder"
    input2 = tf.keras.layers.Input(latent_dims)
    x = input2
    for layer in reversed(hidden_layers):
        x = tf.keras.layers.Dense(layer, activation='relu')(x)
    output2 = tf.keras.layers.Dense(input_dims, activation='sigmoid')(x)
    
    decoder = tf.keras.Model(input2, output2)

    "Autoencoder"
    auto = tf.keras.Model(input, decoder(encoder(input)))
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto

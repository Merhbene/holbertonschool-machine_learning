#!/usr/bin/env python3

import tensorflow.keras as keras

def autoencoder(input_dims, hidden_layers, latent_dims):
    if hidden_layers is None:
        hidden_layers = []
    
    Ei = keras.layers.Input(shape=(input_dims,))
    X = Ei
    for l in hidden_layers:
        X = keras.layers.Dense(l, activation='relu')(X)
    mu = keras.layers.Dense(latent_dims)(X)
    log_sig = keras.layers.Dense(latent_dims)(X)
    def sample_z(args):
        _mu, _log_sig = args
        m = keras.backend.shape(_mu)[0]
        n = keras.backend.int_shape(_mu)[1]
        epsilon = keras.backend.random_normal(shape=(m, n), mean=0., stddev=1.)
        return _mu + keras.backend.exp(log_sig / 2) * epsilon
    Eo = keras.layers.Lambda(sample_z)([mu, log_sig])
    encoder = keras.Model(inputs=Ei, outputs=[Eo, mu, log_sig])
    Di = keras.layers.Input(shape=(latent_dims,))
    X = Di
    for l in reversed(hidden_layers):
        X = keras.layers.Dense(l, activation='relu')(X)
    Do = keras.layers.Dense(input_dims, activation='sigmoid')(X)
    decoder = keras.Model(inputs=Di, outputs=Do)
    auto = keras.Model(inputs=Ei, outputs=decoder(encoder(Ei)[0]))
    def total_loss(x, x_decoded):
        content_loss = keras.backend.sum(keras.backend.binary_crossentropy(x, x_decoded), axis=1)
        kl_loss = 0.5 * keras.backend.sum(keras.backend.exp(log_sig) + keras.backend.square(mu) - 1 - log_sig, axis=1)
        return content_loss + kl_loss
    
    auto.compile(optimizer='adam', loss=total_loss)

    return encoder, decoder, auto

#!/usr/bin/env python3
"""Autoencoder module"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):

    inputs = keras.Input(shape=(input_dims,))
    h = keras.layers.Dense(hidden_layers[0], activation='relu')(inputs)
    for dim in hidden_layers[1:]:
        h = keras.layers.Dense(dim, activation='relu')(h)
    z_mean = keras.layers.Dense(latent_dims)(h)
    z_log_sigma = keras.layers.Dense(latent_dims)(h)

    def sampling(args):
        z_mean, z_log_sigma = args
        epsilon = keras.backend.random_normal(
            shape=(keras.backend.shape(z_mean)[0], latent_dims),
            mean=0.,
            stddev=0.1
        )
        return z_mean + keras.backend.exp(z_log_sigma) * epsilon

    z = keras.layers.Lambda(sampling)([z_mean, z_log_sigma])
    encoder = keras.Model(inputs, [z_mean, z_log_sigma, z])

    latent_inputs = keras.Input(shape=(latent_dims,))
    x = keras.layers.Dense(hidden_layers[-1], activation='relu')(latent_inputs)
    for dim in hidden_layers[-2::-1]:
        x = keras.layers.Dense(dim, activation='relu')(x)
    decoded_outputs = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.Model(latent_inputs, decoded_outputs)
    outputs = decoder(encoder(inputs)[2])
    auto = keras.Model(inputs, outputs)


    def total_loss(inputs, outputs):
        content_loss = keras.backend.sum(keras.backend.binary_crossentropy(x, x_decoded), axis=1)
        kl_loss = 0.5 * keras.backend.sum(keras.backend.exp(z_log_sigma) + keras.backend.square(z_mean) - 1 - z_log_sigma, axis=1)
        return content_loss + kl_loss

    auto.compile(optimizer='adam', loss=total_loss)

    return encoder, decoder, auto

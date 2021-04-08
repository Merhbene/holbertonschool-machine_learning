#!/usr/bin/env python3
""" create a convolutional autoencoder """
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    "Encoder model"
    Input = keras.layers.Input(shape=input_dims)
    x = Input
    for f in filters:
        x = keras.layers.Conv2D(f, kernel_size=(3, 3), padding='same', activation='relu')(x)
        x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    Output = x

    encoder = keras.Model(inputs=Input, outputs=Output)

    "Decoder model"
    Input1 = keras.layers.Input(shape=latent_dims)
    x = Input1
    for f in reversed(filters[1:]):
        x = keras.layers.Conv2D(f, kernel_size=(3, 3), padding='same', activation='relu')(x)
        x = keras.layers.UpSampling2D((2, 2))(x)

    x = keras.layers.Conv2D(filters[0], kernel_size=(3, 3), padding='valid', activation='relu')(x)
    x = keras.layers.UpSampling2D((2, 2))(x)
    Output1 = keras.layers.Conv2D(input_dims[2], kernel_size=(3, 3), activation='sigmoid', padding='same')(x)

    decoder = keras.Model(inputs=Input1, outputs=Output1)

    " Autoencoder "
    auto = keras.Model(inputs=Input, outputs=decoder(encoder(Input)))
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto

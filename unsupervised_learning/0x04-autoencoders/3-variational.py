#!/usr/bin/env python3
"""
    Autoencoders
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """ creates a convolutional autoencoder
        input_dims is an integer containing the dimensions of the model input
        hidden_layers is a list containing the number of filters for each
            convolutional layer in the encoder, respectively
        the hidden layers should be reversed for the decoder
        latent_dims is a tuple of integers containing the dimensions
            of the latent space representation
        Each convolution in the encoder should use a kernel size of (3, 3)
            with same padding and relu activation, followed by max pooling
            of size (2, 2)
        Each convolution in the decoder, except for the last two, should use a
            filter size of (3, 3) with same padding and relu activation,
            followed by upsampling of size (2, 2)
        The second to last convolution should instead use valid padding
        The last convolution should have the same number of filters as number
            of channels in input_dims with sigmoid activation and no upsampling
        Returns: encoder, decoder, auto
            encoder is the encoder model
            decoder is the decoder model
            auto is the full autoencoder model
    """
    inputs = keras.Input(shape=(input_dims,))
    layer_enc = keras.layers.Dense(hidden_layers[0], activation='relu')(inputs)
    hl = len(hidden_layers) - 1
    for hl in range(1, len(hidden_layers)):
        layer_enc = keras.layers.Dense(hidden_layers[hl],
                                       activation='relu')(layer_enc)
    latent_enc = layer_enc
    z_mean = keras.layers.Dense(latent_dims)(latent_enc)
    z_log_sigma = keras.layers.Dense(latent_dims)(latent_enc)


    def sampling(args):
        z_mean, z_log_sigma = args
        epsilon = keras.backend.random_normal(
            shape=(keras.backend.shape(z_mean)[0], latent_dims),
            mean=0,
            stddev=1
        )
        return z_mean + keras.backend.exp(z_log_sigma / 2) * epsilon

    z = keras.layers.Lambda(sampling)([z_mean, z_log_sigma])
    encoded_input = keras.Input(shape=(latent_dims,))
    latent = keras.layers.Dense(hidden_layers[hl],
                                activation='relu')(encoded_input)

    c = 1
    for hl in range(len(hidden_layers) - 2, -1, -1):
        decod = keras.layers.Dense(hidden_layers[hl],
                                   activation='relu')(latent if c else decod)
        c = 0
    decoded = keras.layers.Dense(input_dims,
                                 activation='sigmoid')(latent if c else decod)
    encoder = keras.Model(inputs, [z, z_mean, z_log_sigma])
    decoder = keras.Model(encoded_input, decoded)
    outputs = decoder(encoder(inputs)[2])

    def vae_loss(inputs, outputs):
        reconstruction_loss = keras.losses.binary_crossentropy(inputs, outputs)
        reconstruction_loss *= input_dims
        kl_loss = 1 + z_log_sigma - keras.backend.square(z_mean) \
            - keras.backend.exp(z_log_sigma)
        kl_loss = keras.backend.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = keras.backend.mean(reconstruction_loss + kl_loss)
        return vae_loss

    vae = keras.Model(inputs, outputs)
    vae.compile(optimizer="Adam", loss=vae_loss)
    return encoder, decoder, vae

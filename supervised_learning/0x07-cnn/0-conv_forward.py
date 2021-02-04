#!/usr/bin/env python3
""" Convolutional Forward Prop """
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """perform forward propagation over a convolutional
    layer of a neural network"""
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape

    sh, sw = stride[0], stride[1]

    if padding == 'valid':
        ph = pw = 0

    else:
        ph = int((((h_prev - 1) * sh + kh - h_prev) / 2) + (kh % 2 == 0))
        pw = int((((w_prev - 1) * sw + kw - w_prev) / 2) + (kh % 2 == 0))

    oh = int(((h_prev + 2 * ph - kh) / sh) + 1)
    ow = int(((w_prev + 2 * pw - kw) / sw) + 1)

    output_dim = (m, oh, ow, c_new)

    outputs = np.zeros(output_dim)

    padded_images = np.pad(A_prev, pad_width=((0, 0),
                         (ph, ph), (pw, pw), (0, 0)),
                         mode='constant', constant_values=0)

    for i in range(oh):
        for j in range(ow):
            x = (i * sh) + kh
            y = (j * sw) + kw

            M = padded_images[:, (i * sh):x, (j * sw):y, :]
            for k in range(c_new):
                outputs[:, i, j, k] = np.tensordot(M, W[:, :, :, k], axes=3)
                # axes=3 puisqu'on fait une convolution sur 2 cubes(3D)

    A = activation(outputs + b)

    return A

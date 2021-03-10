#!/usr/bin/env python3
""" Convolutional Back Prop """
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """that perform back propagation over a convolutional
    layer of a neural network"""
    m, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride
    ph = pw = 0

    if padding == 'same':
        ph = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))

    A_prev = np.pad(A_prev, pad_width=((0, 0),
                    (ph, ph), (pw, pw), (0, 0)),
                    mode='constant', constant_values=0)

    dW = np.zeros(W.shape)
    dA = np.zeros(A_prev.shape)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    for img in range(m):

        for h in range(h_new):
            for w in range(w_new):
                for f in range(c_new):

                    filter = W[:, :, :, f]
                    dz = dZ[img, h, w, f]  # int
                    slice_A = A_prev[img, h*sh:h*sh+kh, w*sw:w*sw+kw, :]

                    dA[img, h*sh:h*sh+kh, w*sw:w*sw+kw, :] += dz * filter
                    dW[:, :, :, f] += slice_A * dz

    dA = dA[:, ph:dA.shape[1]-ph, pw:dA.shape[2]-pw, :]

    return dA, dW, db

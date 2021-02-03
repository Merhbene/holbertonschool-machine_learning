#!/usr/bin/env python3
""" Convolutional Back Prop """
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):

    
    
    m, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride
    ph = pw = 0

    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    if padding == 'same':
       ph = int((((h_prev - 1) * sh + kh - h_prev) / 2) + 1)
       pw = int((((w_prev - 1) * sw + kw - w_prev) / 2) + 1)

    A_prev = np.pad(A_prev, pad_width=((0, 0),
                    (ph, ph), (pw, pw), (0, 0)),
                    mode='constant', constant_values=0)

    dW = np.zeros(W.shape)
    dA = np.zeros(A_prev.shape)
    for img in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for f in range(c_new):
                    tmp_W = W[:, :, :, f]

                    tmp_dz = dZ[img, h, w, f] # int
                    dA[img, h*sh:h*sh+kh, w*sw:w*sw+kw, :] += tmp_dz * tmp_W

                    tmp_A_prev = A_prev[img, h*sh:h*sh+kh, w*sw:w*sw+kw, :]
                    dW[:, :, :, f] += tmp_A_prev * tmp_dz

    dA = dA[:, ph:dA.shape[1]-ph, pw:dA.shape[2]-pw, :]

    return dA, dW, db

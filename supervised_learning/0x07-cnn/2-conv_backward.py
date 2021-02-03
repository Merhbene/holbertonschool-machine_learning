#!/usr/bin/env python3
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    m, h_new, w_new, c_new = dZ.shape
    _, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, _ = W.shape
    sh, sw = stride
    ph = 0
    pw = 0

    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    if padding == 'same':
       ph = int((((h_prev - 1) * sh + kh - h_prev) / 2) + 1)
       pw = int((((w_prev - 1) * sw + kw - w_prev) / 2) + 1)

    A_prev = np.pad(A_prev, pad_width=((0, 0),
                    (ph, ph), (pw, pw), (0, 0)),
                    mode='constant', constant_values=0)

    dW = np.zeros_like(W)
    dx = np.zeros_like(A_prev)
    for m_i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for f in range(c_new):
                    tmp_W = W[:, :, :, f]

                    tmp_dz = dZ[m_i, h, w, f]
                    dx[m_i, h*sh:h*sh+kh, w*sw:w*sw+kw, :] += tmp_dz * tmp_W

                    tmp_A_prev = A_prev[m_i, h*sh:h*sh+kh, w*sw:w*sw+kw, :]
                    dW[:, :, :, f] += tmp_A_prev * tmp_dz

    dx = dx[:, ph:dx.shape[1]-ph, pw:dx.shape[2]-pw, :]

    return dx, dW, db

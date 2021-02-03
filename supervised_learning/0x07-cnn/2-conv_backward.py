#!/usr/bin/env python3
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):

    m, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape

    sh, sw = stride[0], stride[1]

    if padding == 'valid':
       ph = pw = 0

    else:
       ph = int((((h - 1) * sh + kh - h_prev) / 2) + (kh % 2 == 0))
       pw = int((((w - 1) * sw + kw - w_prev) / 2) + (kh % 2 == 0))

    #A_prev = np.pad(A_prev, pad_width=((0, 0),(ph, ph),(pw, pw),(0, 0)), mode='constant', constant_values=0)

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

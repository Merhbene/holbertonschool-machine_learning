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

    A_prev = np.pad(A_prev, pad_width=((0, 0),(ph, ph),(pw, pw),(0, 0)), mode='constant', constant_values=0)

    dW = np.zeros(W.shape)
    db = np.sum(dZ,axis=(0,1,2),keepdims=True)
    dA = np.zeros(A_prev.shape)


    for img in range(m):

        for i in range(h_new):
            for j in range(w_new):
              for f in range(c_new):
                

                    x = (i * sh) + kh
                    y = (j * sw) + kw

                    M = A_prev[img, (i * sh):x, (j * sw):y, :]
                    N = W[:, :, :, f]
                    k = dZ[img, i, j, f] #int

                    dW[:, :, :, f] += M *  k

                    dA[img,(i * sh):x, (j * sw):y, :] += N * k

    dA = dA[:, ph:h_prev, pw:w_prev,: ]

    return dW , db , dA

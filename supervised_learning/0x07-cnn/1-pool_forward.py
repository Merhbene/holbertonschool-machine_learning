#!/usr/bin/env python3
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):

    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape[0], kernel_shape[1]

    sh, sw = stride[0], stride[1]

    oh = int(((h  - kh) / sh) + 1)
    ow = int(((w - kw) / sw) + 1)

    output_dim = (m, oh, ow, c_prev)

    outputs = np.zeros(output_dim)



    for i in range(oh):
        for j in range(ow):
            x = (i * sh) + kh
            y = (j * sw) + kw
            M = A_prev[:, (i * sh):x, (j * sw):y, :]
            if mode == 'max': 
               outputs[:, i, j, :] = np.max(M,axis=(1, 2))
            else:
               outputs[:, i, j, :] = np.average(M,axis=(1, 2))

    return outputs
    #A = activation(outputs + b)
    return outputs

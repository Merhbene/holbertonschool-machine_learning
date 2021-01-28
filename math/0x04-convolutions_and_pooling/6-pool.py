#!/usr/bin/env python3 
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    m, h, w, c = images.shape
    kh, kw = kernel_shape[0], kernel_shape[1]

    sh, sw = stride[0], stride[1]
    oh = int((h - kh) / sh) + 1
    ow = int((w - kw) / sw) + 1

    output_dim = (m, oh, ow, c)

    outputs = np.zeros(output_dim)

    for i in range(output_dim[1]):
        for j in range(output_dim[2]):
            x = (i * sh) + kh
            y = (j * sw) + kw
            M = images[:, (i * sh):x, (j * sw):y, :]
            if mode == 'max': 
               outputs[:, i, j, :] = np.max(M,axis=(1, 2))
            else:
               outputs[:, i, j, :] = np.average(M,axis=(1, 2))

    return outputs

#!/usr/bin/env python3
import numpy as np

def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    m, h, w = images.shape
    kh, kw = kernel.shape

    sh, sw = stride[0], stride[1]

    if padding == 'valid':
       ph = pw = 0

    elif padding == 'same':
       ph = int((((h - 1) * sh + kh - h) / 2) + 1)
       pw = int((((w - 1) * sw + kw - w) / 2) + 1)

    else: 
       ph, pw = padding[0], padding[1]


    oh = int(((h + 2 * ph - kh) / sh) + 1)
    ow = int(((w + 2 * pw - kw) / sw) + 1)


    output_dim = (m, oh, ow)

    outputs = np.zeros(output_dim)

    padded_images = np.pad(images, pad_width=((0, 0),(ph, ph),(pw, pw)), mode='constant',  constant_values=0)

    for i in range(output_dim[1]):
        for j in range(output_dim[2]):
            x = (i * sh) + kh
            y = (j * sw) + kw

            M = padded_images[:, (i * sh):x, (j * sw):y]
            outputs[:, i, j] = np.tensordot(M, kernel)

    return outputs
  

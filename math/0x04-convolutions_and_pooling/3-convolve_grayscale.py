#!/usr/bin/env python3
"""
    Strided Convolution
"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):

    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride[0], stride[1]

    if padding == "valid":
       ph = pw = 0

    elif padding == 'same':
       ph = int((((h - 1) * sh + kh - h) / 2) + (kh % 2 == 0))
       pw = int((((w - 1) * sw + kw - w) / 2) +  (kw % 2 == 0))

    else:
       ph = padding[0]
       pw = padding[1]

    oh = int(((h + 2 * ph - kh) / sh) + 1)
    ow = int(((w + 2 * pw - kw) / sw) + 1)

    # convolution output
    output = np.zeros((m, oh, ow))

    # Add zero padding to the input image
    image_padded = np.pad(images, pad_width=((0, 0),(ph, ph),(pw, pw)), mode='constant',  constant_values=0)


    # Loop over every pixel of the output
    for i in range(oh):
        for j in range(ow):
            x = (i * sh) + kh
            y = (j * sw) + kw

            M = image_padded[:, (i * sh):x, (j * sw):y]
            output[:, i, j] = np.tensordot(M, kernel)
    return output

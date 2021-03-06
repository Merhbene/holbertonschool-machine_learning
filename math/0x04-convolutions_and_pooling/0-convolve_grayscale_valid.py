#!/usr/bin/env python3
import numpy as np
"Performs a valid convolution on grayscale images"


def convolve_grayscale_valid(images, kernel):
    """images contains multiple grayscale images"""
    """kernel contains the kernel for the convolution"""

    m, h, w = images.shape
    kh, kw = kernel.shape

    # output shape
    output_dim = (m, h - kh + 1, w - kw + 1)

    # creating outputs
    outputs = np.zeros(output_dim)

    # vectorizing the m images
    # image = np.arange(0, m)

    # iterating over the output array and generating the convolution
    for i in range(output_dim[1]):
        for j in range(output_dim[2]):
            x = i + kh
            y = j + kw
            # outputs[image, i, j] = np.sum(np.multiply(images[image, i: x, j: y], kernel), axis=(1, 2)) "marche"
            # outputs[image, i, j] = (images[image, i:x, j:y] * kernel).sum() "ne marche pas"

            M = images[:, i:x, j:y]
            outputs[:, i, j] = np.tensordot(M, kernel)

    return outputs

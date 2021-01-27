#!/usr/bin/env python3
import numpy as np


def convolve_grayscale_valid(images, kernel):

    m, h, w = images.shape
    kh, kw = kernel.shape

    # output shape 
    output_dim = (m, h - kh + 1, w - kw +1)

    # creating outputs 
    outputs = np.zeros(output_dim)

    # vectorizing the m images
    image = np.arange(0, m)

    # iterating over the output array and generating the convolution
    for i in range(output_dim[0]):
        for j in range(output_dim[1]):
            x = i + kh
            y = j + kw
            outputs[image, i, j] = np.sum(np.multiply(
                images[image, i: x, j: y], kernel), axis=(1, 2))

    return outputs

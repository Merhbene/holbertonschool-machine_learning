#!/usr/bin/env python3
import numpy as np
"Performs a valid convolution on grayscale images"


def convolve_grayscale_padding(images, kernel, padding): 
    """images is a numpy.ndarray with shape (m, h, w)
    containing multiple grayscale images
    kernel is a numpy.ndarray with shape (kh, kw)
    containing the kernel for the convolution """
    m, h, w = images.shape
    kh, kw = kernel.shape

    ph, pw = padding[0], padding[1]

    # output_height and output_width

    oh = h + 2 * ph - kh + 1
    ow = w + 2 * pw - kw + 1

    # output shape
    output_dim = (m, oh, ow)

    # creating outputs
    outputs = np.zeros(output_dim)

    # vectorizing the m images
    #image = np.arange(0, m)
    padded_images = np.pad(images, pad_width=((0, 0),(ph, ph),(pw, pw)), mode='constant',  constant_values=0)

    # iterating over the output array and generating the convolution
    for i in range(output_dim[1]):
        for j in range(output_dim[2]):
            x = i + kh
            y = j + kw
            #outputs[image, i, j] = np.sum(np.multiply(padded_images[image, i: x, j: y], kernel), axis=(1, 2)) 
            # outputs[image, i, j] = (images[image, i:x, j:y] * kernel).sum() "ne marche pas"

            M = padded_images[:, i:x, j:y]
            outputs[:, i, j] = np.tensordot(M, kernel)

    return outputs

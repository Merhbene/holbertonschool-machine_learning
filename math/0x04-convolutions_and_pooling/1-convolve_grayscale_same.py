#!/usr/bin/env python3
import numpy as np
"Performs a valid convolution on grayscale images"


def convolve_grayscale_same(images, kernel): 
    """images is a numpy.ndarray with shape (m, h, w)
    containing multiple grayscale images
    kernel is a numpy.ndarray with shape (kh, kw)
    containing the kernel for the convolution """
    m, h, w = images.shape
    kh, kw = kernel.shape

    ph = int((kh - 1) / 2) # the total rows added 
    pw = int((kw - 1) / 2) # the total coloumns added 
    "to give the input and output the same height and width"

    if kh % 2 == 0:
        ph = int(kh / 2)

    if kw % 2 == 0:
        pw = int(kw / 2)

    # output_height and output_width
    # H = i_h + 2pad - k_h + 1, W = i_w + 2pad - k_w + 1
    oh = h + 2 * ph - kh + int(kh % 2 == 1)
    ow = w + 2 * pw - kw + int(kw % 2 == 1)

    # output shape
    output_dim = (m, oh, ow)

    # creating outputs
    outputs = np.zeros(output_dim)

    # vectorizing the m images
    #image = np.arange(0, m)
    padded_images = np.pad(images, pad_width=((0, 0),(ph, ph),(pw, pw)),mode='constant',  constant_values=0)

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

#!/usr/bin/env python3
import numpy as np


def convolve_grayscale_valid(images, kernel):

    m, h, w = images.shape
    kh, kw = kernel.shape

    # output_height and output_width
    o_h = h - kh + 1
    o_w = w - kw + 1

    # creating outputs of size: n_images, o_h x o_w
    outputs = np.zeros((m, o_h, o_w))

    # vectorizing the n_images
    image = np.arange(0, m)

    # iterating over the output array and generating the convolution
    for x in range(o_h):
        for y in range(o_w):
            x1 = x + kh
            y1 = y + kw
            outputs[image, x, y] = np.sum(np.multiply(
                images[image, x: x1, y: y1], kernel), axis=(1, 2))

    return outputs

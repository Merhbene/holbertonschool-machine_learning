#!/usr/bin/env python3
"""
    Strided Convolution
"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):

    m, input_h, input_w = images.shape
    kernel_h, kernel_w = kernel.shape
    sh, sw = stride

    if padding == "valid":
        ph = 0
        pw = 0
    elif padding == "same":
        ph = int(np.ceil((input_h - 1) * sh + kernel_h - input_h) / 2)
        pw = int(np.ceil((input_w - 1) * sw + kernel_w - input_w) / 2)
    else:
        ph = padding[0]
        pw = padding[1]

    output_height = int(((input_h + 2 * ph - kernel_h) / sh)) + 1
    output_width = int(((input_w + 2 * pw - kernel_w) / sw)) + 1

    # convolution output
    output = np.zeros((m, output_height, output_width))

    # Add zero padding to the input image
    image_padded = np.pad(images,
                          pad_width=((0, 0), (ph, ph), (pw, pw)),
                          mode='constant')

    # Loop over every pixel of the output
    for i in range(output_height):
        for j in range(output_width):
            x = i * sh
            y = j * sw
            # element-wise multiplication of the kernel and the image
            img_slice = image_padded[:, x:x+kernel_h, y:y+kernel_w]
            output[:, i, j] = np.tensordot(img_slice, kernel)
    return output

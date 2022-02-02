#!/usr/bin/env python3
"""
    Strided Convolution
"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Method:
        performs a convolution on grayscale images.
    @images(numpy.ndarray), shape (m, h, w)
        containing multiple grayscale images.
        - m: the number of images
        - h: the height in pixels of the images
        - w: the width in pixels of the images
    @kernel(numpy.ndarray), shape (kh, kw)
        containing the kernel for the convolution
        - kh: the height of the kernel
        - kw: the width of the kernel
    @padding: tuple of (ph, pw)
        - ph: the padding for the height of the image
        - pw: the padding for the width of the image
     @stride: tuple of (sh, sw)
        - sh: the stride for the height of the image
        - sw: the stride for the width of the image
    Returns:
        a numpy.ndarray containing the convolved images.
    """
    m, input_h, input_w = images.shape
    kernel_h, kernel_w = kernel.shape
    sh, sw = stride

    if padding == "valid":
        ph = 0
        pw = 0

    elif padding == 'same':
        ph = int((((input_h - 1) * sh + kernel_h - input_h) / 2) + (kernel_h % 2 == 0))
        pw = int((((input_w - 1) * sw + kernel_w - input_w) / 2) +  (kernel_w % 2 == 0))

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
#!/usr/bin/env python3
import numpy as np


def convolve_grayscale_valid1(images, kernel):
    m, h, w = images.shape
    kh, kw = kernel.shape
    conv_dim = (m, h - kh + 1, w - kw +1)
    conv = np.zeros(conv_dim)
    Img = np.arange(0, m)

    for i in range(conv_dim[1]):
      for j in range(conv_dim[2]):
        conv[Img, i, j] = (images[Img, i:i + kh, j:j + kw] * kernel).sum()

    return conv

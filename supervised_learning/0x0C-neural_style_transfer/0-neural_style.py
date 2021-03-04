#!/usr/bin/env python3
import numpy as np
import tensorflow as tf


class NST :

    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'
    tf.enable_eager_execution()
    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        if type (style_image) is not np.ndarray or style_image.ndim != 3 or style_image.shape[2] !=3 :
            raise TypeError('style_image must be a numpy.ndarray with shape (h, w, 3)')
        if type (content_image) is not np.ndarray or style_image.ndim != 3 or style_image.shape[2] !=3 :
            raise TypeError('content_image must be a numpy.ndarray with shape (h, w, 3)')  
        if (type (alpha) is not int and type (alpha) is not float)  or alpha < 0 :
            raise TypeError('alpha must be a non-negative number')
        if (type (beta) is not int and type (beta) is not float)  or beta< 0 :
            raise TypeError('beta must be a non-negative number')        

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta 

    @staticmethod
    def scale_image(image):
        if type (image) is not np.ndarray or image.ndim != 3 or image.shape[2] !=3 :
            raise TypeError('image must be a numpy.ndarray with shape (h, w, 3)')
        h, w, _ = image.shape
        max_dim = 512
        scale = max_dim / max(h, w) 
        new_shape = (int(h*scale), int(w*scale))
        image=np.expand_dims(image, axis=0)
        scaled_image = tf.image.resize_bicubic(image, new_shape)
        scaled_image = tf.clip_by_value(scaled_image / 255 ,0,1)

        return scaled_image 

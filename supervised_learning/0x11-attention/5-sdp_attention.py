#!/usr/bin/env python3
"""Attention module"""
import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    "calculates the scaled dot product attention"
    """
    Q is a tensor with its last two dimensions as (..., seq_len_q, dk) containing the query matrix
    K is a tensor with its last two dimensions as (..., seq_len_v, dk) containing the key matrix
    V is a tensor with its last two dimensions as (..., seq_len_v, dv) containing the value matrix
    """
    Q_k = tf.matmul(Q, k, transpose_b=True)
    dk = tf.cast(tf.shape(Q)[-1], dtype=tf.float32)
    Q_k_scaled = Q_k / tf.math.sqrt(dk)

    if mask is not None:
        Q_k_scaled += mask * -1e9

    weights = tf.nn.softmax(Q_k_scaled)
    attention = tf.matmul(weights, V)

    return attention, weights

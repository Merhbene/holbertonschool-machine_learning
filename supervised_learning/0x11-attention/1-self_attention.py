#!/usr/bin/env python3`
"""class SelfAttention"""
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """calculate the attention for machine translation"""
    def __init__(self, units):
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)
        # V a Dense layer to be applied to the tanh of the sum of the outputs of W and U
        self.gru = tf.keras.layers.GRU(units=units,  return_sequences=True, return_state=True, recurrent_initializer="glorot_uniform")

    def __call__(self, s_prev, hidden_states):
        "s_prev containing the previous decoder hidden state"
        s_prev = tf.expand_dims(s_prev, 1)
        W_s = self.W(s_prev)
        "hidden_states is a tensor of shape (batch, input_seq_len, units)containing the outputs of the encoder"
        U_h = self.U(hidden_states)

        fc = self.V(tf.nn.tanh(W_s + U_h))
        attention_weights = tf.nn.softmax(fc, axis=1)

        # context contains the context vector for the decoder
        context = attention_weights * hidden_states
        context = tf.reduce_sum(context, axis=1)

        return context, attention_weights

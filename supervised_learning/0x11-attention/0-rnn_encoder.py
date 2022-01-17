#!/usr/bin/env python3`
"""Attention module"""
import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """RNNEncoder class -- Encode for machine translation"""
    def __init__(self, vocab, embedding, units, batch):
        self.batch = batch #  the batch size
        self.units = units # the number of hidden units in the RNN cell
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab, output_dim=embedding)
        self.gru = tf.keras.layers.GRU(units=units,  return_sequences=True, return_state=True, recurrent_initializer="glorot_uniform")

    def initialize_hidden_state(self):
        "Initializes the hidden states for the RNN cell to a tensor of zeros"
        initializer = tf.keras.initializers.Zeros()
        return initializer(shape=(self.batch, self.units))

    def __call__(self, x, initial):
        "Instance Call"
        embeds = self.embedding(x)
        outputs, hidden = self.gru(inputs=embeds, initial_state=initial)   
        return outputs, hidden

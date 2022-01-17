#!/usr/bin/env python3`
import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):

    def __init__(self, vocab, embedding, units, batch):
        self.batch = batch #  the batch size
        self.units = units # the number of hidden units in the RNN cell
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab, output_dim=embedding)
        self.gru = tf.keras.layers.GRU(units=units,  return_sequences=True, return_state=True, recurrent_initializer="glorot_uniform")

    def initialize_hidden_state(self):
        initializer = tf.keras.initializers.Zeros()
        return initializer(shape=(self.batch, self.units))

    def __call__(self, x, initial):
        embeds = self.embedding(x)
        outputs, hidden = self.gru(inputs=embeds, initial_state=initial)   
        return outputs, hidden

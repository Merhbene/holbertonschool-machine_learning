#!/usr/bin/env python3`
"""class RNNDecoder"""
import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):

    def __init__(self, vocab, embedding, units, batch):
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab, output_dim=embedding)
        self.gru = tf.keras.layers.GRU(units=units, return_sequences=True,
                                       return_state=True, recurrent_initializer="glorot_uniform")
        self.F = tf.keras.layers.Dense(units=vocab)

    def __call__(self,x, s_prev, hidden_states):
        _, units = s_prev.shape
        context, weights = SelfAttention(units)(s_prev, hidden_states)

        emb = self.embedding(x)
        context = tf.expand_dims(context, 1)

        concat = tf.concat([context, emb], axis=2)

        outputs, hidden = self.gru(inputs=concat)

        outputs = tf.reshape(outputs, (outputs.shape[0], outputs.shape[2]))
        y = self.F(outputs)

        return y, hidden

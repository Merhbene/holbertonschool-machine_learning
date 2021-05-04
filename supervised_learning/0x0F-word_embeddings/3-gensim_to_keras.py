#!/usr/bin/env python3
"""Extract Word2Vec"""


def gensim_to_keras(model):
    "converts a gensim word2vec model to a keras Embedding layer"
    model.wv.get_keras_embedding(train_embeddings=False)

    return model

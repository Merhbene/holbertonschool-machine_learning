#!/usr/bin/env python3
"""train Word2Vec"""
from gensim.models import Word2Vec


def word2vec_model(sentences, size=100, min_count=5, window=5, negative=5,
                   cbow=True, iterations=5, seed=0, workers=1):
    "create and trains a gensim word2vec model"

    model = Word2Vec(sentences=sentences, size=size, min_count=min_count,
                                   window=window, negative=negative, sg=cbow, seed=seed,
                                   iter=iterations, workers=workers)
    model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)  # train word vectors
    return model

#!/usr/bin/env python3
"""Word Embeddings module"""
import numpy as np

def word2vec_model(
    sentences,
    size=100,
    min_count=5,
    window=5,
    negative=5,
    cbow=True,
    iterations=5,
    seed=0,
    workers=1
):
    """Creates and trains a Word2Vec model
    Arguments:
        sentences {list[str]} -- Is a list of strings
    Keyword Arguments:
        size {int} -- Dimensinality of embedding layer (default: {100})
        min_count {int} -- Minimum number of occurences (default: {5})
        window {int} -- Maximum distance (default: {5})
        negative {int} -- Is the size of negative sampling (default: {5})
        cbow {bool} -- Indicates CBOW or Skip-gram (default: {True})
        iterations {int} -- Is the number of iterations (default: {5})
        seed {int} -- Is the seed for the random number (default: {0})
        workers {int} -- Is the number of workers (default: {1})
    Returns:
        model -- Trained model
    """
    return Word2Vec(
        sentences,
        size=size,
        min_count=min_count,
        window=window,
        negative=negative,
        iter=iterations,
        workers=workers,
        seed=seed,
        sg=not cbow
    )

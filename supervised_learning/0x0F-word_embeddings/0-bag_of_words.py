#!/usr/bin/env python3
""" Bag of words """
from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    "creates a bag of words embedding matrix"
    vectorizer = CountVectorizer(vocabulary=vocab)
    X = vectorizer.fit_transform(sentences)
    features = vectorizer.get_feature_names()
    embeddings = X.toarray()

    return embeddings, features

#!/usr/bin/env python3
"Cumulative N-gram BLEU score"
import numpy as np


def ngram_list(L, n):
    "contiguous sequence of n items"
    c = len(L)
    M = []
    for i in range(c):
        ngram = L[i:i + n]
        if len(ngram) == n:
            listToStr = ' '.join(map(str, ngram))
            M.append(listToStr)
    return M


def cumulative_bleu(references, sentence, n):
    "calculate the cumulative n-gram BLEU score for a sentence"
    c = len(sentence)
    r_list = [(abs(len(r) - c), i) for i, r in enumerate(references)]
    r_ind = min(r_list)[1]
    r = len(references[r_ind])

    if c > r:
        BP = 1
    else:
        BP = np.exp(1 - (r / c))

    weights = [1 / n for i in range(n)]

    Bleu = 1

    for i in range (1, n + 1):

        sentence_i = ngram_list(sentence, i)
        references_i = [ngram_list(ref, i) for ref in references]

        d = {s: sentence_i.count(s) for s in sentence_i}
        count = d.values()

        L = []
        for ref in references_i:
            d = {s: ref.count(s) for s in sentence_i}
            L.append(list(d.values()))
        cc = np.array(L)
        count_clip = cc.max(axis=0)

        precision_i = sum(count_clip) / sum(count)

        Bleu *= precision_i 

    return BP * (Bleu ** (1 / n))

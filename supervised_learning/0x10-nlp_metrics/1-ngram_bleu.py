#!/usr/bin/env python3
"N-gram BLEU score"
import numpy as np

def ngram_list(L, n):
    "contiguous sequence of n items"
    c = len(L)
    M = []
    for i in range(c):
        l = L[i : i + n]
        if len(l) == n:
            listToStr = ' '.join(map(str, l))
            M.append(listToStr)
    return M

def ngram_bleu(references, sentence, n):
    "calculates the n-gram BLEU score for a sentence"
    c = len(sentence)

    r_list = [(abs(len(r) - c), i) for i, r in enumerate(references)]
    r_ind = min(r_list)[1]
    r = len(references[r_ind])

    sentence = ngram_list(sentence, n)
    references = [ngram_list(ref, n) for ref in references]

    d = {s: sentence.count(s) for s in sentence}
    count = d.values()

    L = []
    for ref in references:
        d = {s: ref.count(s) for s in sentence}
        L.append(list(d.values()))
    cc = np.array(L)
    count_clip = cc.max(axis=0)

    p = sum(count_clip) / sum(count)

    if c > r:
        BP = 1
    else:
        BP = np.exp(1 - (r / c))
    return BP * p

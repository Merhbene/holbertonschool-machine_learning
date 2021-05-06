#!/usr/bin/env python3
"Unigram BLEU score"
import numpy as np


def uni_bleu(references, sentence):
    "calculates the unigram BLEU score for a sentence"

    d = {s: sentence.count(s) for s in sentence}
    count = d.values()

    L = []
    for ref in references:
        d = {s: ref.count(s) for s in sentence}
        L.append(list(d.values()))
    cc = np.array(L)
    count_clip = cc.max(axis=0)

    p = sum(count_clip) / sum(count)
    # print(sum(count_clip) , sum(count))
    # print(count, count_clip)
    c = len(sentence)
    """
    r is taken to be the sum of the lengths of the sentences whose lengths
     are closest to the lengths of the candidate sentences
    """
    r_list = [(abs(len(r) - c), i) for i, r in enumerate(references)]
    r_ind = min(r_list)[1]
    r = len(references[r_ind])

    """
    Method2
    r_list = np.array([np.abs(len(s)-c) for s in references])
    r_ind = np.argwhere(r_list == np.min(r_list))
    lens = np.array([len(s) for s in references])[r_ind]
    r = np.min(lens)
    """
    if c > r:
        BP = 1
    else:
        BP = np.exp(1 - (r / c))
    # print(BP,p)
    return BP * p

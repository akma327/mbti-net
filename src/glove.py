import numpy as np


def loadWordVectors(filepath, dimensions=50):
    wordVectors = {}
    f = open(filepath, 'r')
    for line in f:
        linfo = line.strip().split()
        word = linfo[0]
        embedding = np.array(map(float, linfo[1:]))
        wordVectors[word] = embedding
    return wordVectors



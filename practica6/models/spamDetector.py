import numpy as np
from models import readFile
from models import process_email
from models import kernelLineal

vocab = readFile.data4_1()
m = 500
n = 1899


def detection():
    features = np.empty((m, n + 1))
    for i in range(1, m + 1):
        # Leo el spam
        spam = readFile.data4_2(i)
        # process message
        token = process_email.email2TokenList(spam)
        # compruebo que las palabras de ese spam estan en mi lista de vocab
        f = emailFeautures(token)
        features[i - 1, :] = f
    return features


def emailInd(token):
    ind = [vocab[i] for i in token if i in vocab]
    return ind


def emailFeautures(token):
    features = np.zeros(n + 1)
    ind = emailInd(token)
    for i in ind:
        features[i] = 1
    return features

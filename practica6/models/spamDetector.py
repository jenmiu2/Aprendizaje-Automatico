import numpy as np
from models import readFile
from models import process_email
from models import kernelLineal

vocab = readFile.data4_1()
lVocab = len(vocab)
n = 1899


def detection():

    #for i in range(1, 501):
      #  print(i)
    # Leo el spam
    spam = readFile.data4_2(1)
    # process message
    token = process_email.email2TokenList(spam)
    # compruebo que las palabras de ese spam estan en mi lista de vocab
    f = emailFeautures(token)
    print(np.sum(f))
    # entreno los textos con eso


def emailInd(token):
    ind = [vocab[i] for i in token if i in vocab]
    return ind


def emailFeautures(token):
    features = np.zeros((lVocab, 1))
    ind = emailInd(token)
    for i in ind:
        features[i] = 1

    return features

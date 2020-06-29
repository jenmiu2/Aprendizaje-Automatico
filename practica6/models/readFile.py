from models import get_vocab_dict
from scipy.io import loadmat
import numpy as np
import codecs


def data1():
    data_ = loadmat("data/ex6data1.mat")
    x = data_['X']
    y = data_['y'].ravel()
    return x, y


def data2():
    data_ = loadmat("data/ex6data2.mat")
    x = data_['X']
    y = data_['y'].ravel()
    return x, y


def data3():
    data_ = loadmat("data/ex6data3.mat")
    x = data_['X']
    xVal = data_['Xval']
    y = data_['y'].ravel().reshape(-1)
    yVal = data_['yval'].ravel()
    return x, xVal, y, yVal


def data4_1():
    vocabulary = get_vocab_dict.getVocabDict()
    return vocabulary


def data4_2(ind):
    emailContent = codecs.open('{0}/{1:04d}.txt'.format('data/spam/', ind),
                               'r',
                               encoding='utf-8',
                               errors='ignore').read()
    return emailContent


def data4_3(ind):
    emailContent = codecs.open('{0}/{1:04d}.txt'.format('data/easy-ham/', ind),
                               'r',
                               encoding='utf-8',
                               errors='ignore').read()
    return emailContent


def data4_4(ind):
    emailContent = codecs.open('{0}/{1:04d}.txt'.format('data/hard-ham/', ind),
                               'r',
                               encoding='utf-8',
                               errors='ignore').read()
    return emailContent

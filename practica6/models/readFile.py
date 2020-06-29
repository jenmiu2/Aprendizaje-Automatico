from scipy.io import loadmat
import numpy as np


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
    y = data_['y'].ravel()
    yVal = data_['yval'].ravel()
    return x, xVal, y, yVal

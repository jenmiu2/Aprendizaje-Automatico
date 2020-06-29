from models import readFile
from models import showData
from models import kernelLineal
from models import kernelGaussiano
import numpy as np


def parte1():
    x, y = readFile.data1()

    showData.initial(x, y)
    svm = kernelLineal.initial(x, y, 1.0)
    showData.visualize_boundary(x, y, svm, 1)

    svm100 = kernelLineal.initial(x, y, 100)
    showData.visualize_boundary(x, y, svm100, 100)


def parte2():
    x, y = readFile.data2()
    showData.initial(x, y)

    svm = kernelGaussiano.initial(x, y, 1.0, 0.1)
    showData.visualize_boundary(x, y, svm, 1)


def parte3():
    # read data
    x, y, xVal, yVal = readFile.data3()

    # create c´s and sigma´s
    Cs = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    sigmas = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    length = len(sigmas)

    # create scores
    scores = np.zeros(shape=(length, length))

    for i, c in enumerate(Cs):
        for j, sigma in enumerate(sigmas):
            svm = kernelGaussiano.initial(x, y, c=c, sigma=sigma)
            scores[j: i] = svm.score(x, y)

    print(scores)


parte3()

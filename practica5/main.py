import numpy as np
from models import RegLinealReg
from models import readFile
from models import curvasAprendizaje
from models import showData as sd

x, y, xVal, yVal = readFile.image()

def parte1():
        M = len(x)
        X = np.hstack((np.ones(shape=(M, 1)), x))
        theta = np.ones(2)

        cost, grad = RegLinealReg.LinearGradienteCost(theta, X, y)
        print("Cost: {}".format(cost))
        print("Grad : {}".format(grad))

        fmin = RegLinealReg.minGradient(theta, X, y)
        print("{}".format(fmin))
        sd.part1(x, y, fmin['x'])


def parte2():
    M = len(x)
    MVal = len(xVal)

    theta = np.ones(2)

    X = np.hstack((np.ones((M, 1)), x))
    XVal = np.hstack((np.ones((MVal, 1)), xVal))
    errTrain, errVal = curvasAprendizaje.learningCurve(X, y, theta, XVal, yVal)

    sd.parte2(errTrain, errVal, M)


parte2()

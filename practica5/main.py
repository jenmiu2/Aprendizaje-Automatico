import matplotlib.pyplot as plt
import numpy as np
from models import RegLinealReg
from models import readFile
from models import curvasAprendizaje
from models import showData as sd

x, y, xVal, yVal = readFile.image()

def parte1():
        M = len(x)
        X = np.hstack((np.ones((M, 1)), x))
        theta = np.ones(2)

        cost, grad = RegLinealReg.LinearGradienteCost(theta, X, y.ravel())
        print("Cost: {}".format(cost))
        print("Grad : {}".format(grad))

        fmin = RegLinealReg.minGradient(X, y.ravel(), theta)
        print("{}".format(fmin))
        sd.part1(x, y, fmin['x'])


def parte2():
    X = np.hstack((np.ones(shape=(x.shape[0], 1)), x))
    XVal = np.hstack((np.ones(shape=(xVal.shape[0], 1)), xVal))
    errTrain, errVal = curvasAprendizaje.learningCurve(X, y, XVal, yVal)
    print(errTrain)
    print(errVal)


parte1()

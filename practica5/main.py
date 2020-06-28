import numpy as np
from models import RegLinealReg
from models import readFile
from models import curvasAprendizaje
from models import showData as sd
from models import RegPolinomial

x, y, xVal, yVal = readFile.image()
xTest, yTest = readFile.image_Test()
M = len(x)
MVal = len(xVal)
p = 8
theta = np.ones(2)
thetaPoly = np.ones(p)

X = np.hstack((np.ones((M, 1)), x))
XVal = np.hstack((np.ones((MVal, 1)), xVal))

def parte1():
    cost, grad = RegLinealReg.LinearGradienteCost(theta, X, y)
    print("Cost: {}".format(cost))
    print("Grad : {}".format(grad))

    fmin = RegLinealReg.minGradient(theta, X, y)
    print("{}".format(fmin))
    sd.part1(x, y, fmin['x'])


def parte2():
    errTrain, errVal = curvasAprendizaje.learningCurve(X, y, theta, XVal, yVal)
    sd.parte2(errTrain, errVal, M, fig=2)


def parte3():
    # Normalize x
    matrixPoly = RegPolinomial.matrixH(matrix=x, p=8)
    matrixPolyNormalize, mu, sigma = RegPolinomial.matrixNormalize(matrixPoly)

    # Normalize xVal
    matrixPolyVal = RegPolinomial.toNormalize(matrix=xVal, mu=mu, sigma=sigma)

    # Normalize xTest
    matrixPolyTest = RegPolinomial.toNormalize(matrix=xTest, mu=mu, sigma=sigma)

    # Minimize x poly
    fmin = RegLinealReg.minGradient(thetaPoly, matrixPolyNormalize, y, lam=0, maxiter=1000)
    print(fmin)
    sd.parte3_1(theta=fmin['x'], p=8, mu=mu, sigma=sigma, x=x, y=y)

    # Appy learning curve
    errTrain, errVal = curvasAprendizaje.learningCurve(matrixPolyNormalize, y, thetaPoly, matrixPolyVal, yVal, lam=0)

    sd.parte2(errTrain, errVal, M, fig=3)

def parte4():
    # Normalize x
    matrixPoly = RegPolinomial.matrixH(matrix=x, p=8)
    matrixPolyNormalize, mu, sigma = RegPolinomial.matrixNormalize(matrixPoly)

    # Normalize xVal
    matrixPolyVal = RegPolinomial.toNormalize(matrix=xVal, mu=mu, sigma=sigma)

    # Normalize xTest
    matrixPolyTest = RegPolinomial.toNormalize(matrix=xTest, mu=mu, sigma=sigma)

    # Lambda values
    lams = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
    m = len(lams)
    errTrain, errVal = np.zeros(m), np.zeros(m)

    for i in range(m):
        lam = lams[i]
        fmin = RegLinealReg.minGradient(thetaPoly, matrixPolyNormalize, y, lam)
        errTrain[i] = RegLinealReg.error(fmin['x'], matrixPolyNormalize, y)
        errVal[i] = RegLinealReg.error(fmin['x'], matrixPolyVal, yVal)

    sd.parte3_2(errTrain, errVal, lams)

    lam = 3
    fmin = RegLinealReg.minGradient(thetaPoly, matrixPolyNormalize, y, lam)
    error_val = RegLinealReg.error(fmin['x'], matrixPolyVal, yVal)
    print(error_val)
    error_test = RegLinealReg.error(fmin['x'], matrixPolyTest, yTest)
    print(error_test)

parte3()
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
thetaPoly = np.ones(p + 1)
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
    matrixPolyNormalize = np.concatenate([np.ones((M, 1)), matrixPolyNormalize], axis=1)

    # Normalize xVal
    matrixPolyVal = RegPolinomial.toNormalize(matrix=xVal, mu=mu, sigma=sigma)
    matrixPolyVal = np.concatenate([np.ones((MVal, 1)), matrixPolyVal], axis=1)

    # Minimize x poly
    fmin = RegLinealReg.minGradient(thetaPoly, matrixPolyNormalize, y, lam=0, maxiter=1000)
    print(fmin)
    sd.parte3_1(theta=fmin['x'], p=8, mu=mu, sigma=sigma, x=x, y=y)

    # Appy learning curve
    errTrain, errVal = curvasAprendizaje.learningCurve(matrixPolyNormalize, y, thetaPoly, matrixPolyVal, yVal, lam=0)
    sd.parte2(errTrain, errVal, M, fig=3)

def parte4():
    # Normalize x
    matrix = RegPolinomial.matrixH(matrix=x, p=8)
    matrixPolyNormalize, mu, sigma = RegPolinomial.matrixNormalize(matrix)
    matrixPoly = np.concatenate([np.ones((M, 1)), matrixPolyNormalize], axis=1)

    # Normalize xVal
    matrixPolyValNormalize = RegPolinomial.toNormalize(matrix=xVal, mu=mu, sigma=sigma)
    matrixPolyVal = np.concatenate([np.ones((MVal, 1)), matrixPolyValNormalize], axis=1)

    # Normalize xVal
    matrixPolyTestNormalize = RegPolinomial.toNormalize(matrix=xTest, mu=mu, sigma=sigma)
    matrixPolyTest = np.concatenate([np.ones((MVal, 1)), matrixPolyTestNormalize], axis=1)


    # Lambda values
    lams = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
    m = len(lams)
    errTrain, errVal = np.zeros(m), np.zeros(m)

    for i, lam in enumerate(lams):
        fmin = RegLinealReg.minGradient(thetaPoly, matrixPoly, y, lam=lam)
        errTrain[i] = RegLinealReg.error(fmin['x'], matrixPoly, y)
        errVal[i] = RegLinealReg.error(fmin['x'], matrixPolyVal, yVal)

    sd.parte3_2(errTrain, errVal, lams)

    lam = 3
    fmin = RegLinealReg.minGradient(thetaPoly, matrixPoly, y, lam)
    errVal = RegLinealReg.error(fmin['x'], matrixPolyVal, yVal)
    errTest = RegLinealReg.error(fmin['x'], matrixPolyTest, yTest)
    print("Error Validation: {}\n\rError Testing: {}\n\r".format(errVal, errTest))

parte3()
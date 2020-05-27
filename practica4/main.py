from models import readFile as rf
from models import GradientFunction as gf
import numpy as np


def practica4():
    x, y = rf.read_img()
    theta1, theta2 = rf.read_weight()

    # Initialize params
    inputSize = x.shape[1]
    hiddenSize = theta1.shape[0]
    numLabel = 10

    initTheta1 = gf.randomWeight(inputSize, hiddenSize)
    initTheta2 = gf.randomWeight(hiddenSize, numLabel)
    params_ns = np.concatenate((initTheta1.ravel(), initTheta2.ravel()))

    # Cost Calculation without regression
    cost, costReg, grad1, grad2, grad1Reg, grad2Reg = gf.gradient(params_ns, inputSize, hiddenSize, numLabel, x, y)

    print("Cost without Regression: {}".format(cost))
    print("Cost with Regression: {}".format(costReg))

    print("Grad 1 without Regression: {}".format(grad1))
    print("Grad 1 with Regression: {}".format(grad1Reg))

    print("Grad 2 without Regression: {}".format(grad2))
    print("Grad 2 with Regression: {}".format(grad2Reg))

practica4()

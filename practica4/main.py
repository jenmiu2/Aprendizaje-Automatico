from models import readFile as rf
from models import GradientFunction as gf
from models import checkNNGradients
import numpy as np


def practica4():
    x, y = rf.read_img()
    theta1, theta2 = rf.read_weight()



    # BackPropCalculation
    '''
    cost, costReg, grad1, grad2, grad1Reg, grad2Reg = gf.gradient(params_ns, inputSize, hiddenSize, numLabel, x, y)

    print("Cost without Regression: {}".format(cost))
    print("Cost with Regression: {}".format(costReg))

    print("Grad 1 without Regression: {}".format(grad1))
    print("Grad 1 with Regression: {}".format(grad1Reg))

    print("Grad 2 without Regression: {}".format(grad2))
    print("Grad 2 with Regression: {}".format(grad2Reg))

    # Check NN Gradient
    grad = np.concatenate((grad1.ravel(), grad2.ravel()))
    tol = gf.checkNNGradient(cost, grad, params_ns)

    print("The result of the checking in grad : {}".format(tol[0]))
    '''

    prediction = gf.backpropagationLearning(x, y, theta1, theta2)

    prediction = predict(theta1, Theta2, X)
    print("Training Set Accuracy:", sum(pred3[:, np.newaxis] == y)[0] / 5000 * 100, "%")
practica4()

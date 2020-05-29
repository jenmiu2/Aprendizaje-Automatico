from models import readFile as rf
from models import GradientFunction as gf
from models import checkNNGradients as check
import numpy as np


def practica4():
    x, y = rf.read_img()
    theta1, theta2 = rf.read_weight()



    # BackPropCalculation

    inputSize = x.shape[1]
    hiddenSize = theta1.shape[0]
    numLabel = 10
    params_ns = np.concatenate((theta1.ravel(), theta2.ravel()))
    cost, grad = gf.gradient(params_ns, inputSize,hiddenSize, numLabel,x, y)

    print("Cost: {}".format(cost))
    print("Grad: {}".format(grad))


    '''
    # Check NN Gradient

    tol = check.checkNNGradients(gf.gradient, 1)

    print("The result of the checking in grad : {}".format(tol[0]))
    '''



    prediction = gf.backpropagationLearning(x, y, theta1, theta2)
    jac = prediction.x
    print("Result Prediction: {}".format(jac))
    theta1 = jac[:((inputSize + 1) * hiddenSize)].reshape(hiddenSize, inputSize + 1)
    theta2 = jac[((inputSize + 1) * hiddenSize):].reshape(numLabel, hiddenSize + 1)
    a1, z2, aux2, a3, h = gf.forwardPropagation(x, theta1, theta2)
    kmax = np.argmax(h, 1) + 1
    accuracy = sum(kmax[:,np.newaxis] == y) / kmax.shape[0] * 100
    print("Accuracy Backward Propagation: {}%".format(accuracy))

practica4()

from models import readFile
from models import showData
from models import kernelLineal
from models import kernelGaussiano
from models import spamDetector



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
    x, xVal, y, yVal = readFile.data3()
    showData.initial(x, y)

    #Find best parameter
    c, s = kernelGaussiano.findParam(x, y, xVal, yVal)

    svm = kernelGaussiano.initial(x, y, c=c, sigma=s)
    showData.visualize_boundary(x, y, svm, 2)


def parte4():
    xTrain, xVal, xTest, yTrain, yVal, yTest = spamDetector.createTrainTestData()

    # Find best parameter
    c, maxScore = spamDetector.findParam(xTrain, yTrain.ravel(), xVal, yVal.ravel())
    svm = kernelLineal.initial(xTrain, yTrain.ravel(), c=c)

    print("Value of C: {}".format(c))
    print("Training Accuracy: {}".format(svm.score(xTrain, yTrain.ravel()) * 100))
    print("Vaidation Accuracy: {}".format(maxScore * 100))
    print("Test Accuracy: {}".format(svm.score(xTest, yTest.ravel()) * 100))

parte4()
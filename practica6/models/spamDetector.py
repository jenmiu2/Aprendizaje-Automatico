import numpy as np
from models import readFile
from models import process_email
from models import kernelLineal
from sklearn.model_selection import train_test_split

vocab = readFile.data4_1()
mSpam = 500
mEasyHam = 2551
mHardHam = 250
n = 1899


def createEasyHam():
    features = np.empty((mEasyHam, n + 1))
    for i in range(1, mEasyHam + 1):
        # Leo el spam
        easyHam = readFile.data4_3(i)
        # process message
        token = process_email.email2TokenList(easyHam)
        # compruebo que las palabras de ese easy ham estan en mi lista de vocab
        f = emailFeautures(token)
        features[i - 1, :] = f
    labelEasyHam = np.zeros((mEasyHam, 1))
    return features, labelEasyHam


def createHardHam():
    features = np.empty((mHardHam, n + 1))
    for i in range(1, mHardHam + 1):
        # Leo el spam
        hardHam = readFile.data4_4(i)
        # process message
        token = process_email.email2TokenList(hardHam)
        # compruebo que las palabras de ese hard ha, estan en mi lista de vocab
        f = emailFeautures(token)
        features[i - 1, :] = f
    labelHardHam = np.zeros((mHardHam, 1))
    return features, labelHardHam


def createSpam():
    features = np.empty((mSpam, n + 1))
    for i in range(1, mSpam + 1):
        # Leo el spam
        spam = readFile.data4_2(i)
        # process message
        token = process_email.email2TokenList(spam)
        # compruebo que las palabras de ese spam estan en mi lista de vocab
        f = emailFeautures(token)
        features[i - 1, :] = f
    labelSpam = np.ones((mSpam, 1))
    return features, labelSpam


def createTrainTestData():
    featuresSpam, labelSpam = createSpam()
    featuresEasyHam, labelEasyHam = createEasyHam()
    featuresHardHam, labelHardHam = createHardHam()

    X = featuresSpam
    X = np.vstack((X, featuresEasyHam))
    X = np.vstack((X, featuresHardHam))

    Y = labelSpam
    Y = np.vstack((Y, labelEasyHam))
    Y = np.vstack((Y, labelHardHam))


    # It creates a split for test
    x, xTest, y, yTest = train_test_split(X, Y, test_size=0.30, random_state=42)

    # It creates a split for training and validation
    xTrain, xVal, yTrain, yVal = train_test_split(x, y, test_size=0.50, random_state=42)

    return xTrain, xVal, xTest, yTrain, yVal, yTest


def emailInd(token):
    ind = [vocab[i] for i in token if i in vocab]
    return ind


def emailFeautures(token):
    features = np.zeros(n + 1)
    ind = emailInd(token)
    for i in ind:
        features[i] = 1
    return features


def findParam(x, y, xVal, yVal):
    # create c´s
    Cs = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    length = len(Cs)

    # create scores
    scores = np.zeros(shape=length)

    for i, c in enumerate(Cs):
        svm = kernelLineal.initial(x, y, c=c)
        scores[i] = svm.score(xVal, yVal)

    maxScore = np.amax(scores)
    pos = np.argwhere(scores == maxScore)[-1]
    c = Cs[pos[0]]

    return c, maxScore

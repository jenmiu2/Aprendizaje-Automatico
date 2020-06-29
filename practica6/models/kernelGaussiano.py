from sklearn.svm import SVC
import numpy as np

def initial(x, y, c, sigma):
    gamma = 1 / (2 * sigma**2)
    svm = SVC(kernel='rbf', C=c, gamma= gamma)
    svm.fit(x, y)
    return svm


def findParam(x, y, xVal, yVal):
    # create c´s and sigma´s
    Cs = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    sigmas = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    length = len(sigmas)

    # create scores
    scores = np.zeros(shape=(length, length))

    for i, c in enumerate(Cs):
        for j, sigma in enumerate(sigmas):
            svm = initial(x, y, c=c, sigma=sigma)
            scores[j, i] = svm.score(xVal, yVal)

    maxScore = np.amax(scores)
    pos = np.argwhere(scores == maxScore)[-1]
    c = Cs[pos[0]]
    s = sigmas[pos[1]]

    return c, s
import numpy as np


def matrixH(matrix, p):
    M = len(matrix)
    polyMatrix = np.zeros((M, p))
    for i in range(p):
        polyMatrix[:, i] = matrix[:, 0] ** (i + 1)
    return polyMatrix


def matrixNormalize(matrix):
    # Mean of  the matrix
    mu = np.mean(matrix, axis=0)
    matrixMean = matrix - mu

    # Standard deviations of the matrix
    sigma = np.std(matrixMean, axis=0, ddof=1)
    matrixNorm = matrixMean / sigma

    return matrixNorm, mu, sigma


def toNormalize(matrix, mu, sigma, p=8):
    polyMatrix = matrixH(matrix, p=p)
    muMatrix = (polyMatrix - mu)
    normMatrix = muMatrix / sigma
    return normMatrix

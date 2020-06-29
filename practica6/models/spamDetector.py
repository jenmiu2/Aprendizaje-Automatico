import numpy as np
from models import readFile
from models import kernelLineal


def detection(n):
    arr = np.zeros(n)
    vocab = readFile.data4_1()
    for i in range(1, 501):
        spam = readFile.data4_2(i)
        kernelLineal

import matplotlib.pyplot as plt
import numpy as np


def part1(x, y, theta):
    plt.figure()
    plt.plot(x, y, 'x', color='b')
    line = np.linspace(-50, 40, 2)
    plt.plot(line, theta[0] + theta[1] * line)
    plt.xlim(-50, 40)
    plt.ylim(-5, 40)
    plt.show()
    plt.savefig('data/fig1.png')


def parte2(errTrain, errVal, m):
    plt.plot(np.arange(1, m + 1), errTrain)
    plt.plot(np.arange(1, m + 1), errVal)
    plt.show()
    plt.savefig('data/fig2.png', dpi=300)


def parte3():
    print("")
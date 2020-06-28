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


def parte2(errTrain, errVal, m, fig):
    plt.plot(np.arange(1, m + 1), errTrain)
    plt.plot(np.arange(1, m + 1), errVal)
    plt.show()
    plt.savefig('data/fig2-{}.png'.format(fig), dpi=300)


def parte3_1(theta, p, mu, sigma, x, y):
    line2 = np.linspace(-110, 50, 100)

    line1 = theta[0] * np.ones(100)
    for i in range(1, p):
        line1 += theta[i] * (line2 ** i - mu[i - 1]) / sigma[i - 1]

    plt.figure()
    plt.plot(x, y, 'x', color='r')
    plt.plot(line2, line1, 'b--')
    plt.show()
    plt.savefig('data/fig3-1.png', dpi=300)


def parte3_2(errTrain, errVal, lam):
    plt.plot(lam, errTrain, '-o')
    plt.plot(lam, errVal, '-o', lw=2)
    plt.show()
    plt.savefig('data/fig3-2.png', dpi=300)

import numpy as np
import matplotlib.pyplot as plt


def initial(x, y):
    plt.figure(figsize=(4, 6))
    pos = (y == 1).ravel()
    neg = (y == 0).ravel()

    plt.figure()
    plt.scatter(x[pos, 0], x[pos, 1], color='black', marker='+', label='pos')
    plt.scatter(x[neg, 0], x[neg, 1], color='yellow', edgecolors='black', marker='o', label='neg')
    plt.legend(numpoints=1, loc=1)
    plt.savefig('data/fig1.png', dpi=300)
    plt.show()


def visualize_boundary(x, y, svm, fig):
    x1 = np.linspace(x[:, 0].min(), x[:, 0].max(), 100)
    x2 = np.linspace(x[:, 1].min(), x[:, 1].max(), 100)
    x1, x2 = np.meshgrid(x1, x2)
    yp = svm.predict(np.array(
        [x1.ravel(),
         x2.ravel()]).T).reshape(x1.shape)

    pos = (y == 1).ravel()
    neg = (y == 0).ravel()

    plt.figure()
    plt.scatter(x[pos, 0], x[pos, 1], color='black', marker='+')
    plt.scatter(x[neg, 0], x[neg, 1], color='yellow', edgecolors='black', marker='o')
    plt.contour(x1, x2, yp)
    plt.show()
    plt.savefig('data/fig2-{}'.format(fig))

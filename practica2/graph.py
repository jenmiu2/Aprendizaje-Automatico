from matplotlib import pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D


def show_sig_data(x, x_sigm):
    isAccepted = np.where(x == 1)
    notAccepted = np.where(x == 0)

    plt.plot(x_sigm[0], x_sigm[1], c='b', label='sigmoide')
    plt.scatter(x[isAccepted, 0], x[isAccepted, 1], marker="x", c='k', label='accepted')
    plt.scatter(x[notAccepted, 0], x[notAccepted, 1], marker="o", c='#e6ac00', label='not accepted')

    plt.xlabel('Examn 1 Score')  # Add an x-label to the axes.
    plt.ylabel('Examn 2 Score')  # Add a y-label to the axes.
    plt.legend()  # Add a legend.

    plt.figure(figsize=(8, 6), dpi=80)
    plt.show()

def show_graph(x):
    return 0
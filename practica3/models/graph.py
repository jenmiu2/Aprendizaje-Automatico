from matplotlib import pyplot as plt
import numpy as np


def paint_random_choice(x, y, random=10):
    sample = np.random.choice(x.shape[0], random)
    plt.imshow(x[sample, :].reshape(-1, 20).T)
    plt.axis('off')
    plt.show()

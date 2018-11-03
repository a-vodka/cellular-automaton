import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import random
from MircoCellularAutomaton import *

random.seed(0)
prob_matrix = np.array([[ 0.1, 0.1, 0.1], [0.1, 1, 0.1], [ 0.1, 0.1,  0.1]], dtype=float)

ca = MircoCellularAutomaton(200, 200, neighbour='moore', neighbour_matrix=prob_matrix)
ca.initial_cells(20)
ca.calculate(verbose=True)

sobel = scipy.ndimage.prewitt(ca.data)
# sobel[sobel != 0] = 1

plt.subplot(1, 2, 1)
plt.imshow(ca.data, interpolation='none')
plt.subplot(1, 2, 2)

plt.imshow(sobel, interpolation='none', cmap='gray')
plt.tight_layout()
plt.show()

np.savetxt('img20.txt',ca.data, delimiter=';')
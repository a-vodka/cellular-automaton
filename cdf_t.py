import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import scipy.stats
import skimage.measure
from skimage.io import imread
from scipy.integrate import cumtrapz
from skimage.filters import threshold_otsu, threshold_isodata
import skimage.morphology



from scipy.stats import norm

number = 100.

number += 0
param = 35
iter_number = np.linspace(0., 100., 101, endpoint=True)

print(iter_number)


print()

num = np.zeros_like(iter_number)
n_cells = norm.cdf(0, param, 15)+norm.cdf(100, param, 15)
for i in range(num.size):
    n_cells += number * (norm.cdf(i, param, 15) - norm.cdf(i - 1., param, 15))
    if n_cells > 1:
        num[i] = int(n_cells)
        n_cells -= int(n_cells)

print(np.sum(num), n_cells)

plt.plot(iter_number, num)
plt.plot(iter_number, np.cumsum(num)/number)
plt.show()
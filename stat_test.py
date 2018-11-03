import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import random


data = np.loadtxt('img4.txt', delimiter=';')
(w, h) = data.shape
print data
dist = []
for i in range(0, w, 3):
    xline = data[i, :]
    xline_diff = np.ediff1d(xline, to_begin=0)
    dist_line = np.ediff1d(np.nonzero(xline_diff))
    dist = np.append(dist, dist_line)


print np.mean(dist), np.std(dist)
plt.hist(dist)
plt.show()

dist1 = []
for i in range(0, h, 3):
    yline = data[:, i]
    yline_diff = np.ediff1d(yline, to_begin=0)
    dist_line = np.ediff1d(np.nonzero(yline_diff))
    dist1 = np.append(dist1, dist_line)

print np.mean(dist1), np.std(dist1)
plt.hist(dist1)
plt.show()


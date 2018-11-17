import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import random
import skimage.measure
from MircoCellularAutomaton import *

random.seed(0)
prob_matrix = np.array([[ 0.1, 0.1, 0.1], [0.1, 1, 0.1], [ 0.1, 0.1,  0.1]], dtype=float)

ca = MircoCellularAutomaton(100, 100, neighbour='moore', neighbour_matrix=prob_matrix, periodic=False)
ca.initial_cells(100)
#ca.initial_cell_mesh()
ca.calculate(verbose=True)

plt.subplot(1, 2, 1)
plt.imshow(ca.data, interpolation='none')
plt.subplot(1, 2, 2)

bwimage = ca.toBlackAndWhite()

label_img = skimage.measure.label(ca.data)
regions = skimage.measure.regionprops(label_img)

fig, ax = plt.subplots()
ax.imshow(ca.data, cmap=plt.cm.gray)

for props in regions:
    y0, x0 = props.centroid
    orientation = props.orientation
    x1 = x0 + np.cos(orientation) * 0.5 * props.major_axis_length
    y1 = y0 - np.sin(orientation) * 0.5 * props.major_axis_length
    x2 = x0 - np.sin(orientation) * 0.5 * props.minor_axis_length
    y2 = y0 - np.cos(orientation) * 0.5 * props.minor_axis_length

    ax.plot((x0, x1), (y0, y1), '-r', linewidth=0.5)
    ax.plot((x0, x2), (y0, y2), '-r', linewidth=0.5)
    ax.plot(x0, y0, '.g', markersize=3)

    minr, minc, maxr, maxc = props.bbox
    bx = (minc, maxc, maxc, minc, minc)
    by = (minr, minr, maxr, maxr, minr)
    ax.plot(bx, by, '-b', linewidth=2.5)

#ax.axis((0, 600, 600, 0))
plt.show()

print label_img, regions

plt.imshow(bwimage, interpolation='none', cmap='gray')
plt.tight_layout()
plt.show()

#np.savetxt('img20.txt',ca.data, delimiter=';')
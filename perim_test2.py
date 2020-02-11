import numpy as np
import skimage.measure
import skimage.morphology
import skimage.filters

from scipy import ndimage as ndi
from scipy import signal

import matplotlib.pyplot as plt

STREL_4 = np.array([[0, 1, 0],
                    [1, 1, 1],
                    [0, 1, 0]], dtype=np.uint8)
STREL_8 = np.ones((3, 3), dtype=np.uint8)


def picks_area(image, neighbourhood=4):
    if neighbourhood == 4:
        strel = STREL_4
    else:
        strel = STREL_8
    image = image.astype(np.uint8)
    eroded_image = ndi.binary_erosion(image, strel, border_value=0)
    border_image = image - eroded_image

    perimeter_weights = np.zeros(50, dtype=np.double)
    perimeter_weights[[5, 7, 15, 17, 25, 27]] = 0.25
    perimeter_weights[[21, 33]] = 1
    perimeter_weights[[13, 23]] = 0.125

    perimeter_image = ndi.convolve(border_image, np.array([[10, 2, 10],
                                                           [2, 1, 2],
                                                           [10, 2, 10]]),
                                   mode='constant', cval=0)

    # You can also write
    # return perimeter_weights[perimeter_image].sum()
    # but that was measured as taking much longer than bincount + np.dot (5x
    # as much time)
    perimeter_histogram = np.bincount(perimeter_image.ravel(), minlength=50)
    total_perimeter = np.dot(perimeter_histogram, perimeter_weights)

    v = np.count_nonzero(eroded_image)

    if v == 0:
        s = total_perimeter
    else:
        s = v + total_perimeter / 2. - 1.

    return s


def bwarea(bw):
    four = np.ones((2, 2))
    two = np.diag([1, 1])
    fours = signal.convolve2d(bw, four)
    twos = signal.convolve2d(bw, two)

    nQ1 = np.sum(fours == 1)
    nQ3 = np.sum(fours == 3)
    nQ4 = np.sum(fours == 4)
    nQD = np.sum(np.logical_and(fours == 2, twos != 1))
    nQ2 = np.sum(np.logical_and(fours == 2, twos == 1))

    total = 0.25 * nQ1 + 0.5 * nQ2 + 0.875 * nQ3 + nQ4 + 0.75 * nQD

    return total


image = np.array([[0, 0, 1, 0, 0, 0],
                  [0, 1, 1, 1, 0, 0],
                  [0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 1, 1],
                  [0, 0, 1, 0, 0, 1],
                  [0, 1, 1, 1, 0, 0],
                  [1, 1, 1, 1, 1, 0],
                  [0, 1, 1, 1, 0, 0],
                  [0, 0, 1, 0, 0, 0]])

label_img = skimage.measure.label(image)
regions = skimage.measure.regionprops(label_img)

exact_areas = {5: 2, 3: 0.5, 13: 8}

print 'area', 'picks_area', 'bw_area', 'exact_areas'
for props in regions:
    print props.area, picks_area(props.convex_image), bwarea(props.convex_image), exact_areas[props.area]



rs = np.logspace(start=0.42, stop=1.7, base=10)

ps = np.empty_like(rs)
ar_picks = np.empty_like(rs)
ar0 = np.empty_like(rs)
ar_bw = np.empty_like(rs)

for i in range(rs.size):
    r = rs[i]
    region = (np.arange(-r - 1, r + 2) ** 2 + np.arange(-r - 1, r + 2)[:, np.newaxis] ** 2 <= r ** 2).astype('int')
    rr = skimage.measure.regionprops(region)
    ps[i] = rr[0].perimeter
    ar_picks[i] = picks_area(rr[0].convex_image)
    ar0[i] = rr[0].area
    ar_bw[i] = bwarea(rr[0].convex_image)


plt.plot(rs, ar_picks, label='Picks')
plt.plot(rs, ar0, label='standard')
plt.plot(rs, ar_bw, label='bw_area')
plt.plot(rs, np.pi * rs**2, label='theoretical')


plt.ylabel('area')
plt.xlabel('radius')
plt.legend()
plt.grid()
plt.show()


plt.plot(rs, 4 * np.pi * ar_picks / ps ** 2, label='Picks area')
plt.plot(rs, 4 * np.pi * ar0 / ps ** 2, label='standard area')
plt.plot(rs, 4 * np.pi * ar_bw / ps ** 2, label='bw_area')
plt.ylabel('Cs')
plt.xlabel('radius')
plt.legend()
plt.grid()
plt.show()


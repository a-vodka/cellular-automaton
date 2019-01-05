import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb
from skimage.io import imread
import skimage.measure
from scipy.integrate import cumtrapz

image = imread('pure-iron.jpg', as_grey=True)
face = image
face1d = face.reshape(face.size)

# plt.imshow(face, cmap=plt.cm.gray)
# plt.show()

hist, bin_edges = np.histogram(face1d, bins=256, density=True)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

# plt.plot(bin_centers, hist)


# plt.show()

inthist = cumtrapz(hist, bin_centers, initial=0)

alpha = 0.05
lc = bin_centers[np.argmin(np.abs(inthist - alpha))]
rc = bin_centers[np.argmin(np.abs(inthist - (1.0 - alpha)))]
fval = (lc + rc) / 2.0

print fval
thresh = threshold_otsu(image)
# bw = closing(image > 220, square(3))
bw = image > fval

bw = np.zeros_like(image)
bw[image > fval] = 1.

#bw = bw[0:350, 0:550]

print thresh
# remove artifacts connected to image border
#cleared = clear_border(bw)

cleared = bw

label_img = skimage.measure.label(cleared)
regions = skimage.measure.regionprops(label_img)

fig, ax = plt.subplots()
ax.imshow(bw, interpolation='none')
plt.tight_layout()
plt.show()

(w, h) = image.shape
area = np.empty(len(regions))
perimeter = np.empty(len(regions))
orient = np.empty(len(regions))
scale_factor = np.empty(len(regions))

i = 0
for props in regions:
    y0, x0 = props.centroid
    area[i] = props.area
    perimeter[i] = props.perimeter
    orientation = props.orientation
    orient[i] = props.orientation
    if props.minor_axis_length:
        scale_factor[i] = props.major_axis_length / props.minor_axis_length
    else:
        scale_factor[i] = np.inf
    x1 = x0 + np.cos(orientation) * 0.5 * props.major_axis_length
    y1 = y0 - np.sin(orientation) * 0.5 * props.major_axis_length
    x2 = x0 - np.sin(orientation) * 0.5 * props.minor_axis_length
    y2 = y0 - np.cos(orientation) * 0.5 * props.minor_axis_length

    ax.plot((x0, x1), (y0, y1), '-r', linewidth=0.5)
    ax.plot((x0, x2), (y0, y2), '-r', linewidth=0.5)
    ax.plot(x0, y0, '.g', markersize=3)

    # minr, minc, maxr, maxc = props.bbox
    # bx = (minc, maxc, maxc, minc, minc)
    # by = (minr, minr, maxr, maxr, minr)
    # ax.plot(bx, by, '-b', linewidth=2.5)
    i += 1
# ax.axis((0, ca.w, ca.h, 0))
plt.tight_layout()
plt.show()

cond = np.logical_and(area > 10, scale_factor < np.inf)
cond = np.logical_and(cond, area < 10000)
area = area[cond]
perimeter = perimeter[cond]
orient = orient[cond]
scale_factor = scale_factor[cond]

norm_area = area / w / h
Cs = 4. * np.pi * area / (perimeter ** 2)
# print label_img, regions

fig, ax = plt.subplots()
ax.imshow(bw, interpolation='none', cmap='gray')

# ax.plot(ca.centers[1, :], ca.centers[0, :], '.g', markersize=3)
plt.tight_layout()
plt.show()

plt.hist(norm_area)
plt.show()

plt.hist(Cs)
plt.show()

plt.hist(orient)
plt.show()

plt.hist(scale_factor)
plt.show()
# np.savetxt('img20.txt',ca.data, delimiter=';')

print norm_area.size, Cs.size, orient.size, scale_factor.size

t = '\t'
print 'area[i]','norm_area[i]', t, 'Cs[i]', t, 'orient', t, 'scale_factor'
for i in range(norm_area.size):
    print area[i],t,norm_area[i], t, Cs[i], t, orient[i], t, scale_factor[i]

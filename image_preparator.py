import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import scipy.stats
import skimage.measure
from skimage.io import imread
from MircoCellularAutomaton import *
from scipy.integrate import cumtrapz
from skimage.filters import threshold_otsu, threshold_isodata
import skimage.morphology
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import os
from skimage import filters


def main():
    filename = "pure_H62copper.png"
    base = os.path.basename(filename)
    file_title = os.path.splitext(base)[0]
    print(filename, file_title)
    image = imread(filename, as_gray=True)

    print(threshold_isodata(image))
    face = image
    plt.imshow(face)
    plt.show()
    face1d = face.reshape(face.size)

    plt.hist(face1d,bins=256, density=True)
    plt.show()

    hist, bin_edges = np.histogram(face1d, bins=256, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    inthist = cumtrapz(hist, bin_centers, initial=0)
    alpha = 0.05
    lc = bin_centers[np.argmin(np.abs(inthist - alpha))]
    rc = bin_centers[np.argmin(np.abs(inthist - (1.0 - alpha)))]
    fval = (lc + rc) / 2.0

    thresh = threshold_otsu(image)
    print(fval, thresh)
    bw = np.zeros_like(image)
    #bw[image > 0.781] = 1.
    bw[image > fval] = 1.

    bw = skimage.morphology.opening(bw,  skimage.morphology.square(3))
    bw = skimage.morphology.opening(bw)
    data = bw
    bwimage = bw
    # plt.imshow(bw, interpolation='none', cmap='gray')
    # plt.tight_layout()
    # plt.show()
    #ax[0, 0].imshow(data, interpolation='none', cmap='gray')

    fig_m = plt.figure(figsize=(5.65, 5.13), dpi=300)
    ax_m = fig_m.add_subplot(1, 1, 1)

    ax_m.imshow(data, interpolation='none', cmap='gray')
    ax_m.set_yticklabels([])
    ax_m.set_xticklabels([])
    fig_m.tight_layout()
    fig_m.savefig(file_title+'_micro.png', dpi=300)
    fig_m.show()
    plt.show()

    np.save("./models2/"+file_title, data)



if __name__ == "__main__":
    main()

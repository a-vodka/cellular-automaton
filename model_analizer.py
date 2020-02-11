import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import scipy.stats
import skimage.measure
from skimage.io import imread
#from MircoCellularAutomaton import *
from scipy.integrate import cumtrapz
from skimage.filters import threshold_otsu, threshold_isodata
import skimage.morphology
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import os
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import gaussian_kde


def plot_hist(data, ax):
    # na_geparam = scipy.stats.genextreme.fit(data)
    na_lognparam = scipy.stats.lognorm.fit(data)
    data_mean, data_std = np.mean(data), np.std(data)

    ax.hist(data, density=True)
    na_x = np.linspace(np.min(data), np.max(data), 1000)

    # ax.plot(na_x, scipy.stats.genextreme.pdf(na_x, *na_geparam))
    ax.plot(na_x, scipy.stats.lognorm.pdf(na_x, *na_lognparam))
    s = r'$<E>={0:.3e}$'.format(data_mean)
    s += '\n'
    s += r'$\sqrt{\mathrm{var}[K_\sigma]}=' + '{0:.3e}$'.format(data_std)
    plt.text(0.5, 0.8, "{0} {1}".format(*na_lognparam), transform=ax.transAxes, horizontalalignment='center',
             verticalalignment='center', fontsize=6)
    plt.text(0.3, 0.88, s, transform=ax.transAxes, horizontalalignment='left', verticalalignment='center', fontsize=6)


def vdk_perimeter(image):
    (w, h) = image.shape
    image = image.astype(np.uint8)
    data = np.zeros((w + 2, h + 2), dtype=image.dtype)
    data[1:-1, 1:-1] = image
    dilat = skimage.morphology.binary_dilation(data)
    newdata = dilat - data

    kernel = np.array([[10, 2, 10],
                       [2, 1, 2],
                       [10, 2, 10]])

    T = skimage.filters.edges.convolve(newdata, kernel, mode='constant', cval=0)

    cat_a = np.array([5, 15, 7, 25, 27, 17])
    cat_b = np.array([21, 33])
    cat_c = np.array([13, 23])
    cat_a_num = np.count_nonzero(np.isin(T, cat_a))
    cat_b_num = np.count_nonzero(np.isin(T, cat_b))
    cat_c_num = np.count_nonzero(np.isin(T, cat_c))

    perim = cat_a_num + cat_b_num * np.sqrt(2.) + cat_c_num * (1. + np.sqrt(2.)) / 2.

    return perim


def plot_hist(data, ax, ax1, i, yticks):
    hist, bin_edges = np.histogram(data)
    edg = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    w = bin_edges[1] - bin_edges[0]
    # print hist, edg, bin_edges, w, yticks

    ax.bar(edg, hist, zs=i, zdir='y', alpha=0.8, width=w)
    kde = gaussian_kde(data)

    xdata = np.linspace(data.min(), data.max(), 300)
    kdeval = kde.evaluate(xdata)

    ax1.plot(xdata, kdeval, label=yticks[i])
    ax1.grid()

    pass


def main():
    path = './models/'

    files = os.listdir(path)
    numfiles = len(files)
    # norm_area = np.zeros((numfiles, 700))
    # cs = np.zeros((numfiles, 700))
    # scale_factor = np.zeros((numfiles, 700))
    # orient = np.zeros((numfiles, 700))
    yticks = np.empty(numfiles, dtype='U25')
    i = 0

    fig_na = plt.figure(dpi=300)
    fig_cs = plt.figure(dpi=300)
    fig_scf = plt.figure(dpi=300)
    fig_ori = plt.figure(dpi=300)

    ax_na = fig_na.add_subplot(111, projection='3d')
    ax_cs = fig_cs.add_subplot(111, projection='3d')
    ax_scf = fig_scf.add_subplot(111, projection='3d')
    ax_ori = fig_ori.add_subplot(111, projection='3d')

    fig1_na = plt.figure(dpi=300)
    fig1_cs = plt.figure(dpi=300)
    fig1_scf = plt.figure(dpi=300)
    fig1_ori = plt.figure(dpi=300)

    ax1_na = fig1_na.add_subplot(111)
    ax1_cs = fig1_cs.add_subplot(111)
    ax1_scf = fig1_scf.add_subplot(111)
    ax1_ori = fig1_ori.add_subplot(111)

    for name in files:

        if i == numfiles:
            break

        if name.endswith('.npy'):
            #print name
            yticks[i] = name.replace('.npy', '')
            _na, _cs, _scf, _ori = process(path + name)
            # norm_area[i, :_na.size] = _na
            # cs[i, :_cs.size] = _cs
            # scale_factor[i, :_scf.size] = _scf
            # orient[i, :_ori.size] = _ori

            plot_hist(_na, ax_na, ax1_na, i, yticks)
            plot_hist(_cs, ax_cs, ax1_cs, i, yticks)
            plot_hist(_scf, ax_scf, ax1_scf, i, yticks)
            plot_hist(_ori, ax_ori, ax1_ori, i, yticks)

            t = '\t'
            print(name, t, np.mean(_na), t, np.std(_na), t, np.mean(_cs), t, np.std(_cs), t, np.mean(_scf), t, np.std(
                _scf), t, np.mean(_ori), t, np.std(_ori))

            i += 1

    #    ax.set_xlabel('X')
    #    ax.set_ylabel('Y')
    #    ax.set_zlabel('Z')

    # On the y axis let's only label the discrete values that we have data for.
    ax_na.set_yticklabels(yticks)
    ax_cs.set_yticklabels(yticks)
    ax_scf.set_yticklabels(yticks)
    ax_ori.set_yticklabels(yticks)

#    fig_na.tight_layout()
    fig_na.savefig('3d_hist_na.png')
    fig_na.show()

#    fig_cs.tight_layout()
    fig_cs.savefig('3d_hist_cs.png')
    fig_cs.show()

#    fig_scf.tight_layout()
    fig_scf.savefig('3d_hist_scf.png')
    fig_scf.show()

#    fig_ori.tight_layout()
    fig_ori.savefig('3d_hist_ori.png')
    fig_ori.show()

#    fig1_na.tight_layout()
    fig1_na.legend()
    #    fig1_na.grid()
    fig1_na.savefig('2s_hist_kde_na.png')
    fig1_na.show()

#    fig1_cs.tight_layout()
    fig1_cs.legend()
    #    fig1_cs.grid()
    fig1_cs.savefig('2s_hist_kde_cs.png')
    fig1_cs.show()

#    fig1_scf.tight_layout()
    fig1_scf.legend()
    #    fig1_scf.grid()
    fig1_scf.savefig('2s_hist_kde_scf.png')
    fig1_scf.show()

#    fig1_ori.tight_layout()
    fig1_ori.legend()
    #    fig1_ori.grid()
    fig1_ori.savefig('2s_hist_kde_ori.png')
    fig1_ori.show()

    plt.show()

    pass


def process(filename):
    data = np.load(filename)

    label_img = skimage.measure.label(data, neighbors=4, background=0)
    regions = skimage.measure.regionprops(label_img, coordinates='xy')

    area = np.zeros(len(regions))
    perimeter = np.zeros(len(regions))
    orient = np.zeros(len(regions))
    scale_factor = np.zeros(len(regions))

    # fig, ax = plt.subplots(3, 2, figsize=(5.8, 8.3), dpi=300)
    (w, h) = data.shape
    bwimage = data

    i = 0
    for props in regions:
        if props.area < 10:
            continue
        y0, x0 = props.centroid
        area[i] = props.area
        perimeter[i] = np.max([props.perimeter, vdk_perimeter(props.convex_image)])
        # perimeter[i] = vdk_perimeter(props.convex_image)
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

        # ax[0, 1].plot((x0, x1), (y0, y1), '-r', linewidth=0.5)
        # ax[0, 1].plot((x0, x2), (y0, y2), '-r', linewidth=0.5)

        # ax[0, 1].plot(x0, y0, '.r', markersize=1)

        cs_i = 4. * np.pi * area[i] / (perimeter[i] ** 2)

        if cs_i > 1:
            plt.close()
            print(i, cs_i, props._slice, area[i], perimeter[i], vdk_perimeter(props.convex_image))
            plt.imshow(props.convex_image)
            plt.show()

        i += 1

#    print 'Image size =', data.shape
#    print 'Num of detected grains = ', i
#    print 'Grains per square pixel =', float(i) / w / h

    # ax[0, 1].axis((0, h, w, 0))
    # ax[0, 1].plot(ca.centers[1, :], ca.centers[0, :], '.g', markersize=1)
    # ax[0, 1].imshow(bwimage, interpolation='none', cmap='gray')
    # ax[0, 1].legend()

    cond = np.logical_and(area > 10, scale_factor < np.inf)
    area = area[cond]
    perimeter = perimeter[cond]
    orient = orient[cond]
    scale_factor = scale_factor[cond]

    norm_area = area / w / h
    cs = 4. * np.pi * area / (perimeter ** 2)

    # print label_img, regions

    return norm_area, cs, scale_factor, orient

    # plot_hist(norm_area, ax[1, 0])
    # plot_hist(cs, ax[1, 1])
    # plot_hist(scale_factor, ax[2, 1])

    # ax[2, 0].hist(orient)

    # plt.tight_layout()
    # plt.savefig('diag.png')
    # plt.show()

    # np.savetxt('img20.txt',ca.data, delimiter=';')

    # print norm_area.size, Cs.size, orient.size, scale_factor.size

    # t = '\t'
    # print 'norm_area[i]', t, 'Cs[i]', t, 'orient', t, 'scale_factor'
    # for i in range(norm_area.size):
    #    print norm_area[i], t, Cs[i], t, orient[i], t, scale_factor[i]


main()

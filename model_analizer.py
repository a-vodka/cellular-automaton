import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import scipy.stats
import skimage.measure
from skimage.io import imread
# from MircoCellularAutomaton import *
from scipy.integrate import cumtrapz
from skimage.filters import threshold_otsu, threshold_isodata
import skimage.morphology
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import os
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import gaussian_kde
from mpl_toolkits.axes_grid1 import make_axes_locatable


def multiple_formatter(denominator=4, number=np.pi, latex='\pi'):
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    def _multiple_formatter(x, pos):
        den = denominator
        num = np.int(np.rint(den * x / number))
        com = gcd(num, den)
        (num, den) = (int(num / com), int(den / com))
        if den == 1:
            if num == 0:
                return r'$0$'
            if num == 1:
                return r'$%s$' % latex
            elif num == -1:
                return r'$-%s$' % latex
            else:
                return r'$%s%s$' % (num, latex)
        else:
            if num == 1:
                return r'$\frac{%s}{%s}$' % (latex, den)
            elif num == -1:
                return r'$\frac{-%s}{%s}$' % (latex, den)
            else:
                return r'$\frac{%s%s}{%s}$' % (num, latex, den)

    return _multiple_formatter


class Multiple:
    def __init__(self, denominator=4, number=np.pi, latex='\pi'):
        self.denominator = denominator
        self.number = number
        self.latex = latex

    def locator(self):
        return plt.MultipleLocator(self.number / self.denominator)

    def formatter(self):
        return plt.FuncFormatter(multiple_formatter(self.denominator, self.number, self.latex))


def rnd_cmap(n):
    if n < 2:
        n = 2
    n = int(n)
    colors = np.empty([n, 4])
    colors[1] = [1., 1., 1., 1.]  # white background
    colors[0] = [0., 0., 0., 1.]  # white background
    for i in range(2, n):
        colors[i] = [np.random.random_sample(), np.random.random_sample(), np.random.random_sample(), 1]

    cm = ListedColormap(colors, name='my_list')
    return cm


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

    # ax.bar(edg, hist, zs=i, zdir='y', alpha=0.8, width=w)
    kde = gaussian_kde(data)

    xdata = np.linspace(data.min(), data.max(), 300)
    kdeval = kde.evaluate(xdata)

    ax1.plot(xdata, kdeval, label=yticks[i])
    ax1.grid()

    pass


def main():
    path = './models2/'

    files = os.listdir(path)
    numfiles = len(files)
    # norm_area = np.zeros((numfiles, 700))
    # cs = np.zeros((numfiles, 700))
    # scale_factor = np.zeros((numfiles, 700))
    # orient = np.zeros((numfiles, 700))
    yticks = np.empty(numfiles, dtype='U25')
    i = 0

    fig1_na = plt.figure(dpi=100)
    fig1_cs = plt.figure(dpi=100)
    fig1_scf = plt.figure(dpi=100)
    fig1_ori = plt.figure(dpi=100)

    ax1_na = fig1_na.add_subplot(111, title="Na")
    ax1_cs = fig1_cs.add_subplot(111, title="Cs")
    ax1_scf = fig1_scf.add_subplot(111, title="Scf")
    ax1_ori = fig1_ori.add_subplot(111, title="Orient")

    ax1_ori.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 4))
    ax1_ori.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
    ax1_ori.xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
    ax1_ori.set_xlim(left=0, right=np.pi)

    for name in files:

        if i == numfiles:
            break

        if name.endswith('.npy'):
            # print name
            yticks[i] = name.replace('.npy', '')
            _na, _cs, _scf, _ori = process(path + name, plot_hist=True, plot_orientation=True)
            # norm_area[i, :_na.size] = _na
            # cs[i, :_cs.size] = _cs
            # scale_factor[i, :_scf.size] = _scf
            # orient[i, :_ori.size] = _ori

            plot_hist(_na, None, ax1_na, i, yticks)
            plot_hist(_cs, None, ax1_cs, i, yticks)
            plot_hist(_scf, None, ax1_scf, i, yticks)
            plot_hist(_ori, None, ax1_ori, i, yticks)

            t = '\t'
            print(name, t, np.mean(_na), t, np.std(_na), t, np.mean(_cs), t, np.std(_cs), t, np.mean(_scf), t, np.std(
                _scf), t, np.mean(_ori), t, np.std(_ori))

            i += 1

    fig1_na.tight_layout()
    fig1_na.legend()

    #    fig1_na.grid()
    fig1_na.savefig('2s_hist_kde_na.png')
    fig1_na.show()

    fig1_cs.tight_layout()
    fig1_cs.legend()

    #    fig1_cs.grid()
    #    fig1_cs.savefig('2s_hist_kde_cs.png')
    fig1_cs.show()

    fig1_scf.tight_layout()
    fig1_scf.legend()

    #    fig1_scf.grid()
    #    fig1_scf.savefig('2s_hist_kde_scf.png')
    fig1_scf.show()

    #    fig1_ori.tight_layout()
    fig1_ori.legend()

    #    fig1_ori.grid()
    #    fig1_ori.savefig('2s_hist_kde_ori.png')
    fig1_ori.show()

    plt.show()

    pass


def process(filename, plot_orientation=False, plot_hist=False):
    data = np.load(filename)

    label_img = skimage.measure.label(data, background=0)
    regions = skimage.measure.regionprops(label_img)

    area = np.zeros(len(regions))
    perimeter = np.zeros(len(regions))
    orient = np.zeros(len(regions))
    scale_factor = np.zeros(len(regions))
    x = np.zeros(len(regions))
    y = np.zeros(len(regions))

    fig, ax = plt.subplots(1, 1)
    fig.tight_layout()
    ax.set_aspect(1.)

    (w, h) = data.shape
    bwimage = data

    i = 0
    for props in regions:
        if props.area < 10:
            # print("warning: in {} ares of cell {} is {}".format(filename, i, props.area))
            continue

        y0, x0 = props.centroid
        x[i], y[i] = x0, y0

        area[i] = props.area
        perimeter[i] = np.max([props.perimeter, vdk_perimeter(props.convex_image)])
        # perimeter[i] = vdk_perimeter(props.convex_image)

        orient[i] = props.orientation + np.pi / 2
        if props.minor_axis_length:
            scale_factor[i] = props.major_axis_length / props.minor_axis_length
        else:
            scale_factor[i] = np.inf
            print("warning: scale factor for {} cell if infinity".format(i))

        if plot_orientation:
            x1 = x0 + np.cos(orient[i]) * 0.5 * props.major_axis_length
            y1 = y0 - np.sin(orient[i]) * 0.5 * props.major_axis_length
            x2 = x0 - np.sin(orient[i]) * 0.5 * props.minor_axis_length
            y2 = y0 - np.cos(orient[i]) * 0.5 * props.minor_axis_length

            ax.plot((x0, x1), (y0, y1), '-r', linewidth=0.5)
            ax.plot((x0, x2), (y0, y2), '-g', linewidth=0.5)
            ax.plot(x0, y0, '.r', markersize=1)

        cs_i = 4. * np.pi * area[i] / (perimeter[i] ** 2)

        if cs_i > 1:
            print("warning: Cs for {} cell > 1".format(i))
            plt.close()
            print(i, cs_i, props._slice, area[i], perimeter[i], vdk_perimeter(props.convex_image))
            plt.imshow(props.convex_image)
            plt.show()

        i += 1

    ax.axis((0, h, w, 0))
    # ax[0, 1].plot(ca.centers[1, :], ca.centers[0, :], '.g', markersize=1)
    ax.imshow(bwimage, interpolation='none', cmap=rnd_cmap(np.max(bwimage)), origin="lower")

    # ax.legend()
    if plot_hist:
        # create new axes on the right and on the top of the current axes
        # The first argument of the new_vertical(new_horizontal) method is
        # the height (width) of the axes to be created in inches.
        divider = make_axes_locatable(ax)
        axHistx = divider.append_axes("top", 1.2, pad=0.1, sharex=ax)
        axHisty = divider.append_axes("right", 1.2, pad=0.1, sharey=ax)

        # make some labels invisible
        axHistx.xaxis.set_tick_params(labelbottom=False)
        axHisty.yaxis.set_tick_params(labelleft=False)

        bins = int(np.sqrt(i))

        axHistx.hist(x, edgecolor="black", bins=bins)
        axHisty.hist(y, orientation='horizontal', edgecolor="black", bins=bins)

    cond = np.logical_and(area > 10, scale_factor < np.inf)
    area = area[cond]
    perimeter = perimeter[cond]
    orient = orient[cond]
    scale_factor = scale_factor[cond]

    print('Image size =', data.shape)
    print('Number of pixels =', w * h)
    print('Num of detected grains = ', area.size)
    print('Grains per square pixel =', float(i) / w / h)

    norm_area = area / w / h
    cs = 4. * np.pi * area / (perimeter ** 2)

    return norm_area, cs, scale_factor, orient


if __name__ == "__main__":
    main()

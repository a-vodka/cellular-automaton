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


# prob_matrix = np.array([[0.1, 0.1, 0.1], [0.1, 1, 0.1], [0.1, 0.1, 0.1]], dtype=float)
# prob_matrix = np.array([[np.sqrt(2) / 2., 1, np.sqrt(2) / 2.], [1, 1, 1], [np.sqrt(2) / 2., 1, np.sqrt(2) / 2.]],
#                       dtype=float)


def get_pobability_matrix(rx, ry, angle=0, w=999, h=999, verbose=False):
    import skimage.draw
    import skimage.transform

    image = np.zeros((w, h))

    _rx = (rx - 0.5) * w / 3.
    _ry = (ry - 0.5) * h / 3.

    rr, cc = skimage.draw.ellipse(w / 2, h / 2, _rx * 0.999, _ry * 0.999)

    # rr, cc = skimage.draw.polygon( [0, w, w/2, 0], [0, 0, h, 0] ) # triangle

    image[rr, cc] = 1

    image = skimage.transform.rotate(image, angle=angle, order=1)

    prob_matrix = np.zeros((3, 3), dtype=float)

    ws = w / 3.0
    hs = h / 3.0

    for i in range(3):
        for j in range(3):
            w_low, w_high = int(ws * i), int(ws * (i + 1))
            h_low, h_high = int(hs * j), int(hs * (j + 1))
            prob_matrix[i, j] = np.count_nonzero(image[w_low:w_high, h_low:h_high]) / (ws * hs)

    if verbose:
        print prob_matrix
        plt.imshow(image)

        bx = (0, w, w, 0, w)
        by = (hs, hs, 2 * hs, 2 * hs, 2 * hs)
        plt.plot(bx, by, '-b', linewidth=2.0)

        bx = (ws, ws, 2 * ws, 2 * ws, 2 * ws)
        by = (0, h, h, 0, h)
        plt.plot(bx, by, '-b', linewidth=2.0)
        plt.axis((0, w, h, 0))
        plt.show()

    return prob_matrix


def plot_hist(data, ax):
    #na_geparam = scipy.stats.genextreme.fit(data)
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


def main():
    np.random.seed()

    fig, ax = plt.subplots(3, 2, figsize=(5.8, 8.3), dpi=300)
    plt.close()
    if True:
        # prob_matrix = get_pobability_matrix(2.0, 2.0, angle=0, verbose=False)

        prob_matrix = np.array([[1, 1, 0], [1, 1, 1], [0, 1, 1]], dtype=float)

        #ca = MircoCellularAutomaton(513, 565, neighbour='custom', neighbour_matrix=prob_matrix, periodic=False)

        ca = MircoCellularAutomaton(100, 100, neighbour='moore', periodic=False)

        #ca.initial_cells(10)
        ca.initial_cell_mesh()

#        ca.calculate(verbose=True)

        while True:
            is_converged = ca.calculate(verbose=True, max_iter=1)
            plt.imshow(ca.data, interpolation='none')
            plt.tight_layout()
            plt.show()
            if is_converged:
                break

        exit()
        bwimage = ca.to_black_and_white()
        ax[0, 0].imshow(data, interpolation='none')

    else:

        image = imread('fig-2.gif', as_grey=True)
        from skimage import filters
        print threshold_isodata(image)
        face = image
        #        plt.imshow(face)
        #        plt.show()
        face1d = face.reshape(face.size)

        hist, bin_edges = np.histogram(face1d, bins=256, density=True)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        inthist = cumtrapz(hist, bin_centers, initial=0)
        alpha = 0.05
        lc = bin_centers[np.argmin(np.abs(inthist - alpha))]
        rc = bin_centers[np.argmin(np.abs(inthist - (1.0 - alpha)))]
        fval = (lc + rc) / 2.0

        thresh = threshold_otsu(image)
        print fval, thresh
        bw = np.zeros_like(image)
        bw[image > 0.781] = 1.
        data = bw
        # bw = skimage.morphology.opening(bw,  skimage.morphology.square(3))
        # bw = skimage.morphology.opening(bw)
        bwimage = bw
        # plt.imshow(bw, interpolation='none', cmap='gray')
        # plt.tight_layout()
        # plt.show()
        ax[0, 0].imshow(data, interpolation='none', cmap='gray')

    (w, h) = data.shape
    ax[0, 0].axis((0, h, w, 0))

    #    plt.imshow(data, interpolation='none')
    #    plt.tight_layout()
    #    plt.show()

    label_img = skimage.measure.label(data, neighbors=4, background=0)
    regions = skimage.measure.regionprops(label_img, coordinates='xy')

    area = np.zeros(len(regions))
    perimeter = np.zeros(len(regions))
    orient = np.zeros(len(regions))
    scale_factor = np.zeros(len(regions))

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

        ax[0, 1].plot((x0, x1), (y0, y1), '-r', linewidth=0.5)
        ax[0, 1].plot((x0, x2), (y0, y2), '-r', linewidth=0.5)

        ax[0, 1].plot(x0, y0, '.r', markersize=1)

        cs_i = 4. * np.pi * area[i] / (perimeter[i] ** 2)
        print i
        if cs_i > 1:
            plt.close()
            print i, cs_i, props._slice, area[i], perimeter[i], vdk_perimeter(props.convex_image)
            plt.imshow(props.convex_image)
            plt.show()

        i += 1

    print 'Image size =', data.shape
    print 'Num of detected grains = ', i
    print 'Grains per square pixel =', float(i) / w / h

    ax[0, 1].axis((0, h, w, 0))
    # ax[0, 1].plot(ca.centers[1, :], ca.centers[0, :], '.g', markersize=1)
    ax[0, 1].imshow(bwimage, interpolation='none', cmap='gray')
    ax[0, 1].legend()

    cond = np.logical_and(area > 10, scale_factor < np.inf)
    area = area[cond]
    perimeter = perimeter[cond]
    orient = orient[cond]
    scale_factor = scale_factor[cond]

    norm_area = area / w / h
    cs = 4. * np.pi * area / (perimeter ** 2)

    # print label_img, regions

    plot_hist(norm_area, ax[1, 0])
    plot_hist(cs, ax[1, 1])
    plot_hist(scale_factor, ax[2, 1])

    ax[2, 0].hist(orient)

    plt.tight_layout()
    plt.savefig('diag.png')
    plt.show()

    # np.savetxt('img20.txt',ca.data, delimiter=';')

    # print norm_area.size, Cs.size, orient.size, scale_factor.size

    # t = '\t'
    # print 'norm_area[i]', t, 'Cs[i]', t, 'orient', t, 'scale_factor'
    # for i in range(norm_area.size):
    #    print norm_area[i], t, Cs[i], t, orient[i], t, scale_factor[i]


if __name__ == '__main__':
    main()

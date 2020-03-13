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

    image = skimage.transform.rotate(image, angle=angle - 90, order=1)

    prob_matrix = np.zeros((3, 3), dtype=float)

    ws = w / 3.0
    hs = h / 3.0

    for i in range(3):
        for j in range(3):
            w_low, w_high = int(ws * i), int(ws * (i + 1))
            h_low, h_high = int(hs * j), int(hs * (j + 1))
            prob_matrix[i, j] = np.count_nonzero(image[w_low:w_high, h_low:h_high]) / (ws * hs)

    if verbose:
        print(prob_matrix)

        colors = [(1, 1, 1), (0.7, 0.7, 0.7)]
        cm = LinearSegmentedColormap.from_list(
            'my_list', colors, N=2)

        fig = plt.figure(figsize=(3, 3))
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(image, cmap=cm)

        bx = (0, w, w, 0, w)
        by = (hs, hs, 2 * hs, 2 * hs, 2 * hs)
        ax.plot(bx, by, '-k', linewidth=1.0)

        bx = (ws, ws, 2 * ws, 2 * ws, 2 * ws)
        by = (0, h, h, 0, h)
        ax.plot(bx, by, '-k', linewidth=1.0)
        ax.axis((0, w, h, 0))
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        # prob_matrix = np.array([[0, 1, 1], [1, 1, 1], [1, 1, 0]], dtype=float)
        for i in range(3):
            for j in range(3):
                ax.text((1. + 2 * i) / 6., (1. + 2 * j) / 6., "{0:.3f}".format(prob_matrix[2 - j, i]),
                        horizontalalignment='center', verticalalignment='center', fontsize='xx-large',
                        transform=ax.transAxes)

        fig.tight_layout()
        #fig.savefig("./neibours/prob_matirx5.png", dpi=300)
        #ig.savefig("./neibours/prob_matirx5.eps", dpi=300)
        fig.show()
        plt.show()
        plt.close()

    return prob_matrix


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


def main():
    np.random.seed()

    cm = []

    # prob_matrix = get_pobability_matrix(2.0, 2.0 * 2.0 / 3.0, angle=45, verbose=False)
    prob_matrix = get_pobability_matrix(1.7, 2.0, angle=0, verbose=True)


    left_prob_matrix = np.array([[1, 1, 0], [1, 1, 1], [0, 1, 1]], dtype=float)
    right_prob_matrix = np.array([[0, 1, 1], [1, 1, 1], [1, 1, 0]], dtype=float)
    Num_grains = 685

    # call = lambda n: Num_grains * 0.2 * scipy.stats.norm.cdf(n, 3, 1) + Num_grains *0.8 * scipy.stats.norm.cdf(n, 25, 5)
    # call = lambda n: Num_grains * scipy.stats.norm.cdf(n, 20, 2)

    def ngr(n):
        # res = np.array(np.exp(0.05 * (n - 1)) - 1)
        #res = np.array(0.008 * n ** 3)

        res = np.array( 10 * n )

        # res[res > Num_grains] = Num_grains
        return res

    n = np.arange(0, 100, 1)
    plt.plot(n, ngr(n))
    plt.xlim([np.min(n), np.max(n)])
    plt.ylim([0, None])
    plt.xlabel("Iteration number, n")
    plt.ylabel("Number of grains, N")
    plt.grid()
    plt.show()

    ca = MircoCellularAutomaton(513, 565, neighbour='custom', neighbour_matrix=prob_matrix, periodic=False,
                                animation=False, centers_func=ngr)
    # ca = MircoCellularAutomaton(513, 565, neighbour='moore', periodic=False, animation=True)
    # ca = MircoCellularAutomaton(513, 565, neighbour='von_neumann', periodic=False, animation=True)

    # ca = MircoCellularAutomaton(513, 565, neighbour='custom', neighbour_matrix=left_prob_matrix, periodic=False, animation=True)
    # ca = MircoCellularAutomaton(513, 565, neighbour='custom', neighbour_matrix=right_prob_matrix, periodic=False, animation=True)

    # ca.initial_cells(Num_grains)
    # ca.initial_cell_mesh()

    colors = np.empty((Num_grains, 4))
    colors[0] = [1., 1., 1., 1.]  # white background
    for i in range(1, Num_grains):
        colors[i] = [np.random.random_sample(), np.random.random_sample(), np.random.random_sample(), 1]

    cm = ListedColormap(colors, name='my_list')

    #
    while False:
        is_converged = ca.calculate(verbose=True, max_iter=5)
        if is_converged:
            break
        fig1 = plt.figure(figsize=(3, 3))
        ax1 = fig1.add_subplot(111)
        ax1.imshow(ca.data, interpolation='none', cmap=cm)
        fig1.tight_layout()
        fig1.show()
    else:
        ca.calculate(verbose=True)

    data = ca.data

    # ca.save_animation_mpeg('elipse_prob.avi')
    # ca.save_animation_gif('elipse_prob.gif')

    # bwimage = ca.to_black_and_white()

    plt.imshow(data, interpolation='none', cmap=cm)
    plt.show()

    np.save("./models2/test", ca.data)
    print("------ Starting model analyzer ------")
    import model_analizer
    model_analizer.main()


main()

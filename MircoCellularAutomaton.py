import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

import skimage.draw
import skimage.transform


class MircoCellularAutomaton:

    def __init__(self, width, height, neighbour='moore', neighbour_matrix=None, periodic=False, animation=False,
                 centers_func=None):
        self.newdata = None
        self.data = np.zeros((width, height), dtype=int)
        self.w = width
        self.h = height
        self.neighbour_method = self.moore
        self.num_of_cells = 0
        self.periodic = periodic
        self.centers = None
        self.it_number = 0
        np.random.RandomState()
        if neighbour == 'moore':
            self.neighbour_method = self.moore
        if neighbour == 'von_neumann':
            self.neighbour_method = self.von_neumann
        if neighbour == 'custom' and neighbour_matrix is not None:
            self.neighbour_method = self.custom_prob
            self.prob_matrix = neighbour_matrix
        self.is_animated = animation
        self.animation_set = []
        self.animation_fig = plt.figure(dpi=150)

        left_prob_matrix = np.array([[1, 1, 0], [1, 1, 1], [0, 1, 1]], dtype=float)
        right_prob_matrix = np.array([[0, 1, 1], [1, 1, 1], [1, 1, 0]], dtype=float)

        self.centers_func = centers_func

    def von_neumann(self, i, j):
        if self.data[i, j]:
            if not self.data[i, j + 1]: self.newdata[i, j + 1] = self.data[i, j]
            if not self.data[i, j - 1]: self.newdata[i, j - 1] = self.data[i, j]
            if not self.data[i + 1, j]: self.newdata[i + 1, j] = self.data[i, j]
            if not self.data[i - 1, j]: self.newdata[i - 1, j] = self.data[i, j]
        pass

    def moore(self, i, j):
        if self.data[i, j]:
            if not self.data[i, j + 1]: self.newdata[i, j + 1] = self.data[i, j]
            if not self.data[i, j - 1]: self.newdata[i, j - 1] = self.data[i, j]
            if not self.data[i + 1, j]: self.newdata[i + 1, j] = self.data[i, j]
            if not self.data[i - 1, j]: self.newdata[i - 1, j] = self.data[i, j]

            if not self.data[i + 1, j + 1]: self.newdata[i + 1, j + 1] = self.data[i, j]
            if not self.data[i + 1, j - 1]: self.newdata[i + 1, j - 1] = self.data[i, j]
            if not self.data[i - 1, j + 1]: self.newdata[i - 1, j + 1] = self.data[i, j]
            if not self.data[i - 1, j - 1]: self.newdata[i - 1, j - 1] = self.data[i, j]
        pass

    def custom_prob(self, i, j):
        prob = self.prob_matrix
        rnds = np.random.random_sample
        if self.data[i, j]:

            if not self.data[i, j + 1] and rnds() < prob[1, 2]: self.newdata[i, j + 1] = self.data[i, j]
            if not self.data[i, j - 1] and rnds() < prob[1, 0]: self.newdata[i, j - 1] = self.data[i, j]
            if not self.data[i + 1, j] and rnds() < prob[2, 1]: self.newdata[i + 1, j] = self.data[i, j]
            if not self.data[i - 1, j] and rnds() < prob[0, 1]: self.newdata[i - 1, j] = self.data[i, j]

            if not self.data[i + 1, j + 1] and rnds() < prob[2, 2]: self.newdata[i + 1, j + 1] = self.data[i, j]
            if not self.data[i + 1, j - 1] and rnds() < prob[2, 0]: self.newdata[i + 1, j - 1] = self.data[i, j]
            if not self.data[i - 1, j + 1] and rnds() < prob[0, 2]: self.newdata[i - 1, j + 1] = self.data[i, j]
            if not self.data[i - 1, j - 1] and rnds() < prob[0, 0]: self.newdata[i - 1, j - 1] = self.data[i, j]
        pass

    def next_iteration(self):
        self.newdata = np.copy(self.data)

        if self.periodic:
            offset = -2
        else:
            offset = 1

        for i in range(offset, self.w - 1):
            for j in range(offset, self.h - 1):
                self.neighbour_method(i, j)
        self.data = self.newdata
        pass

    def initial_cells(self, num):
        self.num_of_cells = num
        self.centers = np.empty((2, num), dtype=int)
        self.centers[0, :] = np.random.randint(self.w, size=num)
        self.centers[1, :] = np.random.randint(self.h, size=num)

        for i in range(num):
            self.data[self.centers[0, i], self.centers[1, i]] = i + 1
        pass

    def initial_cell_mesh(self, rows=5, cols=5):
        self.num_of_cells = rows * cols
        col_step = self.h / cols
        row_step = self.w / rows
        k = 1
        self.centers = np.empty((2, self.num_of_cells), dtype=int)
        for i in range(rows):
            for j in range(cols):
                ii = int((i + 0.5) * row_step)
                jj = int((j + 0.5) * col_step)
                self.centers[0, k - 1] = ii
                self.centers[1, k - 1] = jj
                self.data[ii, jj] = k
                k += 1
        pass

    def add_new_centers(self):
        if self.centers_func is None:
            return
        i = 0
        num_of_free_cell = self.w * self.h - np.count_nonzero(self.data)
        num = int(self.centers_func(self.it_number)) - self.num_of_cells
        if num < 0:
            num = 0
            return

        print("It_num={}, Cell num = {}, new cells = {}".format(self.it_number, self.num_of_cells, num))

        if num_of_free_cell < num * 2:
            return
        while i < num:
            x = np.random.randint(self.w)
            y = np.random.randint(self.h)
            if not self.data[x, y]:
                self.data[x, y] = self.num_of_cells + 1
                i += 1
                self.num_of_cells += 1
        pass

    def add_new_centers_exponentially(self, number, param, iter_number):
        num = int(number * np.exp(-param * iter_number))
        print("num = ", num)
        self.add_new_centers(num)

    def add_new_centers_normally(self, number, param, iter_number):
        from scipy.stats import norm

        num = int(number * (norm.cdf(iter_number, param, 15) - norm.cdf(iter_number - 1., param, 15)))
        print("num = ", num)
        self.add_new_centers(num)

    def do_animation(self):
        im = plt.imshow(self.data, interpolation=None, animated=True, aspect='equal')
        self.animation_set.append([im])

    def save_animation(self, filename, writer):
        if self.is_animated:
            self.animation_fig.tight_layout()
            ani = animation.ArtistAnimation(self.animation_fig, self.animation_set, blit=True)
            self.animation_fig.show()
            ani.save(filename=filename, writer=writer)
            self.animation_fig.show()

    def save_animation_mpeg(self, filename):
        self.save_animation(filename=filename, writer=animation.FFMpegWriter(fps=24))

    def save_animation_gif(self, filname):
        self.save_animation(filename=filname, writer=animation.PillowWriter(fps=24))

    def calculate(self, verbose=False, max_iter=np.inf):
        saved_zero_cell = -1
        max_iter += self.it_number

        while self.it_number < max_iter:
            self.next_iteration()
            zero_cell = self.w * self.h - np.count_nonzero(self.data)

            if verbose:
                print(self.it_number, zero_cell, "{:2f}%".format(zero_cell * 100.0 / self.w / self.h))

            if self.is_animated:
                self.do_animation()

            self.add_new_centers()
            # self.add_new_centers_exponentially(10., 0.1, i)
            # self.add_new_centers_normally(100., 30., self.it_number)

            if self.it_number % 20 == 0:
                if saved_zero_cell == zero_cell:
                    return True
                saved_zero_cell = zero_cell

            self.it_number += 1

            if zero_cell == 0:
                return True
        return False

    def to_black_and_white(self):
        bw_image = np.zeros_like(self.data)
        for i in range(-2, self.w - 1):
            for j in range(-2, self.h - 1):
                cond = self.data[i, j] == self.data[i, j + 1] and self.data[i, j] == self.data[i, j - 1] and self.data[
                    i + 1, j] and self.data[i, j] == self.data[i - 1, j]
                if cond:
                    bw_image[i, j] = 1.
                else:
                    bw_image[i, j] = 0.
        return bw_image

    def create_grain_boundaries(self):
        # b_image = np.zeros_like(self.data)
        b_image = np.array(self.data)

        for i in range(1, self.w - 1):
            for j in range(1, self.h - 1):
                cond = self.data[i, j] == self.data[i, j + 1] and self.data[i, j] == self.data[i, j - 1] and self.data[
                    i + 1, j] and self.data[i, j] == self.data[i - 1, j]
                if not cond:
                    b_image[i, j] = self.num_of_cells + 1
                else:
                    b_image[i, j] = self.data[i, j]

        return b_image

    def print_data(self):
        print(self.data)

    @staticmethod
    def get_probability_matrix(rx, ry, angle=0, w=999, h=999, verbose=False):
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
            # fig.savefig("./neibours/prob_matirx5.png", dpi=300)
            # ig.savefig("./neibours/prob_matirx5.eps", dpi=300)
            fig.show()
            plt.show()
            plt.close()

        return prob_matrix

    @staticmethod
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

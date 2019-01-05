import numpy as np


class MircoCellularAutomaton:

    def __init__(self, width, height, neighbour='moore', neighbour_matrix=None, periodic=False):
        self.newdata = None
        self.data = np.zeros((width, height), dtype=int)
        self.w = width
        self.h = height
        self.neighbour_method = self.moore
        self.num_of_cells = 0
        self.periodic = periodic
        self.centers = None
        np.random.RandomState()
        if neighbour == 'moore':
            self.neighbour_method = self.moore
        if neighbour == 'von_neumann':
            self.neighbour_method = self.von_neumann
        if neighbour == 'custom' and neighbour_matrix is not None:
            self.neighbour_method = self.custom_prob
            self.prob_matrix = neighbour_matrix

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
        col_step = self.w / cols
        row_step = self.h / rows
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

    def add_new_centers(self, num):
        i = 0
        num_of_free_cell = self.w * self.h - np.count_nonzero(self.data)
        if num_of_free_cell < num * 2:
            return
        while i < num:
            x = np.random.randint(self.w)
            y = np.random.randint(self.h)
            if not self.data[x, y]:
                self.data[x, y] = self.num_of_cells
                i += 1
                self.num_of_cells += 1
        pass

    def add_new_centers_exponentially(self, number, param, iter_number):
        num = int(number * np.exp(-param * iter_number))
        print "num = ", num
        self.add_new_centers(num)

    def calculate(self, verbose=False, max_iter=np.inf):
        i = 0
        saved_zero_cell = -1
        while i < max_iter:
            self.next_iteration()
            zero_cell = self.w * self.h - np.count_nonzero(self.data)

            if verbose:
                print i, zero_cell, "{0}%".format(zero_cell * 100.0 / self.w / self.h)

            # self.add_new_centers(1)
            # self.add_new_centers_exponentially(10., 0.1, i)


            if i % 10:
                if saved_zero_cell == zero_cell:
                    return True
                saved_zero_cell = zero_cell

            i += 1

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

    def print_data(self):
        print self.data

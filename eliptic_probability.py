import numpy as np
import matplotlib.pyplot as plt
import scipy.signal


def get_pobability_matrix(rx, ry, angle=0, w=999, h=999, verbose=False):
    import skimage.draw
    import skimage.transform

    image = np.zeros((w, h))

    _rx = (rx - 0.5) * w / 3.
    _ry = (ry - 0.5) * h / 3.

    rr, cc = skimage.draw.ellipse(w / 2, h / 2, _rx * 0.999, _ry * 0.999)
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


get_pobability_matrix(1.1, 2.0, angle=30)

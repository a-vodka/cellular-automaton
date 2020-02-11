from PIL import Image
import random
import math
import numpy as np


def metric(x, y):
    # return np.abs(x)+np.abs(y)
    n = 3.
    return (np.abs(x) ** n + np.abs(y) ** n) ** (1.0 / n)


def generate_voronoi_diagram(width, height, num_cells):
    image = Image.new("RGB", (width, height))
    putpixel = image.putpixel
    imgx, imgy = image.size
    nx = np.random.randint(low=0, high=imgx, size=num_cells)
    ny = np.random.randint(low=0, high=imgy, size=num_cells)
    nr = np.random.randint(low=0, high=256, size=num_cells)
    ng = np.random.randint(low=0, high=256, size=num_cells)
    nb = np.random.randint(low=0, high=256, size=num_cells)

    for y in range(imgy):
        for x in range(imgx):
            dmin = metric(imgx - 1, imgy - 1)
            j = -1
            for i in range(num_cells):
                d = metric(nx[i] - x, ny[i] - y)
                if d < dmin:
                    dmin = d
                    j = i
            putpixel((x, y), (nr[j], ng[j], nb[j]))
    image.save("VoronoiDiagram.png", "PNG")
    image.show()


def voronoi(shape=(500, 500), num_cells=25):
    depthmap = np.ones(shape, np.float) * 1e308
    colormap = np.zeros(shape, np.int)
    (imgx, imgy) = shape
    nx = np.random.randint(low=0, high=imgx, size=num_cells)
    ny = np.random.randint(low=0, high=imgy, size=num_cells)

    nr = np.random.randint(low=0, high=256, size=num_cells + 1)
    ng = np.random.randint(low=0, high=256, size=num_cells + 1)
    nb = np.random.randint(low=0, high=256, size=num_cells + 1)

    def hypot(X, Y, n=1.4142):
        return (np.abs(X - x) ** n + np.abs(Y - y) ** n) ** (1.0 / n)

    for i in range(num_cells):
        x = nx[i]
        y = ny[i]
        paraboloid = np.fromfunction(hypot, shape)
        colormap = np.where(paraboloid < depthmap, i + 1, colormap)
        depthmap = np.where(paraboloid <
                            depthmap, paraboloid, depthmap)

    # for i in range(num_cells):
    #    colormap[nx[i] - 1:nx[i] + 2, ny[i] - 1:ny[i] + 2] = 0

    colormap = np.transpose(colormap)

    np.save("./voronoi", colormap)

    pixels = np.empty(colormap.shape + (4,), np.int8)

    pixels[:, :, 3] = 0
    pixels[:, :, 2] = nb[colormap]
    pixels[:, :, 1] = ng[colormap]
    pixels[:, :, 0] = nr[colormap]

    image = Image.frombytes("RGBA", shape, pixels)
    image.save('voronoi-2.png')
    image.show()


if __name__ == '__main__':
    np.random.seed(0)
    voronoi(shape=(513,565), num_cells=714)
    np.random.seed(0)
    # generate_voronoi_diagram(500, 500, 25)

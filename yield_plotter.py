import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import path
import scipy.interpolate
import scipy.stats


def main():
    file_path = './yield_out/'
    files = os.listdir(file_path)

    i = 0
    N = len(files) - 1
    path_obj = np.empty(N, dtype=object)
    for f in files:
        if f.endswith('csv'):
            res = np.loadtxt(file_path + f, delimiter=';', usecols=[0, 1, 3, 4])
            plt.plot(res[:, 0], res[:, 1])
            plt.plot(res[:, 2], res[:, 3])
            path_obj[i] = path.Path(res[:, 0:2])
            i += 1
    plt.show()
    print("Loaded files = {}".format(i))
    NN = 1*int(np.sqrt(i))
    NNN = 200
    arr = np.zeros([NN, 2])
    answ = np.zeros([NN, N], dtype=int)
    arr[:, 0] = np.linspace(55e6, 75e6, NN)
    arr[:, 1] = np.linspace(50e6, 100e6, NN)
    xval = np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2)

    newarr = np.zeros([NNN, 2])
    newarr[:, 0] = np.linspace(55e6, 75e6, NNN)
    newarr[:, 1] = np.linspace(50e6, 100e6, NNN)
    newxval = np.sqrt(newarr[:, 0] ** 2 + newarr[:, 1] ** 2)

    for i in range(N):
        answ[:, i] = path_obj[i].contains_points(arr)

    probability = np.mean(answ, axis=1)
    plt.plot(xval, probability)

    newy = scipy.interpolate.pchip_interpolate(xval, probability, newxval, der=1)

    mean = np.trapz(probability, x=xval)+np.min(xval)
    variance = np.min(xval)**2 + np.trapz(2*xval*probability, x=xval) - mean**2
    print("mean = {}, variance = {}".format(mean, np.sqrt(variance)))
    newcdf = scipy.stats.norm.cdf(newxval, mean, variance**0.5)
    newpdf = scipy.stats.norm.pdf(newxval, mean, variance ** 0.5)

    spln = scipy.interpolate.splrep(xval, probability)

    spln_val = scipy.interpolate.splev(newxval, spln)
    spln_val_der = -scipy.interpolate.splev(newxval, spln, der=1)
    plt.plot(newxval, spln_val_der/np.max(spln_val_der))
    plt.plot(newxval, spln_val)
    plt.plot(newxval, 1-newcdf)
    plt.plot(newxval, newpdf / np.max(newpdf))
    plt.plot(newxval, -newy/np.max(-newy))
    plt.plot(xval, -np.diff(probability, prepend=1))
    plt.show()






if __name__ == "__main__":
    main()

import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import path
import scipy.interpolate
import scipy.stats
import scipy.optimize


def main():
    file_path = './yield_out/'
    files = os.listdir(file_path)

    i = 0
    N = len(files) - 1
    path_obj = np.empty(N, dtype=object)
    plt.axhline(y=0, lw=1, color='k')
    plt.axvline(x=0, lw=1, color='k')
    for f in files:
        if f.endswith('csv'):
            res = np.loadtxt(file_path + f, delimiter=';', usecols=[0, 1, 3, 4])
            plt.plot(res[:, 0] / 1e6, res[:, 1] / 1e6)
            # plt.plot(res[:, 2], res[:, 3])
            path_obj[i] = path.Path(res[:, 0:2])
            i += 1
    plt.xlabel(r"$\sigma_1$, MPa")
    plt.ylabel(r"$\sigma_2$, MPa")

    plt.tight_layout()
    plt.grid()
    plt.savefig('./stress_out/yield_spaghetti.png', dpi=300)
    plt.savefig('./stress_out/yield_spaghetti.eps')
    plt.savefig('./stress_out/yield_spaghetti.pdf')
    plt.show()
    plt.close()
    print("Loaded files = {}".format(i))
    NN = i
    NNN = 200
    arr = np.zeros([NN, 2])
    answ = np.zeros([NN, N], dtype=int)
    rho = np.linspace(30e6, 120e6, NN)
    phi = np.linspace(0, 2 * np.pi, 360)
    mean = np.zeros_like(phi)
    variance = np.zeros_like(phi)

    mean_fitted = np.zeros_like(phi)
    variance_fitted = np.zeros_like(phi)

    probability = np.zeros([NN, phi.size])
    for i in range(phi.size):
        arr[:, 0] = rho * np.cos(phi[i])
        arr[:, 1] = rho * np.sin(phi[i])
        xval = np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2)
        for j in range(N):
            answ[:, j] = path_obj[j].contains_points(arr)
        pr = np.mean(answ, axis=1)
        probability[:, i] = pr
        mean[i] = np.trapz(pr, x=xval) + np.min(xval)
        variance[i] = np.min(xval) ** 2 + np.trapz(2 * xval * pr, x=xval) - mean[i] ** 2

        f = lambda x, mu, sigma: 1. - scipy.stats.norm(mu, sigma).cdf(x)
        mean_fitted[i], variance_fitted[i] = scipy.optimize.curve_fit(f, xval, pr, p0=[mean[i], variance[i] ** 0.5])[0]


    plt.polar(phi, mean, 'r', label='mean')
    plt.polar(phi, mean + 3 * variance ** 0.5, 'b')
    plt.polar(phi, mean - 3 * variance ** 0.5, 'b')
    plt.fill_between(phi, mean - 3 * variance ** 0.5, mean + 3 * variance ** 0.5, alpha=0.2,
                     label='confidence interval')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./stress_out/yield_confidence_int.png', dpi=300)
    plt.savefig('./stress_out/yield_confidence_int.pdf')
    plt.show()
    plt.close()

    plt.ylim([0, 1])
    plt.plot(rho/1e6, probability[:, 0],'b-', label='$\sigma_2=0$ (calculated)')
    plt.plot(rho/1e6, 1 - scipy.stats.norm.cdf(rho, mean[0], variance[0] ** 0.5), 'b--', label='$\sigma_2=0$ (fitted)')
    plt.plot(rho/1e6, probability[:, 45], 'r-', label='$\sigma_1=\sigma_2$ (calculated)')
    plt.plot(rho/1e6, 1 - scipy.stats.norm.cdf(rho, mean[45], variance[45] ** 0.5), 'r--', label='$\sigma_1=\sigma_2$ (fitted)')
    plt.plot(rho/1e6, probability[:, 135], 'g-', label='$\sigma_1=-\sigma_2$ (calculated)')
    plt.plot(rho/1e6, 1 - scipy.stats.norm.cdf(rho, mean[135], variance[135] ** 0.5), 'g--', label='$\sigma_1=-\sigma_2$ (fitted)')

    plt.plot(rho/1e6, probability[:, 90], 'c-', label='$\sigma_1=0$ (calculated)')
    plt.plot(rho/1e6, 1 - scipy.stats.norm.cdf(rho, mean[90], variance[90] ** 0.5), 'c--',
             label='$\sigma_1=0$ (fitted)')
    plt.grid()
    plt.xlabel('$\sigma$, MPa')
    plt.ylabel('Probability')
    plt.tight_layout()
    plt.legend()
    plt.savefig('./stress_out/yield_cdf.png', dpi=300)
    plt.savefig('./stress_out/yield_cdf.eps')
    plt.savefig('./stress_out/yield_cdf.pdf')
    plt.show()
    plt.close()

    plt.plot(rho / 1e6, scipy.stats.norm.pdf(rho, mean[0], variance[0] ** 0.5), 'b--',
             label='$\sigma_2=0$ (fitted)')
    plt.plot(rho / 1e6, scipy.stats.norm.pdf(rho, mean[45], variance[45] ** 0.5), 'r--',
             label='$\sigma_1=\sigma_2$ (fitted)')
    plt.plot(rho / 1e6, scipy.stats.norm.pdf(rho, mean[135], variance[135] ** 0.5), 'g--',
             label='$\sigma_1=-\sigma_2$ (fitted)')
    plt.plot(rho / 1e6, scipy.stats.norm.pdf(rho, mean[90], variance[90] ** 0.5), 'c--',
             label='$\sigma_1=0$ (fitted)')

    plt.grid()
    plt.xlabel('$\sigma$, MPa')
    plt.ylabel('$f(\sigma)$')
    plt.ylim([0, None])
    plt.tight_layout()
    plt.legend()
    plt.savefig('./stress_out/yield_pdf.png', dpi=300)
    plt.savefig('./stress_out/yield_pdf.eps')
    plt.savefig('./stress_out/yield_pdf.pdf')
    plt.show()
    plt.close()

    plt.polar(phi, variance**0.5/1e6, label='std.dev, MPa')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./stress_out/yield_stddev.png', dpi=300)
    plt.savefig('./stress_out/yield_stddev.eps')
    plt.show()


    exit()

    newarr = np.zeros([NNN, 2])
    rho = np.linspace(55e6, 120e6, NNN)
    phi = np.pi / 4
    newarr[:, 0] = rho * np.cos(phi)
    newarr[:, 1] = rho * np.cos(phi)
    newxval = np.sqrt(newarr[:, 0] ** 2 + newarr[:, 1] ** 2)

    for i in range(N):
        answ[:, i] = path_obj[i].contains_points(arr)

    probability = np.mean(answ, axis=1)
    plt.plot(xval, probability)
    mean = np.trapz(probability, x=xval) + np.min(xval)
    variance = np.min(xval) ** 2 + np.trapz(2 * xval * probability, x=xval) - mean ** 2
    print("mean = {:e}, variance = {:e}".format(mean, np.sqrt(variance)))

    f = lambda x, mu, sigma: 1. - scipy.stats.norm(mu, sigma).cdf(x)
    mu, sigma = scipy.optimize.curve_fit(f, xval, probability, p0=[mean, variance ** 0.5])[0]

    print("mean = {:e}, variance = {:e}".format(mu, sigma))

    norm_cdf = scipy.stats.norm.cdf(newxval, mean, variance ** 0.5)
    norm_cdf_fitted = scipy.stats.norm.cdf(newxval, mu, sigma)

    plt.plot(xval, probability, 'b')
    plt.plot(newxval, 1 - norm_cdf, 'g')
    plt.plot(newxval, 1 - norm_cdf_fitted, 'r')
    plt.show()

    exit()
    newy = scipy.interpolate.pchip_interpolate(xval, probability, newxval, der=1)

    newpdf = scipy.stats.norm.pdf(newxval, mean, variance ** 0.5)

    spln = scipy.interpolate.splrep(xval, probability)

    spln_val = scipy.interpolate.splev(newxval, spln)
    spln_val_der = -scipy.interpolate.splev(newxval, spln, der=1)
    plt.plot(newxval, spln_val_der / np.max(spln_val_der))
    plt.plot(newxval, spln_val)
    plt.plot(newxval, 1 - newcdf)
    plt.plot(newxval, newpdf / np.max(newpdf))
    plt.plot(newxval, -newy / np.max(-newy))
    plt.plot(xval, -np.diff(probability, prepend=1))
    plt.show()


if __name__ == "__main__":
    main()

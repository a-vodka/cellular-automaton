import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import path
import scipy.interpolate
import scipy.stats
import scipy.optimize


def kirsch_stress(r, t):
    a = 1
    s = 50e6

    sigma_rr = s / 2 * (1 - a ** 2 / r ** 2) + s / 2 * (1 + 3 * a ** 4 / r ** 4 - 4 * a ** 2 / r ** 2) * np.cos(2 * t)
    sigma_tt = s / 2 * (1 + a ** 2 / r ** 2) - s / 2 * (1 + 3 * a ** 4 / r ** 4) * np.cos(2 * t)
    sigma_rt = -s / 2 * (1 - 3 * a ** 4 / r ** 4 + 2 * a ** 2 / r ** 2) * np.sin(2 * t)

    vm = np.sqrt(((sigma_rr - sigma_tt) ** 2 + sigma_rr ** 2 + sigma_tt ** 2 + 6 *
                  sigma_rt ** 2) / 2)

    s1 = (sigma_rr + sigma_tt) / 2 + np.sqrt((sigma_rr - sigma_tt) ** 2 / 4 + sigma_rt ** 2)
    s2 = (sigma_rr + sigma_tt) / 2 - np.sqrt((sigma_rr - sigma_tt) ** 2 / 4 + sigma_rt ** 2)

    return vm, s1, s2


def kirsch_plot():
    # -- Generate Data -----------------------------------------
    # Using linspace so that the endpoint of 360 is included...
    azimuths = np.radians(np.linspace(0, 360, 360))
    zeniths = np.linspace(1, 3, 100)

    r, theta = np.meshgrid(zeniths, azimuths)
    vm, s1, s2 = kirsch_stress(r, theta)
    pr = calc_prob(s1, s2)
    print(np.max(s1))
    print(np.max(s2))
    print(np.max(vm))

    # -- Plot... ------------------------------------------------
    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
    ax.set_ylim([0, np.max(zeniths)])
    contour = ax.contourf(theta, r, pr, cmap='viridis', vmin=0, vmax=1.0, levels=50)
    contour.set_clim(0.0, 1.0)
    fig.colorbar(contour, ax=ax, ticks=np.linspace(0, 1, 11, endpoint=True))
    ax.set_title("Probability of plastic strain occurrence")
    fig.tight_layout()
    fig.savefig("./out_const/kirsch-pr.png", dpi=300)
    fig.savefig("./out_const/kirsch-pr.eps")
    fig.savefig("./out_const/kirsch-pr.pdf")

    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
    ax.set_ylim([0, np.max(zeniths)])
    contour = ax.contourf(theta, r, s1 / 1e6, cmap='viridis', levels=50)
    fig.colorbar(contour, ax=ax, ticks=np.linspace(np.min(s1 / 1e6), np.max(s1 / 1e6), 11, endpoint=True))
    ax.set_title(r"$\sigma_1$, MPa")
    fig.tight_layout()
    fig.savefig("./out_const/kirsch-s1.png", dpi=300)
    fig.savefig("./out_const/kirsch-s1.eps")
    fig.savefig("./out_const/kirsch-s1.pdf")

    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
    ax.set_ylim([0, np.max(zeniths)])
    contour = ax.contourf(theta, r, s2 / 1e6, cmap='viridis', levels=50)
    fig.colorbar(contour, ax=ax, ticks=np.linspace(np.min(s2 / 1e6), np.max(s2 / 1e6), 11, endpoint=True))
    ax.set_title(r"$\sigma_2$, MPa")
    fig.tight_layout()
    fig.savefig("./out_const/kirsch-s2.png", dpi=300)
    fig.savefig("./out_const/kirsch-s2.eps")
    fig.savefig("./out_const/kirsch-s2.pdf")

    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
    ax.set_ylim([0, np.max(zeniths)])
    contour = ax.contourf(theta, r, vm / 1e6, cmap='viridis', levels=50)
    fig.colorbar(contour, ax=ax, ticks=np.linspace(np.min(vm / 1e6), np.max(vm / 1e6), 11, endpoint=True))
    ax.set_title(r"$\sigma_{vonMizes}}$, MPa")
    fig.tight_layout()
    fig.savefig("./out_const/kirsch-vm.png", dpi=300)
    fig.savefig("./out_const/kirsch-vm.eps")
    fig.savefig("./out_const/kirsch-vm.pdf")

    plt.show()
    pass


def calc_prob(s1, s2):
    file_path = './yield_out/'
    files = os.listdir(file_path)
    N = len(files) - 1
    path_obj = np.empty(N, dtype=object)
    i = 0
    for f in files:
        if f.endswith('csv'):
            res = np.loadtxt(file_path + f, delimiter=';', usecols=[0, 1, 3, 4])
            path_obj[i] = path.Path(res[:, 0:2])
            i += 1

    kk, ll = s1.shape
    answ = np.zeros([kk * ll, N], dtype=int)

    s1_r = np.reshape(s1, kk * ll)
    s2_r = np.reshape(s2, kk * ll)
    for j in range(N):
        answ[:, j] = path_obj[j].contains_points(np.transpose([s1_r, s2_r]))
    pr = 1.0 - np.mean(answ, axis=1)
    print(pr[pr > 0], np.max(pr))
    return pr.reshape(kk, ll)
    pass


def main():
    kirsch_plot()
    exit()
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
    plt.plot(rho / 1e6, probability[:, 0], 'b-', label=r'$\sigma_2=0$ (calculated)')
    plt.plot(rho / 1e6, 1 - scipy.stats.norm.cdf(rho, mean[0], variance[0] ** 0.5), 'b--',
             label=r'$\sigma_2=0$ (fitted)')
    plt.plot(rho / 1e6, probability[:, 45], 'r-', label=r'$\sigma_1=\sigma_2$ (calculated)')
    plt.plot(rho / 1e6, 1 - scipy.stats.norm.cdf(rho, mean[45], variance[45] ** 0.5), 'r--',
             label=r'$\sigma_1=\sigma_2$ (fitted)')
    plt.plot(rho / 1e6, probability[:, 135], 'g-', label=r'$\sigma_1=-\sigma_2$ (calculated)')
    plt.plot(rho / 1e6, 1 - scipy.stats.norm.cdf(rho, mean[135], variance[135] ** 0.5), 'g--',
             label=r'$\sigma_1=-\sigma_2$ (fitted)')

    plt.plot(rho / 1e6, probability[:, 90], 'c-', label=r'$\sigma_1=0$ (calculated)')
    plt.plot(rho / 1e6, 1 - scipy.stats.norm.cdf(rho, mean[90], variance[90] ** 0.5), 'c--',
             label=r'$\sigma_1=0$ (fitted)')
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

    plt.polar(phi, variance ** 0.5 / 1e6, label='std.dev, MPa')
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


if __name__ == "__main__":
    main()

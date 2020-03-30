import numpy as np
import matplotlib.pyplot as plt
import scipy.stats


def plot_hist(data, value_name, units="", pdf=None, filename=None):


    s = r'\left\langle{{ {0} }}\right\rangle = {1:.3f}\:\mathrm{{{2}}}'.format(value_name, np.mean(data), units)
    s += r';\quad'
    s += r'\sqrt{{\mathrm{{var}}\left[{0}\right]}}={1:.3f}\:\mathrm{{{2}}}'.format(value_name, np.std(data), units)
    s = '$' + s + '$'

    fig = plt.figure()
    ax = fig.add_subplot(111, title=s)

    ax.hist(data, density=True, bins=int(np.sqrt(data.size)))
    na_x = np.linspace(np.min(data), np.max(data), 1000)

    if pdf:
        na_param = pdf.fit(data)
        ax.plot(na_x, pdf.pdf(na_x, *na_param))

    x_lab = r'${0}$'.format(value_name)
    if units:
        x_lab += ", {}".format(units)
    ax.set_xlabel(x_lab)
    ax.set_ylabel(r'$f({0})$'.format(value_name))
    fig.tight_layout()

    fig.savefig(filename + ".png", dpi=300)
    fig.savefig(filename)

    print(scipy.stats.shapiro(data))


def main():
    data = np.loadtxt("./consts.csv", delimiter=';', dtype=float, comments='#',
                      usecols=(0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 14, 15, 16))

    Ex = data[:, 0] / 1e9
    Ey = data[:, 1] / 1e9
    Gxy = data[:, 2] / 1e9
    nuxy = data[:, 3]
    etaxxy = data[:, 4]
    etayxy = data[:, 5]

    c11 = data[:, 6] / 1e9
    c12 = data[:, 7] / 1e9
    c16 = data[:, 8] / 1e9
    c22 = data[:, 9] / 1e9
    c26 = data[:, 10] / 1e9
    c66 = data[:, 11] / 1e9

    l1 = data[:, 12] / 1e9
    l2 = data[:, 13] / 1e9
    l3 = data[:, 14] / 1e9

    plot_hist(Ex, "E_x", units="GPa", filename="./ex.eps", pdf=scipy.stats.norm)
    plot_hist(Ey, "E_y", units="GPa", filename="./ey.eps", pdf=scipy.stats.norm)
    plot_hist(Gxy, "G_{xy}", units="GPa", filename="./gxy.eps", pdf=scipy.stats.norm)
    plot_hist(nuxy, "\\nu_{xy}", units="", filename="./nuxy.eps", pdf=scipy.stats.norm)
    plot_hist(etaxxy, "\\eta_{x,xy}", units="", filename="./etaxxy.eps", pdf=scipy.stats.norm)
    plot_hist(etayxy, "\\eta_{y,xy}", units="", filename="./etayxy.eps", pdf=scipy.stats.norm)

    plot_hist(c11, "c_{11}", units="GPa", filename="./c11.eps", pdf=scipy.stats.norm)
    plot_hist(c12, "c_{12}", units="GPa", filename="./c12.eps", pdf=scipy.stats.norm)
    plot_hist(c16, "c_{16}", units="GPa", filename="./c16.eps", pdf=scipy.stats.norm)
    plot_hist(c22, "c_{22}", units="GPa", filename="./c22.eps", pdf=scipy.stats.norm)
    plot_hist(c26, "c_{26}", units="GPa", filename="./c26.eps", pdf=scipy.stats.norm)
    plot_hist(c66, "c_{66}", units="GPa", filename="./c66.eps", pdf=scipy.stats.norm)

    plot_hist(l1, "\lambda_{1}", units="GPa", filename="./l1.eps", pdf=scipy.stats.norm)
    plot_hist(l2, "\lambda_{2}", units="GPa", filename="./l2.eps", pdf=scipy.stats.norm)
    plot_hist(l3, "\lambda_{3}", units="GPa", filename="./l3.eps", pdf=scipy.stats.norm)

    plt.show()

    pass


if __name__ == "__main__":
    main()

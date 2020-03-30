import ExportToDamask
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from MircoCellularAutomaton import *
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import os
import sys

sys.path.insert(1, './out_const/')
import const_proc


def pltVal(X, Y, Z, title="", filename=None):
    ux = np.unique(X)
    uy = np.unique(Y)

    XX, YY = np.meshgrid(ux, uy)

    ZZ = Z.reshape([ux.size, uy.size])

    fig, ax = plt.subplots()
    ax.set_title(title)
    contour = ax.contourf(XX, YY, ZZ, cmap='viridis')
    fig.colorbar(contour, ax=ax)
    fig.tight_layout()
    if filename:
        fig.savefig(filename)
        fig.savefig(filename + ".png", dpi=300)

    return


def formulate_stiff_matrix(de):
    C = np.zeros((6, 6), dtype=np.float)

    e1 = de.get_avg_strain_tensor(1)
    s1 = de.get_avg_stress_tensor(1)
    e2 = de.get_avg_strain_tensor(3)
    s2 = de.get_avg_stress_tensor(3)
    e3 = de.get_avg_strain_tensor(5)
    s3 = de.get_avg_stress_tensor(5)
    e4 = de.get_avg_strain_tensor(7)
    s4 = de.get_avg_stress_tensor(7)

    """
    print('-----------')
    print(e1)
    print(s1)
    print('-----------')
    print(e2)
    print(s2)
    print('-----------')
    print(e3)
    print(s3)
    print('-----------')
    print(e4)
    print(s4)
    print('-----------')
    """
    C[0, :] = np.array([s1[0, 0], s1[1, 1], s1[0, 1], 0, 0, 0])
    C[1, :] = np.array([0, s2[0, 0], 0, s2[1, 1], s2[0, 1], 0])
    C[2, :] = np.array([s3[0, 0], s3[1, 1], s3[0, 1], 0, 0, 0])
    C[3, :] = np.array([0, s3[0, 0], 0, s3[1, 1], s3[0, 1], 0])
    C[4, :] = np.array([0, 0, s4[0, 0], 0, s4[1, 1], s4[0, 1]])
    C[5, :] = np.array([0, s1[0, 0], 0, s1[1, 1], s1[0, 1], 0])

    Eps = np.array([e1[0, 0], e2[1, 1], e3[0, 0], e3[1, 1], 2 * e4[0, 1], e1[1, 1]])

    C[C < 1] = 0
    Eps[Eps < 1e-9] = 0

    if np.linalg.matrix_rank(C) < 6:
        C[5, :] = np.array([0, 0, 0, 0, 0, 0])
        Eps[5] = 0
        C[3, :] = np.array([0, 0, 0, 0, 0, 0])
        Eps[3] = 0

    import sympy as sp
    print(sp.Matrix(C).rref())
    print(C, Eps)

    return C, Eps
    pass


def make_sym_matrix(v):
    m = np.array([[v[0], v[1], v[2]], [v[1], v[3], v[4]], [v[2], v[4], v[5]]])
    return m


def yield_surface():
    # filename = "./models/test.npy"
    # data = np.load(filename)
    # data[0, 0] = 1
    # data[-1, -1] = 100

    Num_grains = 200
    n_p = 50

    def n_grain_cdf(n):
        res = np.array(Num_grains * scipy.stats.norm.cdf(n, n_p / 2, np.sqrt(Num_grains) / 2))
        return res

    # n = np.linspace(0, 101, 100)
    # print(n_grain_cdf(n))
    # plt.plot(n, n_grain_cdf(n))
    # plt.show()

    self_path = os.path.dirname(os.path.realpath(__file__))

    colors = np.empty((Num_grains + 1, 4))
    colors[0] = [1., 1., 1., 1.]  # white background
    for i in range(1, Num_grains):
        colors[i] = [np.random.random_sample(), np.random.random_sample(), np.random.random_sample(), 1]

    colors[-1] = [0, 0, 0, 1]  # grain boundaries
    cm = ListedColormap(colors, name='my_list')

    # nn = 50
    # data = np.random.randint(low=1, high=2 * nn - 1, size=(nn, nn))
    # data = np.zeros([nn, nn], dtype=np.int)
    # for i in range(nn // 10):
    #    for j in range(nn // 10):
    #        data[i * 10:(i + 1) * 10, j * 10:(j + 1) * 10] = i + j + 1

    prob_matrix = MircoCellularAutomaton.get_probability_matrix(1.7, 2.0, angle=0, verbose=False)
    ca = MircoCellularAutomaton(200, 200, neighbour='custom', neighbour_matrix=prob_matrix, periodic=False,
                                animation=False, centers_func=n_grain_cdf)

    # ca = MircoCellularAutomaton(500, 500, neighbour='moore', neighbour_matrix=None, periodic=False,
    #                            animation=False, centers_func=None)

    # ca.initial_cells(Num_grains)
    ca.calculate(verbose=True)

    # ca.save_animation_mpeg('elipse_prob.avi')
    # ca.save_animation_gif('elipse_prob.gif')

    b_image = ca.create_grain_boundaries()

    u_phases = np.unique(b_image)
    phase1 = u_phases[0:-1]
    phase2 = np.array([u_phases[u_phases.size - 1]])

    nz = np.count_nonzero(b_image == phase2)
    w, h = b_image.shape
    area = w * h
    print(nz / area, 1 - nz / area)

    print(phase1, phase2)

    plt.imshow(b_image, interpolation='none', cmap=cm)
    plt.tight_layout()
    plt.savefig(self_path + "/stress_out/micro.eps")
    plt.savefig(self_path + "/stress_out/micro.png", dpi=300)
    # plt.show()
    # plt.imshow(b_image, interpolation='none', cmap=cm)
    # plt.show()
    plt.close()

    de = ExportToDamask.DamaskExporter(data=b_image, project_name="vdk_test",
                                       project_dir="/home/oleksii/PycharmProjects/cellular-automaton/../ex-dam6/")

    npo = 100
    rho = 1e-4
    phi = np.linspace(0, 2 * np.pi, npo, endpoint=True, dtype=np.double)
    # ex = np.linspace(-1e-4, 1e-4, npo, endpoint=True, dtype=np.double)
    # ey = np.linspace(-1e-4, 1e-4, npo, endpoint=True, dtype=np.double)

    ex = rho * np.cos(phi)
    ey = rho * np.sin(phi)

    # xv, yv = np.meshgrid(ex, ey, sparse=False, indexing='ij')

    ex0 = 0
    ey0 = 0
    # print(xv)
    for i in range(phi.size):
        de.tension_x_and_y(xval=ex[i] - ex0, yval=ey[i] - ey0, restore=False)
        ex0 = ex[i]
        ey0 = ey[i]
        # print(xv[i, j], yv[i, j])

    de.create_geom_file()
    de.create_material_config(rand_orient=True, phase1=phase1, phase2=phase2)
    de.run_damask()
    de.post_proc(avg_only=False)

    # sx = np.zeros(9)
    # sy = np.zeros(9)
    # seqv = np.zeros(9)
    # yield_stress_value = 31e6
    yield_stress_value = np.array([95e6, 700e6])
    yield_stress = np.zeros([npo, 3])
    yield_stress095 = np.zeros([npo, 3])
    for i in range(npo):
        print(i)
        de.load_by_ls_num(i + 1)
        max_stress = de.get_max_stress_for_each_phase()
        max_stress095 = de.get_max095_stress_for_each_phase()

        s_avg = de.get_avg_stress_tensor(i + 1)
        s_principals, e_vectors = np.linalg.eig(s_avg)
        a_max = np.argmax(np.abs(e_vectors), axis=1)
        s_principals = s_principals[a_max]
        safety_factor = max_stress / yield_stress_value
        safety_factor095 = max_stress095 / yield_stress_value
        yield_stress[i] = s_principals / np.max(safety_factor)
        yield_stress095[i] = s_principals / np.max(safety_factor095)

    #plt.scatter(yield_stress[:, 0], yield_stress[:, 1])
    #plt.show()
    rnd_int = np.random.random_integers(100000)
    file = open(os.path.dirname(os.path.realpath(__file__)) + "/yield_out/yield{}.csv".format(rnd_int), "a")
    str_line = ""
    for i in range(npo):
        str_line += "{:g};{:g};;{:g};{:g};\n".format(yield_stress[i, 0], yield_stress[i, 1], yield_stress095[i, 0],
                                                     yield_stress095[i, 1])
    file.writelines(str_line)
    file.close()


def main():
    Num_grains = 200
    n_p = 50

    def n_grain_cdf(n):
        res = np.array(Num_grains * scipy.stats.norm.cdf(n, n_p / 2, np.sqrt(Num_grains) / 2))
        return res

    # n = np.linspace(0, 101, 100)
    # print(n_grain_cdf(n))
    # plt.plot(n, n_grain_cdf(n))
    # plt.show()

    self_path = os.path.dirname(os.path.realpath(__file__))

    colors = np.empty((Num_grains + 1, 4))
    colors[0] = [1., 1., 1., 1.]  # white background
    for i in range(1, Num_grains):
        colors[i] = [np.random.random_sample(), np.random.random_sample(), np.random.random_sample(), 1]

    colors[-1] = [0, 0, 0, 1]  # grain boundaries
    cm = ListedColormap(colors, name='my_list')

    # nn = 50
    # data = np.random.randint(low=1, high=2 * nn - 1, size=(nn, nn))
    # data = np.zeros([nn, nn], dtype=np.int)
    # for i in range(nn // 10):
    #    for j in range(nn // 10):
    #        data[i * 10:(i + 1) * 10, j * 10:(j + 1) * 10] = i + j + 1

    prob_matrix = MircoCellularAutomaton.get_probability_matrix(1.7, 2.0, angle=0, verbose=False)
    ca = MircoCellularAutomaton(200, 200, neighbour='custom', neighbour_matrix=prob_matrix, periodic=False,
                                animation=False, centers_func=n_grain_cdf)

    # ca = MircoCellularAutomaton(500, 500, neighbour='moore', neighbour_matrix=None, periodic=False,
    #                            animation=False, centers_func=None)

    # ca.initial_cells(Num_grains)
    ca.calculate(verbose=True)
    data = ca.data

    # ca.save_animation_mpeg('elipse_prob.avi')
    # ca.save_animation_gif('elipse_prob.gif')

    b_image = ca.create_grain_boundaries()

    u_phases = np.unique(b_image)
    phase1 = u_phases[0:-1]
    phase2 = np.array([u_phases[u_phases.size - 1]])

    nz = np.count_nonzero(b_image == phase2)
    w, h = b_image.shape
    area = w * h
    print(nz / area, 1 - nz / area)

    print(phase1, phase2)

    plt.imshow(b_image, interpolation='none', cmap=cm)
    plt.tight_layout()
    plt.savefig(self_path + "/stress_out/micro.eps")
    plt.savefig(self_path + "/stress_out/micro.png", dpi=300)
    # plt.show()
    # plt.imshow(b_image, interpolation='none', cmap=cm)
    # plt.show()

    de = ExportToDamask.DamaskExporter(data=b_image, project_name="vdk_test",
                                       project_dir="/home/oleksii/PycharmProjects/cellular-automaton/../ex-dam5/")

    de.tension_x()
    de.tension_y()
    de.tension_x_and_y()
    de.shear_xy()
    # de.tension_y_and_shear_xy()

    de.create_geom_file()
    de.create_material_config(rand_orient=True, phase1=phase1, phase2=phase2)
    de.run_damask()
    de.post_proc(avg_only=False)
    # exit()
    # de.load_by_ls_num(1)
    # print(de.avg_data)
    # print(de.ls_data)
    # print(de.get_avg_strain_tensor())
    # print(de.get_avg_stress_tensor())

    a, b = formulate_stiff_matrix(de)
    print('----*****---')
    # np.savetxt('./a.csv', a, delimiter=';')

    rank = np.linalg.matrix_rank(a)
    if rank == 6:
        S0 = np.linalg.solve(a, b)
    else:
        S0 = np.linalg.lstsq(a, b, rcond=None)[0]
        print('-------**** rank = {0} ****------'.format(rank))

    #
    print("S0 = ", S0)
    #   S0[np.abs(S0) > 1] = 0

    S = make_sym_matrix(S0)
    print(S)

    C = np.linalg.inv(S)
    print(C)

    print('----')
    cv, _ = np.linalg.eig(C)
    sv, _ = np.linalg.eig(S)
    print('Cv = ', cv)
    print('Sv = ', sv)
    # print(w)
    print('----')

    Ex = 1.0 / S[0, 0]
    Ey = 1.0 / S[1, 1]
    Gxy = 1.0 / S[2, 2]
    nuxy = - S[0, 1] * Ex
    etaxxy = S[0, 2] * Gxy
    etayxy = S[1, 2] * Gxy
    print("Ex = {:g}, Ey = {:g}, Gxy = {:g}, nuxy = {:f}, etaxxy={:f}, etayxy= {:f}".format(Ex, Ey, Gxy, nuxy, etaxxy,
                                                                                            etayxy))

    if True:
        file = open(os.path.dirname(os.path.realpath(__file__)) + "/out_const/consts.csv", "a")
        str_line = "{:g};{:g};{:g};{:f};{:f};{:f};;".format(Ex, Ey, Gxy, nuxy, etaxxy, etayxy)
        str_line += "{:g};{:g};{:g};{:g};{:g};{:g};;".format(C[0, 0], C[0, 1], C[0, 2], C[1, 1], C[1, 2], C[2, 2])
        str_line += "{:g};{:g};{:g};;".format(cv[0], cv[1], cv[2])
        file.writelines(str_line + "\n")
        file.close()

    # print(de.get_stress_tensor())
    # print(de.get_strain_tensor())
    # print(de.get_node_coord())
    # print(de.get_stress_vonMizes())
    # return None
    # exit()
    # plt.matshow(data, cmap='flag')
    # plt.show()

    for i in range(1, 9, 2):
        de.load_by_ls_num(i)

        if True:
            file = open(os.path.dirname(os.path.realpath(__file__)) + "/out_const/min_max_stress.csv", "a")

            max_stress = de.get_max_stress_for_each_phase()
            min_stress = de.get_min_stress_for_each_phase()

            vonMizes_avg = de.get_avg_stress_vonMizes(i)

            max_stress_norm = max_stress / vonMizes_avg
            min_stress_norm = min_stress / vonMizes_avg

            str_line = "{};{};{};{};;;{};{};{};{}".format(min_stress[0], max_stress[0], min_stress_norm[0],
                                                          max_stress_norm[0],
                                                          min_stress[1], max_stress[1], min_stress_norm[1],
                                                          max_stress_norm[1])

            file.writelines(str_line + "\n")
            file.close()

        node_coord = de.get_node_coord()
        titles = {1: "Tension X, von Mizes stress, MPa",
                  3: "Tension Y, von Mizes stress, MPa",
                  5: "Tension X and Y, von Mizes stress, MPa",
                  7: "Shear XY, von Mizes stress, MPa"
                  }
        pltVal(node_coord[:, 0], node_coord[:, 1], de.get_stress_vonMizes() / 1e6, title=titles[i],
               filename="{}/stress_out/sigma{}.eps".format(self_path, i))

        const_proc.plot_hist(de.get_stress_vonMizes() / 1e6, "\sigma", units="MPa",
                             filename="{}/stress_out/sigma_hist{}.eps".format(self_path, i),
                             pdf=None)

        node_coord = de.get_displ_nodal_coord()
        displ = de.get_displ()
        usum = de.get_displ_sum()

        dw, dh = displ.shape
        displ = displ[0:dw // 2, :]
        node_coord = node_coord[0:dw // 2, :]
        usum = usum[0:dw // 2]

        scale = 0.0

        # print(node_coord)
        # print(displ)

        xx = node_coord[:, 0] * (1 + scale * displ[:, 0] / np.max(de.get_displ_sum()))
        yy = node_coord[:, 1] * (1 + scale * displ[:, 1] / np.max(de.get_displ_sum()))

        pltVal(xx, yy, usum)

    # plt.show()


# print(node_coord)


def run_y(n):
    for i in range(n):
        yield_surface()


if __name__ == "__main__":
    # for i in range(100):
    #    main()
    # main()
    for i in range(100):
        yield_surface()
    # plt.show()

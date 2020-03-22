import ExportToDamask
import numpy as np
import matplotlib.pyplot as plt

from MircoCellularAutomaton import *
from matplotlib.colors import LinearSegmentedColormap, ListedColormap


def pltVal(X, Y, Z):
    # print(X, Y, Z)
    import matplotlib.tri as tri
    triang = tri.Triangulation(X, Y)
    interpolator = tri.LinearTriInterpolator(triang, Z)
    ngridx = X.size
    ngridy = Y.size
    # Create grid values first.
    xi = np.linspace(X.min(), X.max(), ngridx)
    yi = np.linspace(Y.min(), Y.max(), ngridy)
    Xi, Yi = np.meshgrid(xi, yi)
    zi = interpolator(Xi, Yi)

    fig, ax = plt.subplots()
    contour = ax.contourf(xi, yi, zi, cmap='viridis')
    fig.colorbar(contour, ax=ax)
    plt.show()

    pass


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


def yild_surface():
    # filename = "./models/test.npy"
    # data = np.load(filename)
    # data[0, 0] = 1
    # data[-1, -1] = 100

    nn = 20
    data = np.random.randint(low=1, high=nn - 1, size=(nn, nn))
    de = ExportToDamask.DamaskExporter(data=data, project_name="vdk_test",
                                       project_dir="/home/oleksii/PycharmProjects/cellular-automaton/../ex-dam/")

    npo = 100
    phi = np.linspace(0, 2 * np.pi, npo, endpoint=True, dtype=np.double)
    # ex = np.linspace(-1e-4, 1e-4, npo, endpoint=True, dtype=np.double)
    # ey = np.linspace(-1e-4, 1e-4, npo, endpoint=True, dtype=np.double)

    ex = 1e-4 * np.cos(phi)
    ey = 1e-4 * np.sin(phi)

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
    de.create_material_config(rand_orient=True)
    de.run_damask()
    de.post_proc(avg_only=False)

    # sx = np.zeros(9)
    # sy = np.zeros(9)
    # seqv = np.zeros(9)
    yield_stress_value = 31e6
    yield_stress = np.zeros([npo, 3])
    for i in range(npo):
        de.load_by_ls_num(i + 1)
        s = de.get_stress_vonMizes()
        maxs_sorted = np.sort(s)
        maxs = maxs_sorted[int(maxs_sorted.size * 0.95)] / yield_stress_value

        s_avg = de.get_avg_stress_tensor(i + 1)
        s_principals, _ = np.linalg.eig(s_avg)
        s_principals = np.array([s_avg[0, 0], s_avg[1, 1], 0])
        # print(s_principals)
        print(s_avg)
        # s_principals = np.sort(s_principals)
        yield_stress[i] = s_principals / maxs

    plt.scatter(yield_stress[:, 0], yield_stress[:, 1])
    # plt.show()
    # de.get_


def main():
    Num_grains = 500
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

    # ca = MircoCellularAutomaton(100, 100, neighbour='custom', neighbour_matrix=prob_matrix, periodic=False,
    #                            animation=False, centers_func=ngr)

    ca = MircoCellularAutomaton(500, 500, neighbour='moore', neighbour_matrix=None, periodic=False,
                                animation=False, centers_func=None)

    ca.initial_cells(Num_grains)
    ca.calculate(verbose=True)
    data = ca.data

    # ca.save_animation_mpeg('elipse_prob.avi')
    # ca.save_animation_gif('elipse_prob.gif')

    b_image = ca.create_grain_boundaries()

    u_phases = np.unique(b_image)
    phase1 = u_phases[0:-1]
    phase2 = np.array([u_phases[u_phases.size-1]])

    print(phase1, phase2)

    #plt.imshow(data, interpolation='none', cmap=cm)
    #plt.show()
    plt.imshow(b_image, interpolation='none', cmap=cm)
    plt.show()

    de = ExportToDamask.DamaskExporter(data=b_image, project_name="vdk_test",
                                       project_dir="/home/oleksii/PycharmProjects/cellular-automaton/../ex-dam/")

    de.tension_x()
    de.tension_y()
    de.tension_x_and_y()
    de.shear_xy()
    # de.tension_y_and_shear_xy()

    de.create_geom_file()
    de.create_material_config(rand_orient=True, phase1=phase1, phase2=phase2)
    de.run_damask()
    de.post_proc(avg_only=False)
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

    # print(de.get_stress_tensor())
    # print(de.get_strain_tensor())
    # print(de.get_node_coord())
    # print(de.get_stress_vonMizes())

    plt.matshow(data, cmap='flag')
    plt.show()

    for i in range(1, 9, 2):
        de.load_by_ls_num(i)

        node_coord = de.get_node_coord()

        pltVal(node_coord[:, 0], node_coord[:, 1], de.get_stress_vonMizes())

        plt.hist(de.get_stress_vonMizes())
        plt.show()

        node_coord = de.get_displ_nodal_coord()
        displ = de.get_displ()

        scale = 0.2

        # print(node_coord)
        # print(displ)

        xx = node_coord[:, 0] * (1 + scale * displ[:, 0] / np.max(de.get_displ_sum()))
        yy = node_coord[:, 1] * (1 + scale * displ[:, 1] / np.max(de.get_displ_sum()))

        pltVal(xx, yy, de.get_displ_sum())


# print(node_coord)


if __name__ == "__main__":
    main()
    # for i in range(5):
    #    yild_surface()
    # plt.show()

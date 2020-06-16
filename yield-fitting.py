import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import scipy.optimize as optimize


def yld2000_2d():
    pass


def Yld2000_phi(alpha, a, S):
    # plane stress
    # a=8 for fcc 6 for bcc
    (alpha1, alpha2, alpha3, alpha4, alpha5, alpha6, alpha7, alpha8) = alpha
    sxx, syy, szz, syz, sxz, sxy = S

    C1_11 = alpha1
    C1_22 = alpha2
    C1_66 = alpha7
    C2_11 = 1.0 / 3.0 * (4 * alpha5 - alpha3)
    C2_12 = 1.0 / 3.0 * (2 * alpha6 - 2 * alpha4)
    C2_21 = 1.0 / 3.0 * (2 * alpha3 - 2 * alpha5)
    C2_22 = 1.0 / 3.0 * (4 * alpha4 - alpha6)
    C2_66 = alpha8

    # X',X''
    X1_xx = C1_11 * sxx
    X1_yy = C1_22 * syy
    X1_xy = C1_66 * sxy
    X2_xx = C2_11 * sxx + C2_12 * syy
    X2_yy = C2_21 * sxx + C2_22 * syy
    X2_xy = C2_66 * sxy
    # X1',X2'
    X1_11 = 1.0 / 2 * (X1_xx + X1_yy + np.sqrt((X1_xx - X1_yy) ** 2 + 4 * X1_xy ** 2))
    X1_22 = 1.0 / 2 * (X1_xx + X1_yy - np.sqrt((X1_xx - X1_yy) ** 2 + 4 * X1_xy ** 2))
    # X1'',X2''
    X2_11 = 1.0 / 2 * (X2_xx + X2_yy + np.sqrt((X2_xx - X2_yy) ** 2 + 4 * X2_xy ** 2))
    X2_22 = 1.0 / 2 * (X2_xx + X2_yy - np.sqrt((X2_xx - X2_yy) ** 2 + 4 * X2_xy ** 2))
    # phi',phi'',phi
    Phi1 = abs(X1_11 - X1_22) ** a
    Phi2 = abs(2 * X2_22 + X2_11) ** a + abs(2 * X2_11 + X2_22) ** a
    Phi = Phi1 + Phi2
    ##        print 'a_eff:',(Phi/2.)**(1./a)
    return (Phi / 2.0) ** (1.0 / a)


def principlas(s):
    sp_pr1 = 0.5 * (s[0, 0] + s[1, 1] + np.sqrt((s[0, 0] - s[1, 1]) ** 2 + 4 * s[0, 1] ** 2))
    sp_pr2 = 0.5 * (s[0, 0] + s[1, 1] - np.sqrt((s[0, 0] - s[1, 1]) ** 2 + 4 * s[0, 1] ** 2))
    return np.array([sp_pr1, sp_pr2])


def vonMizes(S):
    sxx, syy, szz, syz, sxz, sxy = S
    sigma = np.array([
        [sxx, sxy, sxz],
        [sxy, syy, syz],
        [sxy, syz, szz]
    ])
    sp = principlas(sigma)
    return np.sqrt(0.5 * ((sxx - syy) ** 2 + sxx ** 2 + syy ** 2))


def Yld2000_n(alpha, a, S):
    """
    Yld2000 yield criterion
    C: c11,c22,c66  c12=c21=1.0 JAC NOT PASS
    D: d11,d12,d21,d22,d66
    """
    C, D = alpha[0:3], alpha[3:8]
    m = a
    sxx, syy, szz, syz, sxz, sxy = S

    s11, s22, s12 = sxx, syy, sxy
    X = np.array([2.0 * C[0] * s11 - C[0] * s22, 2.0 * C[1] * s22 - C[1] * s11, 3.0 * C[2] * s12]) / 3.0  # a1,a2,a7
    Y = np.array([(8.0 * D[2] - 2.0 * D[0] - 2.0 * D[3] + 2.0 * D[1]) * s11 + (
            4.0 * D[3] - 4.0 * D[1] - 4.0 * D[2] + D[0]) * s22,
                  (4.0 * D[0] - 4.0 * D[2] - 4.0 * D[1] + D[3]) * s11 + (
                          8.0 * D[1] - 2.0 * D[3] - 2.0 * D[0] + 2.0 * D[2]) * s22,
                  9.0 * D[4] * s12]) / 9.0

    def priStrs(sx, sy, sxy):
        temp = np.sqrt((sx - sy) ** 2 + 4.0 * sxy ** 2)
        return 0.5 * (sx + sy + temp), 0.5 * (sx + sy - temp)

    m2 = m / 2.0
    m21 = m2 - 1.0
    (X1, X2), (Y1, Y2) = priStrs(*X), priStrs(*Y)  # Principal values of X, Y
    phi1s, phi21s, phi22s = (X1 - X2) ** 2, (2.0 * Y2 + Y1) ** 2, (2.0 * Y1 + Y2) ** 2
    phi1, phi21, phi22 = phi1s ** m2, phi21s ** m2, phi22s ** m2
    left = phi1 + phi21 + phi22
    r = (0.5 * left) ** (1.0 / m)
    return r


def Yld2000_phi_vdk(alpha, a, S):
    sxx, syy, szz, syz, sxz, sxy = S
    sigma = np.array([
        [sxx, sxy, sxz],
        [sxy, syy, syz],
        [sxy, syz, szz]
    ])
    Lp = 1.0 / 3.0 * np.array([
        [2 * alpha[0], -1 * alpha[0], 0],
        [-1 * alpha[1], 2 * alpha[1], 0],
        [0, 0, 3 * alpha[6]]
    ], dtype=float)

    Lpp = 1.0 / 9.0 * np.array([
        [8 * alpha[4] - 2 * alpha[2] - 2 * alpha[5] + 2 * alpha[3],
         4 * alpha[5] - 4 * alpha[3] - 4 * alpha[4] + alpha[2], 0],
        [4 * alpha[2] - 4 * alpha[4] - 4 * alpha[3] + alpha[5],
         8 * alpha[3] - 2 * alpha[5] - 2 * alpha[2] + 2 * alpha[4],
         0],
        [0, 0, 9 * alpha[7]],
    ], dtype=float)

    # sp = np.matmul(Lp, sigma)
    # spp = np.matmul(Lpp, sigma)

    sp = Lp * sigma
    spp = Lpp * sigma

    # sp = Lp * np.array([sxx, syy, sxy])
    # spp = Lpp * np.array([sxx, syy, sxy])

    sp_pr = principlas(sp)
    spp_pr = principlas(spp)

    phi1 = np.abs(sp_pr[0] - sp_pr[1]) ** a
    phi2 = np.abs(2 * spp_pr[1] + spp_pr[0]) ** a + np.abs(2 * spp_pr[0] + spp_pr[1]) ** a

    phi = phi1 + phi2
    ##        print 'a_eff:',(Phi/2.)**(1./a)
    return (phi / 2.0) ** (1.0 / a)


def main():
    alpha = np.array([1.1335, 0.8830, 1.2782, 1.0511, 1.1370, 1.4534, 1.0794, 0.8473])
    #alpha = np.ones(8)

    rho = 1
    phi = np.linspace(0, 2 * np.pi, 100)
    sx = rho * np.cos(phi)
    sy = rho * np.sin(phi)
    r = np.zeros_like(phi)
    rv = np.zeros_like(phi)
    vm = np.zeros_like(phi)
    rn = np.zeros_like(phi)
    for i in range(phi.size):
        def k1(k):
            rr = Yld2000_phi(alpha, 2, np.array([sx[i] / k, sy[i] / k, 0, 0, 0, 0]))
            return rr - 1

        def k2(k):
            rr = Yld2000_phi_vdk(alpha, 2, np.array([sx[i] / k, sy[i] / k, 0, 0, 0, 0]))
            return rr - 1

        def k3(k):
            rr = vonMizes(np.array([sx[i] / k, sy[i] / k, 0, 0, 0, 0]))
            return rr - 1

        def k4(k):
            rr = Yld2000_n(alpha, 6, np.array([sx[i] / k, sy[i] / k, 0, 0, 0, 0]))
            return rr - 1

        sol = optimize.root_scalar(k1, bracket=[0.01, 3], method='brentq')
        rr = sol.root
        r[i] = ((sx[i] / rr) ** 2 + (sy[i] / rr) ** 2) ** 0.5
        sol = optimize.root_scalar(k2, bracket=[0.01, 5], method='brentq')
        rr = sol.root
        rv[i] = ((sx[i] / rr) ** 2 + (sy[i] / rr) ** 2) ** 0.5
        sol = optimize.root_scalar(k3, bracket=[0.01, 3], method='brentq')
        rr = sol.root
        vm[i] = ((sx[i] / rr) ** 2 + (sy[i] / rr) ** 2) ** 0.5

        sol = optimize.root_scalar(k4, bracket=[0.01, 3], method='brentq')
        rr = sol.root
        rn[i] = ((sx[i] / rr) ** 2 + (sy[i] / rr) ** 2) ** 0.5

        # rr =  Yld2000_phi_vdk(alpha, 8, np.array([sx[i], sy[i], 0, 0, 0, 0]))
        # rv[i] = ((sx[i] / rr) ** 2 + (sy[i] / rr) ** 2) ** 0.5
        #
        # rr = vonMizes(np.array([sx[i], sy[i], 0, 0, 0, 0]))
        # vm[i] = ((sx[i] / rr) ** 2 + (sy[i] / rr) ** 2) ** 0.5

    #plt.polar(phi, r)
    #plt.polar(phi, rv)
    plt.polar(phi, vm)
    plt.polar(phi, rn)
    plt.show()

    pass


if __name__ == "__main__":
    main()

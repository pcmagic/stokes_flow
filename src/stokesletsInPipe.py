import numpy as np
import os
from scipy.io import loadmat
from scipy.special import kv, iv
from numpy import pi, real, imag, exp, sqrt, sum, sin, cos
from petsc4py import PETSc


# see Liron, N., and R. Shahar. "Stokes flow due to a Stokeslet in a pipe." Journal of Fluid Mechanics 86.04 (1978): 727-744.
# class containing functions for detailed expression
# noinspection PyTypeChecker
class detail:
    def __init__(self, threshold, b):
        self._threshold = threshold
        self._b = b
        self._k = np.zeros([0])
        self._n = np.zeros([0])
        self._xn = np.zeros([0])
        self._yn = np.zeros([0])
        self._DmyD_xn = np.zeros([0])
        self._DmyD_yn = np.zeros([0])
        self._xn_k0 = np.zeros([0])
        self._yn_k0 = np.zeros([0])
        self._DmyD_xn_k0 = np.zeros([0])
        self._DmyD_yn_k0 = np.zeros([0])
        self._psi_xn1 = np.zeros([0])
        self._psi_xn2 = np.zeros([0])
        self._psi_xn3 = np.zeros([0])
        self._pi_xn1 = np.zeros([0])
        self._pi_xn2 = np.zeros([0])
        self._pi_xn3 = np.zeros([0])
        self._omega_xn1 = np.zeros([0])
        self._omega_xn2 = np.zeros([0])
        self._omega_xn3 = np.zeros([0])
        self._psi_yn1 = np.zeros([0])
        self._psi_yn2 = np.zeros([0])
        self._psi_yn3 = np.zeros([0])
        self._pi_yn1 = np.zeros([0])
        self._pi_yn2 = np.zeros([0])
        self._pi_yn3 = np.zeros([0])
        self._omega_yn1 = np.zeros([0])
        self._omega_yn2 = np.zeros([0])
        self._omega_yn3 = np.zeros([0])
        self._psi_xn1_k0 = np.zeros([0])
        self._psi_xn3_k0 = np.zeros([0])
        self._pi_xn1_k0 = np.zeros([0])
        self._pi_xn3_k0 = np.zeros([0])
        self._omega_xn1_k0 = np.zeros([0])
        self._omega_xn3_k0 = np.zeros([0])
        self._psi_yn2_k0 = np.zeros([0])
        self._pi_yn2_k0 = np.zeros([0])
        self._omega_yn2_k0 = np.zeros([0])
        self._finish_xyk = False  # run _set_xyk first
        self._finish_xn = False  # run _solve_prepare_xn first
        self._finish_yn = False  # run _solve_prepare_yn first
        self._finish1 = False  # run _solve_prepare1 first
        self._finish2 = False  # run _solve_prepare2 first
        self._finish3 = False  # run _solve_prepare3 first

    def _set_xyk(self):
        threshold = self._threshold
        kmax = int(threshold - 2)
        nmax = int(threshold / 2)
        n_use, k_use = np.meshgrid(np.arange(1, nmax + 1), np.arange(-kmax, kmax + 1))
        INDEX = (np.abs(k_use) + 2 * n_use) <= threshold
        INDEX[kmax, :] = 0
        k_use = k_use[INDEX]
        n_use = n_use[INDEX]

        t_path = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.normpath(t_path + '/' + 'xn.mat')
        mat_contents = loadmat(full_path)
        xn = mat_contents['xn']
        full_path = os.path.normpath(t_path + '/' + 'yn.mat')
        mat_contents = loadmat(full_path)
        yn = mat_contents['yn']
        xn_use = np.vstack((xn[kmax:0:-1, 0: nmax], xn[0: kmax + 1, 0: nmax]))
        yn_use = np.vstack((yn[kmax:0:-1, 0: nmax], yn[0: kmax + 1, 0: nmax]))
        xn_use = xn_use[INDEX]
        yn_use = yn_use[INDEX]
        xn_k0 = xn[0, 0:nmax]
        yn_k0 = yn[0, 0:nmax]

        self._k = k_use
        self._n = n_use
        self._xn = xn_use
        self._yn = yn_use
        self._xn_k0 = xn_k0
        self._yn_k0 = yn_k0
        self._finish_xyk = True
        return True

    def get_b(self):
        return self._b

    def _solve_prepare_xn(self):
        err_msg = 'run _set_xyk first. '
        assert self._finish_xyk, err_msg

        DmyD = lambda k, s: 2 * s ** (-2) * iv(k, s) * (
            (-1) * s * ((-4) + k ** 2 + s ** 2) * iv((-1) + k, s) ** 2 + 2 * ((-2) + k) * (k * (2 + k) + s ** 2) * iv(
                    (-1) + k, s) * iv(k, s) + s * (k * (4 + k) + s ** 2) * iv(k, s) ** 2)
        DmyDk0 = lambda s: 2 * iv(0, s) * (
            s * iv(0, s) ** 2 + (-4) * iv(0, s) * iv(1, s) + (-1) * s ** (-1) * ((-4) + s ** 2) * iv(1, s) ** 2)
        self._DmyD_xn = DmyD(self._k, self._xn)
        self._DmyD_xn_k0 = DmyDk0(self._xn_k0)
        self._finish_xn = True
        return True

    def _solve_prepare_yn(self):
        err_msg = 'run _set_xyk first. '
        assert self._finish_xyk, err_msg

        DmyD = lambda k, s: 2 * s ** (-2) * iv(k, s) * (
            (-1) * s * ((-4) + k ** 2 + s ** 2) * iv((-1) + k, s) ** 2 + 2 * ((-2) + k) * (k * (2 + k) + s ** 2) * iv(
                    (-1) + k, s) * iv(k, s) + s * (k * (4 + k) + s ** 2) * iv(k, s) ** 2)
        DmyDk0 = lambda s: 2 * iv(0, s) * (
            s * iv(0, s) ** 2 + (-4) * iv(0, s) * iv(1, s) + (-1) * s ** (-1) * ((-4) + s ** 2) * iv(1, s) ** 2)
        self._DmyD_yn = DmyD(self._k, self._yn)
        self._DmyD_yn_k0 = DmyDk0(self._yn_k0)
        self._finish_yn = True
        return True

    def _solve_prepare1(self):
        err_msg = 'run _solve_prepare_xn first. '
        assert self._finish_xn, err_msg

        psi1 = lambda k, s, b: (1 / 16) * pi ** (-2) * (
            s ** 2 * ((iv((-2) + k, s) + iv(k, s)) * iv(1 + k, s) + iv((-1) + k, s) * (iv(k, s) + iv(2 + k, s))) * (
                iv((-1) + k, b * s) * kv((-1) + k, s) + (-2) * b * iv(k, b * s) * kv(k, s) + iv(1 + k, b * s) * kv(
                        1 + k,
                        s)) + (
                -1) * (s * iv((-1) + k, s) + (-1) * ((-1) + k) * iv(k, s)) * (
                iv(1 + k, s) * (
                    b * s * (iv((-2) + k, b * s) + 3 * iv(k, b * s)) * kv((-1) + k, s) + iv((-1) + k, b * s) * (
                        (-2) * s * kv((-2) + k, s) + (-2) * (1 + k) * kv((-1) + k, s)) + (-2) * s * iv(1 + k,
                                                                                                       b * s) * kv(k,
                                                                                                                   s)) + 2 * iv(
                        (-1) + k, s) * (
                    (-1) * s * (iv((-1) + k, b * s) + iv(1 + k, b * s)) * kv(k, s) + 2 * (
                        b * s * iv(k, b * s) + (-1) * (2 + k) * iv(1 + k, b * s)) * kv(
                            1 + k,
                            s))))
        pi1 = lambda k, s, b: (1 / 16) * pi ** (-2) * (iv(k, s) * iv(1 + k, s) * (
            b * s * (iv((-2) + k, b * s) + 3 * iv(k, b * s)) * kv((-1) + k, s) + iv((-1) + k, b * s) * (
                (-2) * s * kv((-2) + k, s) + (-2) * (1 + k) * kv((-1) + k, s)) + (-2) * s * iv(1 + k,
                                                                                               b * s) * kv(
                    k,
                    s)) + (
                                                           -2) * iv((-1) + k, s) * (
                                                           s * iv((-1) + k, b * s) * (
                                                               2 * iv(1 + k, s) * kv((-1) + k, s) + iv(k, s) * kv(k,
                                                                                                                  s)) + (
                                                               -2) * b * s * iv(k, b * s) * (
                                                               2 * iv(1 + k, s) * kv(k, s) + iv(k, s) * kv(1 + k,
                                                                                                           s)) + iv(
                                                                   1 + k, b * s) * (
                                                               2 * s * iv(1 + k, s) * kv(1 + k, s) + iv(k, s) * (
                                                                   s * kv(k, s) + 2 * (2 + k) * kv(1 + k, s)))))
        omega1 = lambda k, s, b: (1 / 16) * pi ** (-2) * s ** (-1) * (s ** 2 * iv((-1) + k, s) ** 2 * (
            (-1) * b * s * iv((-2) + k, b * s) * kv((-1) + k, s) + (-3) * b * s * iv(k, b * s) * kv((-1) + k, s) + (
                -8) * b * k * iv(k, b * s) * kv(k, s) + 2 * iv((-1) + k, b * s) * (
                s * kv((-2) + k, s) + (1 + 3 * k) * kv((-1) + k, s) + (-1) * s * kv(k, s)) + 4 * b * s * iv(k,
                                                                                                            b * s) * kv(
                    1 + k, s) + (-8) * iv(1 + k, b * s) * kv(1 + k, s)) + (-2) * s * iv(
                (-1) + k,
                s) * iv(
                k, s) * (
                                                                          (-1) * b * ((-1) + k) * s * iv((-2) + k,
                                                                                                         b * s) * kv(
                                                                                  (-1) + k, s) + 3 * b * s * iv(k,
                                                                                                                b * s) * kv(
                                                                                  (-1) + k, s) + (
                                                                              -3) * b * k * s * iv(k, b * s) * kv(
                                                                                  (-1) + k, s) + (-8) * b * k ** 2 * iv(
                                                                                  k,
                                                                                  b * s) * kv(
                                                                                  k, s) + 2 * iv((-1) + k,
                                                                                                 b * s) * (
                                                                              ((-1) + k) * s * kv((-2) + k, s) + (
                                                                                  (-1) + 3 * k ** 2) * kv((-1) + k,
                                                                                                          s) + (
                                                                                  -1) * ((-1) + k) * s * kv(k, s)) + (
                                                                              -4) * b * s * iv(
                                                                                  k, b * s) * kv(1 + k,
                                                                                                 s) + 4 * b * k * s * iv(
                                                                                  k, b * s) * kv(1 + k, s) + 8 * iv(
                                                                                  1 + k,
                                                                                  b * s) * kv(
                                                                                  1 + k, s) + (
                                                                              -4) * k * iv(
                                                                                  1 + k, b * s) * kv(1 + k, s)) + iv(k,
                                                                                                                     s) ** 2 * (
                                                                          (-2) * iv((-1) + k, b * s) * (
                                                                              (4 * k * s + s ** 3) * kv((-2) + k, s) + (
                                                                                  4 * k + 4 * k ** 2 + s ** 2 + 3 * k * s ** 2) * kv(
                                                                                      (-1) + k, s) + (-1) * s ** 3 * kv(
                                                                                      k,
                                                                                      s)) + s * (
                                                                              b * (4 * k + s ** 2) * iv((-2) + k,
                                                                                                        b * s) * kv(
                                                                                      (-1) + k,
                                                                                      s) + 8 * iv(
                                                                                      1 + k, b * s) * (
                                                                                  (-1) * k * kv(k, s) + s * kv(1 + k,
                                                                                                               s)) + b * iv(
                                                                                      k,
                                                                                      b * s) * (
                                                                                  3 * (4 * k + s ** 2) * kv((-1) + k,
                                                                                                            s) + (
                                                                                      -4) * s * (
                                                                                      (-2) * k * kv(k, s) + s * kv(
                                                                                              1 + k,
                                                                                              s))))))
        psi1_k0 = lambda s, b: (1 / 16) * pi ** (-2) * iv(1, s) * (
            (-4) * s ** 2 * (iv(0, s) + iv(2, s)) * (b * iv(0, b * s) * kv(0, s) + (-1) * iv(1, b * s) * kv(1, s)) + (
                -8) * s * (iv(0, s) + s * iv(1, s)) * (
                b * iv(0, b * s) * kv(1, s) + (-1) * iv(1, b * s) * kv(2, s)))
        pi1_k0 = lambda s, b: (1 / 2) * pi ** (-2) * iv(1, s) * (
            b * iv(0, b * s) + (-1) * s * iv(1, b * s) * (iv(1, s) * kv(1, s) + iv(0, s) * kv(2, s)))

        self._psi_xn1 = psi1(self._k, self._xn, self._b)
        self._psi_yn1 = psi1(self._k, self._yn, self._b)
        self._pi_xn1 = pi1(self._k, self._xn, self._b)
        self._pi_yn1 = pi1(self._k, self._yn, self._b)
        self._omega_xn1 = omega1(self._k, self._xn, self._b)
        self._omega_yn1 = omega1(self._k, self._yn, self._b)
        self._psi_xn1_k0 = psi1_k0(self._xn_k0, self._b)
        self._omega_xn1_k0 = 0
        self._pi_xn1_k0 = pi1_k0(self._xn_k0, self._b)
        self._finish1 = True
        return True

    def _solve_prepare2(self):
        err_msg = 'run _solve_prepare_yn first. '
        assert self._finish_yn, err_msg

        psi2 = lambda k, s, b: (1 / 16) * pi ** (-2) * (
            s ** 2 * ((iv((-2) + k, s) + iv(k, s)) * iv(1 + k, s) + iv((-1) + k, s) * (iv(k, s) + iv(2 + k, s))) * (
                iv((-1) + k, b * s) * kv((-1) + k, s) + (-1) * iv(1 + k, b * s) * kv(1 + k, s)) + (
                -4) * b ** (-1) * (s * iv((-1) + k, s) + (-1) * ((-1) + k) * iv(k, s)) * (
                b * ((-2) + k) * iv((-1) + k, b * s) * iv(1 + k, s) * kv((-1) + k, s) + (-1) * k * iv(k, b * s) * iv(
                        1 + k, s) * kv(k, s) + iv((-1) + k, s) * (
                    (-1) * k * iv(k, b * s) * kv(k, s) + b * (2 + k) * iv(1 + k, b * s) * kv(1 + k, s))))
        pi2 = lambda k, s, b: (1 / 4) * b ** (-1) * pi ** (-2) * (
            iv(k, s) * iv(1 + k, s) * (
                b * ((-2) + k) * iv((-1) + k, b * s) * kv((-1) + k, s) + (-1) * k * iv(k, b * s) * kv(k, s)) + iv(
                    (-1) + k,
                    s) * (
                (-1) * b * s * iv((-1) + k, b * s) * iv(1 + k, s) * kv((-1) + k, s) + b * s * iv(1 + k, s) * iv(1 + k,
                                                                                                                b * s) * kv(
                        1 + k, s) + iv(k, s) * (
                    (-1) * k * iv(k, b * s) * kv(k, s) + b * (2 + k) * iv(1 + k, b * s) * kv(1 + k, s))))
        omega2 = lambda k, s, b: (1 / 2) * b ** (-1) * pi ** (-2) * s ** (-1) * (
            (-1) * b * s ** 2 * iv((-1) + k, s) ** 2 * (
                iv((-1) + k, b * s) * kv((-1) + k, s) + iv(1 + k, b * s) * kv(1 + k, s)) + b * s * iv((-1) + k, s) * iv(
                    k,
                    s) * (
                ((-2) + 3 * k) * iv((-1) + k, b * s) * kv((-1) + k, s) + ((-2) + k) * iv(1 + k, b * s) * kv(1 + k,
                                                                                                            s)) + iv(k,
                                                                                                                     s) ** 2 * (
                b * (4 * k + (-2) * k ** 2 + s ** 2) * iv((-1) + k, b * s) * kv((-1) + k, s) + 2 * k ** 2 * iv(k,
                                                                                                               b * s) * kv(
                        k, s) + b * s ** 2 * iv(1 + k, b * s) * kv(1 + k, s)))
        omega2_k0 = lambda s, b: pi ** (-2) * (
            s * iv(0, s) ** 2 + (-2) * iv(0, s) * iv(1, s) + (-1) * s * iv(1, s) ** 2) * iv(1, b * s) * kv(1, s)

        self._psi_xn2 = psi2(self._k, self._xn, self._b)
        self._psi_yn2 = psi2(self._k, self._yn, self._b)
        self._pi_xn2 = pi2(self._k, self._xn, self._b)
        self._pi_yn2 = pi2(self._k, self._yn, self._b)
        self._omega_xn2 = omega2(self._k, self._xn, self._b)
        self._omega_yn2 = omega2(self._k, self._yn, self._b)
        self._psi_yn2_k0 = 0
        self._omega_yn2_k0 = omega2_k0(self._yn_k0, self._b)
        self._pi_yn2_k0 = 0
        self._finish2 = True
        return True

    def _solve_prepare3(self):
        err_msg = 'run _solve_prepare_xn first. '
        assert self._finish_xn, err_msg

        psi3 = lambda k, s, b: (1 / 8) * pi ** (-2) * s * (
            ((iv((-2) + k, s) + iv(k, s)) * iv(1 + k, s) + iv((-1) + k, s) * (iv(k, s) + iv(2 + k, s))) * (
                (-1) * b * s * iv((-1) + k, b * s) * kv(k, s) + iv(k, b * s) * (
                    s * kv((-1) + k, s) + 2 * ((-1) + k) * kv(k, s))) + (-2) * (
                s * iv((-1) + k, s) + (-1) * ((-1) + k) * iv(k, s)) * (
                b * iv((-1) + k, b * s) * iv(1 + k, s) * kv((-1) + k, s) + (-1) * iv(k, b * s) * iv(1 + k, s) * kv(k,
                                                                                                                   s) + iv(
                        (-1) + k, s) * (
                    (-1) * iv(k, b * s) * kv(k, s) + b * iv(1 + k, b * s) * kv(1 + k, s))))
        pi3 = lambda k, s, b: (1 / 4) * pi ** (-2) * (
            (-1) * s * iv(k, s) * iv(k, b * s) * iv(1 + k, s) * kv(k, s) + b * s * iv((-1) + k, b * s) * iv(1 + k,
                                                                                                            s) * (
                iv(k, s) * kv((-1) + k, s) + 2 * iv((-1) + k, s) * kv(k, s)) + iv((-1) + k,
                                                                                  s) * (
                (-1) * iv(k, b * s) * (s * iv(k, s) * kv(k, s) + 2 * iv(1 + k, s) * (
                    s * kv((-1) + k, s) + 2 * ((-1) + k) * kv(k, s))) + b * s * iv(k, s) * iv(1 + k, b * s) * kv(1 + k,
                                                                                                                 s)))
        omega3 = lambda k, s, b: (1 / 4) * pi ** (-2) * s ** (-1) * (s * iv(k, s) ** 2 * (
            (-2) * k * iv(k, b * s) * (s * kv((-1) + k, s) + 2 * k * kv(k, s)) + b * iv((-1) + k, b * s) * (
                (4 * k + s ** 2) * kv((-1) + k, s) + 2 * k * s * kv(k, s)) + (-1) * b * s ** 2 * iv(1 + k,
                                                                                                    b * s) * kv(
                    1 + k, s)) + s * iv((-1) + k, s) ** 2 * (2 * k * iv(k, b * s) * (
            s * kv((-1) + k, s) + 2 * ((-1) + k) * kv(k, s)) + (-1) * b * s * iv((-1) + k, b * s) * (
                                                                 s * kv((-1) + k, s) + 2 * k * kv(k,
                                                                                                  s)) + b * s ** 2 * iv(
                1 + k, b * s) * kv(1 + k, s)) + 2 * iv((-1) + k, s) * iv(k, s) * (
                                                                         (-2) * k ** 2 * iv(k, b * s) * (
                                                                             s * kv((-1) + k, s) + 2 * ((-1) + k) * kv(
                                                                                     k,
                                                                                     s)) + b * s * iv(
                                                                                 (-1) + k, b * s) * (
                                                                             ((-1) + k) * s * kv((-1) + k,
                                                                                                 s) + 2 * k ** 2 * kv(k,
                                                                                                                      s)) + (
                                                                             -1) * b * ((-1) + k) * s ** 2 * iv(1 + k,
                                                                                                                b * s) * kv(
                                                                                 1 + k,
                                                                                 s)))

        psi3_k0 = lambda s, b: (1 / 4) * pi ** (-2) * s * iv(1, s) * (b * iv(1, b * s) * (
            (-1) * s * (iv(0, s) + iv(2, s)) * kv(0, s) + (-2) * (iv(0, s) + s * iv(1, s)) * kv(1, s)) + iv(0,
                                                                                                            b * s) * (
                                                                          2 * (s * iv(1, s) + (-1) * iv(2, s)) * kv(0,
                                                                                                                    s) + s * (
                                                                              iv(0, s) + iv(2, s)) * kv(1, s)))
        pi3_k0 = lambda s, b: (1 / 2) * pi ** (-2) * iv(1, s) * (
            b * iv(1, b * s) + (-1) * s * iv(0, b * s) * (iv(2, s) * kv(0, s) + iv(1, s) * kv(1, s)))

        self._psi_xn3 = psi3(self._k, self._xn, self._b)
        self._psi_yn3 = psi3(self._k, self._yn, self._b)
        self._pi_xn3 = pi3(self._k, self._xn, self._b)
        self._pi_yn3 = pi3(self._k, self._yn, self._b)
        self._omega_xn3 = omega3(self._k, self._xn, self._b)
        self._omega_yn3 = omega3(self._k, self._yn, self._b)
        self._psi_xn3_k0 = psi3_k0(self._xn_k0, self._b)
        self._omega_xn3_k0 = 0
        self._pi_xn3_k0 = pi3_k0(self._xn_k0, self._b)
        self._finish3 = True
        return True

    def solve_u1(self, R, Phi, z):
        err_msg = 'run _solve_prepare1 first. '
        assert self._finish1, err_msg

        AFPhi1nL = lambda xn, k, psi1, omega1, pi1, R, z, DmyD: (-2) * exp(1) ** ((-1) * z * imag(xn)) * pi * imag(
                DmyD ** (-1) * exp(1) ** (sqrt(-1 + 0j) * z * real(xn)) * (
                    (-1) * (omega1 + k * pi1) * iv((-1) + k, R * xn) + k * (
                        omega1 + pi1 + k * pi1 + (-1) * psi1) * R ** (
                        -1) * xn ** (-1) * iv(k, R * xn)))
        AFPhi1nR = lambda yn, k, psi1, omega1, pi1, R, z, DmyD: (-1) * exp(1) ** ((-1) * z * imag(yn)) * pi * imag(
                DmyD ** (-1) * (
                    (-1) * (omega1 + k * pi1) * iv((-1) + k, R * yn) + k * (
                        omega1 + pi1 + k * pi1 + (-1) * psi1) * R ** (
                        -1) * yn ** (-1) * iv(k, R * yn)))
        AFR1nL = lambda xn, k, psi1, omega1, pi1, R, z, DmyD: (-2) * exp(1) ** ((-1) * z * imag(xn)) * pi * imag(
                DmyD ** (-1) * exp(1) ** (sqrt(-1 + 0j) * z * real(xn)) * R ** (-1) * xn ** (-1) * (
                    ((-1) * pi1 + psi1) * R * xn * iv((-1) + k, R * xn) + (
                        k * (omega1 + pi1 + k * pi1 + (-1) * psi1) + pi1 * R ** 2 * xn ** 2) * iv(k, R * xn)))
        AFR1nR = lambda yn, k, psi1, omega1, pi1, R, z, DmyD: (-1) * exp(1) ** ((-1) * z * imag(yn)) * pi * imag(
                DmyD ** (-1) * R ** (-1) * yn ** (-1) * (
                    ((-1) * pi1 + psi1) * R * yn * iv((-1) + k, R * yn) + (
                        k * (omega1 + pi1 + k * pi1 + (-1) * psi1) + pi1 * R ** 2 * yn ** 2) * iv(k, R * yn)))
        BFz1nL = lambda xn, k, psi1, omega1, pi1, R, z, DmyD: (-2) * exp(1) ** ((-1) * z * imag(xn)) * pi * real(
                DmyD ** (-1) * exp(1) ** (sqrt(-1 + 0j) * z * real(xn)) * (
                    pi1 * R * xn * iv((-1) + k, R * xn) + (pi1 + (-1) * k * pi1 + psi1) * iv(k, R * xn)))
        BFz1nR = lambda yn, k, psi1, omega1, pi1, R, z, DmyD: (-1) * exp(1) ** ((-1) * z * imag(yn)) * pi * real(
                DmyD ** (-1) * (pi1 * R * yn * iv((-1) + k, R * yn) + (pi1 + (-1) * k * pi1 + psi1) * iv(k, R * yn)))
        uR1_k0 = lambda xn, psi1, omega1, pi1, R, z, DmyD: (-2) * exp(1) ** ((-1) * z * imag(xn)) * pi * imag(
                DmyD ** (-1) * exp(1) ** (sqrt(-1 + 0j) * z * real(xn)) * (
                    pi1 * R * xn * iv(0, R * xn) + ((-1) * pi1 + psi1) * iv(1, R * xn)))
        uz1_k0 = lambda xn, psi1, omega1, pi1, R, z, DmyD: (-2) * exp(1) ** ((-1) * z * imag(xn)) * pi * real(
                DmyD ** (-1) * exp(1) ** (sqrt(-1 + 0j) * z * real(xn)) * (
                    (pi1 + psi1) * iv(0, R * xn) + pi1 * R * xn * iv(1, R * xn)))

        R = np.array(R, dtype=float).flatten()
        z = np.array(z, dtype=float).flatten()
        Phi = np.array(Phi, dtype=float)
        Phi_shape = Phi.shape
        Phi_flags = Phi.flags
        Phi = Phi.flatten()
        err_msg = 'both R and z should be scales. '
        assert R.size == 1 and z.size == 1, err_msg

        uR1 = Phi.copy()
        uPhi1 = Phi.copy()
        uz1 = Phi.copy()
        uR1k0 = sum(uR1_k0(self._xn_k0, self._psi_xn1_k0, self._omega_xn1_k0, self._pi_xn1_k0, R, z, self._DmyD_xn_k0))
        uPhi1k0 = 0
        uz1k0 = sum(uz1_k0(self._xn_k0, self._psi_xn1_k0, self._omega_xn1_k0, self._pi_xn1_k0, R, z, self._DmyD_xn_k0))
        t_AFR1nL = AFR1nL(self._xn, self._k, self._psi_xn1, self._omega_xn1, self._pi_xn1, R, z, self._DmyD_xn)
        t_AFR1nR = AFR1nR(self._yn, self._k, self._psi_yn1, self._omega_yn1, self._pi_yn1, R, z, self._DmyD_yn)
        t_AFPhi1nL = AFPhi1nL(self._xn, self._k, self._psi_xn1, self._omega_xn1, self._pi_xn1, R, z, self._DmyD_xn)
        t_AFPhi1nR = AFPhi1nR(self._yn, self._k, self._psi_yn1, self._omega_yn1, self._pi_yn1, R, z, self._DmyD_yn)
        t_BFz1nL = BFz1nL(self._xn, self._k, self._psi_xn1, self._omega_xn1, self._pi_xn1, R, z, self._DmyD_xn)
        t_BFz1nR = BFz1nR(self._yn, self._k, self._psi_yn1, self._omega_yn1, self._pi_yn1, R, z, self._DmyD_yn)
        for i0, phi in enumerate(Phi):
            uR1[i0] = uR1k0 + sum((t_AFR1nL + t_AFR1nR) * cos(self._k * phi))
            uPhi1[i0] = uPhi1k0 + sum((t_AFPhi1nL + t_AFPhi1nR) * sin(self._k * phi))
            uz1[i0] = uz1k0 + sum((t_BFz1nL + t_BFz1nR) * cos(self._k * phi))
        if Phi_flags['C_CONTIGUOUS']:
            uR1 = uR1.reshape(Phi_shape, order='C')
            uPhi1 = uPhi1.reshape(Phi_shape, order='C')
            uz1 = uz1.reshape(Phi_shape, order='C')
        elif Phi_flags['F_CONTIGUOUS']:
            uR1 = uR1.reshape(Phi_shape, order='F')
            uPhi1 = uPhi1.reshape(Phi_shape, order='F')
            uz1 = uz1.reshape(Phi_shape, order='F')
        else:
            raise ValueError('C_CONTIGUOUS and F_CONTIGUOUS are both False. ')
        return uR1, uPhi1, uz1

    def solve_u2(self, R, Phi, z):
        err_msg = 'run _solve_prepare2 first. '
        assert self._finish2, err_msg

        AFPhi2nL = lambda xn, k, psi2, omega2, pi2, R, z, DmyD: (-2) * exp(1) ** ((-1) * z * imag(xn)) * pi * imag(
                DmyD ** (-1) * exp(1) ** (sqrt(-1 + 0j) * z * real(xn)) * (
                    ((-1) * omega2 + k * pi2) * iv((-1) + k, R * xn) + k * (
                        omega2 + (-1) * (1 + k) * pi2 + psi2) * R ** (
                        -1) * xn ** (-1) * iv(k, R * xn)))
        AFPhi2nR = lambda yn, k, psi2, omega2, pi2, R, z, DmyD: (-1) * exp(1) ** ((-1) * z * imag(yn)) * pi * imag(
                DmyD ** (-1) * (
                    ((-1) * omega2 + k * pi2) * iv((-1) + k, R * yn) + k * (
                        omega2 + (-1) * (1 + k) * pi2 + psi2) * R ** (
                        -1) * yn ** (-1) * iv(k, R * yn)))
        AFR2nL = lambda xn, k, psi2, omega2, pi2, R, z, DmyD: (-2) * exp(1) ** ((-1) * z * imag(xn)) * pi * imag(
                DmyD ** (-1) * exp(1) ** (sqrt(-1 + 0j) * z * real(xn)) * R ** (-1) * xn ** (-1) * (
                    ((-1) * pi2 + psi2) * R * xn * iv((-1) + k, R * xn) + (
                        k * ((-1) * omega2 + pi2 + k * pi2 + (-1) * psi2) + pi2 * R ** 2 * xn ** 2) * iv(k, R * xn)))
        AFR2nR = lambda yn, k, psi2, omega2, pi2, R, z, DmyD: (-1) * exp(1) ** ((-1) * z * imag(yn)) * pi * imag(
                DmyD ** (-1) * R ** (-1) * yn ** (-1) * (
                    ((-1) * pi2 + psi2) * R * yn * iv((-1) + k, R * yn) + (
                        k * ((-1) * omega2 + pi2 + k * pi2 + (-1) * psi2) + pi2 * R ** 2 * yn ** 2) * iv(k, R * yn)))
        BFz2nL = lambda xn, k, psi2, omega2, pi2, R, z, DmyD: (-2) * exp(1) ** ((-1) * z * imag(xn)) * pi * real(
                DmyD ** (-1) * exp(1) ** (sqrt(-1 + 0j) * z * real(xn)) * (
                    pi2 * R * xn * iv((-1) + k, R * xn) + (pi2 + (-1) * k * pi2 + psi2) * iv(k, R * xn)))
        BFz2nR = lambda yn, k, psi2, omega2, pi2, R, z, DmyD: (-1) * exp(1) ** ((-1) * z * imag(yn)) * pi * real(
                DmyD ** (-1) * (pi2 * R * yn * iv((-1) + k, R * yn) + (pi2 + (-1) * k * pi2 + psi2) * iv(k, R * yn)))
        uPhi2_k0 = lambda yn, psi2, omega2, pi2, R, z, DmyD: exp(1) ** ((-1) * z * imag(yn)) * pi * imag(
                DmyD ** (-1) * omega2 * iv(1, R * yn))

        R = np.array(R, dtype=float).flatten()
        z = np.array(z, dtype=float).flatten()
        Phi = np.array(Phi, dtype=float)
        Phi_shape = Phi.shape
        Phi_flags = Phi.flags
        Phi = Phi.flatten()
        err_msg = 'both R and z should be scales. '
        assert R.size == 1 and z.size == 1, err_msg

        uR2 = Phi.copy()
        uPhi2 = Phi.copy()
        uz2 = Phi.copy()
        uR2k0 = 0
        uPhi2k0 = sum(
                uPhi2_k0(self._yn_k0, self._psi_yn2_k0, self._omega_yn2_k0, self._pi_yn2_k0, R, z, self._DmyD_yn_k0))
        uz2k0 = 0
        t_AFR2nL = AFR2nL(self._xn, self._k, self._psi_xn2, self._omega_xn2, self._pi_xn2, R, z, self._DmyD_xn)
        t_AFR2nR = AFR2nR(self._yn, self._k, self._psi_yn2, self._omega_yn2, self._pi_yn2, R, z, self._DmyD_yn)
        t_AFPhi2nL = AFPhi2nL(self._xn, self._k, self._psi_xn2, self._omega_xn2, self._pi_xn2, R, z, self._DmyD_xn)
        t_AFPhi2nR = AFPhi2nR(self._yn, self._k, self._psi_yn2, self._omega_yn2, self._pi_yn2, R, z, self._DmyD_yn)
        t_BFz2nL = BFz2nL(self._xn, self._k, self._psi_xn2, self._omega_xn2, self._pi_xn2, R, z, self._DmyD_xn)
        t_BFz2nR = BFz2nR(self._yn, self._k, self._psi_yn2, self._omega_yn2, self._pi_yn2, R, z, self._DmyD_yn)
        for i0, phi in enumerate(Phi):
            uR2[i0] = uR2k0 + sum((t_AFR2nL + t_AFR2nR) * sin(self._k * phi))
            uPhi2[i0] = uPhi2k0 + sum((t_AFPhi2nL + t_AFPhi2nR) * cos(self._k * phi))
            uz2[i0] = uz2k0 + sum((t_BFz2nL + t_BFz2nR) * sin(self._k * phi))
        if Phi_flags['C_CONTIGUOUS']:
            uR2 = uR2.reshape(Phi_shape, order='C')
            uPhi2 = uPhi2.reshape(Phi_shape, order='C')
            uz2 = uz2.reshape(Phi_shape, order='C')
        elif Phi_flags['F_CONTIGUOUS']:
            uR2 = uR2.reshape(Phi_shape, order='F')
            uPhi2 = uPhi2.reshape(Phi_shape, order='F')
            uz2 = uz2.reshape(Phi_shape, order='F')
        else:
            raise ValueError('C_CONTIGUOUS and F_CONTIGUOUS are both False. ')
        return uR2, uPhi2, uz2

    def solve_u3(self, R, Phi, z):
        err_msg = 'run _solve_prepare3 first. '
        assert self._finish3, err_msg

        BFPhi3nL = lambda xn, k, psi3, omega3, pi3, R, z, DmyD: 2 * exp(1) ** ((-1) * z * imag(xn)) * pi * real(
                DmyD ** (-1) * exp(1) ** (sqrt(-1 + 0j) * z * real(xn)) * (
                    (-1) * (omega3 + k * pi3) * iv((-1) + k, R * xn) + k * (
                        omega3 + pi3 + k * pi3 + (-1) * psi3) * R ** (
                        -1) * xn ** (-1) * iv(k, R * xn)))
        BFPhi3nR = lambda yn, k, psi3, omega3, pi3, R, z, DmyD: exp(1) ** ((-1) * z * imag(yn)) * pi * real(
                DmyD ** (-1) * (
                    (-1) * (omega3 + k * pi3) * iv((-1) + k, R * yn) + k * (
                        omega3 + pi3 + k * pi3 + (-1) * psi3) * R ** (
                        -1) * yn ** (-1) * iv(k, R * yn)))
        BFR3nL = lambda xn, k, psi3, omega3, pi3, R, z, DmyD: 2 * exp(1) ** ((-1) * z * imag(xn)) * pi * real(
                DmyD ** (-1) * exp(1) ** (sqrt(-1 + 0j) * z * real(xn)) * R ** (-1) * xn ** (-1) * (
                    ((-1) * pi3 + psi3) * R * xn * iv((-1) + k, R * xn) + (
                        k * (omega3 + pi3 + k * pi3 + (-1) * psi3) + pi3 * R ** 2 * xn ** 2) * iv(k, R * xn)))
        BFR3nR = lambda yn, k, psi3, omega3, pi3, R, z, DmyD: exp(1) ** ((-1) * z * imag(yn)) * pi * real(
                DmyD ** (-1) * R ** (-1) * yn ** (-1) * (
                    ((-1) * pi3 + psi3) * R * yn * iv((-1) + k, R * yn) + (
                        k * (omega3 + pi3 + k * pi3 + (-1) * psi3) + pi3 * R ** 2 * yn ** 2) * iv(k, R * yn)))
        AFz3nL = lambda xn, k, psi3, omega3, pi3, R, z, DmyD: (-2) * exp(1) ** ((-1) * z * imag(xn)) * pi * imag(
                DmyD ** (-1) * exp(1) ** (sqrt(-1 + 0j) * z * real(xn)) * (
                    pi3 * R * xn * iv((-1) + k, R * xn) + (pi3 + (-1) * k * pi3 + psi3) * iv(k, R * xn)))
        AFz3nR = lambda yn, k, psi3, omega3, pi3, R, z, DmyD: (-1) * exp(1) ** ((-1) * z * imag(yn)) * pi * imag(
                DmyD ** (-1) * (pi3 * R * yn * iv((-1) + k, R * yn) + (pi3 + (-1) * k * pi3 + psi3) * iv(k, R * yn)))
        uR3_k0 = lambda xn, psi3, omega3, pi3, R, z, DmyD: 2 * exp(1) ** ((-1) * z * imag(xn)) * pi * real(
                DmyD ** (-1) * exp(1) ** (sqrt(-1 + 0j) * z * real(xn)) * (
                    pi3 * R * xn * iv(0, R * xn) + ((-1) * pi3 + psi3) * iv(1, R * xn)))
        uz3_k0 = lambda xn, psi3, omega3, pi3, R, z, DmyD: (-2) * exp(1) ** ((-1) * z * imag(xn)) * pi * imag(
                DmyD ** (-1) * exp(1) ** (sqrt(-1 + 0j) * z * real(xn)) * (
                    (pi3 + psi3) * iv(0, R * xn) + pi3 * R * xn * iv(1, R * xn)))

        R = np.array(R, dtype=float).flatten()
        z = np.array(z, dtype=float).flatten()
        Phi = np.array(Phi, dtype=float)
        Phi_shape = Phi.shape
        Phi_flags = Phi.flags
        Phi = Phi.flatten()
        err_msg = 'both R and z should be scales. '
        assert R.size == 1 and z.size == 1, err_msg

        uR3 = Phi.copy()
        uPhi3 = Phi.copy()
        uz3 = Phi.copy()
        uR3k0 = sum(uR3_k0(self._xn_k0, self._psi_xn3_k0, self._omega_xn3_k0, self._pi_xn3_k0, R, z, self._DmyD_xn_k0))
        uPhi3k0 = 0
        uz3k0 = sum(uz3_k0(self._xn_k0, self._psi_xn3_k0, self._omega_xn3_k0, self._pi_xn3_k0, R, z, self._DmyD_xn_k0))
        t_BFR3nL = BFR3nL(self._xn, self._k, self._psi_xn3, self._omega_xn3, self._pi_xn3, R, z, self._DmyD_xn)
        t_BFR3nR = BFR3nR(self._yn, self._k, self._psi_yn3, self._omega_yn3, self._pi_yn3, R, z, self._DmyD_yn)
        t_BFPhi3nL = BFPhi3nL(self._xn, self._k, self._psi_xn3, self._omega_xn3, self._pi_xn3, R, z, self._DmyD_xn)
        t_BFPhi3nR = BFPhi3nR(self._yn, self._k, self._psi_yn3, self._omega_yn3, self._pi_yn3, R, z, self._DmyD_yn)
        t_AFz3nL = AFz3nL(self._xn, self._k, self._psi_xn3, self._omega_xn3, self._pi_xn3, R, z, self._DmyD_xn)
        t_AFz3nR = AFz3nR(self._yn, self._k, self._psi_yn3, self._omega_yn3, self._pi_yn3, R, z, self._DmyD_yn)
        for i0, phi in enumerate(Phi):
            uR3[i0] = uR3k0 + sum((t_BFR3nL + t_BFR3nR) * cos(self._k * phi))
            uPhi3[i0] = uPhi3k0 + sum((t_BFPhi3nL + t_BFPhi3nR) * sin(self._k * phi))
            uz3[i0] = uz3k0 + sum((t_AFz3nL + t_AFz3nR) * cos(self._k * phi))
        if Phi_flags['C_CONTIGUOUS']:
            uR3 = uR3.reshape(Phi_shape, order='C')
            uPhi3 = uPhi3.reshape(Phi_shape, order='C')
            uz3 = uz3.reshape(Phi_shape, order='C')
        elif Phi_flags['F_CONTIGUOUS']:
            uR3 = uR3.reshape(Phi_shape, order='F')
            uPhi3 = uPhi3.reshape(Phi_shape, order='F')
            uz3 = uz3.reshape(Phi_shape, order='F')
        else:
            raise ValueError('C_CONTIGUOUS and F_CONTIGUOUS are both False. ')
        return uR3, uPhi3, uz3

    def solve_prepare(self):
        self._set_xyk()
        self._solve_prepare_xn()
        self._solve_prepare_yn()
        self._solve_prepare1()
        self._solve_prepare2()
        self._solve_prepare3()
        return True

    def solve_u(self, R, Phi, z):
        uR1, uPhi1, uz1 = self.solve_u1(R, Phi, z)
        uR2, uPhi2, uz2 = self.solve_u2(R, Phi, z)
        uR3, uPhi3, uz3 = self.solve_u3(R, Phi, z)
        return uR1, uPhi1, uz1, uR2, uPhi2, uz2, uR3, uPhi3, uz3

    def solve_uxyz(self, nodes):
        comm = PETSc.COMM_WORLD.tompi4py()
        rank = comm.Get_rank()
        phi = np.arctan2(nodes[:, 1], nodes[:, 0])
        rho = np.sqrt(nodes[:, 0] ** 2 + nodes[:, 1] ** 2)
        z = nodes[:, 2]
        u1 = []
        u2 = []
        u3 = []
        dmda = PETSc.DMDA().create(sizes=(nodes.shape[0],), dof=1,
                                   stencil_width=0, comm=PETSc.COMM_WORLD)
        dmda.setFromOptions()
        dmda.setUp()
        for i0 in range(dmda.getRanges()[0][0], dmda.getRanges()[0][1]):
            t_rho = rho[i0]
            t_phi = phi[i0]
            t_z = z[i0]
            abs_z = np.abs(t_z)
            sign_z = np.sign(t_z)
            if np.isclose(abs_z, 1):
                uR1, uPhi1, uz1, uR2, uPhi2, uz2, uR3, uPhi3, uz3 = self.solve_u(t_rho, t_phi, abs_z)
                ux1 = np.cos(t_phi) * uR1 - np.sin(t_phi) * uPhi1
                ux2 = np.cos(t_phi) * uR2 - np.sin(t_phi) * uPhi2
                ux3 = np.cos(t_phi) * uR3 - np.sin(t_phi) * uPhi3
                uy1 = np.sin(t_phi) * uR1 + np.cos(t_phi) * uPhi1
                uy2 = np.sin(t_phi) * uR2 + np.cos(t_phi) * uPhi2
                uy3 = np.sin(t_phi) * uR3 + np.cos(t_phi) * uPhi3
            else:
                ux1 = 0
                uy1 = 0
                uz1 = 0
                ux2 = 0
                uy2 = 0
                uz2 = 0
                ux3 = 0
                uy3 = 0
                uz3 = 0
            u1.append((ux1, uy1, sign_z * uz1))
            u2.append((ux2, uy2, sign_z * uz2))
            u3.append((sign_z * ux3, sign_z * uy3, uz3))
        u1_all = np.vstack(comm.allgather(u1))
        u2_all = np.vstack(comm.allgather(u2))
        u3_all = np.vstack(comm.allgather(u3))
        return u1_all, u2_all, u3_all


class detail_light(detail):
    def __init__(self, threshold):
        super().__init__(threshold=threshold, b=0)

    def set_b(self, b):
        self._b = b
        return True

    def solve_prepare_light(self):
        self._set_xyk()
        self._solve_prepare_xn()
        self._solve_prepare_yn()
        return True

    def solve_prepare_b(self):
        self._solve_prepare1()
        self._solve_prepare2()
        self._solve_prepare3()
        return True

    def solve_u1_light(self, R, Phi, z):
        err_msg = 'run _solve_prepare1 first. '
        assert self._finish1, err_msg

        AFPhi1nL = lambda xn, k, psi1, omega1, pi1, R, z, DmyD: (-2) * exp(1) ** ((-1) * z * imag(xn)) * pi * imag(
                DmyD ** (-1) * exp(1) ** (sqrt(-1 + 0j) * z * real(xn)) * (
                    (-1) * (omega1 + k * pi1) * iv((-1) + k, R * xn) + k * (
                        omega1 + pi1 + k * pi1 + (-1) * psi1) * R ** (
                        -1) * xn ** (-1) * iv(k, R * xn)))
        AFPhi1nR = lambda yn, k, psi1, omega1, pi1, R, z, DmyD: (-1) * exp(1) ** ((-1) * z * imag(yn)) * pi * imag(
                DmyD ** (-1) * (
                    (-1) * (omega1 + k * pi1) * iv((-1) + k, R * yn) + k * (
                        omega1 + pi1 + k * pi1 + (-1) * psi1) * R ** (
                        -1) * yn ** (-1) * iv(k, R * yn)))
        AFR1nL = lambda xn, k, psi1, omega1, pi1, R, z, DmyD: (-2) * exp(1) ** ((-1) * z * imag(xn)) * pi * imag(
                DmyD ** (-1) * exp(1) ** (sqrt(-1 + 0j) * z * real(xn)) * R ** (-1) * xn ** (-1) * (
                    ((-1) * pi1 + psi1) * R * xn * iv((-1) + k, R * xn) + (
                        k * (omega1 + pi1 + k * pi1 + (-1) * psi1) + pi1 * R ** 2 * xn ** 2) * iv(k, R * xn)))
        AFR1nR = lambda yn, k, psi1, omega1, pi1, R, z, DmyD: (-1) * exp(1) ** ((-1) * z * imag(yn)) * pi * imag(
                DmyD ** (-1) * R ** (-1) * yn ** (-1) * (
                    ((-1) * pi1 + psi1) * R * yn * iv((-1) + k, R * yn) + (
                        k * (omega1 + pi1 + k * pi1 + (-1) * psi1) + pi1 * R ** 2 * yn ** 2) * iv(k, R * yn)))
        BFz1nL = lambda xn, k, psi1, omega1, pi1, R, z, DmyD: (-2) * exp(1) ** ((-1) * z * imag(xn)) * pi * real(
                DmyD ** (-1) * exp(1) ** (sqrt(-1 + 0j) * z * real(xn)) * (
                    pi1 * R * xn * iv((-1) + k, R * xn) + (pi1 + (-1) * k * pi1 + psi1) * iv(k, R * xn)))
        BFz1nR = lambda yn, k, psi1, omega1, pi1, R, z, DmyD: (-1) * exp(1) ** ((-1) * z * imag(yn)) * pi * real(
                DmyD ** (-1) * (pi1 * R * yn * iv((-1) + k, R * yn) + (pi1 + (-1) * k * pi1 + psi1) * iv(k, R * yn)))
        uR1_k0 = lambda xn, psi1, omega1, pi1, R, z, DmyD: (-2) * exp(1) ** ((-1) * z * imag(xn)) * pi * imag(
                DmyD ** (-1) * exp(1) ** (sqrt(-1 + 0j) * z * real(xn)) * (
                    pi1 * R * xn * iv(0, R * xn) + ((-1) * pi1 + psi1) * iv(1, R * xn)))
        uz1_k0 = lambda xn, psi1, omega1, pi1, R, z, DmyD: (-2) * exp(1) ** ((-1) * z * imag(xn)) * pi * real(
                DmyD ** (-1) * exp(1) ** (sqrt(-1 + 0j) * z * real(xn)) * (
                    (pi1 + psi1) * iv(0, R * xn) + pi1 * R * xn * iv(1, R * xn)))

        uR1k0 = sum(uR1_k0(self._xn_k0, self._psi_xn1_k0, self._omega_xn1_k0, self._pi_xn1_k0, R, z, self._DmyD_xn_k0))
        uPhi1k0 = 0
        uz1k0 = sum(uz1_k0(self._xn_k0, self._psi_xn1_k0, self._omega_xn1_k0, self._pi_xn1_k0, R, z, self._DmyD_xn_k0))
        t_AFR1nL = AFR1nL(self._xn, self._k, self._psi_xn1, self._omega_xn1, self._pi_xn1, R, z, self._DmyD_xn)
        t_AFR1nR = AFR1nR(self._yn, self._k, self._psi_yn1, self._omega_yn1, self._pi_yn1, R, z, self._DmyD_yn)
        t_AFPhi1nL = AFPhi1nL(self._xn, self._k, self._psi_xn1, self._omega_xn1, self._pi_xn1, R, z, self._DmyD_xn)
        t_AFPhi1nR = AFPhi1nR(self._yn, self._k, self._psi_yn1, self._omega_yn1, self._pi_yn1, R, z, self._DmyD_yn)
        t_BFz1nL = BFz1nL(self._xn, self._k, self._psi_xn1, self._omega_xn1, self._pi_xn1, R, z, self._DmyD_xn)
        t_BFz1nR = BFz1nR(self._yn, self._k, self._psi_yn1, self._omega_yn1, self._pi_yn1, R, z, self._DmyD_yn)
        uR1 = uR1k0 + sum((t_AFR1nL + t_AFR1nR) * cos(self._k * Phi))
        uPhi1 = uPhi1k0 + sum((t_AFPhi1nL + t_AFPhi1nR) * sin(self._k * Phi))
        uz1 = uz1k0 + sum((t_BFz1nL + t_BFz1nR) * cos(self._k * Phi))
        return uR1, uPhi1, uz1

    def solve_u2_light(self, R, Phi, z):
        err_msg = 'run _solve_prepare2 first. '
        assert self._finish2, err_msg

        AFPhi2nL = lambda xn, k, psi2, omega2, pi2, R, z, DmyD: (-2) * exp(1) ** ((-1) * z * imag(xn)) * pi * imag(
                DmyD ** (-1) * exp(1) ** (sqrt(-1 + 0j) * z * real(xn)) * (
                    ((-1) * omega2 + k * pi2) * iv((-1) + k, R * xn) + k * (
                        omega2 + (-1) * (1 + k) * pi2 + psi2) * R ** (
                        -1) * xn ** (-1) * iv(k, R * xn)))
        AFPhi2nR = lambda yn, k, psi2, omega2, pi2, R, z, DmyD: (-1) * exp(1) ** ((-1) * z * imag(yn)) * pi * imag(
                DmyD ** (-1) * (
                    ((-1) * omega2 + k * pi2) * iv((-1) + k, R * yn) + k * (
                        omega2 + (-1) * (1 + k) * pi2 + psi2) * R ** (
                        -1) * yn ** (-1) * iv(k, R * yn)))
        AFR2nL = lambda xn, k, psi2, omega2, pi2, R, z, DmyD: (-2) * exp(1) ** ((-1) * z * imag(xn)) * pi * imag(
                DmyD ** (-1) * exp(1) ** (sqrt(-1 + 0j) * z * real(xn)) * R ** (-1) * xn ** (-1) * (
                    ((-1) * pi2 + psi2) * R * xn * iv((-1) + k, R * xn) + (
                        k * ((-1) * omega2 + pi2 + k * pi2 + (-1) * psi2) + pi2 * R ** 2 * xn ** 2) * iv(k, R * xn)))
        AFR2nR = lambda yn, k, psi2, omega2, pi2, R, z, DmyD: (-1) * exp(1) ** ((-1) * z * imag(yn)) * pi * imag(
                DmyD ** (-1) * R ** (-1) * yn ** (-1) * (
                    ((-1) * pi2 + psi2) * R * yn * iv((-1) + k, R * yn) + (
                        k * ((-1) * omega2 + pi2 + k * pi2 + (-1) * psi2) + pi2 * R ** 2 * yn ** 2) * iv(k, R * yn)))
        BFz2nL = lambda xn, k, psi2, omega2, pi2, R, z, DmyD: (-2) * exp(1) ** ((-1) * z * imag(xn)) * pi * real(
                DmyD ** (-1) * exp(1) ** (sqrt(-1 + 0j) * z * real(xn)) * (
                    pi2 * R * xn * iv((-1) + k, R * xn) + (pi2 + (-1) * k * pi2 + psi2) * iv(k, R * xn)))
        BFz2nR = lambda yn, k, psi2, omega2, pi2, R, z, DmyD: (-1) * exp(1) ** ((-1) * z * imag(yn)) * pi * real(
                DmyD ** (-1) * (pi2 * R * yn * iv((-1) + k, R * yn) + (pi2 + (-1) * k * pi2 + psi2) * iv(k, R * yn)))
        uPhi2_k0 = lambda yn, psi2, omega2, pi2, R, z, DmyD: exp(1) ** ((-1) * z * imag(yn)) * pi * imag(
                DmyD ** (-1) * omega2 * iv(1, R * yn))

        uR2k0 = 0
        uPhi2k0 = sum(
                uPhi2_k0(self._yn_k0, self._psi_yn2_k0, self._omega_yn2_k0, self._pi_yn2_k0, R, z, self._DmyD_yn_k0))
        uz2k0 = 0
        t_AFR2nL = AFR2nL(self._xn, self._k, self._psi_xn2, self._omega_xn2, self._pi_xn2, R, z, self._DmyD_xn)
        t_AFR2nR = AFR2nR(self._yn, self._k, self._psi_yn2, self._omega_yn2, self._pi_yn2, R, z, self._DmyD_yn)
        t_AFPhi2nL = AFPhi2nL(self._xn, self._k, self._psi_xn2, self._omega_xn2, self._pi_xn2, R, z, self._DmyD_xn)
        t_AFPhi2nR = AFPhi2nR(self._yn, self._k, self._psi_yn2, self._omega_yn2, self._pi_yn2, R, z, self._DmyD_yn)
        t_BFz2nL = BFz2nL(self._xn, self._k, self._psi_xn2, self._omega_xn2, self._pi_xn2, R, z, self._DmyD_xn)
        t_BFz2nR = BFz2nR(self._yn, self._k, self._psi_yn2, self._omega_yn2, self._pi_yn2, R, z, self._DmyD_yn)
        uR2 = uR2k0 + sum((t_AFR2nL + t_AFR2nR) * sin(self._k * Phi))
        uPhi2 = uPhi2k0 + sum((t_AFPhi2nL + t_AFPhi2nR) * cos(self._k * Phi))
        uz2 = uz2k0 + sum((t_BFz2nL + t_BFz2nR) * sin(self._k * Phi))
        return uR2, uPhi2, uz2

    def solve_u3_light(self, R, Phi, z):
        err_msg = 'run _solve_prepare3 first. '
        assert self._finish3, err_msg

        BFPhi3nL = lambda xn, k, psi3, omega3, pi3, R, z, DmyD: 2 * exp(1) ** ((-1) * z * imag(xn)) * pi * real(
                DmyD ** (-1) * exp(1) ** (sqrt(-1 + 0j) * z * real(xn)) * (
                    (-1) * (omega3 + k * pi3) * iv((-1) + k, R * xn) + k * (
                        omega3 + pi3 + k * pi3 + (-1) * psi3) * R ** (
                        -1) * xn ** (-1) * iv(k, R * xn)))
        BFPhi3nR = lambda yn, k, psi3, omega3, pi3, R, z, DmyD: exp(1) ** ((-1) * z * imag(yn)) * pi * real(
                DmyD ** (-1) * (
                    (-1) * (omega3 + k * pi3) * iv((-1) + k, R * yn) + k * (
                        omega3 + pi3 + k * pi3 + (-1) * psi3) * R ** (
                        -1) * yn ** (-1) * iv(k, R * yn)))
        BFR3nL = lambda xn, k, psi3, omega3, pi3, R, z, DmyD: 2 * exp(1) ** ((-1) * z * imag(xn)) * pi * real(
                DmyD ** (-1) * exp(1) ** (sqrt(-1 + 0j) * z * real(xn)) * R ** (-1) * xn ** (-1) * (
                    ((-1) * pi3 + psi3) * R * xn * iv((-1) + k, R * xn) + (
                        k * (omega3 + pi3 + k * pi3 + (-1) * psi3) + pi3 * R ** 2 * xn ** 2) * iv(k, R * xn)))
        BFR3nR = lambda yn, k, psi3, omega3, pi3, R, z, DmyD: exp(1) ** ((-1) * z * imag(yn)) * pi * real(
                DmyD ** (-1) * R ** (-1) * yn ** (-1) * (
                    ((-1) * pi3 + psi3) * R * yn * iv((-1) + k, R * yn) + (
                        k * (omega3 + pi3 + k * pi3 + (-1) * psi3) + pi3 * R ** 2 * yn ** 2) * iv(k, R * yn)))
        AFz3nL = lambda xn, k, psi3, omega3, pi3, R, z, DmyD: (-2) * exp(1) ** ((-1) * z * imag(xn)) * pi * imag(
                DmyD ** (-1) * exp(1) ** (sqrt(-1 + 0j) * z * real(xn)) * (
                    pi3 * R * xn * iv((-1) + k, R * xn) + (pi3 + (-1) * k * pi3 + psi3) * iv(k, R * xn)))
        AFz3nR = lambda yn, k, psi3, omega3, pi3, R, z, DmyD: (-1) * exp(1) ** ((-1) * z * imag(yn)) * pi * imag(
                DmyD ** (-1) * (pi3 * R * yn * iv((-1) + k, R * yn) + (pi3 + (-1) * k * pi3 + psi3) * iv(k, R * yn)))
        uR3_k0 = lambda xn, psi3, omega3, pi3, R, z, DmyD: 2 * exp(1) ** ((-1) * z * imag(xn)) * pi * real(
                DmyD ** (-1) * exp(1) ** (sqrt(-1 + 0j) * z * real(xn)) * (
                    pi3 * R * xn * iv(0, R * xn) + ((-1) * pi3 + psi3) * iv(1, R * xn)))
        uz3_k0 = lambda xn, psi3, omega3, pi3, R, z, DmyD: (-2) * exp(1) ** ((-1) * z * imag(xn)) * pi * imag(
                DmyD ** (-1) * exp(1) ** (sqrt(-1 + 0j) * z * real(xn)) * (
                    (pi3 + psi3) * iv(0, R * xn) + pi3 * R * xn * iv(1, R * xn)))

        uR3k0 = sum(uR3_k0(self._xn_k0, self._psi_xn3_k0, self._omega_xn3_k0, self._pi_xn3_k0, R, z, self._DmyD_xn_k0))
        uPhi3k0 = 0
        uz3k0 = sum(uz3_k0(self._xn_k0, self._psi_xn3_k0, self._omega_xn3_k0, self._pi_xn3_k0, R, z, self._DmyD_xn_k0))
        t_BFR3nL = BFR3nL(self._xn, self._k, self._psi_xn3, self._omega_xn3, self._pi_xn3, R, z, self._DmyD_xn)
        t_BFR3nR = BFR3nR(self._yn, self._k, self._psi_yn3, self._omega_yn3, self._pi_yn3, R, z, self._DmyD_yn)
        t_BFPhi3nL = BFPhi3nL(self._xn, self._k, self._psi_xn3, self._omega_xn3, self._pi_xn3, R, z, self._DmyD_xn)
        t_BFPhi3nR = BFPhi3nR(self._yn, self._k, self._psi_yn3, self._omega_yn3, self._pi_yn3, R, z, self._DmyD_yn)
        t_AFz3nL = AFz3nL(self._xn, self._k, self._psi_xn3, self._omega_xn3, self._pi_xn3, R, z, self._DmyD_xn)
        t_AFz3nR = AFz3nR(self._yn, self._k, self._psi_yn3, self._omega_yn3, self._pi_yn3, R, z, self._DmyD_yn)
        uR3 = uR3k0 + sum((t_BFR3nL + t_BFR3nR) * cos(self._k * Phi))
        uPhi3 = uPhi3k0 + sum((t_BFPhi3nL + t_BFPhi3nR) * sin(self._k * Phi))
        uz3 = uz3k0 + sum((t_AFz3nL + t_AFz3nR) * cos(self._k * Phi))
        return uR3, uPhi3, uz3

    def solve_u_light(self, R, Phi, z):
        uR1, uPhi1, uz1 = self.solve_u1_light(R, Phi, z)
        uR2, uPhi2, uz2 = self.solve_u2_light(R, Phi, z)
        uR3, uPhi3, uz3 = self.solve_u3_light(R, Phi, z)
        return uR1, uPhi1, uz1, uR2, uPhi2, uz2, uR3, uPhi3, uz3

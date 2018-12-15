import sphere_rs
import cProfile
from petsc4py import PETSc
import numpy as np
from numpy import pi

# deltaLength = 0.05 ** np.arange(0.25, 1.05, 0.1)
# n = np.rint(16 / deltaLength / deltaLength)
# OptDB = PETSc.Options()
# OptDB.setValue('e', 3)
# OptDB.setValue('legendre_m', 3)
# OptDB.setValue('legendre_k', 3)
# sphere_rs.two_step()

# cProfile.run('sphere_rs.main_fun()')

deltaLength = np.array((0.1, 0.15, 0.2, 0.25, 0.3))
epsilon = np.arange(0.7, 3, 0.2)
# deltaLength = np.array((0.25, 0.5))
# epsilon = np.array((0.1, 0.25))
deltaLength, epsilon = np.meshgrid(deltaLength, epsilon)
deltaLength = deltaLength.flatten()
epsilon = epsilon.flatten()
err = np.zeros((epsilon.size))
ini_err = err.copy()
residualNorm = epsilon.copy()
ini_residualNorm = epsilon.copy()
OptDB = PETSc.Options()
for i0 in range(epsilon.size):
    d = deltaLength[i0]
    e = epsilon[i0]
    fileHandle = 'sphereInTunnel_%d_%f_%f' % (i0, d, e)
    PETSc.Sys.Print(fileHandle)

# def two_para_regularized_stokeslets_matrix_3d(vnodes,
#                                               fnodes,
#                                               **kwargs):
#     h1 = {'1': lambda r, e: (1 / 8) * pi ** (-1) * (e ** 2 + r ** 2) ** (-3 / 2) * (2 * e ** 2 + r ** 2),
#           '2': lambda r, e: (1 / 32) * pi ** (-1) * (e ** 2 + r ** 2) ** (-5 / 2) * (10 * e ** 4 + 11 * e ** 2 * r ** 2 + 4 * r ** 4)}
#     h2 = {'1': lambda r, e: (1 / 8) * pi ** (-1) * (e ** 2 + r ** 2) ** (-3 / 2),
#           '2': lambda r, e: (1 / 32) * pi ** (-1) * (e ** 2 + r ** 2) ** (-5 / 2) * (7 * e ** 2 + 4 * r ** 2)}
#
#     e = kwargs['delta']
#     n = str(kwargs['twoPara_n'])
#     n_vnode = vnodes.size
#     n_fnode = fnodes.size
#
#     n_unknown = 3
#     m = np.zeros((n_vnode, n_fnode))
#     for i0 in range(n_vnode):
#         delta_xi = fnodes - vnodes[i0 // 3]
#         temp1 = delta_xi ** 2
#         delta_r = np.sqrt(temp1.sum(axis=1))
#         i = i0 % 3
#         for j in range(3):
#             m[i0, j::n_unknown] = int(i == j) * h1[n](delta_r, e) + delta_xi[:, i] * delta_xi[:, j] * h2[n](delta_r, e)
#     return m
#
#
# if __name__ == '__main__':
#     n = 10
#     vnodes = np.random.sample((n, 3))
#     fnodes = vnodes.copy()
#     kwargs = {'delta': 0.01,
#               'twoPara_n': 1}
#     m = two_para_regularized_stokeslets_matrix_3d(vnodes, fnodes, **kwargs)
pass

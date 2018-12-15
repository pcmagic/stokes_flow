import sys
import petsc4py

petsc4py.init(sys.argv)

from petsc4py import PETSc
from src.geo import *
import numpy as np
from src.StokesFlowMethod import light_stokeslets_matrix_3d
from scipy.sparse.linalg import gmres
from time import time

tunn1 = geo.tunnel_geo()
tunn1.create_deltatheta(2 * np.pi / 30, 1, 3, with_cover=1)
tunn2 = tunn1.copy()
tunn2.node_zoom_radius(0.8)
geo_com = geo.geoComposit([tunn1, tunn2])
fig1 = geo_com.show_nodes()

# class gmres_counter(object):
#     def __init__(self, disp=True):
#         self._disp = disp
#         self.niter = 0
#
#     def __call__(self, rk=None):
#         self.niter += 1
#         if self._disp:
#             print('iter %3i\trk = %s' % (self.niter, str(rk)))
#
#
# OptDB = PETSc.Options()
# nth = OptDB.getInt('nth', 3)
# t0 = time()
#
# geo1 = tunnel_geo()
# geo3 = tunnel_geo()
# dth = np.pi * 2 / nth
# geo2 = geo1.create_deltatheta(dth=dth, radius=0.2, length=1, epsilon=1, with_cover=1, factor=1)
# geo1.set_rigid_velocity(np.ones(6))
# geo3.create_deltatheta(dth=dth * 0.6, radius=0.2, length=1, epsilon=1, with_cover=1, factor=1)
# geo3.set_rigid_velocity(np.ones(6))
# # geo1.show_nodes()
# # geo2.show_nodes()
# m1 = light_stokeslets_matrix_3d(geo1.get_nodes(), geo2.get_nodes())
# m3 = light_stokeslets_matrix_3d(geo3.get_nodes(), geo2.get_nodes())
# u1 = geo1.get_velocity()
# u3 = geo3.get_velocity()
# counter = gmres_counter()
# itr = 1000
# # itr = 100
# print('nth=%d, # of nodes=%d' % (nth, u1.size))
# f, info = gmres(m1, u1, restrt=itr, maxiter=itr, tol=1e-100, callback=counter)
# t1 = time()
# print(counter.niter)
# print('nth=%d, # of nodes=%d, use time=%f' % (nth, u1.size, (t1 - t0)))
# print(info, np.linalg.norm((np.dot(m1, f) - u1) / u1), np.linalg.norm((np.dot(m3, f) - u3) / u3))
#
# delta = (99 / 2) * pi ** (-1) * ((-1) + r) ** 3 * (
#         (-5) + (-15) * r + (-30) * r ** 2 + 1126 * r ** 3 + (-2847) * r ** 4 + 1911 * r ** 5)
# delta = (99 / 2) * epsilon ** (-11) * np.pi ** (-1) * (epsilon + (-1) * r) ** 3 * (
#         5 * epsilon ** 5 + 15 * epsilon ** 4 * r + 30 * epsilon ** 3 * r ** 2 + (
#     -1126) * epsilon ** 2 * r ** 3 + 2847 * epsilon * r ** 4 + (-1911) * r ** 5)
# delta = (6006 / 5) * epsilon ** (-13) * np.pi ** (-1) * (epsilon + (-1) * r) ** 3 * (
#             epsilon ** 7 + 3 * epsilon ** 6 * r + 6 * epsilon ** 5 * r ** 2 + (
#         -980) * epsilon ** 4 * r ** 3 + 5955 * epsilon ** 3 * r ** 4 + (
#                 -13938) * epsilon ** 2 * r ** 5 + 14416 * epsilon * r ** 6 + (-5508) * r ** 7)

h1a = lambda r, e: (1 / 40) * e ** (-11) * np.pi ** (-1) * (88 * e ** 10 + r ** 2 * ((-1320) * e ** 8 + r ** 3 * (
            67914 * e ** 5 + r * ((-264000) * e ** 4 + r * (
                441045 * e ** 3 + r * ((-385000) * e ** 2 + (173030 * e + (-31752) * r) * r))))))
h2a = lambda r, e: (1 / 8) * e ** (-11) * np.pi ** (-1) * (132 * e ** 8 + r ** 3 * ((-9702) * e ** 5 + r * (
            39600 * e ** 4 + r * ((-68607) * e ** 3 + r * (61600 * e ** 2 + r * ((-28314) * e + 5292 * r))))))

h1b = lambda r, e: (1 / 40) * e ** (-11) * np.pi ** (-1) * ((-31752) * r ** 10 + e * (173030 * r ** 9 + e * (
            (-385000) * r ** 8 + e * (441045 * r ** 7 + e * (
                (-264000) * r ** 6 + e * (67914 * r ** 5 + e ** 3 * (88 * e ** 2 + (-1320) * r ** 2)))))))
h2b = lambda r, e: (1 / 8) * e ** (-11) * np.pi ** (-1) * (5292 * r ** 8 + e * ((-28314) * r ** 7 + e * (
            61600 * r ** 6 + e * ((-68607) * r ** 5 + e * (39600 * r ** 4 + e * (132 * e ** 3 + (-9702) * r ** 3))))))

h1c = lambda r, e: (1 / 40) * e ** (-11) * np.pi ** (-1) * (88 * e ** 10 + r ** 2 * (
        (-1320) * e ** 8 + r ** 3 * (67914 * e ** 5 + r * (
        (-264000) * e ** 4 + r * (
        441045 * e ** 3 + r * ((-385000) * e ** 2 + (173030 * e + (-31752) * r) * r))))))
h2c = lambda r, e: (1 / 8) * e ** (-11) * np.pi ** (-1) * (
        132 * e ** 8 + r ** 3 * ((-9702) * e ** 5 + r * (
        39600 * e ** 4 + r * ((-68607) * e ** 3 + r * (61600 * e ** 2 + r * ((-28314) * e + 5292 * r))))))

import sys
import petsc4py

petsc4py.init(sys.argv)

from petsc4py import PETSc
from src.geo import *
import numpy as np
from src.StokesFlowMethod import light_stokeslets_matrix_3d
from scipy.sparse.linalg import gmres
from time import time


class gmres_counter(object):
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0

    def __call__(self, rk=None):
        self.niter += 1
        if self._disp:
            print('iter %3i\trk = %s' % (self.niter, str(rk)))


OptDB = PETSc.Options()
nth = OptDB.getInt('nth', 3)
t0 = time()

geo1 = tunnel_geo()
geo3 = tunnel_geo()
dth = np.pi * 2 / nth
geo2 = geo1.create_deltatheta(dth=dth, radius=0.2, length=1, epsilon=1, with_cover=True, factor=1)
geo1.set_rigid_velocity(np.ones(6))
geo3.create_deltatheta(dth=dth * 0.6, radius=0.2, length=1, epsilon=1, with_cover=True, factor=1)
geo3.set_rigid_velocity(np.ones(6))
# geo1.show_nodes()
# geo2.show_nodes()
m1 = light_stokeslets_matrix_3d(geo1.get_nodes(), geo2.get_nodes())
m3 = light_stokeslets_matrix_3d(geo3.get_nodes(), geo2.get_nodes())
u1 = geo1.get_velocity()
u3 = geo3.get_velocity()
counter = gmres_counter()
itr = 1000
# itr = 100
print('nth=%d, # of nodes=%d' % (nth, u1.size))
f, info = gmres(m1, u1, restrt=itr, maxiter=itr, tol=1e-100, callback=counter)
t1 = time()
print(counter.niter)
print('nth=%d, # of nodes=%d, use time=%f' % (nth, u1.size, (t1 - t0)))
print(info, np.linalg.norm((np.dot(m1, f) - u1) / u1), np.linalg.norm((np.dot(m3, f) - u3) / u3))

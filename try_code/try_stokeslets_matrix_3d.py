import numpy as np

from src.StokesFlowMethod import stokeslets_matrix_3d
from src.geo import *
from src.stokes_flow import StokesFlowObj

n = 6000
r1 = 1
r2 = 2
geo1 = sphere_geo()
geo1.create_n(n, r1)
geo2 = sphere_geo()
geo2.create_n(n, r2)
obj = StokesFlowObj()
obj.set_data(geo1, geo2)
m = stokeslets_matrix_3d(obj, obj)
# print(n, m.shape)
print(m[3:6, 3:6])

from os import path as ospath
import sys
import numpy as np

t_path = sys.path[0]
t_path = ospath.dirname(t_path)
if ospath.isdir(t_path):
    sys.path = [t_path] + sys.path
else:
    err_msg = "can not add path father path"
    raise ValueError(err_msg)
from src.StokesFlowMethod import stokeslets_matrix_3d
from src.geo import *
from src.stokes_flow import stokesFlowObj

n = 6000
r1 = 1
r2 = 2
geo1 = sphere_geo()
geo1.create_n(n, r1)
geo2 = sphere_geo()
geo2.create_n(n, r2)
obj = stokesFlowObj()
obj.set_data(geo1, geo2)
m = stokeslets_matrix_3d(obj, obj)
# print(n, m.shape)
print(m[3:6, 3:6])
# 20170807
# stokeslets in two plate case
# zhang fan, zhang ji


import sys
import petsc4py

petsc4py.init(sys.argv)
from os import path as ospath

t_path = sys.path[0]
t_path = ospath.dirname(t_path)
if ospath.isdir(t_path):
    sys.path = [t_path] + sys.path
else:
    err_msg = "can not add path father path"
    raise ValueError(err_msg)

import numpy as np
from src import stokes_flow as sf
from src.stokes_flow import problem_dic, obj_dic
from petsc4py import PETSc
from src.geo import *
from time import time
import pickle
from scipy.io import savemat
from src.ref_solution import *
from scipy.io import loadmat
import warnings
from memory_profiler import profile
from src.stokesTwoPlate import tank


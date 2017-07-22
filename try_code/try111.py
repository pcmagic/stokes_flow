# coding=utf-8
# try codes, call functions at stokes_flow.py

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
# from src.ref_solution import *
from scipy.io import loadmat
import warnings
from memory_profiler import profile
from petsc4py.PETSc import Sys

ugeo = geo()
ugeo.set_nodes((0, 0, 0), deltalength=0)
t_u_pkg = PETSc.DMComposite().create(comm=PETSc.COMM_WORLD)
t_u_pkg.addDM(ugeo.get_dmda())
t_u_pkg.setFromOptions()
t_u_pkg.setUp()
u_isglb = t_u_pkg.getGlobalISs()
ugeo.set_glbIdx(u_isglb[0].getIndices())
_, u_glbIdx_all = ugeo.get_glbIdx()
print(type(u_isglb[0].getIndices()))
print(u_isglb[0].getIndices().shape)
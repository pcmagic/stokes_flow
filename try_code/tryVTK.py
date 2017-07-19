# coding=utf-8

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
from pyvtk import *
import numpy as np
from scipy.io import loadmat
from src import stokes_flow as sf
from src.stokes_flow import problem_dic, obj_dic
from src.geo import *


def main_fun():
    matname = 'around'
    if matname[-4:] != '.mat':
        matname = matname + '.mat'
    bnodesHeadle = 'bnodes'
    belemsHeadle = 'belems'
    fileHeadle = 'tryVTK'

    bgeo = geo()
    bgeo.mat_nodes(filename=matname, mat_handle=bnodesHeadle)
    bgeo.mat_elmes(filename=matname, mat_handle=belemsHeadle, elemtype='tetra')

    bnodes = bgeo.get_nodes()
    belems, elemtype = bgeo.get_mesh()
    err_msg = 'mesh type is NOT tetrahedron. '
    assert elemtype == 'tetra', err_msg

    u = bnodes
    vtk = VtkData(
            UnstructuredGrid(bnodes,
                             tetra=belems,
                             ),
            PointData(Vectors(u, name='velocity')),
            ' '
    )
    vtk.tofile(fileHeadle)

if __name__ == '__main__':
    main_fun()

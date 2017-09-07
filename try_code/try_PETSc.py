# coding=utf-8
# main codes, call functions at stokes_flow.py
# Zhang Ji, 20170519

import sys
import petsc4py

petsc4py.init(sys.argv)

import numpy as np
# from src import stokes_flow as sf
# from src.stokes_flow import problem_dic, obj_dic
from petsc4py import PETSc
# from src.geo import *
from time import time
import pickle
from scipy.io import savemat
# from src.ref_solution import *
from scipy.io import loadmat
import warnings
from memory_profiler import profile
from petsc4py.PETSc import Sys


def main_fun(**main_kwargs):
    n1 = 5
    n2 = 5
    dof1 = 3
    dof2 = 3
    stencil_width = 1

    comm = PETSc.COMM_WORLD.tompi4py()
    rank = comm.Get_rank()
    da1 = PETSc.DMDA().create(sizes=(n1,), dof=dof1, stencil_width=stencil_width, comm=PETSc.COMM_WORLD)
    # coord1 = np.array(((0, 0, 0),
    #                    (0, 0, 1),
    #                    (0, 1, 0),
    #                    (1, 0, 0),
    #                    (0, 1, 1))).flatten()
    coord1 = np.arange(15)
    c1 = PETSc.Vec().createWithArray(coord1)
    da1.setCoordinates(c1)
    da1.setFromOptions()
    da1.setUp()
    da2 = PETSc.DMDA().create(sizes=(n2,), dof=dof2, stencil_width=stencil_width, comm=PETSc.COMM_WORLD)
    c2 = PETSc.Vec().createWithArray(coord1 + 0.1)
    da2.setCoordinates(c2)
    da2.setFromOptions()
    da2.setUp()
    da_pkg = PETSc.DMComposite().create(comm=PETSc.COMM_WORLD)
    da_pkg.addDM(da1)
    da_pkg.addDM(da2)
    da_pkg.setFromOptions()
    da_pkg.setUp()
    isglb = da_pkg.getGlobalISs()
    isloc = da_pkg.getLocalISs()
    lgmap = da_pkg.getLGMap()

    X = da_pkg.createGlobalVector()
    PETSc.Sys.Print(X.getSizes()[1])
    B = da_pkg.createGlobalVector()
    M = PETSc.Mat().create(comm=PETSc.COMM_WORLD)
    M.setSizes((X.getSizes(), B.getSizes()))
    M.setType('dense')
    M.setFromOptions()
    M.setUp()
    M.setLGMap(lgmap, lgmap)

    X1 = da1.createGlobalVector()
    # PETSc.Sys.Print(range(da1.getRanges()[0][0], da1.getRanges()[0][1]))
    # PETSc.Sys.Print(da1.getSizes()[0])
    # c1 = da1.getCoordinates()
    # c1.view()
    # c1_loc = da1.getCoordinatesLocal()
    # if rank == 3:
    #     c1_loc.view()
    # X2 = da2.createGlobalVector()
    # c2 = da2.getCoordinates()
    # c2.view()
    # c2_loc = da2.getCoordinatesLocal()
    # c2_loc.view()

    # for i0 in isglb[0].getIndices():
    #     for i1 in data:
    #         M.setValue(i0, i1, 110)
    # for i0 in isglb[1].getIndices():
    #     for i1 in data:
    #         M.setValue(i0, i1, 210)
    # data = np.hstack(comm.allgather(isglb[1].getIndices()))
    # for i0 in isglb[0].getIndices():
    #     for i1 in data:
    #         M.setValue(i0, i1, 120)
    # for i0 in isglb[1].getIndices():
    #     for i1 in data:
    #         M.setValue(i0, i1, 220)
    # M.assemble()
    # myview = PETSc.Viewer().createASCII('M.txt', 'w')
    # myview(M)


    return True


if __name__ == '__main__':
    main_fun()

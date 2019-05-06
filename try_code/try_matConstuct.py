# coding=utf-8
import sys

import numpy as np
import petsc4py
from mpi4py import MPI
from petsc4py import PETSc

petsc4py.init(sys.argv)
mSizes = (2, 2)
mij = uniqueList()

# # create sub-matrices mij
# for i in range(len(mSizes)):
#     for j in range(len(mSizes)):
#         temp_m = PETSc.Mat().create(comm=PETSc.COMM_WORLD)
#         temp_m.setSizes(((None, mSizes[i]), (None, mSizes[j])))
#         temp_m.setType('dense')
#         temp_m.setFromOptions()
#         temp_m.setUp()
#         temp_m[:, :] = np.random.random_sample((mSizes[i], mSizes[j]))
#         temp_m.assemble()
#         temp_m.view()
#         mij.append(temp_m)
#
# # Now we have four sub-matrices. I would like to construct them into a big matrix M.
# M = PETSc.Mat().create(comm=PETSc.COMM_WORLD)
# M.setSizes(((None, np.sum(mSizes)), (None, np.sum(mSizes))))
# M.setType('dense')
# M.setFromOptions()
# M.setUp()
# mLocations = np.insert(np.cumsum(mSizes), 0, [0])  # mLocations = [0, mSizes]
# for i in range(len(mSizes)):
#     for j in range(len(mSizes)):
#         temp_m = mij[i * len(mSizes) + j].getDenseArray()
#         temp_m_start, temp_m_end = mij[i * len(mSizes) + j].getOwnershipRange()
#         rank = MPI.COMM_WORLD.Get_rank()
#         PETSc.Sys.Print('rank:', rank, '   ', i, '   ', j, '   ', temp_m, '   ', temp_m.shape, '   ', temp_m_start, '   ',
#               temp_m_end)
#         for k in range(temp_m_start, temp_m_end):
#             # PETSc.Sys.Print('i: %d, j: %d, k: %d, mLocations[i]: %d'%(i, j, k, mLocations[i]))
#             M.setValues(mLocations[i] + k, np.arange(mLocations[j], mLocations[j + 1], dtype='int32'),
#                         temp_m[k - temp_m_start, :])
# M.assemble()
# M.view()

temp_m_petsc = PETSc.Mat().create(comm=PETSc.COMM_WORLD)
temp_m_petsc.setSizes(((None, mSizes[0]), (None, mSizes[1])))
temp_m_petsc.setType('dense')
temp_m_petsc.setFromOptions()
temp_m_petsc.setUp()
temp_m_petsc[:, :] = np.array(((3, 0), (0, 2)))
temp_m_petsc.assemble()
# temp_m_petsc.view()

# I_petsc = PETSc.Mat().create(comm=PETSc.COMM_WORLD)
# temp_m_petsc.convert(mat_type='dense', out=I_petsc)
# I_petsc.zeroEntries()
# I_petsc.shift(alpha=1)
# I_petsc.view()
#
# inv_petsc = PETSc.Mat().create(comm=PETSc.COMM_WORLD)
# temp_m_petsc.convert(mat_type='dense', out=inv_petsc)
# rperm, cperm = temp_m_petsc.getOrdering(ord_type='MATORDERINGNATURAL')
# temp_m_petsc.factorLU(rperm, cperm)
# temp_m_petsc.matSolve(I_petsc, inv_petsc)
# inv_petsc.view()

a = PETSc.Mat().create(comm=PETSc.COMM_WORLD)
temp_m_petsc.convert(mat_type='dense', out=a)
a[:, :] = np.array(((1, 2), (3, 4)))
a.assemble()
b = PETSc.Mat().create(comm=PETSc.COMM_WORLD)
temp_m_petsc.convert(mat_type='dense', out=b)
b[:, :] = np.array(((5, 6), (7, 8)))
b.assemble()
c = PETSc.Mat().create(comm=PETSc.COMM_WORLD)
temp_m_petsc.convert(mat_type='dense', out=c)
a.matMult(b, c)
a.view()
b.view()
c.view()
b.matMult(a, c)
c.view()
b.matMult(a, c)
c.view()

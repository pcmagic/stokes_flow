import sys
import petsc4py

petsc4py.init(sys.argv)
from ecoli_in_pipe import head_tail

# import numpy as np
# from scipy.interpolate import interp1d
# from petsc4py import PETSc
# from ecoli_in_pipe import single_ecoli, ecoliInPipe, head_tail, ecoli_U
# from codeStore import ecoli_common
#
#
# def call_head_tial(uz_factor=1., wz_factor=1.):
#     PETSc.Sys.Print('')
#     PETSc.Sys.Print('################################################### uz_factor = %f, wz_factor = %f' %
#                     (uz_factor, wz_factor))
#     t_head_U = head_U.copy()
#     t_tail_U = tail_U.copy()
#     t_head_U[2] = t_head_U[2] * uz_factor
#     t_tail_U[2] = t_tail_U[2] * uz_factor
#     # C1 = t_head_U[5] - t_tail_U[5]
#     # C2 = t_head_U[5] / t_tail_U[5]
#     # t_head_U[5] = wz_factor * C1 * C2 / (wz_factor * C2 - 1)
#     # t_tail_U[5] = C1 / (wz_factor * C2 - 1)
#     t_head_U[5] = wz_factor * t_head_U[5]
#     t_kwargs = {'head_U': t_head_U,
#                 'tail_U': t_tail_U, }
#     total_force = head_tail.main_fun()
#     return total_force
#
#
# OptDB = PETSc.Options()
# fileHandle = OptDB.getString('f', 'ecoliInPipe')
# OptDB.setValue('f', fileHandle)
# main_kwargs = {'fileHandle': fileHandle}
# # head_U, tail_U, ref_U = ecoli_common.ecoli_restart(**main_kwargs)
# # ecoli_common.ecoli_restart(**main_kwargs)
# head_U = np.array([0, 0, 1, 0, 0, 1])
# tail_U = np.array([0, 0, 1, 0, 0, 1])
# call_head_tial()

head_tail.main_fun()

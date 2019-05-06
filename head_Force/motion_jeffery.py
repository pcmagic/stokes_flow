# coding=utf-8

import sys
import petsc4py

petsc4py.init(sys.argv)

import numpy as np
from time import time
from scipy.io import savemat
# from src.stokes_flow import problem_dic, obj_dic
# from src.geo import *
# from petsc4py import PETSc
# from src import stokes_flow as sf
# from src.myio import *
# from src.StokesFlowMethod import light_stokeslets_matrix_3d
from src.support_class import *
# from src.objComposite import createEcoliComp_ellipse
# from src.myvtk import save_singleEcoli_vtk
# import ecoli_in_pipe.ecoli_common as ec
# import os
from src.jeffery_model import *
from src import jeffery_model as jm
from codeStore import support_fun as spf


# import import_my_lib

# # Todo: rewrite input and print process.
# def get_problem_kwargs(**main_kwargs):
#     OptDB = PETSc.Options()
#     fileHandle = OptDB.getString('f', 'motion_ecoli')
#     OptDB.setValue('f', fileHandle)
#     problem_kwargs = ec.get_problem_kwargs()
#     problem_kwargs['fileHandle'] = fileHandle
#
#     kwargs_list = (get_shearFlow_kwargs(), main_kwargs,)
#     for t_kwargs in kwargs_list:
#         for key in t_kwargs:
#             problem_kwargs[key] = t_kwargs[key]
#
#     # vtk_matname = OptDB.getString('vtk_matname', 'pipe_dbg')
#     # t_path = os.path.dirname(os.path.abspath(__file__))
#     # vtk_matname = os.path.normpath(os.path.join(t_path, vtk_matname))
#     # problem_kwargs['vtk_matname'] = vtk_matname
#     max_iter = OptDB.getInt('max_iter', 10)
#     update_order = OptDB.getInt('update_order', 1)
#     eval_dt = OptDB.getReal('eval_dt', 0.1)
#     problem_kwargs['max_iter'] = max_iter
#     problem_kwargs['update_order'] = update_order
#     problem_kwargs['eval_dt'] = eval_dt
#     return problem_kwargs
#
#
# def print_case_info(**problem_kwargs):
#     caseIntro = '-->Ecoli in infinite shear flow case, force free case, use regularized Stokeslets. '
#     ec.print_case_info(caseIntro, **problem_kwargs)
#
#     max_iter = problem_kwargs['max_iter']
#     update_order = problem_kwargs['update_order']
#     eval_dt = problem_kwargs['eval_dt']
#     PETSc.Sys.Print('Iteration Loop: max_iter %d, update_order %d, eval_dt %f' %
#                     (max_iter, update_order, eval_dt))
#
#     print_shearFlow_info(**problem_kwargs)
#     return True
#

# @profile
def main_fun(**main_kwargs):
    alpha = 1
    eval_dt = 0.001
    max_iter = 10
    fileHandle = 'SingleStokesletsJefferyProblem'

    #     norm = np.random.sample(3)
    norm = np.array((1, 0, 0))
    center = np.array((1, 1, 1))
    ellipse_kwargs = {'name':     'ellipse0',
                      'center':   center,
                      'norm':     norm / np.linalg.norm(norm),
                      'velocity': 0.000,
                      'lbd':      (alpha ** 2 - 1) / (alpha ** 2 + 1)}
    ellipse_obj = jm.JefferyObj(**ellipse_kwargs)
    problem = jm.SingleStokesletsJefferyProblem(StokesletsStrength=(1, 0, 0))
    problem.add_obj(ellipse_obj)

    # evaluation loop
    t0 = time()
    for idx in range(1, max_iter + 1):
        problem.update_location(eval_dt, print_handle='%d / %d' % (idx, max_iter))
    t1 = time()
    print('%s: run %d loops using %f' % (fileHandle, max_iter, (t1 - t0)))
    #     print(alpha, norm, center, problem.planeShearRate)
    #     print(norm / np.linalg.norm(norm))
    #     print(np.vstack(ellipse_obj.norm_hist))

    center_hist = np.vstack(ellipse_obj.center_hist)
    U_hist = np.vstack(ellipse_obj.U_hist)
    norm_hist = np.vstack(ellipse_obj.norm_hist)
    print(norm_hist)
    return True


if __name__ == '__main__':
    main_fun()

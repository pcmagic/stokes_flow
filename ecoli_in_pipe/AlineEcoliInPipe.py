# coding=utf-8

import sys

import petsc4py

petsc4py.init(sys.argv)

import numpy as np
# import pickle
# from time import time
# from scipy.io import loadmat
# from src.stokes_flow import problem_dic, obj_dic
# from src.geo import *
from petsc4py import PETSc
from src import stokes_flow as sf
from src.myio import *
from src.support_class import *
from src.objComposite import createEcoliComp_tunnel
from src.myvtk import save_singleEcoli_vtk
from ecoli_in_pipe.ecoli_common import *


# def get_problem_kwargs(**main_kwargs):
#     problem_kwargs = get_solver_kwargs()
#     OptDB = PETSc.Options()
#     fileHandle = OptDB.getString('f', 'ecoliInPipe')
#     OptDB.setValue('f', fileHandle)
#     problem_kwargs['fileHandle'] = fileHandle
#
#     kwargs_list = (get_vtk_tetra_kwargs(), get_ecoli_kwargs(), get_forcefree_kwargs(), main_kwargs,)
#     for t_kwargs in kwargs_list:
#         for key in t_kwargs:
#             problem_kwargs[key] = t_kwargs[key]
#     return problem_kwargs
#
#
# def print_case_info(**problem_kwargs):
#     fileHandle = problem_kwargs['fileHandle']
#     PETSc.Sys.Print('-->Ecoli in pipe case, force free case.')
#     print_solver_info(**problem_kwargs)
#     print_forcefree_info(**problem_kwargs)
#     print_ecoli_info(fileHandle, **problem_kwargs)
#     return True


# @profile
def main_fun(**main_kwargs):
    OptDB = PETSc.Options()
    fileHandle = OptDB.getString('f', 'ecoliInPipe')
    OptDB.setValue('f', fileHandle)
    main_kwargs['fileHandle'] = fileHandle
    problem_kwargs = get_problem_kwargs(**main_kwargs)

    if not problem_kwargs['restart']:
        forcepipe = problem_kwargs['forcepipe']
        print_case_info(**problem_kwargs)
        ecoli_comp0 = createEcoliComp_tunnel(name='ecoli_0', **problem_kwargs)
        ecoli_comp1 = createEcoliComp_tunnel(name='ecoli_0', **problem_kwargs)
        ecoli_length = problem_kwargs['ls'] + problem_kwargs['dist_hs'] + problem_kwargs['ph'] * problem_kwargs['ch']
        ecoli_comp1.move(np.array((0, 0, 1.3 * ecoli_length)))
        problem = sf.stokesletsInPipeforcefreeProblem(**problem_kwargs)
        problem.set_prepare(forcepipe)
        problem.do_solve_process((ecoli_comp0,), pick_M=True)
        # post process
        head_U, tail_U = print_single_ecoli_forcefree_result(ecoli_comp0, **problem_kwargs)
        ecoli_U = ecoli_comp0.get_ref_U()
        save_singleEcoli_vtk(problem, createHandle=createEcoliComp_tunnel)
    else:
        head_U, tail_U, ecoli_U = ecoli_restart(**main_kwargs)
    return head_U, tail_U, ecoli_U


if __name__ == '__main__':
    main_fun()

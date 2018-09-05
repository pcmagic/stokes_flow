# coding=utf-8
# 1. generate velocity and force nodes of sphere using MATLAB,
# 2. for each force node, get b, solve surrounding velocity boundary condition (pipe and cover, named boundary velocity) using formula from Liron's paper, save .mat file
# 3. read .mat file, for each boundary velocity, solve associated boundary force.
# 4. solve sphere M matrix using boundary force.
# 5. solve problem and check.

import sys

import petsc4py

petsc4py.init(sys.argv)

import numpy as np
import pickle
# from time import time
# from scipy.io import loadmat
# from src.stokes_flow import problem_dic, obj_dic
# from src.geo import *
from petsc4py import PETSc
from src import stokes_flow as sf
from src.myio import *
# from src.support_class import *
from src.objComposite import *
from src.myvtk import save_singleEcoli_vtk
from ecoli_in_pipe.ecoli_common import *


# def get_problem_kwargs(**main_kwargs):
#     problem_kwargs = get_solver_kwargs()
#     OptDB = PETSc.Options()
#     fileHeadle = OptDB.getString('f', 'singleEcoliPro')
#     OptDB.setValue('f', fileHeadle)
#     problem_kwargs['fileHeadle'] = fileHeadle
#
#     kwargs_list = (get_vtk_tetra_kwargs(), get_ecoli_kwargs(), get_forcefree_kwargs(), main_kwargs,)
#     for t_kwargs in kwargs_list:
#         for key in t_kwargs:
#             problem_kwargs[key] = t_kwargs[key]
#     return problem_kwargs
#
#
# def print_case_info(**problem_kwargs):
#     fileHeadle = problem_kwargs['fileHeadle']
#     PETSc.Sys.Print('-->Ecoli in free space, force free case.')
#     print_solver_info(**problem_kwargs)
#     print_forcefree_info(**problem_kwargs)
#     print_ecoli_info(fileHeadle, **problem_kwargs)
#     return True


# @profile
def main_fun(**main_kwargs):
    OptDB = PETSc.Options()
    fileHeadle = OptDB.getString('f', 'singleEcoliPro')
    OptDB.setValue('f', fileHeadle)
    main_kwargs['fileHeadle'] = fileHeadle
    problem_kwargs = get_problem_kwargs(**main_kwargs)

    if not problem_kwargs['restart']:
        print_case_info(**problem_kwargs)
        ecoliHeadType = OptDB.getString('ecoliHeadType', 'tunnel')
        if 'ellipse' in ecoliHeadType:
            ecoli_comp = createEcoliComp_ellipse(name='ecoli_0', **problem_kwargs)
        elif 'tunnel' in ecoliHeadType:
            ecoli_comp = createEcoliComp_tunnel(name='ecoli_0', **problem_kwargs)
        else:
            err_msg = 'wrong ecoliHeadType'
            raise ValueError(err_msg)
        # ecoli_comp.show_u_nodes(linestyle=' ')
        # # dbg
        # for obj in ecoli_comp.get_obj_list():
        #     filename = fileHeadle + '_' + str(obj)
        #     obj.get_u_geo().save_nodes(filename + '_U')
        #     obj.get_f_geo().save_nodes(filename + '_f')
        problem = sf.forcefreeProblem(**problem_kwargs)
        problem.do_solve_process(ecoli_comp, pick_M=True)
        # # debug
        # problem.saveM_ASCII('%s_M.txt' % fileHeadle)
        # problem.saveF_ASCII('%s_F.txt' % fileHeadle)
        # problem.saveV_ASCII('%s_V.txt' % fileHeadle)

        # post process
        head_U, tail_U = print_single_ecoli_forcefree_result(ecoli_comp, **problem_kwargs)
        ecoli_U = ecoli_comp.get_ref_U()
        save_singleEcoli_vtk(problem, createHandle=createEcoliComp_tunnel)
    else:
        head_U, tail_U, ecoli_U = ecoli_restart(**main_kwargs)
    return head_U, tail_U, ecoli_U


if __name__ == '__main__':
    main_fun()

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
# import pickle
# from time import time
# from scipy.io import loadmat
# from src.stokes_flow import problem_dic, obj_dic
# from src.geo import *
from petsc4py import PETSc
from src import stokes_flow as sf
from src.myio import *
from src.objComposite import createEcoliComp_tunnel
from src.myvtk import *


def get_problem_kwargs(**main_kwargs):
    problem_kwargs = get_solver_kwargs()
    OptDB = PETSc.Options()
    fileHeadle = OptDB.getString('f', 'singleEcoliPro')
    problem_kwargs['fileHeadle'] = fileHeadle

    kwargs_list = (main_kwargs, get_vtk_tetra_kwargs(), get_ecoli_kwargs(), get_forceFree_kwargs())
    for t_kwargs in kwargs_list:
        for key in t_kwargs:
            problem_kwargs[key] = t_kwargs[key]

    zoom_factor = problem_kwargs['zoom_factor']
    rh1 = problem_kwargs['rh1']
    if 'ecoli_U' not in main_kwargs.keys():
        ecoli_Uz = OptDB.getReal('ecoli_Uz', 0) * zoom_factor * rh1
        problem_kwargs['ecoli_U'] = np.array((0, 0, ecoli_Uz, 0, 0, 0))
    return problem_kwargs


def print_case_info(**problem_kwargs):
    fileHeadle = problem_kwargs['fileHeadle']
    print_solver_info(**problem_kwargs)
    print_forceFree_info(**problem_kwargs)
    print_ecoli_info(fileHeadle, **problem_kwargs)
    return True


# @profile
def main_fun(**main_kwargs):
    problem_kwargs = get_problem_kwargs(**main_kwargs)

    if not problem_kwargs['restart']:
        print_case_info(**problem_kwargs)
        ecoli_comp = createEcoliComp_tunnel(name='ecoli_0', **problem_kwargs)
        problem = sf.dual_potential_problem(**problem_kwargs)
        for obj in ecoli_comp.get_obj_list()[1:]:
            problem.add_obj(obj)
        problem.print_info()
        problem.create_matrix()
        problem.solve()

        # post process
        # print_single_ecoli_forceFree_result(ecoli_comp, **problem_kwargs)
        save_singleEcoli_U_vtk(problem, createHandle=createEcoliComp_tunnel)
    else:
        pass

    return True


if __name__ == '__main__':
    main_fun()

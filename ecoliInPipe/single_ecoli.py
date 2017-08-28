# coding=utf-8
# 1. generate velocity and force nodes of sphere using MATLAB,
# 2. for each force node, get b, solve surrounding velocity boundary condition (pipe and cover, named boundary velocity) using formula from Liron's paper, save .mat file
# 3. read .mat file, for each boundary velocity, solve associated boundary force.
# 4. solve sphere M matrix using boundary force.
# 5. solve problem and check.

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

# import numpy as np
# import pickle
# from time import time
# from scipy.io import loadmat
# from src.stokes_flow import problem_dic, obj_dic
# from src.geo import *
from petsc4py import PETSc
from src import stokes_flow as sf
from src.myio import *
from src.objComposite import createEcoliComp_tunnel
from src.myvtk import save_singleEcoli_vtk


def get_problem_kwargs(**main_kwargs):
    problem_kwargs = get_solver_kwargs()
    OptDB = PETSc.Options()
    fileHeadle = OptDB.getString('f', 'singleEcoliPro')
    problem_kwargs['fileHeadle'] = fileHeadle

    kwargs_list = (main_kwargs, get_vtk_tetra_kwargs(), get_ecoli_kwargs(), get_forceFree_kwargs())
    for t_kwargs in kwargs_list:
        for key in t_kwargs:
            problem_kwargs[key] = t_kwargs[key]
    return problem_kwargs


def print_case_info(**problem_kwargs):
    fileHeadle = problem_kwargs['fileHeadle']
    print_solver_info_forceFree(**problem_kwargs)
    print_forceFree_info(**problem_kwargs)
    print_ecoli_info(fileHeadle, **problem_kwargs)
    return True


# @profile
def main_fun(**main_kwargs):
    comm = PETSc.COMM_WORLD.tompi4py()
    rank = comm.Get_rank()
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    fileHeadle = problem_kwargs['fileHeadle']
    center = problem_kwargs['center']

    if not problem_kwargs['restart']:
        print_case_info(**problem_kwargs)
        ecoli_comp = createEcoliComp_tunnel(name='ecoli_0', **problem_kwargs)
        # ecoli_comp.show_u_nodes(linestyle=' ')
        obj_list = (ecoli_comp,)
        problem = sf.forceFreeProblem(**problem_kwargs)
        problem.do_solve_process(obj_list)
        # # debug
        # problem.saveM_ASCII('%s_M.txt' % fileHeadle)
        # problem.saveF_ASCII('%s_F.txt' % fileHeadle)
        # problem.saveV_ASCII('%s_V.txt' % fileHeadle)

        print_single_ecoli_forceFree_result(ecoli_comp, **problem_kwargs)
        # PETSc.Sys.Print(problem.get_total_force(center=center))
        # PETSc.Sys.Print(ecoli_comp.get_total_force())

        if problem_kwargs['pickProblem']:
            problem.pickmyself(fileHeadle)
        save_singleEcoli_vtk(problem)
    else:
        pass

    return True


if __name__ == '__main__':
    main_fun()

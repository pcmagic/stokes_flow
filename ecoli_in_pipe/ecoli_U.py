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
from src.objComposite import createEcoli_tunnel
from src.myvtk import save_singleEcoli_vtk


def get_problem_kwargs(**main_kwargs):
    problem_kwargs = get_solver_kwargs()

    kwargs_list = (main_kwargs, get_vtk_tetra_kwargs(), get_ecoli_kwargs(), get_forceFree_kwargs())
    for t_kwargs in kwargs_list:
        for key in t_kwargs:
            problem_kwargs[key] = t_kwargs[key]

    zoom_factor = problem_kwargs['zoom_factor']
    rh1 = problem_kwargs['rh1']

    OptDB = PETSc.Options()
    fileHeadle = OptDB.getString('f', 'singleEcoli_U')
    problem_kwargs['fileHeadle'] = fileHeadle
    if 'ecoli_U' in main_kwargs.keys():
        problem_kwargs['ecoli_U'] = main_kwargs['ecoli_U']
    else:
        ecoli_Uz = OptDB.getReal('ecoli_Uz', 0.001) * zoom_factor * rh1
        problem_kwargs['ecoli_U'] = np.array((0, 0, ecoli_Uz, 0, 0, 0))
    return problem_kwargs


def print_case_info(**problem_kwargs):
    fileHeadle = problem_kwargs['fileHeadle']
    print_solver_info(**problem_kwargs)
    print_forceFree_info(**problem_kwargs)
    print_ecoli_info(fileHeadle, **problem_kwargs)
    print_ecoli_U_info(fileHeadle, **problem_kwargs)
    return True


# @profile
def main_fun(**main_kwargs):
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    fileHeadle = problem_kwargs['fileHeadle']
    ecoli_U = problem_kwargs['ecoli_U']
    rel_Us = problem_kwargs['rel_Us']
    rel_Uh = problem_kwargs['rel_Uh']
    center = problem_kwargs['center']

    if not problem_kwargs['restart']:
        print_case_info(**problem_kwargs)
        vsobj, vhobj0, vhobj1, vTobj = createEcoli_tunnel(**problem_kwargs)
        vsobj.set_rigid_velocity(rel_Us + ecoli_U, center=center)
        vhobj0.set_rigid_velocity(rel_Uh + ecoli_U, center=center)
        vhobj1.set_rigid_velocity(rel_Uh + ecoli_U, center=center)
        vTobj.set_rigid_velocity(rel_Uh + ecoli_U, center=center)

        obj_list = (vsobj, vhobj0, vhobj1, vTobj,)
        problem = sf.stokesFlowProblem(**problem_kwargs)
        problem.do_solve_process(obj_list)
        # # debug
        # problem.saveM_ASCII('%s_M.txt' % fileHeadle)
        # problem.saveF_ASCII('%s_F.txt' % fileHeadle)
        # problem.saveV_ASCII('%s_V.txt' % fileHeadle)

        t_force = vsobj.get_total_force()
        PETSc.Sys.Print('---->>>head resultant is', t_force / 6 / np.pi)
        t_force = 0
        for t_obj in (vhobj0, vhobj1, vTobj, ):
            t_force = t_force + t_obj.get_total_force()
        PETSc.Sys.Print('---->>>tail resultant is', t_force / 6 / np.pi)
        if problem_kwargs['pickProblem']:
            problem.pickmyself(fileHeadle)
        save_singleEcoli_vtk(problem, ecoli_U)
    else:
        pass

    return True


if __name__ == '__main__':
    main_fun()

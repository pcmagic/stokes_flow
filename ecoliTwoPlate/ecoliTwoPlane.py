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
from src.objComposite import createEcoliComp_ellipse
from src.myvtk import save_singleEcoli_vtk


def get_problem_kwargs(**main_kwargs):
    problem_kwargs = get_solver_kwargs()
    OptDB = PETSc.Options()
    fileHandle = OptDB.getString('f', 'ecoliTwoPlane')
    problem_kwargs['fileHandle'] = fileHandle
    problem_kwargs['twoPlateHeight'] = 10

    kwargs_list = (main_kwargs, get_vtk_tetra_kwargs(), get_ecoli_kwargs(), get_forcefree_kwargs())
    for t_kwargs in kwargs_list:
        for key in t_kwargs:
            problem_kwargs[key] = t_kwargs[key]
    return problem_kwargs


def print_case_info(**problem_kwargs):
    fileHandle = problem_kwargs['fileHandle']
    print_solver_info(**problem_kwargs)
    print_forcefree_info(**problem_kwargs)
    print_ecoli_info(fileHandle, **problem_kwargs)
    return True


# @profile
def main_fun(**main_kwargs):
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    fileHandle = problem_kwargs['fileHandle']

    if not problem_kwargs['restart']:
        print_case_info(**problem_kwargs)
        ecoli_comp = createEcoliComp_ellipse(name='ecoli_0', **problem_kwargs)
        problem = sf.ForceFreeProblem(**problem_kwargs)
        problem.do_solve_process((ecoli_comp,), pick_M=True)
        head_U, tail_U = print_single_ecoli_forcefree_result(ecoli_comp, **problem_kwargs)
        save_singleEcoli_vtk(problem, createHandle=createEcoliComp_ellipse)
    else:
        pass
        # with open(fileHandle + '_pick.bin', 'rb') as input:
        #     unpick = pickle.Unpickler(input)
        #     problem = unpick.load( )
        #     problem.unpick_myself( )
        # problem_kwargs = problem.get_kwargs( )
        # forcepipe = problem_kwargs['forcepipe']
        # rh1 = problem_kwargs['rh1']
        # zoom_factor = problem_kwargs['zoom_factor']
        # rel_Us = problem_kwargs['rel_Us']
        # rel_Uh = problem_kwargs['rel_Uh']
        # prb_index = problem_kwargs['prb_index']
        # with_T_geo = len(problem.get_obj_list( )[0].get_obj_list( )) == 4
        # ecoli_comp = problem.get_obj_list( )[0]
        # if with_T_geo:
        #     vsobj, vhobj0, vhobj1, vTobj = ecoli_comp.get_obj_list( )
        # else:
        #     vsobj, vhobj0, vhobj1 = ecoli_comp.get_obj_list( )
        #
        # problem_kwargs1 = get_problem_kwargs(**main_kwargs)
        # problem_kwargs['matname'] = problem_kwargs1['matname']
        # problem_kwargs['bnodesHeadle'] = problem_kwargs1['bnodesHeadle']
        # problem_kwargs['belemsHeadle'] = problem_kwargs1['belemsHeadle']
        # problem_kwargs['ffweight'] = problem_kwargs1['ffweight']
        # problem.set_kwargs(**problem_kwargs)
        # print_case_info(**problem_kwargs)
        # problem.print_info( )
        #
        # OptDB = PETSc.Options( )
        # if OptDB.getBool('check_MPISIZE', True):
        #     err_msg = 'problem was picked with MPI size %d, current MPI size %d is wrong. ' % (
        #         problem_kwargs['MPISIZE'], problem_kwargs1['MPISIZE'],)
        #     assert problem_kwargs['MPISIZE'] == problem_kwargs1['MPISIZE'], err_msg
        #
        # problem.set_force_free( )
        # problem.solve( )
        # print_single_ecoli_forcefree_result(ecoli_comp, **problem_kwargs)
        #
        # # save_singleEcoli_vtk(problem)

    return True


if __name__ == '__main__':
    main_fun()

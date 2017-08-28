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
# sys.path = ['/home/zhangji/stokes_flow-master'] + sys.path

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
    fileHeadle = OptDB.getString('f', 'ecoliInPipe')
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
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    fileHeadle = problem_kwargs['fileHeadle']
    forcepipe = problem_kwargs['forcepipe']

    if not problem_kwargs['restart']:
        print_case_info(**problem_kwargs)
        ecoli_comp = createEcoliComp_tunnel(name='ecoli_0', **problem_kwargs)
        problem = sf.stokesletsInPipeForceFreeProblem(**problem_kwargs)
        problem.set_prepare(forcepipe)
        problem.add_obj(ecoli_comp)

        if problem_kwargs['pickProblem']:
            problem.pickmyself(fileHeadle, check=True)
        problem.print_info()
        problem.create_matrix()
        problem.solve()
        # debug
        # problem.saveM_ASCII('%s_M.txt' % fileHeadle)

        print_single_ecoli_forceFree_result(ecoli_comp, **problem_kwargs)

        if problem_kwargs['pickProblem']:
            problem.pickmyself(fileHeadle, pick_M=True)
        save_singleEcoli_vtk(problem)
    else:
        pass
        # with open(fileHeadle + '_pick.bin', 'rb') as input:
        #     unpick = pickle.Unpickler(input)
        #     problem = unpick.load( )
        #     problem.unpickmyself( )
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
        # print_single_ecoli_forceFree_result(ecoli_comp, **problem_kwargs)
        #
        # # save_singleEcoli_vtk(problem)

    return True


if __name__ == '__main__':
    main_fun()

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
from src.stokes_flow import problem_dic, obj_dic
# from src.geo import *
from petsc4py import PETSc
from src import stokes_flow as sf
from src.myio import *
from src.objComposite import createEcoli_tunnel
from src.myvtk import *
from src.support_class import *


def get_problem_kwargs(**main_kwargs):
    problem_kwargs = get_solver_kwargs()

    kwargs_list = (get_vtk_tetra_kwargs(), get_ecoli_kwargs(), get_forceFree_kwargs(), main_kwargs,)
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
        ecoli_Uz = OptDB.getReal('ecoli_Uz', 0) * zoom_factor * rh1
        problem_kwargs['ecoli_U'] = np.array((0, 0, ecoli_Uz, 0, 0, 0))

    ecoli_part = OptDB.getString('ecoli_part', 'full')
    problem_kwargs['ecoli_part'] = ecoli_part
    return problem_kwargs


def print_case_info(**problem_kwargs):
    fileHeadle = problem_kwargs['fileHeadle']
    print_solver_info(**problem_kwargs)
    print_ecoli_info(fileHeadle, **problem_kwargs)
    print_ecoli_U_info(fileHeadle, **problem_kwargs)
    return True


# @profile
def main_fun(**main_kwargs):
    def set_part_velocity(weight):
        kwargs = problem.get_kwargs()
        t_rel_Us = (rel_Us + ecoli_U) * weight
        t_rel_Uh = (rel_Uh + ecoli_U) * weight
        kwargs['ecoli_U'] = np.zeros(6)
        kwargs['rel_Us'] = t_rel_Us
        kwargs['rel_Uh'] = t_rel_Uh
        problem.set_kwargs(**kwargs)

        head_obj.set_rigid_velocity(t_rel_Us, center=center)
        for t_obj in tail_obj:
            t_obj.set_rigid_velocity(t_rel_Uh, center=center)
        return True

    main_kwargs['rel_Us'] = np.array((0, 0, 1, 0, 0, 1))
    main_kwargs['rel_Uh'] = np.array((0, 0, 1, 0, 0, 1))
    main_kwargs['ecoli_U'] = np.zeros(6)
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    ecoli_U = problem_kwargs['ecoli_U']
    rel_Us = problem_kwargs['rel_Us']
    rel_Uh = problem_kwargs['rel_Uh']
    center = problem_kwargs['center']
    ecoli_part = problem_kwargs['ecoli_part']
    matrix_method = problem_kwargs['matrix_method']
    with_T_geo = problem_kwargs['with_T_geo']
    pickProblem = problem_kwargs['pickProblem']
    fileHeadle = problem_kwargs['fileHeadle']

    if not problem_kwargs['restart']:
        print_case_info(**problem_kwargs)
        all_part = createEcoli_tunnel(**problem_kwargs)
        head_obj = all_part[0]
        if with_T_geo:
            tail_obj = all_part[1:]
        else:
            tail_obj = all_part[1:3]
        t_obj_list = {'head': (head_obj,),
                      'tail': (tail_obj,),
                      'full': (head_obj, tail_obj)}
        obj_list = t_obj_list[ecoli_part]

        problem = problem_dic[matrix_method](**problem_kwargs)
        if 'stokesletsInPipe' in matrix_method:
            forcepipe = problem_kwargs['forcepipe']
            problem.set_prepare(forcepipe)
        for obj in tube_flatten(obj_list):
            problem.add_obj(obj)
        if pickProblem:
            problem.pickmyself(fileHeadle, check=True)
        problem.print_info()

        # # dbg
        # n_node_full = np.sum([tobj.get_n_f_node() for tobj in tube_flatten(obj_list)])
        # err_msg = 'amount of node is %d, too much to solve. ' % n_node_full
        # assert n_node_full < 19000, err_msg
        problem.create_matrix()

        # 1. translation
        set_part_velocity(weight=(1, 1, 1, 0, 0, 0))
        # problem.show_velocity(length_factor=0.001)
        problem.create_F_U()
        problem.solve()
        print_single_ecoli_force_result(problem, part=ecoli_part, prefix='tran', **problem_kwargs)
        save_singleEcoli_U_vtk(problem, createHandle=createEcoli_tunnel, part=ecoli_part)

        # 2. rotation
        set_part_velocity(weight=(0, 0, 0, 1, 1, 1))
        # problem.show_velocity(length_factor=0.01)
        problem.create_F_U()
        problem.solve()
        print_single_ecoli_force_result(problem, part=ecoli_part, prefix='rota', **problem_kwargs)
        save_singleEcoli_U_vtk(problem, createHandle=createEcoli_tunnel, part=ecoli_part)

        # 3. move
        set_part_velocity(weight=(1, 1, 1, 1, 1, 1))
        # problem.show_velocity(length_factor=0.01)
        problem.create_F_U()
        problem.solve()
        print_single_ecoli_force_result(problem, part=ecoli_part, prefix='move', **problem_kwargs)
        save_singleEcoli_U_vtk(problem, createHandle=createEcoli_tunnel, part=ecoli_part)

        if pickProblem:
            problem.pickmyself(fileHeadle, pick_M=True, unpick=False)
    else:
        pass

    return True


if __name__ == '__main__':
    main_fun()

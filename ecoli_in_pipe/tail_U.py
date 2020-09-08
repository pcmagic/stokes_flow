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
from src.geo import *
from petsc4py import PETSc
from src import stokes_flow as sf
from src.myio import *
from src.objComposite import *
from src.myvtk import *
from src.support_class import *
from codeStore.helix_common import *


# @profile
def main_fun(**main_kwargs):
    OptDB = PETSc.Options()
    fileHandle = OptDB.getString('f', 'tail_U')
    OptDB.setValue('f', fileHandle)
    main_kwargs['fileHandle'] = fileHandle
    field_range = np.array([[-3, -3, -3], [3, 3, 3]])
    n_grid = np.array([1, 1, 1]) * OptDB.getInt('n_grid', 10)
    main_kwargs['field_range'] = field_range
    main_kwargs['n_grid'] = n_grid
    main_kwargs['region_type'] = 'rectangle'
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    matrix_method = problem_kwargs['matrix_method']
    pickProblem = problem_kwargs['pickProblem']
    fileHandle = problem_kwargs['fileHandle']
    save_vtk = problem_kwargs['save_vtk']

    if not problem_kwargs['restart']:
        print_case_info(**problem_kwargs)
        tail_obj_list = create_ecoli_tail(moveh=np.zeros(3), **problem_kwargs)

        problem = problem_dic[matrix_method](**problem_kwargs)
        if 'stokesletsInPipe' in matrix_method:
            forcepipe = problem_kwargs['forcepipe']
            problem.set_prepare(forcepipe)
        for tobj in tail_obj_list:
            problem.add_obj(tobj)

        # # dbg
        # problem.show_u_nodes()
        # assert 1 == 2

        if pickProblem:
            problem.pickmyself('%s_tran' % fileHandle, ifcheck=True)
        problem.print_info()
        problem.create_matrix()

        # 1. translation
        for tobj in tail_obj_list:
            tobj.set_rigid_velocity(np.array((0, 0, 1, 0, 0, 0)))
        problem.create_F_U()
        problem.solve()
        if problem_kwargs['pickProblem']:
            problem.pickmyself('%s_tran' % fileHandle, pick_M=False, mat_destroy=False)
        print_single_ecoli_force_result(problem, part='tail', prefix='tran', **problem_kwargs)
        problem.vtk_self('%s_tran' % fileHandle)
        if save_vtk:
            problem.vtk_velocity('%s_tran' % fileHandle)

        # 2. rotation
        for tobj in tail_obj_list:
            tobj.set_rigid_velocity(np.array((0, 0, 0, 0, 0, 1)))
        problem.create_F_U()
        problem.solve()
        if problem_kwargs['pickProblem']:
            problem.pickmyself('%s_rota' % fileHandle, pick_M=False, mat_destroy=False)
        print_single_ecoli_force_result(problem, part='tail', prefix='rota', **problem_kwargs)
        problem.vtk_self('%s_rota' % fileHandle)
        if save_vtk:
            problem.vtk_velocity('%s_rota' % fileHandle)
    return True


def self_repeat_tail(**main_kwargs):
    OptDB = PETSc.Options()
    fileHandle = OptDB.getString('f', 'tail_U')
    OptDB.setValue('f', fileHandle)
    main_kwargs['fileHandle'] = fileHandle
    field_range = np.array([[-3, -3, -3], [3, 3, 3]])
    n_grid = np.array([1, 1, 1]) * OptDB.getInt('n_grid', 10)
    main_kwargs['field_range'] = field_range
    main_kwargs['n_grid'] = n_grid
    main_kwargs['region_type'] = 'rectangle'
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    matrix_method = problem_kwargs['matrix_method']
    pickProblem = problem_kwargs['pickProblem']
    fileHandle = problem_kwargs['fileHandle']
    save_vtk = problem_kwargs['save_vtk']

    if not problem_kwargs['restart']:
        print_case_info(**problem_kwargs)
        _ = create_selfRepeat_tail(np.zeros(3), **problem_kwargs)
        tail_list, tail_start_list, tail_body0_list, tail_end_list = _
        part_obj_list = list(tube_flatten((tail_start_list, tail_body0_list, tail_end_list)))
        problem = problem_dic[matrix_method](**problem_kwargs)
        if 'stokesletsInPipe' in matrix_method:
            forcepipe = problem_kwargs['forcepipe']
            problem.set_prepare(forcepipe)
        for t1_list in (tail_start_list, tail_body0_list, tail_end_list):
            for obj_pair in zip(t1_list, tail_list, ):
                problem.add_obj(obj_pair)
        # # dbg
        # tgeo = base_geo()
        # tnodes = np.vstack([t1.get_u_geo().get_all_nodes() for t1 in tail_list])
        # tgeo.set_nodes(tnodes, 0)
        # tgeo.show_nodes()
        # problem.show_u_nodes()
        # assert 1 == 2

        if pickProblem:
            problem.pickmyself('%s_tran' % fileHandle, ifcheck=True)
        problem.print_info()
        problem.create_matrix()

        # 1. translation
        for tobj in part_obj_list:
            tobj.set_rigid_velocity(np.array((0, 0, 1, 0, 0, 0)))
        problem.create_F_U()
        problem.solve()
        if problem_kwargs['pickProblem']:
            problem.pickmyself('%s_tran' % fileHandle, pick_M=False, mat_destroy=False)
        print_single_ecoli_force_result(problem, part='tail', prefix='tran', **problem_kwargs)
        problem.vtk_self('%s_tran' % fileHandle)
        if save_vtk:
            problem.vtk_velocity('%s_tran' % fileHandle)

        # 2. rotation
        for tobj in part_obj_list:
            tobj.set_rigid_velocity(np.array((0, 0, 0, 0, 0, 1)))
        problem.create_F_U()
        problem.solve()
        if problem_kwargs['pickProblem']:
            problem.pickmyself('%s_rota' % fileHandle, pick_M=False, mat_destroy=False)
        print_single_ecoli_force_result(problem, part='tail', prefix='rota', **problem_kwargs)
        problem.vtk_self('%s_rota' % fileHandle)
        if save_vtk:
            problem.vtk_velocity('%s_rota' % fileHandle)
    return True


def self_rotate_tail(**main_kwargs):
    OptDB = PETSc.Options()
    fileHandle = OptDB.getString('f', 'tail_U')
    OptDB.setValue('f', fileHandle)
    main_kwargs['fileHandle'] = fileHandle
    field_range = np.array([[-3, -3, -3], [3, 3, 3]])
    n_grid = np.array([1, 1, 1]) * OptDB.getInt('n_grid', 10)
    main_kwargs['field_range'] = field_range
    main_kwargs['n_grid'] = n_grid
    main_kwargs['region_type'] = 'rectangle'
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    matrix_method = problem_kwargs['matrix_method']
    pickProblem = problem_kwargs['pickProblem']
    fileHandle = problem_kwargs['fileHandle']
    save_vtk = problem_kwargs['save_vtk']
    n_tail = problem_kwargs['n_tail']

    if not problem_kwargs['restart']:
        print_case_info(**problem_kwargs)
        tail_obj_list = create_ecoli_tail(moveh=np.zeros(3), **problem_kwargs)
        tail_obj = tail_obj_list[0]
        problem_kwargs['problem_center'] = tail_obj.get_u_geo().get_center()
        problem_kwargs['problem_norm'] = tail_obj.get_u_geo().get_geo_norm()
        problem_kwargs['problem_n_copy'] = n_tail

        problem = problem_dic[matrix_method](**problem_kwargs)
        problem.add_obj(tail_obj)

        # # dbg
        # problem.show_u_nodes()
        # err_msg = 'self_rotate_3d_petsc function is modified. '
        # assert 1 == 2, err_msg

        if pickProblem:
            problem.pickmyself('%s_tran' % fileHandle, ifcheck=True)
        problem.print_info()
        problem.create_matrix()

        # 1. translation
        problem.set_rigid_velocity(1, 0)
        problem.create_F_U()
        problem.solve()
        if problem_kwargs['pickProblem']:
            problem.pickmyself('%s_tran' % fileHandle, pick_M=False, mat_destroy=False)
        print_single_ecoli_force_result(problem, part='tail', prefix='tran', **problem_kwargs)
        problem.vtk_self('%s_tran' % fileHandle)
        if save_vtk:
            problem.vtk_velocity('%s_tran' % fileHandle)

        # 2. rotation
        problem.set_rigid_velocity(0, 1)
        problem.create_F_U()
        problem.solve()
        if problem_kwargs['pickProblem']:
            problem.pickmyself('%s_rota' % fileHandle, pick_M=False, mat_destroy=False)
        print_single_ecoli_force_result(problem, part='tail', prefix='rota', **problem_kwargs)
        problem.vtk_self('%s_rota' % fileHandle)
        if save_vtk:
            problem.vtk_velocity('%s_rota' % fileHandle)
    return True


def dbg_SelfRepeat_FatHelix():
    repeat_n = 3  # repeat tail repeat_n times
    # problem_kwargs['repeat_n'] = repeat_n
    # repeat_n = problem_kwargs['repeat_n']

    tugeo = SelfRepeat_FatHelix(repeat_n)
    tugeo.create_deltatheta(dth=0.2, radius=0.1, R1=1, R2=1, B=0.2, n_c=3, with_cover=1)
    # tugeo.show_nodes()
    tugeo.show_all_nodes()


if __name__ == '__main__':
    OptDB = PETSc.Options()
    # if OptDB.getBool('self_repeat_tail', False):
    #     OptDB.setValue('main_fun', False)
    #     self_repeat_tail()
    #
    # if OptDB.getBool('main_fun', True):
    #     main_fun()

    matrix_method = OptDB.getString('sm', 'rs_stokeslets')
    assert matrix_method in ('pf', 'pf_selfRepeat',
                             'pf_selfRotate', 'rs_selfRotate', 'lg_rs_selfRotate')
    if matrix_method == 'pf_selfRepeat':
        self_repeat_tail()
    if matrix_method in ('pf_selfRotate', 'rs_selfRotate', 'lg_rs_selfRotate'):
        self_rotate_tail()
    elif matrix_method == 'pf':
        main_fun()

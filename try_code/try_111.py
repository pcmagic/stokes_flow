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
from codeStore.ecoli_common import *


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

    if not problem_kwargs['restart']:
        print_case_info(**problem_kwargs)
        tail_obj_list = createEcoli_tunnel(**problem_kwargs)[1]
        obj_center = tail_obj_list[0].get_u_geo().get_center()
        for tobj in tail_obj_list:
            t1 = tobj.get_u_geo().get_center()
            assert np.allclose(t1, tobj.get_f_geo().get_center())
            assert np.allclose(t1, obj_center)
            tobj.move(-obj_center)

        problem = problem_dic[matrix_method](**problem_kwargs)
        if 'stokesletsInPipe' in matrix_method:
            forcepipe = problem_kwargs['forcepipe']
            problem.set_prepare(forcepipe)
        for tobj in tail_obj_list:
            problem.add_obj(tobj)
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
        problem.vtk_velocity('%s_rota' % fileHandle)
    return True


if __name__ == '__main__':
    main_fun()

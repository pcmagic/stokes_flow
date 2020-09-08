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
# from src import stokes_flow as sf
from src.myio import *
from src.objComposite import *
# from src.myvtk import *
# from src.support_class import *
from codeStore.helix_common import *


def get_problem_kwargs(**main_kwargs):
    problem_kwargs = get_solver_kwargs()

    kwargs_list = (get_vtk_tetra_kwargs(), get_ecoli_kwargs(), get_forcefree_kwargs(), main_kwargs,)
    for t_kwargs in kwargs_list:
        for key in t_kwargs:
            problem_kwargs[key] = t_kwargs[key]
    return problem_kwargs


def print_case_info(caseIntro='-->(some introduce here)', **problem_kwargs):
    fileHandle = problem_kwargs['fileHandle']
    PETSc.Sys.Print(caseIntro)
    print_solver_info(**problem_kwargs)
    print_forcefree_info(**problem_kwargs)
    print_ecoli_info(fileHandle, **problem_kwargs)
    return True


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
    center = problem_kwargs['center']
    #

    rs1 = problem_kwargs['rs1']
    rs2 = problem_kwargs['rs2']
    assert rs1 == rs2
    ls = problem_kwargs['ls']
    es = problem_kwargs['es']
    zoom_factor = problem_kwargs['zoom_factor']
    ds = problem_kwargs['ds']

    if not problem_kwargs['restart']:
        print_case_info(**problem_kwargs)
        tail_obj_list = create_ecoli_tail(moveh=np.zeros(3), **problem_kwargs)
        # add a pipe inside the helical tail.
        vsobj = obj_dic[matrix_method]()
        vsgeo = tunnel_geo()
        dth = ds / rs1
        fsgeo = vsgeo.create_deltatheta(dth, rs1, ls, es, 1, 1, True)
        vsobj.set_data(fsgeo, vsgeo, name='sphere_0')
        vsobj.zoom(zoom_factor)

        # with pipe at center
        problem = problem_dic[matrix_method](**problem_kwargs)
        for tobj in tail_obj_list:
            problem.add_obj(tobj)
        problem.add_obj(vsobj)
        # problem.show_u_nodes(linestyle='')
        # assert 1 == 2
        if pickProblem:
            problem.pickmyself('%s_tran' % fileHandle, ifcheck=True)
        problem.print_info()
        problem.create_matrix()
        # AtBtCt_full(problem, save_vtk=False, pick_M=False, print_each=False,
        #             center=center)
        # translation
        for tobj in tail_obj_list:
            tobj.set_rigid_velocity(np.array((0, 0, 1, 0, 0, 0)))
        problem.create_F_U()
        problem.solve()
        tres = np.sum([tobj.get_total_force() for tobj in tail_obj_list], axis=0)
        PETSc.Sys.Print('-->>At')
        PETSc.Sys.Print(tres[:3])
        PETSc.Sys.Print('-->>Bt1')
        PETSc.Sys.Print(tres[3:])
        # rotation
        for tobj in tail_obj_list:
            tobj.set_rigid_velocity(np.array((0, 0, 0, 0, 0, 1)))
        problem.create_F_U()
        problem.solve()
        tres = np.sum([tobj.get_total_force() for tobj in tail_obj_list], axis=0)
        PETSc.Sys.Print('-->>Bt2')
        PETSc.Sys.Print(tres[:3])
        PETSc.Sys.Print('-->>Ct')
        PETSc.Sys.Print(tres[3:])

        # without pipe at center
        problem = problem_dic[matrix_method](**problem_kwargs)
        for tobj in tail_obj_list:
            problem.add_obj(tobj)
        if pickProblem:
            problem.pickmyself('%s_tran' % fileHandle, ifcheck=True)
        problem.print_info()
        problem.create_matrix()
        # AtBtCt_full(problem, save_vtk=False, pick_M=False, print_each=False,
        #             center=center)
        # translation
        for tobj in tail_obj_list:
            tobj.set_rigid_velocity(np.array((0, 0, 1, 0, 0, 0)))
        problem.create_F_U()
        problem.solve()
        tres = np.sum([tobj.get_total_force() for tobj in tail_obj_list], axis=0)
        PETSc.Sys.Print('-->>At')
        PETSc.Sys.Print(tres[:3])
        PETSc.Sys.Print('-->>Bt1')
        PETSc.Sys.Print(tres[3:])
        # rotation
        for tobj in tail_obj_list:
            tobj.set_rigid_velocity(np.array((0, 0, 0, 0, 0, 1)))
        problem.create_F_U()
        problem.solve()
        tres = np.sum([tobj.get_total_force() for tobj in tail_obj_list], axis=0)
        PETSc.Sys.Print('-->>Bt2')
        PETSc.Sys.Print(tres[:3])
        PETSc.Sys.Print('-->>Ct')
        PETSc.Sys.Print(tres[3:])
    return True


if __name__ == '__main__':
    OptDB = PETSc.Options()
    # if OptDB.getBool('self_repeat_tail', False):
    #     OptDB.setValue('main_fun', False)
    #     self_repeat_tail()
    #
    # if OptDB.getBool('main_fun', True):
    #     main_fun()

    # matrix_method = OptDB.getString('sm', 'rs_stokeslets')
    # assert matrix_method in ('pf', 'pf_selfRepeat', 'pf_selfRotate')
    # if matrix_method == 'pf_selfRepeat':
    #     self_repeat_tail()
    # if matrix_method == 'pf_selfRotate':
    #     self_rotate_tail()
    # elif matrix_method == 'pf':
    #     main_fun()
    main_fun()

# coding=utf-8

import sys

import petsc4py

petsc4py.init(sys.argv)

import numpy as np
# from time import time
# from scipy.io import loadmat
# from src.stokes_flow import problem_dic, obj_dic
from src.geo import *
from petsc4py import PETSc
from src import stokes_flow as sf
from codeStore.ecoli_common import *


# @profile
def main_fun(**main_kwargs):
    OptDB = PETSc.Options()
    fileHandle = OptDB.getString('f', 'head_U')
    OptDB.setValue('f', fileHandle)
    main_kwargs['fileHandle'] = fileHandle
    problem_kwargs = get_problem_kwargs(**main_kwargs)

    if not problem_kwargs['restart']:
        print_case_info(**problem_kwargs)
        # cylinder
        rs1 = problem_kwargs['rs1']
        rs2 = problem_kwargs['rs2']
        err_msg = 'the symmetric assumption needs rs1==rs2'
        assert rs1 == rs2, err_msg
        ls = problem_kwargs['ls']
        ds = problem_kwargs['ds']
        u_geo = revolve_pipe()
        f_geo = u_geo.create_half_deltaz_v2(ds, ls, rs1)

        revolve_obj2 = sf.StokesFlowRingObj()
        revolve_obj2.set_data(f_geo, u_geo, name='sphereObj_0')
        problem2 = sf.StokesletsRingInPipeProblemSymz(**problem_kwargs)
        problem2.add_obj(revolve_obj2)
        if problem_kwargs['pickProblem']:
            problem2.pickmyself('%s_tran' % fileHandle, ifcheck=True)
        problem2.print_info()
        problem2.create_matrix()

        # translation
        revolve_obj2.set_rigid_velocity(np.array((0, 0, 1, 0, 0, 0)))
        problem2.create_F_U()
        problem2.solve()
        if problem_kwargs['pickProblem']:
            problem2.pickmyself('%s_tran' % fileHandle, pick_M=False, mat_destroy=False)
        PETSc.Sys.Print('translational resistance is %f ' %
                        (revolve_obj2.get_total_force()[2] * 2))
        problem2.vtk_obj('%s_tran' % fileHandle)

        # rotation
        revolve_obj2.set_rigid_velocity(np.array((0, 0, 0, 0, 0, 1)))
        problem2.create_F_U()
        problem2.solve()
        if problem_kwargs['pickProblem']:
            problem2.pickmyself('%s_rot' % fileHandle, pick_M=False)
        PETSc.Sys.Print('rotational resistance is %f ' %
                        (revolve_obj2.get_total_force()[5] * 2))
        problem2.vtk_obj('%s_rot' % fileHandle)
    return True


def None_symz(**main_kwargs):
    OptDB = PETSc.Options()
    fileHandle = OptDB.getString('f', 'try_stokesletsRingInPipe')
    OptDB.setValue('f', fileHandle)
    main_kwargs['fileHandle'] = fileHandle
    problem_kwargs = get_problem_kwargs(**main_kwargs)

    if not problem_kwargs['restart']:
        print_case_info(**problem_kwargs)
        # cylinder
        rs1 = problem_kwargs['rs1']
        rs2 = problem_kwargs['rs2']
        err_msg = 'the symmetric assumption needs rs1==rs2'
        assert rs1 == rs2, err_msg
        ls = problem_kwargs['ls']
        ds = problem_kwargs['ds']
        u_geo = revolve_pipe()
        f_geo = u_geo.create_deltaz(ds, ls, rs1)

        revolve_obj2 = sf.StokesFlowRingObj()
        revolve_obj2.set_data(f_geo, u_geo, name='sphereObj_0')
        problem2 = sf.StokesletsRingInPipeProblem(**problem_kwargs)
        problem2.add_obj(revolve_obj2)
        if problem_kwargs['pickProblem']:
            problem2.pickmyself('%s_tran' % fileHandle, ifcheck=True)
        problem2.print_info()
        problem2.create_matrix()

        # translation
        revolve_obj2.set_rigid_velocity(np.array((0, 0, 1, 0, 0, 0)))
        problem2.create_F_U()
        problem2.solve()
        if problem_kwargs['pickProblem']:
            problem2.pickmyself('%s_tran' % fileHandle, pick_M=False, mat_destroy=False)
        PETSc.Sys.Print('translational resistance is %f ' %
                        (revolve_obj2.get_total_force()[2]))
        problem2.vtk_obj('%s_tran' % fileHandle)

        # rotation
        revolve_obj2.set_rigid_velocity(np.array((0, 0, 0, 0, 0, 1)))
        problem2.create_F_U()
        problem2.solve()
        if problem_kwargs['pickProblem']:
            problem2.pickmyself('%s_rot' % fileHandle, pick_M=False)
        PETSc.Sys.Print('rotational resistance is %f ' %
                        (revolve_obj2.get_total_force()[5]))
        problem2.vtk_obj('%s_rot' % fileHandle)
    return True


def main_bulk(**main_kwargs):
    OptDB = PETSc.Options()
    fileHandle = OptDB.getString('f', 'try_stokesletsRing')
    OptDB.setValue('f', fileHandle)
    main_kwargs['fileHandle'] = fileHandle
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    # print('dbg')
    # problem_kwargs['n_c'] = 100

    if not problem_kwargs['restart']:
        print_case_info(**problem_kwargs)
        # cylinder in bulk fluid
        rs1 = problem_kwargs['rs1']
        rs2 = problem_kwargs['rs2']
        err_msg = 'the symmetric assumption needs rs1==rs2'
        assert rs1 == rs2, err_msg
        ls = problem_kwargs['ls']
        ds = problem_kwargs['ds']
        u_geo = revolve_pipe()
        f_geo = u_geo.create_deltaz(ds, ls, rs1)
        u_geo.set_rigid_velocity(np.array((0, 0, 1, 0, 0, 0)))
        revolve_obj = sf.StokesFlowRingObj()
        revolve_obj.set_data(f_geo, u_geo, name='pipeObj_0')
        problem = sf.StokesletsRingProblem(**problem_kwargs)

        problem.add_obj(revolve_obj)
        if problem_kwargs['pickProblem']:
            problem.pickmyself('%s_tran' % fileHandle, ifcheck=True)
        problem.print_info()
        problem.create_matrix()

        # translation
        revolve_obj.set_rigid_velocity(np.array((0, 0, 1, 0, 0, 0)))
        problem.create_F_U()
        problem.solve()
        if problem_kwargs['pickProblem']:
            problem.pickmyself('%s_tran' % fileHandle, pick_M=False, mat_destroy=False)
        PETSc.Sys.Print('translational resistance is %f ' %
                        (revolve_obj.get_total_force()[2]))
        problem.vtk_obj('%s_tran' % fileHandle)

        # rotation
        revolve_obj.set_rigid_velocity(np.array((0, 0, 0, 0, 0, 1)))
        problem.create_F_U()
        problem.solve()
        if problem_kwargs['pickProblem']:
            problem.pickmyself('%s_rot' % fileHandle, pick_M=False)
        PETSc.Sys.Print('rotational resistance is %f ' %
                        (revolve_obj.get_total_force()[5]))
        problem.vtk_obj('%s_rot' % fileHandle)
    return True


if __name__ == '__main__':
    OptDB = PETSc.Options()
    if OptDB.getBool('None_symz', False):
        OptDB.setValue('main_fun', False)
        None_symz()

    if OptDB.getBool('main_bulk', False):
        OptDB.setValue('main_fun', False)
        main_bulk()

    if OptDB.getBool('main_fun', True):
        main_fun()


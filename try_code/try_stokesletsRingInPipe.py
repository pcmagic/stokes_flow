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
    fileHandle = OptDB.getString('f', 'try_stokesletsRingInPipe')
    OptDB.setValue('f', fileHandle)
    main_kwargs['fileHandle'] = fileHandle
    problem_kwargs = get_problem_kwargs(**main_kwargs)

    if not problem_kwargs['restart']:
        print_case_info(**problem_kwargs)

        # # sphere
        # rs1 = problem_kwargs['rs1']
        # rs2 = problem_kwargs['rs2']
        # ds = problem_kwargs['ds']
        # u_geo = revolve_ellipse()
        # f_geo = u_geo.create_deltaz(ds, rs1, rs2)
        # u_geo.set_rigid_velocity(np.array((0, 0, 1, 0, 0, 0)))
        # revolve_obj = sf.StokesFlowRingObj()
        # revolve_obj.set_data(f_geo, u_geo, name='sphereObj_0')
        # problem = sf.StokesletsRingInPipeProblem(**problem_kwargs)
        # problem.do_solve_process((revolve_obj,), pick_M=False)
        # # revolve_obj.show_force(length_factor=0.1, show_nodes=False)
        # print(revolve_obj.get_total_force()[:3] / (6 * np.pi * rs2))

        # # sphere
        # rs1 = problem_kwargs['rs1']
        # rs2 = problem_kwargs['rs2']
        # ds = problem_kwargs['ds']
        # u_geo = revolve_ellipse()
        # f_geo = u_geo.create_half_deltaz(ds, rs1, rs2)
        # u_geo.set_rigid_velocity(np.array((0, 0, 1, 0, 0, 0)))
        # revolve_obj = sf.StokesFlowRingObj()
        # revolve_obj.set_data(f_geo, u_geo, name='sphereObj_0')
        # problem = sf.StokesletsRingInPipeProblemSymz(**problem_kwargs)
        # problem.do_solve_process((revolve_obj,), pick_M=False)
        # # revolve_obj.show_force(length_factor=0.1, show_nodes=False)
        # print(revolve_obj.get_total_force()[:3] / (6 * np.pi * rs2) * 2)

        # sphere in bulk flow
        epsilon = -0.5
        rs1 = problem_kwargs['rs1']
        rs2 = problem_kwargs['rs2']
        ds = problem_kwargs['ds']
        ds = ds / 5
        problem_kwargs['n_c'] = 100
        n_c = problem_kwargs['n_c']
        print('dbg, n_c=%d' % n_c)
        u_geo = revolve_ellipse()
        f_geo = u_geo.create_delta(ds, rs1, rs2, epsilon)
        u_geo.set_rigid_velocity(np.array((0, 0, 1, 0, 0, 0)))
        revolve_obj = sf.StokesFlowRingObj()
        revolve_obj.set_data(f_geo, u_geo, name='sphereObj_0')
        problem = sf.StokesletsRingProblem(**problem_kwargs)
        problem.do_solve_process((revolve_obj,), pick_M=False)
        # problem.show_force(length_factor=0.3 * n_c)
        print(revolve_obj.get_force().reshape((-1, 3)))
        print(revolve_obj.get_total_force()[:3] / (6 * np.pi * rs2))
        print(revolve_obj.get_total_force()[3:] / (8 * np.pi * rs2 ** 3))

        # # pipe in bulk flow
        # rs1 = problem_kwargs['rs1']
        # ls = problem_kwargs['ls']
        # ds = problem_kwargs['ds']
        # problem_kwargs['n_c'] = 100
        # n_c = problem_kwargs['n_c']
        # print('dbg, n_c=%d' % n_c)
        # u_geo = revolve_pipe()
        # f_geo = u_geo.create_deltaz(ds, ls, rs1)
        # u_geo.set_rigid_velocity(np.array((0, 0, 1, 0, 0, 0)))
        # revolve_obj = sf.StokesFlowRingObj()
        # revolve_obj.set_data(f_geo, u_geo, name='pipeObj_0')
        # problem = sf.StokesletsRingProblem(**problem_kwargs)
        # problem.do_solve_process((revolve_obj,), pick_M=False)
        # # problem.show_f_u_nodes()
        # # problem.show_force(length_factor=0.3 * n_c)
        # print(revolve_obj.get_total_force())

        # # pipe in bulk flow
        # rs1 = problem_kwargs['rs1']
        # ls = problem_kwargs['ls']
        # # ls = 4
        # ds = problem_kwargs['ds']
        # u_geo = revolve_pipe()
        # f_geo = u_geo.create_deltaz(ds, ls, rs1)
        # u_geo.set_rigid_velocity(np.array((0, 0, 1, 0, 0, 0)))
        # revolve_obj = sf.StokesFlowRingObj()
        # revolve_obj.set_data(f_geo, u_geo, name='pipeObj_0')
        # problem = sf.StokesletsRingProblem(**problem_kwargs)
        # problem.do_solve_process((revolve_obj,), pick_M=False)
        # print(revolve_obj.get_total_force()[:3])
    return True


def main_bulk(**main_kwargs):
    OptDB = PETSc.Options()
    fileHandle = OptDB.getString('f', 'try_stokesletsRing')
    OptDB.setValue('f', fileHandle)
    main_kwargs['fileHandle'] = fileHandle
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    print('dbg')
    problem_kwargs['n_c'] = 100
    n_c = problem_kwargs['n_c']
    print('dbg, n_c=%d' % n_c)
    epsilon = -0.5

    if not problem_kwargs['restart']:
        print_case_info(**problem_kwargs)
        # cylinder in bulk fluid
        rs1 = problem_kwargs['rs1']
        rs2 = problem_kwargs['rs2']
        err_msg = 'the symmetric assumption needs rs1==rs2'
        assert rs1 == rs2, err_msg
        ds = problem_kwargs['ds']
        u_geo = revolve_ellipse()
        f_geo = u_geo.create_delta(ds, rs1, rs2, epsilon=epsilon)
        u_geo.set_rigid_velocity(np.array((0, 0, 1, 0, 0, 0)))
        revolve_obj = sf.StokesFlowRingObj()
        revolve_obj.set_data(f_geo, u_geo, name='sphereObj_0')
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
        # problem.vtk_obj('%s_tran' % fileHandle)
        # print(revolve_obj.get_force().reshape((-1, 3)))
        # print()
        # print()
        # print()
        revolve_obj.show_force(length_factor=0.5*n_c)

        # rotation
        revolve_obj.set_rigid_velocity(np.array((0, 0, 0, 0, 0, 1)))
        problem.create_F_U()
        problem.solve()
        if problem_kwargs['pickProblem']:
            problem.pickmyself('%s_rot' % fileHandle, pick_M=False)
        PETSc.Sys.Print('rotational resistance is %f ' %
                        (revolve_obj.get_total_force()[5]))
        # problem.vtk_obj('%s_rot' % fileHandle)
        # print(revolve_obj.get_force().reshape((-1, 3)))
        revolve_obj.show_force(length_factor=0.5*n_c)
    return True


def main_bulk_full(**main_kwargs):
    OptDB = PETSc.Options()
    fileHandle = OptDB.getString('f', 'try_stokesletsRing')
    OptDB.setValue('f', fileHandle)
    main_kwargs['fileHandle'] = fileHandle
    main_kwargs['matrix_method'] = 'pf'
    main_kwargs['n_c'] = 20
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    print('dbg')
    n_c = problem_kwargs['n_c']
    print('dbg, n_c=%d' % n_c)
    epsilon = -0.5

    if not problem_kwargs['restart']:
        print_case_info(**problem_kwargs)
        # cylinder in bulk fluid
        rs1 = problem_kwargs['rs1']
        rs2 = problem_kwargs['rs2']
        err_msg = 'the symmetric assumption needs rs1==rs2'
        assert rs1 == rs2, err_msg
        ds = problem_kwargs['ds']
        u_geo = revolve_ellipse()
        f_geo = u_geo.create_delta(ds, rs1, rs2, epsilon=epsilon)
        u_geo.set_rigid_velocity(np.array((0, 0, 1, 0, 0, 0)))
        revolve_obj = sf.StokesletsRingObjFull()
        revolve_obj.set_data(f_geo, u_geo, name='sphereObj_0', n_c=n_c)
        # revolve_obj.show_f_nodes()
        # revolve_obj.show_u_nodes()
        # revolve_obj.show_f_u_nodes()
        problem = sf.StokesFlowProblem(**problem_kwargs)

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
        PETSc.Sys.Print('translational resistance is %s ' %
                        (revolve_obj.get_total_force() / (6 * np.pi * rs1)))
        # problem.vtk_obj('%s_tran' % fileHandle)
        # revolve_obj.show_slice_force(length_factor=0.5*n_c)
        # revolve_obj.show_slice_force(length_factor=0.5*n_c, idx=int(n_c/2))
        print(revolve_obj.get_force().reshape((-1, 3))[::n_c])
        print()
        print()
        print()

        # rotation
        revolve_obj.set_rigid_velocity(np.array((0, 0, 0, 0, 0, 1)))
        problem.create_F_U()
        problem.solve()
        if problem_kwargs['pickProblem']:
            problem.pickmyself('%s_rot' % fileHandle, pick_M=False)
        PETSc.Sys.Print('rotational resistance is %s ' %
                        (revolve_obj.get_total_force() / (6 * np.pi * rs1 ** 3)))
        # problem.vtk_obj('%s_rot' % fileHandle)
        # revolve_obj.show_slice_force(length_factor=0.5*n_c)
        # revolve_obj.show_slice_force(length_factor=0.5 * n_c, idx=int(n_c / 2))
        print(revolve_obj.get_force().reshape((-1, 3))[::n_c])
    return True


if __name__ == '__main__':
    # main_fun()
    main_bulk()
    # main_bulk_full()

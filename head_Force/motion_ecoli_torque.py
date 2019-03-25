# coding=utf-8

import sys
import petsc4py

petsc4py.init(sys.argv)

import numpy as np
from time import time
from scipy.io import savemat
# from src.stokes_flow import problem_dic, obj_dic
from src.geo import *
from petsc4py import PETSc
from src import stokes_flow as sf
from src.myio import *
from src.StokesFlowMethod import light_stokeslets_matrix_3d
from src.support_class import *
from src.objComposite import *
# from src.myvtk import save_singleEcoli_vtk
import ecoli_in_pipe.ecoli_common as ec
import os


# import import_my_lib

# Todo: rewrite input and print process.
def get_problem_kwargs(**main_kwargs):
    OptDB = PETSc.Options()
    fileHandle = OptDB.getString('f', 'motion_ecoli_torque')
    OptDB.setValue('f', fileHandle)
    problem_kwargs = ec.get_problem_kwargs()
    problem_kwargs['fileHandle'] = fileHandle

    ecoli_velocity = OptDB.getReal('ecoli_velocity', 1)
    problem_kwargs['ecoli_velocity'] = ecoli_velocity

    kwargs_list = (get_shearFlow_kwargs(), get_update_kwargs(), main_kwargs,)
    for t_kwargs in kwargs_list:
        for key in t_kwargs:
            problem_kwargs[key] = t_kwargs[key]

    # vtk_matname = OptDB.getString('vtk_matname', 'pipe_dbg')
    # t_path = os.path.dirname(os.path.abspath(__file__))
    # vtk_matname = os.path.normpath(os.path.join(t_path, vtk_matname))
    # problem_kwargs['vtk_matname'] = vtk_matname
    return problem_kwargs


def print_case_info(**problem_kwargs):
    caseIntro = '-->Ecoli in infinite shear flow case, given speed and torque free case. '
    ec.print_case_info(caseIntro, **problem_kwargs)
    ecoli_velocity = problem_kwargs['ecoli_velocity']
    PETSc.Sys.Print('    ecoli_velocity %f' % ecoli_velocity)
    print_update_info(**problem_kwargs)
    print_shearFlow_info(**problem_kwargs)
    return True


# @profile
def main_fun(**main_kwargs):
    comm = PETSc.COMM_WORLD.tompi4py()
    rank = comm.Get_rank()
    # # dbg
    # main_kwargs['ecoli_velocity'] = -1.75439131e-02
    # # main_kwargs['ffweightx'] = 1
    # # main_kwargs['ffweighty'] = 1
    # # main_kwargs['ffweightz'] = 1
    # # main_kwargs['ffweightT'] = 1
    # main_kwargs['max_iter'] = 1
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    print_case_info(**problem_kwargs)
    fileHandle = problem_kwargs['fileHandle']
    max_iter = problem_kwargs['max_iter']
    eval_dt = problem_kwargs['eval_dt']
    ecoli_velocity = problem_kwargs['ecoli_velocity']
    iter_tor = 1e-1

    if not problem_kwargs['restart']:
        # create ecoli
        ecoli_comp = create_ecoli_2part(**problem_kwargs)
        # create check obj
        check_kwargs = problem_kwargs.copy()
        check_kwargs['nth'] = problem_kwargs['nth'] - 2 if problem_kwargs['nth'] >= 10 else problem_kwargs['nth'] + 1
        check_kwargs['ds'] = problem_kwargs['ds'] * 1.2
        check_kwargs['hfct'] = 1
        check_kwargs['Tfct'] = 1
        ecoli_comp_check = create_ecoli_2part(**check_kwargs)

        head_rel_U = ecoli_comp.get_rel_U_list()[0]
        tail_rel_U = ecoli_comp.get_rel_U_list()[1]
        problem = sf.ShearFlowForceFreeIterateProblem(**problem_kwargs)
        problem.add_obj(ecoli_comp)
        problem.set_iterate_comp(ecoli_comp)
        problem.print_info()
        problem_ff = sf.ShearFlowForceFreeProblem(**problem_kwargs)
        problem_ff.add_obj(ecoli_comp)
        planeShearRate = problem.get_planeShearRate()

        # calculate torque
        t2 = time()
        PETSc.Sys.Print(' ')
        PETSc.Sys.Print('############################ Current loop %05d / %05d ############################' %
                        (0, max_iter))
        PETSc.Sys.Print('calculate the motor spin of the ecoli that keeps |ref_U|==ecoli_velocity in free space')
        # 1) ini guess
        problem_ff.set_planeShearRate(np.zeros(3))
        problem.set_planeShearRate(np.zeros(3))
        problem_ff.create_matrix()
        problem_ff.solve()
        ref_U = ecoli_comp.get_ref_U()
        fct = ecoli_velocity / np.linalg.norm(ref_U[:3])
        PETSc.Sys.Print('  ini ref_U in free space', ref_U * fct)
        # 2) optimize force and torque free
        problem.create_matrix()
        ref_U, _, _ = problem.do_iterate2(ini_refU1=ref_U, tolerate=iter_tor)
        # 3) check accurate of force.
        ecoli_comp_check.dbg_set_rel_U_list([head_rel_U, tail_rel_U])
        ecoli_comp_check.set_ref_U(ref_U)
        velocity_err_list = problem.vtk_check(fileHandle, ecoli_comp_check)
        PETSc.Sys.Print('velocity error of head (total, x, y, z): ', next(velocity_err_list))
        PETSc.Sys.Print('velocity error of tail (total, x, y, z): ', next(velocity_err_list))
        # 4) set parameters
        fct = ecoli_velocity / np.linalg.norm(ref_U[:3])
        ecoli_comp.dbg_set_rel_U_list([head_rel_U * fct, tail_rel_U * fct])
        ecoli_comp.set_ref_U(ref_U * fct)
        ecoli_comp_check.dbg_set_rel_U_list([head_rel_U * fct, tail_rel_U * fct])
        ecoli_comp_check.set_ref_U(ref_U * fct)
        problem.set_planeShearRate(planeShearRate)
        problem_ff.set_planeShearRate(planeShearRate)
        # 5) save and print
        if rank == 0:
            idx = 0
            ti = idx * eval_dt
            savemat('%s_%05d' % (fileHandle, idx), {
                'ti':             ti,
                'planeShearRate': planeShearRate,
                'ecoli_center':   np.vstack(ecoli_comp.get_center()),
                'ecoli_nodes':    np.vstack([tobj.get_u_nodes() for tobj in ecoli_comp.get_obj_list()]),
                'ecoli_f':        np.hstack([np.zeros_like(tobj.get_force())
                                             for tobj in ecoli_comp.get_obj_list()]).reshape(-1, 3),
                'ecoli_u':        np.hstack([np.zeros_like(tobj.get_re_velocity())
                                             for tobj in ecoli_comp.get_obj_list()]).reshape(-1, 3),
                'ecoli_norm':     np.vstack(ecoli_comp.get_norm()),
                'ecoli_U':        np.vstack(ecoli_comp.get_ref_U()),
                'tail_rel_U':     np.vstack(ecoli_comp.get_rel_U_list()[1])}, oned_as='column', )
        PETSc.Sys.Print('  ref_U in free space', ref_U * fct)
        PETSc.Sys.Print('  |ref_U| in free space', np.linalg.norm(ref_U[:3]) * fct, np.linalg.norm(ref_U[3:]) * fct)
        PETSc.Sys.Print('  tail_rel_U in free space', tail_rel_U * fct)
        print_single_ecoli_force_result(ecoli_comp, prefix='', part='full', **problem_kwargs)
        t3 = time()
        PETSc.Sys.Print('#################### Current loop %05d / %05d uses: %08.3fs ####################' %
                        (0, max_iter, (t3 - t2)))

        # evaluation loop
        t0 = time()
        for idx in range(1, max_iter + 1):
            t2 = time()
            PETSc.Sys.Print()
            PETSc.Sys.Print('############################ Current loop %05d / %05d ############################' %
                            (idx, max_iter))
            # 1) ini guess
            problem_ff.create_matrix()
            problem_ff.solve()
            ref_U = ecoli_comp.get_ref_U()
            PETSc.Sys.Print('  ini ref_U in shear flow', ref_U)
            # 2) optimize force and torque free
            problem.create_matrix()
            ref_U, _, _ = problem.do_iterate2(ini_refU1=ref_U, tolerate=iter_tor)
            ecoli_comp.set_ref_U(ref_U)
            # 3) check accurate of force.
            ecoli_comp_check.set_ref_U(ref_U)
            velocity_err_list = problem.vtk_check(fileHandle, ecoli_comp_check)
            PETSc.Sys.Print('velocity error of head (total, x, y, z): ', next(velocity_err_list))
            PETSc.Sys.Print('velocity error of tail (total, x, y, z): ', next(velocity_err_list))
            # 4) save and print
            if rank == 0:
                ti = idx * eval_dt
                savemat('%s_%05d' % (fileHandle, idx), {
                    'ti':             ti,
                    'planeShearRate': planeShearRate,
                    'ecoli_center':   np.vstack(ecoli_comp.get_center()),
                    'ecoli_nodes':    np.vstack([tobj.get_u_nodes() for tobj in ecoli_comp.get_obj_list()]),
                    'ecoli_f':        np.hstack([tobj.get_force() for tobj in ecoli_comp.get_obj_list()]).reshape(-1,
                                                                                                                  3),
                    'ecoli_u':        np.hstack([tobj.get_re_velocity() for tobj in ecoli_comp.get_obj_list()]
                                                ).reshape(-1, 3),
                    'ecoli_norm':     np.vstack(ecoli_comp.get_norm()),
                    'ecoli_U':        np.vstack(ecoli_comp.get_ref_U()),
                    'tail_rel_U':     np.vstack(ecoli_comp.get_rel_U_list()[1])}, oned_as='column', )
            print_single_ecoli_force_result(ecoli_comp, prefix='', part='full', **problem_kwargs)
            # 5) update
            problem.update_location(eval_dt, print_handle='%d / %d' % (idx, max_iter))
            t3 = time()
            PETSc.Sys.Print('#################### Current loop %05d / %05d uses: %08.3fs ####################' %
                            (idx, max_iter, (t3 - t2)))

        t1 = time()
        PETSc.Sys.Print('%s: run %d loops using %f' % (fileHandle, max_iter, (t1 - t0)))

        problem.destroy()
        if rank == 0:
            savemat(fileHandle,
                    {'ecoli_center': np.vstack(ecoli_comp.get_center_hist()),
                     'ecoli_norm':   np.vstack(ecoli_comp.get_norm_hist()),
                     'ecoli_U':      np.vstack(ecoli_comp.get_ref_U_hist()),
                     't':            (np.arange(max_iter) + 1) * eval_dt},
                    oned_as='column')
    else:
        pass
    return True


def main_fun_noIter(**main_kwargs):
    comm = PETSc.COMM_WORLD.tompi4py()
    rank = comm.Get_rank()
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    print_case_info(**problem_kwargs)
    fileHandle = problem_kwargs['fileHandle']
    max_iter = problem_kwargs['max_iter']
    eval_dt = problem_kwargs['eval_dt']
    ecoli_velocity = problem_kwargs['ecoli_velocity']

    if not problem_kwargs['restart']:
        # create ecoli
        ecoli_comp = create_ecoli_2part(**problem_kwargs)
        head_rel_U = ecoli_comp.get_rel_U_list()[0]
        tail_rel_U = ecoli_comp.get_rel_U_list()[1]
        problem_ff = sf.ShearFlowForceFreeProblem(**problem_kwargs)
        problem_ff.add_obj(ecoli_comp)
        problem_ff.print_info()
        planeShearRate = problem_ff.get_planeShearRate()

        # calculate torque
        t2 = time()
        PETSc.Sys.Print(' ')
        PETSc.Sys.Print('############################ Current loop %05d / %05d ############################' %
                        (0, max_iter))
        PETSc.Sys.Print('calculate the motor spin of the ecoli that keeps |ref_U|==ecoli_velocity in free space')
        # 1) ini guess
        problem_ff.create_matrix()
        problem_ff.solve()
        ref_U = ecoli_comp.get_ref_U()
        # 4) set parameters
        fct = ecoli_velocity / np.linalg.norm(ref_U[:3])
        ecoli_comp.dbg_set_rel_U_list([head_rel_U * fct, tail_rel_U * fct])
        ecoli_comp.set_ref_U(ref_U * fct)
        # 5) save and print
        if rank == 0:
            idx = 0
            ti = idx * eval_dt
            savemat('%s_%05d' % (fileHandle, idx), {
                'ti':             ti,
                'planeShearRate': planeShearRate,
                'ecoli_center':   np.vstack(ecoli_comp.get_center()),
                'ecoli_nodes':    np.vstack([tobj.get_u_nodes() for tobj in ecoli_comp.get_obj_list()]),
                'ecoli_f':        np.hstack([np.zeros_like(tobj.get_force())
                                             for tobj in ecoli_comp.get_obj_list()]).reshape(-1, 3),
                'ecoli_u':        np.hstack([np.zeros_like(tobj.get_re_velocity())
                                             for tobj in ecoli_comp.get_obj_list()]).reshape(-1, 3),
                'ecoli_norm':     np.vstack(ecoli_comp.get_norm()),
                'ecoli_U':        np.vstack(ecoli_comp.get_ref_U()),
                'tail_rel_U':     np.vstack(ecoli_comp.get_rel_U_list()[1])}, oned_as='column', )
        PETSc.Sys.Print('  ref_U in free space', ref_U * fct)
        PETSc.Sys.Print('  |ref_U| in free space', np.linalg.norm(ref_U[:3]) * fct, np.linalg.norm(ref_U[3:]) * fct)
        PETSc.Sys.Print('  tail_rel_U in free space', tail_rel_U * fct)
        print_single_ecoli_force_result(ecoli_comp, prefix='', part='full', **problem_kwargs)
        t3 = time()
        PETSc.Sys.Print('#################### Current loop %05d / %05d uses: %08.3fs ####################' %
                        (0, max_iter, (t3 - t2)))

        # evaluation loop
        t0 = time()
        for idx in range(1, max_iter + 1):
            t2 = time()
            PETSc.Sys.Print()
            PETSc.Sys.Print('############################ Current loop %05d / %05d ############################' %
                            (idx, max_iter))
            # 1) ini guess
            problem_ff.create_matrix()
            problem_ff.solve()
            ref_U = ecoli_comp.get_ref_U()
            # 4) save and print
            if rank == 0:
                ti = idx * eval_dt
                savemat('%s_%05d' % (fileHandle, idx), {
                    'ti':             ti,
                    'planeShearRate': planeShearRate,
                    'ecoli_center':   np.vstack(ecoli_comp.get_center()),
                    'ecoli_nodes':    np.vstack([tobj.get_u_nodes() for tobj in ecoli_comp.get_obj_list()]),
                    'ecoli_f':        np.hstack([tobj.get_force() for tobj in ecoli_comp.get_obj_list()]).reshape(-1,
                                                                                                                  3),
                    'ecoli_u':        np.hstack([tobj.get_re_velocity() for tobj in ecoli_comp.get_obj_list()]
                                                ).reshape(-1, 3),
                    'ecoli_norm':     np.vstack(ecoli_comp.get_norm()),
                    'ecoli_U':        np.vstack(ecoli_comp.get_ref_U()),
                    'tail_rel_U':     np.vstack(ecoli_comp.get_rel_U_list()[1])}, oned_as='column', )
            print_single_ecoli_force_result(ecoli_comp, prefix='', part='full', **problem_kwargs)
            # 5) update
            problem_ff.update_location(eval_dt, print_handle='%d / %d' % (idx, max_iter))
            t3 = time()
            PETSc.Sys.Print('#################### Current loop %05d / %05d uses: %08.3fs ####################' %
                            (idx, max_iter, (t3 - t2)))

        t1 = time()
        PETSc.Sys.Print('%s: run %d loops using %f' % (fileHandle, max_iter, (t1 - t0)))

        if rank == 0:
            savemat(fileHandle,
                    {'ecoli_center': np.vstack(ecoli_comp.get_center_hist()),
                     'ecoli_norm':   np.vstack(ecoli_comp.get_norm_hist()),
                     'ecoli_U':      np.vstack(ecoli_comp.get_ref_U_hist()),
                     't':            (np.arange(max_iter) + 1) * eval_dt},
                    oned_as='column')
    else:
        pass
    return True


def passive_fun_noIter(**main_kwargs):
    comm = PETSc.COMM_WORLD.tompi4py()
    rank = comm.Get_rank()
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    print_case_info(**problem_kwargs)
    fileHandle = problem_kwargs['fileHandle']
    max_iter = problem_kwargs['max_iter']
    eval_dt = problem_kwargs['eval_dt']

    if not problem_kwargs['restart']:
        # create ecoli
        ecoli_comp = create_ecoli_2part(**problem_kwargs)
        ecoli_comp.dbg_set_rel_U_list([np.zeros(6), np.zeros(6)])
        problem_ff = sf.ShearFlowForceFreeProblem(**problem_kwargs)
        problem_ff.add_obj(ecoli_comp)
        problem_ff.print_info()
        planeShearRate = problem_ff.get_planeShearRate()

        # evaluation loop
        t0 = time()
        for idx in range(1, max_iter + 1):
            t2 = time()
            PETSc.Sys.Print()
            PETSc.Sys.Print('############################ Current loop %05d / %05d ############################' %
                            (idx, max_iter))
            # 1) ini guess
            problem_ff.create_matrix()
            problem_ff.solve()
            ref_U = ecoli_comp.get_ref_U()
            # 4) save and print
            if rank == 0:
                ti = idx * eval_dt
                savemat('%s_%05d' % (fileHandle, idx), {
                    'ti':             ti,
                    'planeShearRate': planeShearRate,
                    'ecoli_center':   np.vstack(ecoli_comp.get_center()),
                    'ecoli_nodes':    np.vstack([tobj.get_u_nodes() for tobj in ecoli_comp.get_obj_list()]),
                    'ecoli_f':        np.hstack([tobj.get_force() for tobj in ecoli_comp.get_obj_list()]).reshape(-1,
                                                                                                                  3),
                    'ecoli_u':        np.hstack([tobj.get_re_velocity() for tobj in ecoli_comp.get_obj_list()]
                                                ).reshape(-1, 3),
                    'ecoli_norm':     np.vstack(ecoli_comp.get_norm()),
                    'ecoli_U':        np.vstack(ecoli_comp.get_ref_U()),
                    'tail_rel_U':     np.vstack(ecoli_comp.get_rel_U_list()[1])}, oned_as='column', )
            PETSc.Sys.Print('  true ref_U in free space', ref_U)
            # 5) update
            problem_ff.update_location(eval_dt, print_handle='%d / %d' % (idx, max_iter))
            t3 = time()
            PETSc.Sys.Print('#################### Current loop %05d / %05d uses: %08.3fs ####################' %
                            (idx, max_iter, (t3 - t2)))

        t1 = time()
        PETSc.Sys.Print('%s: run %d loops using %f' % (fileHandle, max_iter, (t1 - t0)))

        if rank == 0:
            savemat(fileHandle,
                    {'ecoli_center': np.vstack(ecoli_comp.get_center_hist()),
                     'ecoli_norm':   np.vstack(ecoli_comp.get_norm_hist()),
                     'ecoli_U':      np.vstack(ecoli_comp.get_ref_U_hist()),
                     't':            (np.arange(max_iter) + 1) * eval_dt},
                    oned_as='column')
    else:
        pass
    return True


if __name__ == '__main__':
    OptDB = PETSc.Options()
    if OptDB.getBool('main_fun_noIter', False):
        OptDB.setValue('main_fun', False)
        main_fun_noIter()

    if OptDB.getBool('passive_fun_noIter', False):
        OptDB.setValue('main_fun', False)
        passive_fun_noIter()

    if OptDB.getBool('main_fun', True):
        main_fun()

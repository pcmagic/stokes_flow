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
    fileHandle = OptDB.getString('f', 'motion_helix_passive')
    OptDB.setValue('f', fileHandle)
    problem_kwargs = ec.get_problem_kwargs()
    problem_kwargs['fileHandle'] = fileHandle
    ini_rot_theta = OptDB.getReal('ini_rot_theta', 0)
    ini_rot_phi = OptDB.getReal('ini_rot_phi', 0)
    problem_kwargs['ini_rot_theta'] = ini_rot_theta
    problem_kwargs['ini_rot_phi'] = ini_rot_phi

    kwargs_list = (get_shearFlow_kwargs(), get_update_kwargs(), main_kwargs,)
    for t_kwargs in kwargs_list:
        for key in t_kwargs:
            problem_kwargs[key] = t_kwargs[key]
    return problem_kwargs


def print_case_info(**problem_kwargs):
    caseIntro = '-->Passive helix in infinite shear flow case, force and torque free case. '
    ec.print_case_info(caseIntro, **problem_kwargs)
    print_update_info(**problem_kwargs)
    print_shearFlow_info(**problem_kwargs)
    ini_rot_theta = problem_kwargs['ini_rot_theta']
    ini_rot_phi = problem_kwargs['ini_rot_phi']
    PETSc.Sys.Print('    ini_rot_theta: %f, ini_rot_phi: %f ' % (ini_rot_theta, ini_rot_phi))
    return True


# @profile
def main_fun(**main_kwargs):
    comm = PETSc.COMM_WORLD.tompi4py()
    rank = comm.Get_rank()
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    print_case_info(**problem_kwargs)
    fileHandle = problem_kwargs['fileHandle']
    max_iter = problem_kwargs['max_iter']
    eval_dt = problem_kwargs['eval_dt']
    ini_rot_theta = problem_kwargs['ini_rot_theta']
    ini_rot_phi = problem_kwargs['ini_rot_phi']
    iter_tor = 1e-2

    if not problem_kwargs['restart']:
        # create helix
        _, tail_obj_list = createEcoli_ellipse(name='ecoli0', **problem_kwargs)
        tail_obj = sf.StokesFlowObj()
        tail_obj.set_name('tail_obj')
        tail_obj.combine(tail_obj_list)
        tail_obj.move(-tail_obj.get_u_geo().get_center())
        tail_obj.node_rotation(np.array((0, 1, 0)), theta=ini_rot_theta)
        tail_obj.node_rotation(np.array((0, 0, 1)), theta=ini_rot_phi)
        t_norm = tail_obj.get_u_geo().get_geo_norm()
        helix_comp = sf.ForceFreeComposite(center=np.zeros(3), norm=t_norm, name='helix_0')
        helix_comp.add_obj(obj=tail_obj, rel_U=np.zeros(6))
        problem_ff = sf.ShearFlowForceFreeProblem(**problem_kwargs)
        problem_ff.add_obj(helix_comp)
        problem_ff.print_info()
        problem = sf.ShearFlowForceFreeIterateProblem(**problem_kwargs)
        problem.add_obj(helix_comp)
        problem.set_iterate_comp(helix_comp)
        planeShearRate = problem_ff.get_planeShearRate()

        # evaluation loop
        t0 = time()
        for idx in range(1, max_iter + 1):
            t2 = time()
            PETSc.Sys.Print()
            PETSc.Sys.Print('############################ Current loop %05d / %05d ############################' %
                            (idx, max_iter))
            # 1) ini guess
            ref_U0 = helix_comp.get_ref_U()
            problem_ff.create_matrix()
            problem_ff.solve()
            ref_U1 = helix_comp.get_ref_U()
            PETSc.Sys.Print('  ini ref_U0 in shear flow %s' % str(ref_U0))
            PETSc.Sys.Print('  ini ref_U1 in shear flow %s' % str(ref_U1))
            # 2) optimize force and torque free
            problem.create_matrix()
            ref_U = problem.do_iterate3(ini_refU0=ref_U0, ini_refU1=ref_U1, tolerate=iter_tor)
            helix_comp.set_ref_U(ref_U)
            # 4) save and print
            if rank == 0:
                ti = idx * eval_dt
                savemat('%s_%05d' % (fileHandle, idx), {
                    'ti':             ti,
                    'planeShearRate': planeShearRate,
                    'ecoli_center':   np.vstack(helix_comp.get_center()),
                    'ecoli_nodes':    np.vstack([tobj.get_u_nodes() for tobj in helix_comp.get_obj_list()]),
                    'ecoli_f':        np.hstack([tobj.get_force() for tobj in helix_comp.get_obj_list()]).reshape(-1,
                                                                                                                  3),
                    'ecoli_u':        np.hstack([tobj.get_re_velocity() for tobj in helix_comp.get_obj_list()]
                                                ).reshape(-1, 3),
                    'ecoli_norm':     np.vstack(helix_comp.get_norm()),
                    'ecoli_U':        np.vstack(helix_comp.get_ref_U()), }, oned_as='column', )
            PETSc.Sys.Print('  true ref_U in shear flow', ref_U)
            tU = np.linalg.norm(ref_U[:3])
            tW = np.linalg.norm(ref_U[3:])
            terr = (ref_U1 - ref_U) / [tU, tU, tU, tW, tW, tW]
            PETSc.Sys.Print('  error of direct method', terr)
            # 5) update
            problem.update_location(eval_dt, print_handle='%d / %d' % (idx, max_iter))
            t3 = time()
            PETSc.Sys.Print('#################### Current loop %05d / %05d uses: %08.3fs ####################' %
                            (idx, max_iter, (t3 - t2)))

        t1 = time()
        PETSc.Sys.Print('%s: run %d loops using %f' % (fileHandle, max_iter, (t1 - t0)))

        problem_ff.destroy()
        problem.destroy()
        if rank == 0:
            savemat(fileHandle,
                    {'ecoli_center': np.vstack(helix_comp.get_center_hist()),
                     'ecoli_norm':   np.vstack(helix_comp.get_norm_hist()),
                     'ecoli_U':      np.vstack(helix_comp.get_ref_U_hist()),
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
    ini_rot_theta = problem_kwargs['ini_rot_theta']
    ini_rot_phi = problem_kwargs['ini_rot_phi']

    if not problem_kwargs['restart']:
        # create helix
        _, tail_obj_list = createEcoli_ellipse(name='ecoli0', **problem_kwargs)
        tail_obj = sf.StokesFlowObj()
        tail_obj.set_name('tail_obj')
        tail_obj.combine(tail_obj_list)
        tail_obj.move(-tail_obj.get_u_geo().get_center())
        tail_obj.node_rotation(np.array((0, 1, 0)), theta=ini_rot_theta)
        tail_obj.node_rotation(np.array((0, 0, 1)), theta=ini_rot_phi)
        t_norm = tail_obj.get_u_geo().get_geo_norm()
        helix_comp = sf.ForceFreeComposite(center=np.zeros(3), norm=t_norm, name='helix_0')
        helix_comp.add_obj(obj=tail_obj, rel_U=np.zeros(6))
        problem_ff = sf.ShearFlowForceFreeProblem(**problem_kwargs)
        problem_ff.add_obj(helix_comp)
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
            ref_U1 = helix_comp.get_ref_U()
            PETSc.Sys.Print('  ini ref_U1 in shear flow %s' % str(ref_U1))
            # 4) save and print
            if rank == 0:
                ti = idx * eval_dt
                savemat('%s_%05d' % (fileHandle, idx), {
                    'ti':             ti,
                    'planeShearRate': planeShearRate,
                    'ecoli_center':   np.vstack(helix_comp.get_center()),
                    'ecoli_nodes':    np.vstack([tobj.get_u_nodes() for tobj in helix_comp.get_obj_list()]),
                    'ecoli_f':        np.hstack([tobj.get_force() for tobj in helix_comp.get_obj_list()]
                                                ).reshape(-1, 3),
                    'ecoli_u':        np.hstack([tobj.get_re_velocity() for tobj in helix_comp.get_obj_list()]
                                                ).reshape(-1, 3),
                    'ecoli_norm':     np.vstack(helix_comp.get_norm()),
                    'ecoli_U':        np.vstack(helix_comp.get_ref_U()), }, oned_as='column', )
            # 5) update
            problem_ff.update_location(eval_dt, print_handle='%d / %d' % (idx, max_iter))
            t3 = time()
            PETSc.Sys.Print('#################### Current loop %05d / %05d uses: %08.3fs ####################' %
                            (idx, max_iter, (t3 - t2)))
        t1 = time()
        PETSc.Sys.Print('%s: run %d loops using %f' % (fileHandle, max_iter, (t1 - t0)))

        if rank == 0:
            savemat(fileHandle,
                    {'ecoli_center': np.vstack(helix_comp.get_center_hist()),
                     'ecoli_norm':   np.vstack(helix_comp.get_norm_hist()),
                     'ecoli_U':      np.vstack(helix_comp.get_ref_U_hist()),
                     't':            (np.arange(max_iter) + 1) * eval_dt},
                    oned_as='column')
    else:
        pass
    return True


def main_fun_table(**main_kwargs):
    comm = PETSc.COMM_WORLD.tompi4py()
    rank = comm.Get_rank()
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    print_case_info(**problem_kwargs)
    fileHandle = problem_kwargs['fileHandle']
    max_iter = problem_kwargs['max_iter']
    eval_dt = problem_kwargs['eval_dt']
    ini_rot_theta = problem_kwargs['ini_rot_theta']
    ini_rot_phi = problem_kwargs['ini_rot_phi']

    if not problem_kwargs['restart']:
        # create helix
        _, tail_obj_list = createEcoli_ellipse(name='ecoli0', **problem_kwargs)
        tail_obj = sf.StokesFlowObj()
        tail_obj.set_name('tail_obj')
        tail_obj.combine(tail_obj_list)
        tail_obj.move(-tail_obj.get_u_geo().get_center())
        tail_obj.node_rotation(np.array((0, 1, 0)), theta=ini_rot_theta)
        tail_obj.node_rotation(np.array((0, 0, 1)), theta=ini_rot_phi)
        t_norm = tail_obj.get_u_geo().get_geo_norm()
        helix_comp = sf.ForceFreeComposite(center=np.zeros(3), norm=t_norm, name='helix_0')
        helix_comp.add_obj(obj=tail_obj, rel_U=np.zeros(6))
        problem_ff = sf.ShearFlowForceFreeLookUpTable(**problem_kwargs)
        problem_ff.add_obj(helix_comp)
        problem_ff.print_info()
        planeShearRate = problem_ff.get_planeShearRate()
        # problem_ff.create_empty_M()
        problem_ff.create_matrix()
        problem_ff.load_table('hlxB01_tau1a')

        # evaluation loop
        t0 = time()
        for idx in range(1, max_iter + 1):
            t2 = time()
            PETSc.Sys.Print()
            PETSc.Sys.Print('############################ Current loop %05d / %05d ############################' %
                            (idx, max_iter))
            # 1) ini guess
            problem_ff.solve()
            ref_U1 = helix_comp.get_ref_U()
            PETSc.Sys.Print('  ini ref_U1 in shear flow %s' % str(ref_U1))
            # 4) save and print
            if rank == 0:
                ti = idx * eval_dt
                savemat('%s_%05d' % (fileHandle, idx), {
                    'ti':             ti,
                    'planeShearRate': planeShearRate,
                    'ecoli_center':   np.vstack(helix_comp.get_center()),
                    'ecoli_nodes':    np.vstack([tobj.get_u_nodes() for tobj in helix_comp.get_obj_list()]),
                    'ecoli_f':        np.hstack([tobj.get_force() for tobj in helix_comp.get_obj_list()]
                                                ).reshape(-1, 3),
                    'ecoli_u':        np.hstack([tobj.get_re_velocity() for tobj in helix_comp.get_obj_list()]
                                                ).reshape(-1, 3),
                    'ecoli_norm':     np.vstack(helix_comp.get_norm()),
                    'ecoli_U':        np.vstack(helix_comp.get_ref_U()), }, oned_as='column', )
            # 5) update
            problem_ff.update_location(eval_dt, print_handle='%d / %d' % (idx, max_iter))
            t3 = time()
            PETSc.Sys.Print('#################### Current loop %05d / %05d uses: %08.3fs ####################' %
                            (idx, max_iter, (t3 - t2)))
        t1 = time()
        PETSc.Sys.Print('%s: run %d loops using %f' % (fileHandle, max_iter, (t1 - t0)))

        if rank == 0:
            savemat(fileHandle,
                    {'ecoli_center': np.vstack(helix_comp.get_center_hist()),
                     'ecoli_norm':   np.vstack(helix_comp.get_norm_hist()),
                     'ecoli_U':      np.vstack(helix_comp.get_ref_U_hist()),
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

    if OptDB.getBool('main_fun_table', False):
        OptDB.setValue('main_fun', False)
        main_fun_table()

    if OptDB.getBool('main_fun', True):
        main_fun()

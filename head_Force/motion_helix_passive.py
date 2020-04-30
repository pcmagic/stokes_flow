# coding=utf-8

import sys
import petsc4py

petsc4py.init(sys.argv)

import numpy as np
from time import time
from scipy.io import savemat
# from src.stokes_flow import problem_dic, obj_dic
from petsc4py import PETSc
from src import stokes_flow as sf
from src.myio import *
# from src.StokesFlowMethod import light_stokeslets_matrix_3d
from src.support_class import *
from src.objComposite import *
# from src.myvtk import save_singleEcoli_vtk
import codeStore.ecoli_common as ec
# import os
from scanf import scanf
import pickle


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

    # restart
    OptDB = PETSc.Options()
    rst_file_name = OptDB.getString('rst_file_name', ' ')
    rst_file_name = check_file_extension(rst_file_name, '_pick.bin')
    problem_kwargs['rst_file_name'] = rst_file_name

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
    if problem_kwargs['restart']:
        rst_file_name = problem_kwargs['rst_file_name']
        PETSc.Sys.Print('restart model, previous file is %s' % rst_file_name)
    return True


def get_theta_phi_psi(nodes0, nodes1, P0, P1, C0, C1):
    """
    :param nodes0: nodes of obj at step0.
    :param nodes1: nodes of obj at step1.
    :param P0: norm of obj at step0.
    :param P1: norm of obj at step1.
    :param C0: center of obj at step0.
    :param C1: center of obj at step1.
    """
    P0 = P0 / np.linalg.norm(P0)
    P1 = P1 / np.linalg.norm(P1)
    theta0 = np.arccos(P0[2])
    t0 = np.arctan2(P0[1], P0[0])
    phi0 = t0 + 2 * np.pi if t0 < 0 else t0  # (-pi,pi) -> (0, 2pi)
    theta1 = np.arccos(P1[2])
    t1 = np.arctan2(P1[1], P1[0])
    phi1 = t1 + 2 * np.pi if t1 < 0 else t1  # (-pi,pi) -> (0, 2pi)

    r0 = nodes0[0] - C0
    n0 = np.dot(r0, P0) * P0
    P20 = r0 - n0
    P20 = vector_rotation_norm(P20, norm=np.array((0, 0, 1)), theta=-phi0)  # rotate back
    P20 = vector_rotation_norm(P20, norm=np.array((0, 1, 0)), theta=-theta0)
    P20 = P20 / np.linalg.norm(P20)
    r1 = nodes1[0] - C1
    n1 = np.dot(r1, P1) * P1
    P21 = r1 - n1
    P21 = vector_rotation_norm(P21, norm=np.array((0, 0, 1)), theta=-phi1)  # rotate back
    P21 = vector_rotation_norm(P21, norm=np.array((0, 1, 0)), theta=-theta1)
    P21 = P21 / np.linalg.norm(P21)
    sign = np.sign(np.dot(np.array((0, 0, 1)), np.cross(P20, P21)))
    t_psi = sign * np.arccos(np.clip(np.dot(P20, P21), -1, 1))
    psi1 = t_psi + 2 * np.pi if t_psi < 0 else t_psi  # (-pi,pi) -> (0, 2pi)
    return theta1, phi1, psi1


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
            ref_U = problem.do_iterate3(ini_refU0=ref_U0, ini_refU1=ref_U1, rtol=iter_tor)
            helix_comp.set_ref_U(ref_U)
            # 4) save and print
            if rank == 0:
                ti = idx * eval_dt
                savemat('%s_%05d.mat' % (fileHandle, idx), {
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
            savemat('%s.mat' % fileHandle,
                    {'ecoli_center': np.vstack(helix_comp.get_center_hist()),
                     'ecoli_norm':   np.vstack(helix_comp.get_norm_hist()),
                     'ecoli_U':      np.vstack(helix_comp.get_ref_U_hist()),
                     't':            (np.arange(max_iter) + 1) * eval_dt},
                    oned_as='column')
    else:
        pass
    return True


def eval_loop_noIter(problem_ff, **problem_kwargs):
    comm = PETSc.COMM_WORLD.tompi4py()
    rank = comm.Get_rank()
    helix_comp = problem_ff.get_obj_list()[0]
    fileHandle = problem_kwargs['fileHandle']
    ini_iter = problem_kwargs['ini_iter']
    max_iter = problem_kwargs['max_iter']
    eval_dt = problem_kwargs['eval_dt']
    pickProblem = problem_kwargs['pickProblem']
    planeShearRate = problem_ff.get_planeShearRate()

    # evaluation loop
    t0 = time()  # starting time of whole loop
    t4 = time()  # starting time of pick process
    for idx in range(ini_iter, max_iter + 1):
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
            savemat('%s_%05d.mat' % (fileHandle, idx), {
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
        t5 = time()
        if t5 - t4 > 3600 and pickProblem:  # pick the problem every 3600s
            problem_ff.pickmyself('%s_%05d' % (fileHandle, idx), ifcheck=False, pick_M=False, unpick=True)
        t3 = time()
        PETSc.Sys.Print('#################### Current loop %05d / %05d uses: %08.3fs ####################' %
                        (idx, max_iter, (t3 - t2)))
    t1 = time()
    PETSc.Sys.Print('%s: run %d loops using %f' % (fileHandle, max_iter, (t1 - t0)))

    if rank == 0:
        savemat('%s.mat' % fileHandle,
                {'ecoli_center': np.vstack(helix_comp.get_center_hist()),
                 'ecoli_norm':   np.vstack(helix_comp.get_norm_hist()),
                 'ecoli_U':      np.vstack(helix_comp.get_ref_U_hist()),
                 't':            (np.arange(max_iter) + 1) * eval_dt},
                oned_as='column')
    return True


def main_fun_noIter(**main_kwargs):
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    print_case_info(**problem_kwargs)
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

        problem_kwargs['ini_iter'] = 1
        eval_loop_noIter(problem_ff, **problem_kwargs)
    else:
        rst_file_name = problem_kwargs['rst_file_name']
        ini_Iter = scanf('%*c_%d_pick.bin', rst_file_name)[0] + 1
        with open(rst_file_name, 'rb') as file1:
            unpick = pickle.Unpickler(file1)
            problem_ff = unpick.load()
            problem_ff.unpick_myself()
        problem_ff.print_info()

        problem_kwargs['ini_iter'] = ini_Iter
        problem_kwargs['max_iter'] = ini_Iter + problem_kwargs['max_iter']
        eval_loop_noIter(problem_ff, **problem_kwargs)
    return True


if __name__ == '__main__':
    OptDB = PETSc.Options()
    if OptDB.getBool('main_fun_noIter', False):
        OptDB.setValue('main_fun', False)
        main_fun_noIter()

    if OptDB.getBool('main_fun', True):
        main_fun()

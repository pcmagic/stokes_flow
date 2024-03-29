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
from src.objComposite import *
# from src.myvtk import save_singleEcoli_vtk
import codeStore.ecoli_common as ec
import pickle


# import import_my_lib

# Todo: rewrite input and print process.
def get_problem_kwargs(**main_kwargs):
    OptDB = PETSc.Options()
    fileHandle = OptDB.getString('f', 'loop_table_ecoli')
    OptDB.setValue('f', fileHandle)
    problem_kwargs = ec.get_problem_kwargs()
    problem_kwargs['fileHandle'] = fileHandle

    ecoli_velocity = OptDB.getReal('ecoli_velocity', 1)
    problem_kwargs['ecoli_velocity'] = ecoli_velocity
    n_norm_theta = OptDB.getInt('n_norm_theta', 2)
    n_norm_phi = OptDB.getInt('n_norm_phi', 2)
    norm_psi = OptDB.getReal('norm_psi', 0)
    problem_kwargs['n_norm_theta'] = n_norm_theta
    problem_kwargs['n_norm_phi'] = n_norm_phi
    problem_kwargs['norm_psi'] = norm_psi

    kwargs_list = (get_shearFlow_kwargs(), main_kwargs,)
    for t_kwargs in kwargs_list:
        for key in t_kwargs:
            problem_kwargs[key] = t_kwargs[key]

    # vtk_matname = OptDB.getString('vtk_matname', 'pipe_dbg')
    # t_path = os.path.dirname(os.path.abspath(__file__))
    # vtk_matname = os.path.normpath(os.path.join(t_path, vtk_matname))
    # problem_kwargs['vtk_matname'] = vtk_matname
    pickle_name = '%s_kwargs.pickle' % fileHandle
    with open(pickle_name, 'wb') as handle:
        pickle.dump(problem_kwargs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    PETSc.Sys.Print('---->save kwargs to %s' % pickle_name)
    return problem_kwargs


def print_case_info(**problem_kwargs):
    caseIntro = '-->Ecoli in infinite shear flow case, given speed and torque free case. '
    ec.print_case_info(caseIntro, **problem_kwargs)
    ecoli_velocity = problem_kwargs['ecoli_velocity']
    PETSc.Sys.Print('    ecoli_velocity %f' % ecoli_velocity)

    n_norm_theta = problem_kwargs['n_norm_theta']
    n_norm_phi = problem_kwargs['n_norm_phi']
    norm_psi = problem_kwargs['norm_psi']
    PETSc.Sys.Print('Loop parameter space: n_norm_theta %d, n_norm_phi %d. norm_psi %f' %
                    (n_norm_theta, n_norm_phi, norm_psi))
    print_shearFlow_info(**problem_kwargs)
    return True


def do_solve_once_noIter(problem_ff: sf.ShearFlowForceFreeProblem,
                         ecoli_comp: sf.ForceFreeComposite,
                         fileHandle, norm_theta, norm_phi, norm_psi,
                         planeShearRate, rank, idx, N):
    PETSc.Sys.Print()
    PETSc.Sys.Print('%s %05d / %05d theta=%f, phi=%f, psi=%f %s' %
                    ('#' * 25, idx, N, norm_theta, norm_phi, norm_psi, '#' * 25,))
    problem_ff.create_matrix()
    problem_ff.solve()
    ref_U = ecoli_comp.get_ref_U()
    PETSc.Sys.Print('  ini ref_U in shear flow', ref_U)

    PETSc.Sys.Print('  dbg ecoli_comp.get_rel_U_list', ecoli_comp.get_rel_U_list())
    PETSc.Sys.Print('  dbg problem_ff.get_planeShearRate()', problem_ff.get_planeShearRate())

    if rank == 0:
        mat_name = '%s_th%f_phi%f_psi_%f.mat' % (fileHandle, norm_theta, norm_phi, norm_psi)
        savemat(mat_name, {
            'norm_theta':     norm_theta,
            'norm_phi':       norm_phi,
            'norm_psi':       norm_psi,
            'planeShearRate': planeShearRate,
            'ecoli_center':   np.vstack(ecoli_comp.get_center()),
            'ecoli_nodes':    np.vstack([tobj.get_u_nodes() for tobj in ecoli_comp.get_obj_list()]),
            # 'ecoli_f':        np.hstack([np.zeros_like(tobj.get_force())
            #                              for tobj in ecoli_comp.get_obj_list()]).reshape(-1, 3),
            # 'ecoli_u':        np.hstack([np.zeros_like(tobj.get_re_velocity())
            #                              for tobj in ecoli_comp.get_obj_list()]).reshape(-1, 3),
            'ecoli_norm':     np.vstack(ecoli_comp.get_norm()),
            'ecoli_U':        np.vstack(ecoli_comp.get_ref_U()),
            'tail_rel_U':     np.vstack(ecoli_comp.get_rel_U_list()[1])}, oned_as='column', )
    return True


def do_solve_rotlet_once_noIter(problem_ff: sf.ShearFlowForceFreeProblem,
                                ecoli_comp: sf.ForceFreeComposite,
                                fileHandle, norm_theta, norm_phi, norm_psi, planeShearRate,
                                rank, idx, N):
    PETSc.Sys.Print()
    PETSc.Sys.Print('%s %05d / %05d theta=%f, phi=%f, psi=%f %s' %
                    ('#' * 25, idx, N, norm_theta, norm_phi, norm_psi, '#' * 25,))
    problem_ff.create_matrix()
    problem_ff.solve()
    ref_U = ecoli_comp.get_ref_U()
    PETSc.Sys.Print('  ini ref_U in shear flow', ref_U)

    if rank == 0:
        mat_name = '%s_th%f_phi%f_psi_%f.mat' % (fileHandle, norm_theta, norm_phi, norm_psi)
        savemat(mat_name, {
            'norm_theta':     norm_theta,
            'norm_phi':       norm_phi,
            'norm_psi':       norm_psi,
            'planeShearRate': planeShearRate,
            'ecoli_center':   np.vstack(ecoli_comp.get_center()),
            'ecoli_nodes':    np.vstack([tobj.get_u_nodes() for tobj in ecoli_comp.get_obj_list()]),
            'ecoli_norm':     np.vstack(ecoli_comp.get_norm()),
            'ecoli_U':        np.vstack(ecoli_comp.get_ref_U()),
            'tail_rel_U':     np.zeros(6)}, oned_as='column', )
        return True


def do_solve_once(problem_ff: sf.ShearFlowForceFreeProblem,
                  problem: sf.ShearFlowForceFreeIterateProblem,
                  ecoli_comp: sf.ForceFreeComposite,
                  fileHandle, norm_theta, norm_phi, norm_psi, planeShearRate, rank, idx, N,
                  iter_tor):
    PETSc.Sys.Print()
    PETSc.Sys.Print('%s %05d / %05d theta=%f, phi=%f, psi=%f %s' %
                    ('#' * 25, idx, N, norm_theta, norm_phi, norm_psi, '#' * 25,))
    # 1) ini guess
    ref_U0 = ecoli_comp.get_ref_U()
    problem_ff.create_matrix()
    problem_ff.solve()
    ref_U1 = ecoli_comp.get_ref_U()
    PETSc.Sys.Print('  ini ref_U0 in shear flow %s' % str(ref_U0))
    PETSc.Sys.Print('  ini ref_U1 in shear flow %s' % str(ref_U1))
    # 2) optimize force and torque free
    problem.create_matrix()
    ref_U = problem.do_iterate3(ini_refU0=ref_U0, ini_refU1=ref_U1, rtol=iter_tor)
    ecoli_comp.set_ref_U(ref_U)
    PETSc.Sys.Print('  true ref_U in shear flow', ref_U)
    tU = np.linalg.norm(ref_U[:3])
    tW = np.linalg.norm(ref_U[3:])
    terr = (ref_U1 - ref_U) / [tU, tU, tU, tW, tW, tW]
    PETSc.Sys.Print('  error of direct method', terr)

    if rank == 0:
        mat_name = '%s_th%f_phi%f_psi_%f.mat' % (fileHandle, norm_theta, norm_phi, norm_psi)
        savemat(mat_name, {
            'norm_theta':     norm_theta,
            'norm_phi':       norm_phi,
            'norm_psi':       norm_psi,
            'planeShearRate': planeShearRate,
            'ecoli_center':   np.vstack(ecoli_comp.get_center()),
            'ecoli_nodes':    np.vstack(
                    [tobj.get_u_nodes() for tobj in ecoli_comp.get_obj_list()]),
            # 'ecoli_f':        np.hstack([np.zeros_like(tobj.get_force())
            #                              for tobj in ecoli_comp.get_obj_list()]).reshape(-1, 3),
            # 'ecoli_u':        np.hstack([np.zeros_like(tobj.get_re_velocity())
            #                              for tobj in ecoli_comp.get_obj_list()]).reshape(-1, 3),
            'ecoli_norm':     np.vstack(ecoli_comp.get_norm()),
            'ecoli_U':        np.vstack(ecoli_comp.get_ref_U()),
            'tail_rel_U':     np.vstack(ecoli_comp.get_rel_U_list()[1])}, oned_as='column', )
    return True


def main_fun_noIter(**main_kwargs):
    comm = PETSc.COMM_WORLD.tompi4py()
    rank = comm.Get_rank()
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    print_case_info(**problem_kwargs)
    fileHandle = problem_kwargs['fileHandle']
    ecoli_velocity = problem_kwargs['ecoli_velocity']
    n_norm_theta = problem_kwargs['n_norm_theta']
    n_norm_phi = problem_kwargs['n_norm_phi']
    norm_psi = problem_kwargs['norm_psi']
    N = n_norm_phi * n_norm_theta

    if not problem_kwargs['restart']:
        # create ecoli
        ecoli_comp = create_ecoli_2part(**problem_kwargs)
        ecoli_comp.node_rotation(ecoli_comp.get_norm(), norm_psi)
        head_rel_U = ecoli_comp.get_rel_U_list()[0]
        tail_rel_U = ecoli_comp.get_rel_U_list()[1]
        problem_ff = sf.ShearFlowForceFreeProblem(**problem_kwargs)
        problem_ff.add_obj(ecoli_comp)
        problem_ff.print_info()
        problem_ff.create_matrix()
        planeShearRate = problem_ff.get_planeShearRate()

        # calculate torque
        t2 = time()
        PETSc.Sys.Print(' ')
        PETSc.Sys.Print(
                'calculate the motor spin of the ecoli that keeps |ref_U|==ecoli_velocity in free space')
        # 1) ini guess
        problem_ff.set_planeShearRate(np.zeros(3))
        problem_ff.create_matrix()
        problem_ff.solve()
        ref_U = ecoli_comp.get_ref_U()
        # 4) set parameters
        fct = ecoli_velocity / np.linalg.norm(ref_U[:3])
        ecoli_comp.set_rel_U_list([head_rel_U * fct, tail_rel_U * fct])
        ecoli_comp.set_ref_U(ref_U * fct)
        problem_ff.set_planeShearRate(planeShearRate)
        PETSc.Sys.Print('  ini ref_U in free space', ref_U)
        PETSc.Sys.Print('  true ref_U in free space', ref_U * fct)
        PETSc.Sys.Print('  Now used relative velocity of head and tail are %s and %s' %
                        (str(head_rel_U * fct), str(tail_rel_U * fct)))
        t3 = time()
        PETSc.Sys.Print('    Current process uses: %07.3fs' % (t3 - t2))

        # 1). theta=0, ecoli_norm=(0, 0, 1)
        norm_theta, norm_phi = 0, 0
        t2 = time()
        do_solve_once_noIter(problem_ff, ecoli_comp, fileHandle, norm_theta, norm_phi, norm_psi,
                             planeShearRate, rank, 0, N)
        t3 = time()
        PETSc.Sys.Print('    Current process uses: %07.3fs' % (t3 - t2))
        # 2). loop over parameter space
        for i0, norm_theta in enumerate(np.linspace(0, np.pi, n_norm_theta, endpoint=False)):
            ecoli_comp.node_rotation(np.array((0, 1, 0)), norm_theta)
            for i1, norm_phi in enumerate(np.linspace(0, 2 * np.pi, n_norm_phi, endpoint=False)):
                t2 = time()
                idx = i0 * n_norm_phi + i1 + 1
                ecoli_comp.node_rotation(np.array((0, 0, 1)), norm_phi)
                do_solve_once_noIter(problem_ff, ecoli_comp, fileHandle, norm_theta, norm_phi,
                                     norm_psi, planeShearRate, rank, idx, N)
                ecoli_comp.node_rotation(np.array((0, 0, 1)), -norm_phi)  # rotate back
                t3 = time()
                PETSc.Sys.Print('    Current process uses: %07.3fs' % (t3 - t2))
            ecoli_comp.node_rotation(np.array((0, 1, 0)), -norm_theta)  # rotate back
    else:
        pass
    return True


def test_location_Iter(**main_kwargs):
    comm = PETSc.COMM_WORLD.tompi4py()
    rank = comm.Get_rank()
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    print_case_info(**problem_kwargs)
    fileHandle = problem_kwargs['fileHandle']
    ecoli_velocity = problem_kwargs['ecoli_velocity']
    norm_psi = problem_kwargs['norm_psi']
    norm_theta = problem_kwargs['norm_theta']
    norm_phi = problem_kwargs['norm_phi']
    PETSc.Sys.Print('-->norm_theta=%f, norm_phi=%f' % (norm_theta, norm_phi))
    iter_tor = 1e-3

    if not problem_kwargs['restart']:
        # create ecoli
        ecoli_comp = create_ecoli_2part(**problem_kwargs)
        ecoli_comp.node_rotation(ecoli_comp.get_norm(), norm_psi)
        head_rel_U = ecoli_comp.get_rel_U_list()[0]
        tail_rel_U = ecoli_comp.get_rel_U_list()[1]
        problem_ff = sf.ShearFlowForceFreeProblem(**problem_kwargs)
        problem_ff.add_obj(ecoli_comp)
        problem_ff.print_info()
        problem = sf.ShearFlowForceFreeIterateProblem(**problem_kwargs)
        problem.add_obj(ecoli_comp)
        problem.set_iterate_comp(ecoli_comp)
        planeShearRate = problem_ff.get_planeShearRate()

        # calculate torque
        t2 = time()
        PETSc.Sys.Print(' ')
        PETSc.Sys.Print(
                'calculate the motor spin of the ecoli that keeps |ref_U|==ecoli_velocity in free space')
        # 1) ini guess
        problem_ff.set_planeShearRate(np.zeros(3))
        problem.set_planeShearRate(np.zeros(3))
        problem_ff.create_matrix()
        problem_ff.solve()
        ref_U = ecoli_comp.get_ref_U()
        # 2) optimize force and torque free
        problem.create_matrix()
        ref_U = problem.do_iterate3(ini_refU1=ref_U, rtol=iter_tor)
        PETSc.Sys.Print('  ini ref_U in free space', ref_U)
        # 4) set parameters
        fct = ecoli_velocity / np.linalg.norm(ref_U[:3])
        ecoli_comp.set_rel_U_list([head_rel_U * fct, tail_rel_U * fct])
        ecoli_comp.set_ref_U(ref_U * fct)
        problem_ff.set_planeShearRate(planeShearRate)
        problem.set_planeShearRate(planeShearRate)
        PETSc.Sys.Print('  true ref_U in free space', ref_U * fct)
        PETSc.Sys.Print('  Now used relative velocity of head and tail are %s and %s' %
                        (str(head_rel_U * fct), str(tail_rel_U * fct)))
        t3 = time()
        PETSc.Sys.Print('    Current process uses: %07.3fs' % (t3 - t2))

        # 1). given ecoli_norm.
        t2 = time()
        ecoli_comp.node_rotation(np.array((0, 1, 0)), norm_theta)
        ecoli_comp.node_rotation(np.array((0, 0, 1)), norm_phi)
        do_solve_once(problem_ff, problem, ecoli_comp, fileHandle, norm_theta, norm_phi,
                      norm_psi,
                      planeShearRate, rank, 0, 1, iter_tor)
        ecoli_comp.node_rotation(np.array((0, 0, 1)), -norm_phi)  # rotate back
        ecoli_comp.node_rotation(np.array((0, 1, 0)), -norm_theta)  # rotate back
        t3 = time()
        PETSc.Sys.Print('    Current process uses: %07.3fs' % (t3 - t2))
    else:
        pass
    return True


def test_location_noIter(**main_kwargs):
    comm = PETSc.COMM_WORLD.tompi4py()
    rank = comm.Get_rank()
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    print_case_info(**problem_kwargs)
    fileHandle = problem_kwargs['fileHandle']
    ecoli_velocity = problem_kwargs['ecoli_velocity']
    norm_psi = problem_kwargs['norm_psi']
    norm_theta = problem_kwargs['norm_theta']
    norm_phi = problem_kwargs['norm_phi']
    PETSc.Sys.Print('-->norm_theta=%f, norm_phi=%f' % (norm_theta, norm_phi))
    iter_tor = 1e-3

    if not problem_kwargs['restart']:
        # create ecoli
        ecoli_comp = create_ecoli_2part(**problem_kwargs)
        ecoli_comp.node_rotation(ecoli_comp.get_norm(), norm_psi)
        head_rel_U = ecoli_comp.get_rel_U_list()[0]
        tail_rel_U = ecoli_comp.get_rel_U_list()[1]
        problem_ff = sf.ShearFlowForceFreeProblem(**problem_kwargs)
        problem_ff.add_obj(ecoli_comp)
        problem_ff.print_info()
        planeShearRate = problem_ff.get_planeShearRate()

        # calculate torque
        t2 = time()
        PETSc.Sys.Print(' ')
        PETSc.Sys.Print(
                'calculate the motor spin of the ecoli that keeps |ref_U|==ecoli_velocity in free space')
        # 1) ini guess
        problem_ff.set_planeShearRate(np.zeros(3))
        problem_ff.create_matrix()
        problem_ff.solve()
        ref_U = ecoli_comp.get_ref_U()
        PETSc.Sys.Print('  ini ref_U in free space', ref_U)
        # 4) set parameters
        fct = ecoli_velocity / np.linalg.norm(ref_U[:3])
        ecoli_comp.set_rel_U_list([head_rel_U * fct, tail_rel_U * fct])
        ecoli_comp.set_ref_U(ref_U * fct)
        problem_ff.set_planeShearRate(planeShearRate)
        PETSc.Sys.Print('  true ref_U in free space', ref_U * fct)
        PETSc.Sys.Print('  Now used relative velocity of head and tail are %s and %s' %
                        (str(head_rel_U * fct), str(tail_rel_U * fct)))
        t3 = time()
        PETSc.Sys.Print('    Current process uses: %07.3fs' % (t3 - t2))

        # 1). given ecoli_norm.
        t2 = time()
        ecoli_comp.node_rotation(np.array((0, 1, 0)), norm_theta)
        ecoli_comp.node_rotation(np.array((0, 0, 1)), norm_phi)
        do_solve_once_noIter(problem_ff, ecoli_comp, fileHandle,
                             norm_theta, norm_phi, norm_psi,
                             planeShearRate, rank, 0, 1)
        ecoli_comp.node_rotation(np.array((0, 0, 1)), -norm_phi)  # rotate back
        ecoli_comp.node_rotation(np.array((0, 1, 0)), -norm_theta)  # rotate back
        t3 = time()
        PETSc.Sys.Print('    Current process uses: %07.3fs' % (t3 - t2))
    else:
        pass
    return True


def test_location_noIter_wm(**main_kwargs):
    comm = PETSc.COMM_WORLD.tompi4py()
    rank = comm.Get_rank()
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    print_case_info(**problem_kwargs)
    fileHandle = problem_kwargs['fileHandle']
    norm_psi = problem_kwargs['norm_psi']
    norm_theta = problem_kwargs['norm_theta']
    norm_phi = problem_kwargs['norm_phi']
    PETSc.Sys.Print('-->norm_theta=%f, norm_phi=%f' % (norm_theta, norm_phi))

    if not problem_kwargs['restart']:
        # create ecoli
        ecoli_comp = create_ecoli_2part(**problem_kwargs)
        ecoli_comp.node_rotation(ecoli_comp.get_norm(), norm_psi)
        problem_ff = sf.ShearFlowForceFreeProblem(**problem_kwargs)
        problem_ff.add_obj(ecoli_comp)
        problem_ff.print_info()
        planeShearRate = problem_ff.get_planeShearRate()
        ecoli_comp.node_rotation(np.array((0, 1, 0)), norm_theta)
        ecoli_comp.node_rotation(np.array((0, 0, 1)), norm_phi)
        # ecoli_comp.show_u_nodes()

        # no shear, with wm
        t2 = time()
        PETSc.Sys.Print(' ')
        PETSc.Sys.Print('calculate the velocity of active ecoli in free space ')
        problem_ff.set_planeShearRate(np.zeros(3))
        problem_ff.create_matrix()
        problem_ff.solve()
        ref_U = ecoli_comp.get_ref_U()
        PETSc.Sys.Print('  ini ref_U in free space', ref_U)
        problem_ff.set_planeShearRate(planeShearRate)
        t3 = time()
        PETSc.Sys.Print('    Current process uses: %07.3fs' % (t3 - t2))

        # with shear, no wm
        t2 = time()
        PETSc.Sys.Print(' ')
        PETSc.Sys.Print('calculate the velocity of passive ecoli in shear space ')
        t_rel_list = ecoli_comp.get_rel_U_list()
        ecoli_comp.set_rel_U_list([np.zeros_like(ta) for ta in t_rel_list])
        problem_ff.create_matrix()
        problem_ff.solve()
        ref_U = ecoli_comp.get_ref_U()
        PETSc.Sys.Print('  ini ref_U in free space', ref_U)
        ecoli_comp.set_rel_U_list(t_rel_list)
        t3 = time()
        PETSc.Sys.Print('    Current process uses: %07.3fs' % (t3 - t2))

        # with shear, with wm
        t2 = time()
        PETSc.Sys.Print('calculate the velocity of active ecoli in shear flow ')
        do_solve_once_noIter(problem_ff, ecoli_comp, fileHandle,
                             norm_theta, norm_phi, norm_psi,
                             planeShearRate, rank, 0, 1)
        t3 = time()
        PETSc.Sys.Print('    Current process uses: %07.3fs' % (t3 - t2))
    else:
        pass
    return True


def main_fun_random(**main_kwargs):
    comm = PETSc.COMM_WORLD.tompi4py()
    rank = comm.Get_rank()
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    print_case_info(**problem_kwargs)
    fileHandle = problem_kwargs['fileHandle']
    ecoli_velocity = problem_kwargs['ecoli_velocity']
    n_norm = problem_kwargs['n_norm']
    iter_tor = 1e-3

    if not problem_kwargs['restart']:
        # create ecoli
        ecoli_comp = create_ecoli_2part(**problem_kwargs)
        head_rel_U = ecoli_comp.get_rel_U_list()[0]
        tail_rel_U = ecoli_comp.get_rel_U_list()[1]
        problem_ff = sf.ShearFlowForceFreeProblem(**problem_kwargs)
        problem_ff.add_obj(ecoli_comp)
        problem_ff.print_info()
        problem_ff.create_matrix()
        problem = sf.ShearFlowForceFreeIterateProblem(**problem_kwargs)
        problem.add_obj(ecoli_comp)
        problem.set_iterate_comp(ecoli_comp)
        planeShearRate = problem_ff.get_planeShearRate()

        # calculate torque
        t2 = time()
        PETSc.Sys.Print(' ')
        PETSc.Sys.Print(
                'calculate the motor spin of the ecoli that keeps |ref_U|==ecoli_velocity in free space')
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
        ref_U = problem.do_iterate3(ini_refU1=ref_U, rtol=iter_tor)
        # 4) set parameters
        fct = ecoli_velocity / np.linalg.norm(ref_U[:3])
        ecoli_comp.set_rel_U_list([head_rel_U * fct, tail_rel_U * fct])
        ecoli_comp.set_ref_U(ref_U * fct)
        problem_ff.set_planeShearRate(planeShearRate)
        problem.set_planeShearRate(planeShearRate)
        PETSc.Sys.Print('  true ref_U in free space', ref_U * fct)
        PETSc.Sys.Print('  Now used relative velocity of head and tail are %s and %s' %
                        (str(head_rel_U * fct), str(tail_rel_U * fct)))
        t3 = time()
        PETSc.Sys.Print('    Current process uses: %07.3fs' % (t3 - t2))

        # 1). theta=0, ecoli_norm=(0, 0, 1)
        norm_theta, norm_phi = 0, 0
        t2 = time()
        do_solve_once(problem_ff, problem, ecoli_comp, fileHandle, norm_theta, norm_phi, 0,
                      planeShearRate, rank, 0, n_norm, iter_tor)
        ref_U000 = ecoli_comp.get_ref_U().copy()
        t3 = time()
        PETSc.Sys.Print('    Current process uses: %07.3fs' % (t3 - t2))
        # 2). loop over parameter space
        for idx in range(n_norm):
            norm_theta, norm_phi, norm_psi = np.random.sample(3)
            ecoli_comp.node_rotation(ecoli_comp.get_norm(), norm_psi)
            ecoli_comp.set_ref_U(ref_U000)
            ecoli_comp.node_rotation(np.array((0, 1, 0)), norm_theta)
            t2 = time()
            ecoli_comp.node_rotation(np.array((0, 0, 1)), norm_phi)
            do_solve_once(problem_ff, problem, ecoli_comp, fileHandle, norm_theta, norm_phi,
                          norm_psi,
                          planeShearRate, rank, idx, n_norm, iter_tor)
            ecoli_comp.node_rotation(np.array((0, 0, 1)), -norm_phi)  # rotate back
            t3 = time()
            PETSc.Sys.Print('    Current process uses: %07.3fs' % (t3 - t2))
            ecoli_comp.node_rotation(np.array((0, 1, 0)), -norm_theta)  # rotate back
    else:
        pass
    return True


def main_fun(**main_kwargs):
    comm = PETSc.COMM_WORLD.tompi4py()
    rank = comm.Get_rank()
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    print_case_info(**problem_kwargs)
    fileHandle = problem_kwargs['fileHandle']
    ecoli_velocity = problem_kwargs['ecoli_velocity']
    n_norm_theta = problem_kwargs['n_norm_theta']
    n_norm_phi = problem_kwargs['n_norm_phi']
    norm_psi = problem_kwargs['norm_psi']
    N = n_norm_phi * n_norm_theta
    iter_tor = 1e-3

    if not problem_kwargs['restart']:
        # create ecoli
        ecoli_comp = create_ecoli_2part(**problem_kwargs)
        ecoli_comp.node_rotation(ecoli_comp.get_norm(), norm_psi)
        head_rel_U = ecoli_comp.get_rel_U_list()[0]
        tail_rel_U = ecoli_comp.get_rel_U_list()[1]
        problem_ff = sf.ShearFlowForceFreeProblem(**problem_kwargs)
        problem_ff.add_obj(ecoli_comp)
        problem_ff.print_info()
        problem = sf.ShearFlowForceFreeIterateProblem(**problem_kwargs)
        problem.add_obj(ecoli_comp)
        problem.set_iterate_comp(ecoli_comp)
        planeShearRate = problem_ff.get_planeShearRate()

        # calculate torque
        t2 = time()
        PETSc.Sys.Print(' ')
        PETSc.Sys.Print(
                'calculate the motor spin of the ecoli that keeps |ref_U|==ecoli_velocity in free space')
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
        ref_U = problem.do_iterate3(ini_refU1=ref_U, rtol=iter_tor)
        # 4) set parameters
        fct = ecoli_velocity / np.linalg.norm(ref_U[:3])
        ecoli_comp.set_rel_U_list([head_rel_U * fct, tail_rel_U * fct])
        ecoli_comp.set_ref_U(ref_U * fct)
        problem_ff.set_planeShearRate(planeShearRate)
        problem.set_planeShearRate(planeShearRate)
        PETSc.Sys.Print('  true ref_U in free space', ref_U * fct)
        PETSc.Sys.Print('  Now used relative velocity of head and tail are %s and %s' %
                        (str(head_rel_U * fct), str(tail_rel_U * fct)))
        t3 = time()
        PETSc.Sys.Print('    Current process uses: %07.3fs' % (t3 - t2))

        # 1). theta=0, ecoli_norm=(0, 0, 1)
        norm_theta, norm_phi = 0, 0
        t2 = time()
        do_solve_once(problem_ff, problem, ecoli_comp, fileHandle, norm_theta, norm_phi,
                      norm_psi,
                      planeShearRate, rank, 0, N, iter_tor)
        ref_U000 = ecoli_comp.get_ref_U().copy()
        t3 = time()
        PETSc.Sys.Print('    Current process uses: %07.3fs' % (t3 - t2))
        # 2). loop over parameter space
        for i0, norm_theta in enumerate(np.linspace(0, np.pi, n_norm_theta, endpoint=False)):
            ecoli_comp.set_ref_U(ref_U000)
            ecoli_comp.node_rotation(np.array((0, 1, 0)), norm_theta)
            for i1, norm_phi in enumerate(np.linspace(0, 2 * np.pi, n_norm_phi, endpoint=False)):
                t2 = time()
                idx = i0 * n_norm_phi + i1 + 1
                ecoli_comp.node_rotation(np.array((0, 0, 1)), norm_phi)
                do_solve_once(problem_ff, problem, ecoli_comp, fileHandle, norm_theta, norm_phi,
                              norm_psi,
                              planeShearRate, rank, idx, N, iter_tor)
                ecoli_comp.node_rotation(np.array((0, 0, 1)), -norm_phi)  # rotate back
                t3 = time()
                PETSc.Sys.Print('    Current process uses: %07.3fs' % (t3 - t2))
            ecoli_comp.node_rotation(np.array((0, 1, 0)), -norm_theta)  # rotate back
    else:
        pass
    return True


def main_fun_rotlets(**main_kwargs):
    # in this case, assume the head is infinity small, and been replaced by a rotlet.
    comm = PETSc.COMM_WORLD.tompi4py()
    rank = comm.Get_rank()
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    print_case_info(**problem_kwargs)
    fileHandle = problem_kwargs['fileHandle']
    ecoli_velocity = problem_kwargs['ecoli_velocity']
    n_norm_theta = problem_kwargs['n_norm_theta']
    n_norm_phi = problem_kwargs['n_norm_phi']
    norm_psi = problem_kwargs['norm_psi']
    N = n_norm_phi * n_norm_theta

    if not problem_kwargs['restart']:
        # create ecoli
        rotlet_strength = -1  # just a ini guess.
        ecoli_comp = create_rotlets_tail_2part(rotlet_strength=rotlet_strength,
                                               **problem_kwargs)
        ecoli_comp.node_rotation(ecoli_comp.get_norm(), norm_psi)
        # Prepare Problem
        problem_ff = sf.GivenForceShearFlowProblem(**problem_kwargs)
        problem_ff.add_obj(ecoli_comp)
        problem_ff.print_info()
        planeShearRate = problem_ff.get_planeShearRate()

        # calculate torque
        t2 = time()
        PETSc.Sys.Print('calculate the motor spin of the ecoli that'
                        ' keeps |ref_U|==ecoli_velocity in free space')
        # 1) ini guess
        problem_ff.set_planeShearRate(np.zeros(3))
        problem_ff.create_matrix()
        problem_ff.solve()
        ref_U = ecoli_comp.get_ref_U()
        # 4) set parameters
        fct = ecoli_velocity / np.linalg.norm(ref_U[:3])
        tnorm = ecoli_comp.get_norm()
        torque = tnorm * rotlet_strength * fct
        tail_obj0 = ecoli_comp.get_obj_list()[0]
        tlst = []
        for location, force, StokesletsHandle in tail_obj0.get_point_force_list():
            tlst.append((location, force * fct, StokesletsHandle))
        tail_obj0.dbg_set_point_force_list(tlst)
        ecoli_comp.set_givenF(np.hstack((np.zeros(3), -1 * torque)))
        problem_ff.set_planeShearRate(planeShearRate)
        PETSc.Sys.Print('  ini torque of rotlet in free space', torque / fct)
        PETSc.Sys.Print('  true torque of rotlet in free space', torque)
        PETSc.Sys.Print('  helix velocity', ref_U * fct)
        t3 = time()
        PETSc.Sys.Print('    Current process uses: %07.3fs' % (t3 - t2))

        # 1). theta=0, ecoli_norm=(0, 0, 1)
        norm_theta, norm_phi = 0, 0
        t2 = time()
        do_solve_rotlet_once_noIter(problem_ff, ecoli_comp, fileHandle, norm_theta,
                                    norm_phi, norm_psi, planeShearRate, rank, 0, N)
        t3 = time()
        PETSc.Sys.Print('    Current process uses: %07.3fs' % (t3 - t2))
        # 2). loop over parameter space
        for i0, norm_theta in enumerate(np.linspace(0, np.pi, n_norm_theta, endpoint=False)):
            ecoli_comp.node_rotation(np.array((0, 1, 0)), norm_theta)
            for i1, norm_phi in enumerate(np.linspace(0, 2 * np.pi, n_norm_phi, endpoint=False)):
                t2 = time()
                idx = i0 * n_norm_phi + i1 + 1
                ecoli_comp.node_rotation(np.array((0, 0, 1)), norm_phi)
                do_solve_rotlet_once_noIter(problem_ff, ecoli_comp, fileHandle, norm_theta,
                                            norm_phi, norm_psi, planeShearRate, rank, idx, N)
                ecoli_comp.node_rotation(np.array((0, 0, 1)), -norm_phi)  # rotate back
                t3 = time()
                PETSc.Sys.Print('    Current process uses: %07.3fs' % (t3 - t2))
            ecoli_comp.node_rotation(np.array((0, 1, 0)), -norm_theta)  # rotate back
    else:
        pass
    return True


def main_fun_GivenT(**main_kwargs):
    # in this case, helix is locomotion under a external given torque.
    # the torque direction is coincident with the helix norm.
    # currently, this code is used to dbg the case main_fun_rotlets,
    comm = PETSc.COMM_WORLD.tompi4py()
    rank = comm.Get_rank()
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    print_case_info(**problem_kwargs)
    fileHandle = problem_kwargs['fileHandle']
    ecoli_velocity = problem_kwargs['ecoli_velocity']
    n_norm_theta = problem_kwargs['n_norm_theta']
    n_norm_phi = problem_kwargs['n_norm_phi']
    norm_psi = problem_kwargs['norm_psi']
    N = n_norm_phi * n_norm_theta

    if not problem_kwargs['restart']:
        # create ecoli
        givenT = 1  # just a ini guess.
        _, tail_obj_list = createEcoli_ellipse(name='ecoli0', **problem_kwargs)
        tail_obj = sf.StokesFlowObj()
        tail_obj.set_name('tail_obj')
        tail_obj.combine(tail_obj_list)
        tail_obj.move(-tail_obj.get_u_geo().get_center())
        t_norm = tail_obj.get_u_geo().get_geo_norm()
        ecoli_comp = sf.GivenForceComposite(center=np.zeros(3), norm=t_norm, name='helix_0',
                                            givenF=givenT * t_norm)
        ecoli_comp.add_obj(obj=tail_obj, rel_U=np.zeros(6))
        ecoli_comp.node_rotation(ecoli_comp.get_norm(), norm_psi)

        # Prepare Problem
        problem_ff = sf.GivenForceShearFlowProblem(**problem_kwargs)
        problem_ff.add_obj(ecoli_comp)
        problem_ff.print_info()
        planeShearRate = problem_ff.get_planeShearRate()

        # calculate torque
        t2 = time()
        PETSc.Sys.Print('calculate the motor spin of the ecoli that'
                        ' keeps |ref_U|==ecoli_velocity in free space')
        # 1) ini guess
        problem_ff.set_planeShearRate(np.zeros(3))
        problem_ff.create_matrix()
        problem_ff.solve()
        ref_U = ecoli_comp.get_ref_U()
        # 4) set parameters
        fct = ecoli_velocity / np.linalg.norm(ref_U[:3])
        tnorm = ecoli_comp.get_norm()
        ecoli_comp.set_givenF(np.hstack((np.zeros(3), tnorm * givenT * fct)))
        problem_ff.set_planeShearRate(planeShearRate)
        PETSc.Sys.Print('  ini given torque of helix in free space', tnorm * givenT)
        PETSc.Sys.Print('  true given torque of helix in free space', tnorm * givenT * fct)
        PETSc.Sys.Print('  helix velocity', ref_U * fct)
        t3 = time()
        PETSc.Sys.Print('    Current process uses: %07.3fs' % (t3 - t2))

        # 1). theta=0, ecoli_norm=(0, 0, 1)
        norm_theta, norm_phi = 0, 0
        t2 = time()
        do_solve_rotlet_once_noIter(problem_ff, ecoli_comp, fileHandle, norm_theta,
                                    norm_phi, norm_psi, planeShearRate, rank, 0, N)
        t3 = time()
        PETSc.Sys.Print('    Current process uses: %07.3fs' % (t3 - t2))
        # 2). loop over parameter space
        for i0, norm_theta in enumerate(np.linspace(0, np.pi, n_norm_theta, endpoint=False)):
            ecoli_comp.node_rotation(np.array((0, 1, 0)), norm_theta)
            for i1, norm_phi in enumerate(np.linspace(0, 2 * np.pi, n_norm_phi, endpoint=False)):
                t2 = time()
                idx = i0 * n_norm_phi + i1 + 1
                ecoli_comp.node_rotation(np.array((0, 0, 1)), norm_phi)
                do_solve_rotlet_once_noIter(problem_ff, ecoli_comp, fileHandle, norm_theta,
                                            norm_phi, norm_psi, planeShearRate, rank, idx, N)
                ecoli_comp.node_rotation(np.array((0, 0, 1)), -norm_phi)  # rotate back
                t3 = time()
                PETSc.Sys.Print('    Current process uses: %07.3fs' % (t3 - t2))
            ecoli_comp.node_rotation(np.array((0, 1, 0)), -norm_theta)  # rotate back
    else:
        pass
    return True


if __name__ == '__main__':
    OptDB = PETSc.Options()
    if OptDB.getBool('main_fun_noIter', False):
        OptDB.setValue('main_fun', False)
        main_fun_noIter()

    if OptDB.getBool('main_fun_rotlets', False):
        OptDB.setValue('main_fun', False)
        main_fun_rotlets()

    if OptDB.getBool('test_location_Iter', False):
        OptDB.setValue('main_fun', False)
        norm_theta = OptDB.getReal('norm_theta', 0)
        norm_phi = OptDB.getReal('norm_phi', 0)
        test_location_Iter(norm_theta=norm_theta, norm_phi=norm_phi)

    if OptDB.getBool('test_location_noIter', False):
        OptDB.setValue('main_fun', False)
        norm_theta = OptDB.getReal('norm_theta', 0)
        norm_phi = OptDB.getReal('norm_phi', 0)
        test_location_noIter(norm_theta=norm_theta, norm_phi=norm_phi)

    if OptDB.getBool('test_location_noIter_wm', False):
        OptDB.setValue('main_fun', False)
        norm_theta = OptDB.getReal('norm_theta', 0)
        norm_phi = OptDB.getReal('norm_phi', 0)
        test_location_noIter_wm(norm_theta=norm_theta, norm_phi=norm_phi)

    if OptDB.getBool('main_fun_random', False):
        OptDB.setValue('main_fun', False)
        n_norm = OptDB.getReal('n_norm', 1)
        main_fun_random(n_norm=n_norm)

    if OptDB.getBool('main_fun', True):
        main_fun()

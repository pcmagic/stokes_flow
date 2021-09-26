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

    n_norm_theta = OptDB.getInt('n_norm_theta', 2)
    n_norm_phi = OptDB.getInt('n_norm_phi', 2)
    norm_psi1 = OptDB.getReal('norm_psi1', 0)
    norm_psi2 = OptDB.getReal('norm_psi2', 0)
    th_idx = OptDB.getInt('th_idx', 0)
    problem_kwargs['n_norm_theta'] = n_norm_theta
    problem_kwargs['n_norm_phi'] = n_norm_phi
    problem_kwargs['norm_psi1'] = norm_psi1
    problem_kwargs['norm_psi2'] = norm_psi2
    problem_kwargs['th_idx'] = th_idx
    rel_tail1 = OptDB.getReal('rel_tail1', 0)
    rel_tail2 = OptDB.getReal('rel_tail2', 0)
    problem_kwargs['rel_tail1'] = rel_tail1
    problem_kwargs['rel_tail2'] = rel_tail2

    kwargs_list = (get_shearFlow_kwargs(), main_kwargs,)
    for t_kwargs in kwargs_list:
        for key in t_kwargs:
            problem_kwargs[key] = t_kwargs[key]
    pickle_name = '%s_kwargs.pickle' % fileHandle
    with open(pickle_name, 'wb') as handle:
        pickle.dump(problem_kwargs, handle, protocol=4)
    PETSc.Sys.Print('---->save kwargs to %s' % pickle_name)
    return problem_kwargs


def print_case_info(**problem_kwargs):
    caseIntro = '-->Ecoli in infinite shear flow case, given speed and torque free case. '
    ec.print_case_info(caseIntro, **problem_kwargs)
    th_idx = problem_kwargs['th_idx']
    PETSc.Sys.Print('----> DBG th_idx=%d' % th_idx)

    n_norm_theta = problem_kwargs['n_norm_theta']
    n_norm_phi = problem_kwargs['n_norm_phi']
    norm_psi1 = problem_kwargs['norm_psi1']
    norm_psi2 = problem_kwargs['norm_psi2']
    rel_tail1 = problem_kwargs['rel_tail1']
    rel_tail2 = problem_kwargs['rel_tail2']
    PETSc.Sys.Print('  -->rel_tail1 %f, rel_tail2 %f' % (rel_tail1, rel_tail2))
    PETSc.Sys.Print('Loop parameter space: n_norm_theta %d, n_norm_phi %d. '
                    'norm_psi1 %f, norm_psi2 %f' %
                    (n_norm_theta, n_norm_phi, norm_psi1, norm_psi2))
    print_shearFlow_info(**problem_kwargs)
    return True


def do_solve_once_noIter(problem_ff: sf.ShearFlowForceFreeProblem,
                         ecoli_comp: sf.ForceFreeComposite,
                         fileHandle, norm_theta, norm_phi, norm_psi1, norm_psi2,
                         planeShearRate, rank, idx, N):
    PETSc.Sys.Print()
    PETSc.Sys.Print('%s %05d / %05d theta=%f, phi=%f, psi1=%f psi2=%f %s' %
                    ('#' * 20, idx, N, norm_theta, norm_phi, norm_psi1, norm_psi2, '#' * 20,))
    problem_ff.create_matrix()
    problem_ff.solve()
    ref_U = ecoli_comp.get_ref_U()
    PETSc.Sys.Print('  ini ref_U in shear flow', ref_U)

    if rank == 0:
        mat_name = '%s_th%f_phi%f_psi1-%f_psi2-%f.mat' % \
                   (fileHandle, norm_theta, norm_phi, norm_psi1, norm_psi2)
        savemat(mat_name, {
            'norm_theta':     norm_theta,
            'norm_phi':       norm_phi,
            'norm_psi1':      norm_psi1,
            'norm_psi2':      norm_psi2,
            'planeShearRate': planeShearRate,
            'ecoli_center':   np.vstack(ecoli_comp.get_center()),
            'ecoli_norm':     np.vstack(ecoli_comp.get_norm()),
            'ecoli_U':        np.vstack(ecoli_comp.get_ref_U()),
            'rel_U':          np.vstack(ecoli_comp.get_rel_U_list())}, oned_as='column', )
    return True


def main_fun_noIter(**main_kwargs):
    comm = PETSc.COMM_WORLD.tompi4py()
    rank = comm.Get_rank()
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    print_case_info(**problem_kwargs)
    fileHandle = problem_kwargs['fileHandle']
    n_norm_theta = problem_kwargs['n_norm_theta']
    n_norm_phi = problem_kwargs['n_norm_phi']
    norm_psi1 = problem_kwargs['norm_psi1']
    norm_psi2 = problem_kwargs['norm_psi2']
    N = n_norm_phi * n_norm_theta
    rel_tail1 = problem_kwargs['rel_tail1']
    rel_tail2 = problem_kwargs['rel_tail2']
    th_idx = problem_kwargs['th_idx']

    if not problem_kwargs['restart']:
        # create problem
        ecoli_comp = create_ecoli_dualTail(**problem_kwargs)
        rel_U_list = [np.zeros(6),
                      np.array((0, 0, 0, 0, 0, rel_tail1)),
                      np.array((0, 0, 0, 0, 0, rel_tail2))]
        ecoli_comp.set_rel_U_list(rel_U_list)
        tail_obj1 = ecoli_comp.get_obj_list()[1]
        tail_obj1.node_rotation(tail_obj1.get_u_geo().get_geo_norm(), norm_psi1)
        tail_obj2 = ecoli_comp.get_obj_list()[2]
        tail_obj2.node_rotation(tail_obj2.get_u_geo().get_geo_norm(), norm_psi2)
        problem_ff = sf.ShearFlowForceFreeProblem(**problem_kwargs)
        problem_ff.add_obj(ecoli_comp)
        problem_ff.print_info()
        problem_ff.create_matrix()
        planeShearRate = problem_ff.get_planeShearRate()

        # loop over parameter space
        PETSc.Sys.Print(np.linspace(0, np.pi, n_norm_theta)[th_idx:])
        for i0, norm_theta in enumerate(np.linspace(0, np.pi, n_norm_theta)[th_idx:]):
            ecoli_comp.node_rotation(np.array((0, 1, 0)), norm_theta)
            for i1, norm_phi in enumerate(np.linspace(0, 2 * np.pi, n_norm_phi)):
                t2 = time()
                idx = i0 * n_norm_phi + i1 + 1
                ecoli_comp.node_rotation(np.array((0, 0, 1)), norm_phi)
                # problem_ff.show_velocity(0.001)
                do_solve_once_noIter(problem_ff, ecoli_comp, fileHandle, norm_theta, norm_phi,
                                     norm_psi1, norm_psi2, planeShearRate, rank, idx, N)
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

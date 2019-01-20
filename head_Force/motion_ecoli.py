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
from src.objComposite import createEcoliComp_ellipse
# from src.myvtk import save_singleEcoli_vtk
import ecoli_in_pipe.ecoli_common as ec
import os


# import import_my_lib

# Todo: rewrite input and print process.
def get_problem_kwargs(**main_kwargs):
    OptDB = PETSc.Options()
    fileHandle = OptDB.getString('f', 'motion_ecoli')
    OptDB.setValue('f', fileHandle)
    problem_kwargs = ec.get_problem_kwargs()
    problem_kwargs['fileHandle'] = fileHandle

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
    caseIntro = '-->Ecoli in infinite shear flow case, force free case. '
    ec.print_case_info(caseIntro, **problem_kwargs)
    print_update_info(**problem_kwargs)
    print_shearFlow_info(**problem_kwargs)
    return True


# @profile
def main_fun(**main_kwargs):
    comm = PETSc.COMM_WORLD.tompi4py()
    rank = comm.Get_rank()
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    print_case_info(**problem_kwargs)
    fileHandle = problem_kwargs['fileHandle']
    max_iter = problem_kwargs['max_iter']
    update_order = problem_kwargs['update_order']
    eval_dt = problem_kwargs['eval_dt']
    rs1 = problem_kwargs['rs1']

    if not problem_kwargs['restart']:
        # create obj
        ecoli_comp = createEcoliComp_ellipse(name='ecoli0', **problem_kwargs)
        ecoli_comp.set_update_para(fix_x=False, fix_y=False, fix_z=False,
                                   update_order=update_order, update_fun=Adams_Moulton_Methods)

        # prepare problem
        problem = sf.ShearFlowForceFreeProblem(**problem_kwargs)
        for obj in [ecoli_comp, ]:
            problem.add_obj(obj)
        problem.print_info()
        problem.create_matrix()
        # # dbg
        # problem.solve()
        # print_single_ecoli_force_result(ecoli_comp, prefix='', part='full', **problem_kwargs)
        # ref_U = ecoli_comp.get_ref_U()
        # norm = ecoli_comp.get_norm()
        # PETSc.Sys.Print('    ref_U', ref_U)
        # PETSc.Sys.Print('    norm', norm)
        # tU = np.dot(ref_U[:3], norm) / np.dot(norm, norm)
        # tW = np.dot(ref_U[3:], norm) / np.dot(norm, norm)
        # PETSc.Sys.Print('    |ref_U|', np.hstack((np.linalg.norm(ref_U[:3]), np.linalg.norm(ref_U[3:]))))
        # PETSc.Sys.Print('    ref_U projection on norm', np.hstack((tU, tW)))

        # evaluation loop
        t0 = time()
        for idx in range(1, max_iter + 1):
            problem.solve()
            print_single_ecoli_force_result(ecoli_comp, prefix='', part='full', **problem_kwargs)
            if rank == 0:
                savemat('%s_%05d' % (fileHandle, idx), {
                    'ecoli_center': np.vstack(ecoli_comp.get_center()),
                    'ecoli_nodes':  np.vstack([tobj.get_u_nodes() for tobj in
                                               ecoli_comp.get_obj_list()]),
                    'ecoli_f':      np.hstack(
                            [tobj.get_force() for tobj in ecoli_comp.get_obj_list()]
                    ).reshape(-1, 3),
                    'ecoli_u':      np.hstack([tobj.get_re_velocity() for tobj in
                                               ecoli_comp.get_obj_list()]
                                              ).reshape(-1, 3),
                    'ecoli_norm':   np.vstack(
                            ecoli_comp.get_obj_list()[0].get_u_geo().get_geo_norm()),
                    'ecoli_U':      np.vstack(ecoli_comp.get_ref_U())}, oned_as='column')
            # ref_U = ecoli_comp.get_ref_U()
            # fct = rs1 / np.linalg.norm(ref_U[:3])
            # Todo: check if ecoli out of pipe boundary.
            problem.update_location(eval_dt, print_handle='%d / %d' % (idx, max_iter))
            problem.create_matrix()
            # problem.show_u_nodes()
            # problem.vtk_obj(fileHandle, idx)
            # problem.vtk_tetra('%s_vtkU_%05d' % (fileHandle, idx), vtk_geo)
        t1 = time()
        PETSc.Sys.Print('%s: run %d loops using %f' % (fileHandle, max_iter, (t1 - t0)))
        # # dbg
        # PETSc.Sys.Print(ecoli_comp.get_center_hist())
        # PETSc.Sys.Print(ecoli_comp.get_obj_list()[0].get_obj_norm_hist())
        # PETSc.Sys.Print(ecoli_comp.get_ref_U_hist())

        problem.destroy()
        if rank == 0:
            savemat(fileHandle,
                    {'ecoli_center': np.vstack(ecoli_comp.get_center_hist()),
                     'ecoli_norm':   np.vstack(ecoli_comp.get_norm_hist()),
                     'ecoli_U':      np.vstack(ecoli_comp.get_ref_U_hist())},
                    oned_as='column')
    else:
        pass
    return True


if __name__ == '__main__':
    main_fun()

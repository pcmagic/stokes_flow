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
    fileHandle = OptDB.getString('f', 'motion_ecoli_speed')
    OptDB.setValue('f', fileHandle)
    problem_kwargs = ec.get_problem_kwargs()
    problem_kwargs['fileHandle'] = fileHandle

    ecoli_velocity = OptDB.getReal('ecoli_velocity', 0)
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
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    print_case_info(**problem_kwargs)
    fileHandle = problem_kwargs['fileHandle']
    max_iter = problem_kwargs['max_iter']
    eval_dt = problem_kwargs['eval_dt']
    update_order = problem_kwargs['update_order']
    update_fun = problem_kwargs['update_fun']
    ecoli_velocity = problem_kwargs['ecoli_velocity']

    if not problem_kwargs['restart']:
        # create obj
        head_obj, tail_obj_list = createEcoli_ellipse(name='ecoli0', **problem_kwargs)
        tail_obj = sf.StokesFlowObj()
        tail_obj.combine(tail_obj_list)

        head_geo = head_obj.get_u_geo()
        head_norm = head_geo.get_geo_norm()
        givenU = head_norm * ecoli_velocity
        ecoli_comp = sf.GivenVelocityComposite(center=head_geo.get_center(), norm=head_geo.get_geo_norm(),
                                               givenU=givenU, name='ecoli_0')
        ecoli_comp.add_obj(obj=head_obj, rel_U=np.zeros(6))
        ecoli_comp.add_obj(obj=tail_obj, rel_U=np.zeros(6))
        ecoli_comp.set_update_para(fix_x=False, fix_y=False, fix_z=False,
                                   update_fun=update_fun, update_order=update_order)

        # prepare problem
        problem = sf.GivenVelocityProblem(**problem_kwargs)
        for obj in [ecoli_comp, ]:
            problem.add_obj(obj)
        problem.print_info()
        # # dbg
        # problem.create_matrix()
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
            t2 = time()
            PETSc.Sys.Print()
            PETSc.Sys.Print('#########################################################################################')
            problem.create_matrix()
            problem.solve()
            print_single_ecoli_force_result(ecoli_comp, prefix='', part='full', **problem_kwargs)
            ellipse_norm = head_geo.get_geo_norm()
            givenU = ellipse_norm * ecoli_velocity
            ecoli_comp.set_givenU(givenU)
            problem.update_location(eval_dt, print_handle='%d / %d' % (idx, max_iter))
            re_ecoli_F = ecoli_comp.get_total_force()
            t3 = time()
            PETSc.Sys.Print('----> Current loop %d / %d uses: %fs, re_ecoli_F: %s' %
                            (idx, max_iter, (t3 - t2), str(re_ecoli_F)))
            # if rank == 0:
            #     savemat('%s_%05d' % (fileHandle, idx), {
            #         'ecoli_center': np.vstack(ecoli_comp.get_center()),
            #         'ecoli_nodes':  np.vstack([tobj.get_u_nodes() for tobj in
            #                                    ecoli_comp.get_obj_list()]),
            #         'ecoli_f':      np.hstack(
            #                 [tobj.get_force() for tobj in ecoli_comp.get_obj_list()]
            #         ).reshape(-1, 3),
            #         'ecoli_u':      np.hstack([tobj.get_re_velocity() for tobj in
            #                                    ecoli_comp.get_obj_list()]
            #                                   ).reshape(-1, 3),
            #         'ecoli_norm':   np.vstack(
            #                 ecoli_comp.get_obj_list()[0].get_u_geo().get_geo_norm()),
            #         'ecoli_U':      np.vstack(ecoli_comp.get_ref_U())}, oned_as='column')
        t1 = time()
        PETSc.Sys.Print('%s: run %d loops using %f' % (fileHandle, max_iter, (t1 - t0)))

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

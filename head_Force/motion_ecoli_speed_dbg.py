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
    err_msg = 'ecoli_velocity >= 0'
    assert ecoli_velocity >= 0, err_msg
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
    update_order = problem_kwargs['update_order']
    update_fun = problem_kwargs['update_fun']
    iter_tor = 1e-1

    if not problem_kwargs['restart']:
        ecoli_comp = create_ecoli_2part(**problem_kwargs)

        problem_forcefree = sf.ShearFlowForceFreeProblem(**problem_kwargs)
        problem_forcefree.add_obj(ecoli_comp)
        problem = sf.ShearFlowForceFreeIterateProblem(**problem_kwargs)
        problem.add_obj(ecoli_comp)
        problem.set_iterate_comp(ecoli_comp)
        problem.print_info()
        ec.shear_sovle_velocity_1step(ecoli_comp, problem_forcefree, problem,
                                      v_ecoli_tor=1e-2, iter_tor=iter_tor, **problem_kwargs)

        # # get the ini_guess of the ref_U
        # PETSc.Sys.Print()
        # PETSc.Sys.Print('get the ini_guess of the ref_U')
        # problem = sf.ShearFlowForceFreeProblem(**problem_kwargs)
        # problem.add_obj(ecoli_comp)
        # problem.create_matrix()
        # problem.solve()
        # ref_U1 = ecoli_comp.get_ref_U()
        # problem.set_ffweight(*(problem.get_ffweight()[:4] * [10, 10, 10, 100]))
        # problem.create_F_U()
        # problem.set_force_free()
        # problem.solve()
        # ref_U0 = ecoli_comp.get_ref_U()
        #
        # # prepare problem
        # PETSc.Sys.Print()
        # problem = sf.ShearFlowForceFreeIterateProblem(tolerate=1e-1, **problem_kwargs)
        # problem.add_obj(ecoli_comp)
        # problem.print_info()
        # problem.create_matrix()
        # problem.set_iterate_comp(ecoli_comp)
        # problem.create_matrix()
        # refU, Ftol, Ttol = problem.do_iterate(ini_refU0= ref_U0, ini_refU1=ref_U1)
        # ecoli_comp.set_ref_U(refU)
        # print_single_ecoli_force_result(ecoli_comp, prefix='', part='full', **problem_kwargs)
        # ref_U = ecoli_comp.get_ref_U()
        # center = ecoli_comp.get_center()
        # PETSc.Sys.Print('ref_U %s' % str(ref_U))
        # PETSc.Sys.Print('|ref_U| %s, %s' % (str(np.linalg.norm(ref_U[:3])), str(np.linalg.norm(ref_U[3:]))))
        # PETSc.Sys.Print('U_head %s' % str(ref_U + ecoli_comp.get_rel_U_list()[0]))
        # PETSc.Sys.Print('U_tail %s' % str(ref_U + ecoli_comp.get_rel_U_list()[1]))
        # sumF = np.sum(np.vstack([tobj.get_total_force(center=center) for tobj in ecoli_comp.get_obj_list()]), axis=0)
        # PETSc.Sys.Print('check, sumF is %s' % str(sumF))
        # PETSc.Sys.Print('check, sumF/headF is %s' %
        #                 str(sumF / ecoli_comp.get_obj_list()[0].get_total_force(center=center)))

        # # an iterative method to find motor spin
        # iterateTolerate = 1e-5
        # err_ref_u = np.inf
        # tmp_rel_U = rel_U_list[1]
        # while err_ref_u > iterateTolerate:
        #     problem.create_F_U()
        #     problem.solve()
        #     ref_U = ecoli_comp.get_ref_U()
        #     norm_ref_u = np.linalg.norm(ref_U[:3])
        #     err_ref_u = np.abs((norm_ref_u - ecoli_velocity) / ecoli_velocity)
        #     tmp_rel_U = tmp_rel_U * ecoli_velocity / norm_ref_u
        #     tmp_rel_U_list = [rel_Us, tmp_rel_U]
        #     ecoli_comp.dbg_set_rel_U_list(tmp_rel_U_list)
        #     PETSc.Sys.Print('############################################################################')
        #     PETSc.Sys.Print('ref_U', ref_U)
        #     PETSc.Sys.Print('norm_ref_U', norm_ref_u)
        #     PETSc.Sys.Print('err_ref_U', err_ref_u)
        #     PETSc.Sys.Print('tmp_rel_U', tmp_rel_U)
        # print_single_ecoli_force_result(ecoli_comp, prefix='', part='full', **problem_kwargs)
        # problem.destroy()

        # # dbg, check if force and torque free
        # ref_U = ecoli_comp.get_ref_U()
        # center = ecoli_comp.get_center()
        # head_rel_U, tail_rel_U = ecoli_comp.get_rel_U_list()
        # head_u_geo = head_obj.get_u_geo()
        # head_u_geo.set_rigid_velocity(head_rel_U + ref_U)
        # tail_u_geo = tail_obj.get_u_geo()
        # tail_u_geo.set_rigid_velocity(tail_rel_U + ref_U)
        # problem = sf.StokesFlowProblem(**problem_kwargs)
        # problem.add_obj(head_obj)
        # problem.add_obj(tail_obj)
        # # problem.print_info()
        # problem.create_matrix()
        # problem.solve()
        # sumF = head_obj.get_total_force(center=center) + tail_obj.get_total_force(center=center)
        # PETSc.Sys.Print('check, sumF is %s' % str(sumF))
        # PETSc.Sys.Print('check, sumF/headF is %s' % str(sumF / head_obj.get_total_force(center=center)))
    else:
        pass
    return True


if __name__ == '__main__':
    main_fun()

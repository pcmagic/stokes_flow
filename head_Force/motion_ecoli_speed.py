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
    iter_tor = 1e-1

    if not problem_kwargs['restart']:
        ecoli_comp = create_ecoli_2part(**problem_kwargs)
        problem_forcefree = sf.ShearFlowForceFreeProblem(**problem_kwargs)
        problem_forcefree.add_obj(ecoli_comp)
        problem = sf.ShearFlowForceFreeIterateProblem(**problem_kwargs)
        problem.add_obj(ecoli_comp)
        problem.set_iterate_comp(ecoli_comp)
        problem.print_info()

        # evaluation loop
        t0 = time()
        for idx in range(1, max_iter + 1):
            t2 = time()
            PETSc.Sys.Print()
            PETSc.Sys.Print('#########################################################################################')
            ec.shear_sovle_velocity_1step(ecoli_comp, problem_forcefree, problem,
                                          v_ecoli_tor=1e-2, iter_tor=iter_tor, **problem_kwargs)
            problem.update_location(eval_dt, print_handle='%d / %d' % (idx, max_iter))
            t3 = time()
            PETSc.Sys.Print('----> Current loop %d / %d uses: %fs' % (idx, max_iter, (t3 - t2)))
            if rank == 0:
                savemat('%s_%05d.mat' % (fileHandle, idx), {
                    'ecoli_center': np.vstack(ecoli_comp.get_center()),
                    'ecoli_nodes':  np.vstack([tobj.get_u_nodes() for tobj in ecoli_comp.get_obj_list()]),
                    'ecoli_f':      np.hstack([tobj.get_force() for tobj in ecoli_comp.get_obj_list()]).reshape(-1, 3),
                    'ecoli_u':      np.hstack([tobj.get_re_velocity() for tobj in ecoli_comp.get_obj_list()]
                                              ).reshape(-1, 3),
                    'ecoli_norm':   np.vstack(ecoli_comp.get_obj_list()[0].get_u_geo().get_geo_norm()),
                    'ecoli_U':      np.vstack(ecoli_comp.get_ref_U()),
                    'tail_rel_U':   np.vstack(ecoli_comp.get_rel_U_list()[1])}, oned_as='column', )
        t1 = time()
        PETSc.Sys.Print('%s: run %d loops using %f' % (fileHandle, max_iter, (t1 - t0)))

        problem.destroy()
        problem_forcefree.destroy()
        if rank == 0:
            savemat('%s.mat' % fileHandle,
                    {'ecoli_center': np.vstack(ecoli_comp.get_center_hist()),
                     'ecoli_norm':   np.vstack(ecoli_comp.get_norm_hist()),
                     'ecoli_U':      np.vstack(ecoli_comp.get_ref_U_hist()),
                     't':            (np.arange(max_iter) + 1) * eval_dt},
                    oned_as='column')

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

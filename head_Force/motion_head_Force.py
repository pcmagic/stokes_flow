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
# from src.objComposite import createEcoliComp_tunnel
# from src.myvtk import save_singleEcoli_vtk
import codeStore.ecoli_common as ec


# import import_my_lib

# Todo: rewrite input and print process.
def get_problem_kwargs(**main_kwargs):
    OptDB = PETSc.Options()
    fileHandle = OptDB.getString('f', 'motion_head_Force')
    OptDB.setValue('f', fileHandle)
    problem_kwargs = ec.get_problem_kwargs()
    problem_kwargs['fileHandle'] = fileHandle

    # vtk_matname = OptDB.getString('vtk_matname', 'pipe_dbg')
    # t_path = os.path.dirname(os.path.abspath(__file__))
    # vtk_matname = os.path.normpath(os.path.join(t_path, vtk_matname))
    # problem_kwargs['vtk_matname'] = vtk_matname
    ellipse_F_dist = OptDB.getReal('ellipse_F_dist', 1)
    ellipse_centerx = OptDB.getReal('ellipse_centerx', 0)
    ellipse_centery = OptDB.getReal('ellipse_centery', 0)
    ellipse_centerz = OptDB.getReal('ellipse_centerz', 0)
    ellipse_center = np.array(
            (ellipse_centerx, ellipse_centery, ellipse_centerz))  # center of ecoli
    ecoli_tail_strength = OptDB.getReal('ecoli_tail_strength', 1)
    problem_kwargs['ellipse_center'] = ellipse_center
    problem_kwargs['ecoli_tail_strength'] = ecoli_tail_strength
    problem_kwargs['ellipse_F_dist'] = ellipse_F_dist

    # kwargs_list = (get_pipe_kwargs(), get_PoiseuilleFlow_kwargs(), get_shearFlow_kwargs(), main_kwargs,)
    kwargs_list = (get_pipe_kwargs(), get_shearFlow_kwargs(), get_update_kwargs(), main_kwargs,)
    for t_kwargs in kwargs_list:
        for key in t_kwargs:
            problem_kwargs[key] = t_kwargs[key]
    return problem_kwargs


def print_case_info(**problem_kwargs):
    caseIntro = '-->Ecoli in pipe case, force free case, finite pipe and use regularized Stokeslets. '
    ec.print_case_info(caseIntro, **problem_kwargs)
    ellipse_center = problem_kwargs['ellipse_center']
    ecoli_tail_strength = problem_kwargs['ecoli_tail_strength']
    ellipse_F_dist = problem_kwargs['ellipse_F_dist']
    PETSc.Sys.Print('    ellipse_center %s, ecoli_tail_strength %f, ellipse_F_dist %f' %
                    (str(ellipse_center), ecoli_tail_strength, ellipse_F_dist))

    print_update_info(**problem_kwargs)
    print_pipe_info(**problem_kwargs)
    # print_PoiseuilleFlow_info(**problem_kwargs)
    print_shearFlow_info(**problem_kwargs)
    return True


def create_ellipse_obj(**problem_kwargs):
    ds = problem_kwargs['ds']
    rs1 = problem_kwargs['rs1']
    rs2 = problem_kwargs['rs2']
    rot_norm = problem_kwargs['rot_norm']
    rot_theta = problem_kwargs['rot_theta']
    ellipse_center = problem_kwargs['ellipse_center']
    # matrix_method = problem_kwargs['matrix_method']
    ecoli_tail_strength = problem_kwargs['ecoli_tail_strength']
    ellipse_F_dist = problem_kwargs['ellipse_F_dist']

    ellipse_geo0 = ellipse_base_geo()
    ellipse_geo0.create_delta(ds, rs1, rs2)
    ellipse_geo0.node_rotation(rot_norm, rot_theta)
    ellipse_geo0.move(ellipse_center)
    ellipse_geo0.set_rigid_velocity(np.zeros(6))
    ellipse_obj = sf.FundSoltObj()
    ellipse_obj.set_data(ellipse_geo0, ellipse_geo0, name='head_ellipse')
    F_ellipse = -1 * ellipse_geo0.get_geo_norm() * ecoli_tail_strength
    location = ellipse_F_dist * ellipse_geo0.get_geo_norm() + ellipse_geo0.get_center()
    ellipse_obj.add_point_force(location=location, force=F_ellipse,
                                StokesletsHandle=light_stokeslets_matrix_3d)
    return ellipse_obj


# @profile
def main_fun(**main_kwargs):
    main_kwargs['rot_norm'] = np.array((0, 1, 0))
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    # print(problem_kwargs)
    # print()
    # print()
    # print()
    print_case_info(**problem_kwargs)
    fileHandle = problem_kwargs['fileHandle']
    max_iter = problem_kwargs['max_iter']
    eval_dt = problem_kwargs['eval_dt']
    update_order = problem_kwargs['update_order']
    update_fun = problem_kwargs['update_fun']
    norm_theta = problem_kwargs['norm_theta']
    norm_phi = problem_kwargs['norm_phi']
    PETSc.Sys.Print('-->norm_theta=%f, norm_phi=%f' % (norm_theta, norm_phi))

    if not problem_kwargs['restart']:
        # create obj
        ellipse_obj = create_ellipse_obj(**problem_kwargs)
        ellipse_obj.node_rotation(np.array((0, 1, 0)), norm_theta)
        ellipse_obj.node_rotation(np.array((0, 0, 1)), norm_phi)
        ellipse_geo0 = ellipse_obj.get_u_geo()
        F_ellipse = ellipse_obj.get_point_force_list()[0][1]
        givenF = np.hstack((F_ellipse, np.zeros(3)))
        ecoli_comp = sf.GivenForceComposite(center=ellipse_geo0.get_center(),
                                            norm=ellipse_geo0.get_geo_norm(),
                                            name='ecoli_0', givenF=givenF)
        ecoli_comp.add_obj(obj=ellipse_obj, rel_U=np.zeros(6))
        ecoli_comp.set_update_para(fix_x=False, fix_y=False, fix_z=False,
                                   update_fun=update_fun, update_order=update_order)

        # Prepare Problem
        problem = sf.GivenForceShearFlowProblem(**problem_kwargs)
        problem.add_obj(ecoli_comp)
        problem.print_info()
        problem.create_matrix()

        # evaluation loop
        t0 = time()
        for idx in range(max_iter):
            t2 = time()
            problem.solve()
            problem.update_location(eval_dt, print_handle='%d / %d' % (idx, max_iter))
            problem.create_matrix()
            re_ecoli_F = ecoli_comp.get_total_force()
            # problem.vtk_obj(fileHandle, idx)
            t3 = time()
            PETSc.Sys.Print('----> Current loop %d / %d uses: %fs, re_ecoli_F: %s' %
                            (idx, max_iter, (t3 - t2), str(re_ecoli_F)))
        t1 = time()
        PETSc.Sys.Print('%s: run %d loops using %f' % (fileHandle, max_iter, (t1 - t0)))
        # # dbg
        # PETSc.Sys.Print(ecoli_comp.get_center_hist())
        # PETSc.Sys.Print(ellipse_obj.get_obj_norm_hist())
        # PETSc.Sys.Print(ellipse_obj.get_force_norm_hist()[0])
        # PETSc.Sys.Print(ecoli_comp.get_ref_U_hist())

        problem.destroy()
        comm = PETSc.COMM_WORLD.tompi4py()
        rank = comm.Get_rank()
        if rank == 0:
            savemat('%s.mat' % fileHandle,
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
    norm_theta = OptDB.getReal('norm_theta', 0)
    norm_phi = OptDB.getReal('norm_phi', 0)
    main_fun(norm_theta=norm_theta, norm_phi=norm_phi)

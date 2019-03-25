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
# from src.support_class import *
# from src.objComposite import createEcoliComp_tunnel
# from src.myvtk import save_singleEcoli_vtk
import ecoli_in_pipe.ecoli_common as ec
import os


# import import_my_lib

# Todo: rewrite input and print process.
def get_problem_kwargs(**main_kwargs):
    OptDB = PETSc.Options()
    fileHandle = OptDB.getString('f', 'motion_head_Force_pipe')
    OptDB.setValue('f', fileHandle)
    problem_kwargs = ec.get_problem_kwargs()
    problem_kwargs['fileHandle'] = fileHandle

    kwargs_list = (get_pipe_kwargs(), get_PoiseuilleFlow_kwargs(), get_update_kwargs(), main_kwargs,)
    for t_kwargs in kwargs_list:
        for key in t_kwargs:
            problem_kwargs[key] = t_kwargs[key]

    vtk_matname = OptDB.getString('vtk_matname', 'pipe_dbg')
    t_path = os.path.dirname(os.path.abspath(__file__))
    vtk_matname = os.path.normpath(os.path.join(t_path, vtk_matname))
    ellipse_F_dist = OptDB.getReal('ellipse_F_dist', 1)
    ellipse_centerx = OptDB.getReal('ellipse_centerx', 0)
    ellipse_centery = OptDB.getReal('ellipse_centery', 0)
    ellipse_centerz = OptDB.getReal('ellipse_centerz', 0)
    ellipse_center = np.array((ellipse_centerx, ellipse_centery, ellipse_centerz))  # center of ecoli
    ecoli_tail_strength = OptDB.getReal('ecoli_tail_strength', 1)
    problem_kwargs['vtk_matname'] = vtk_matname
    problem_kwargs['ellipse_center'] = ellipse_center
    problem_kwargs['ecoli_tail_strength'] = ecoli_tail_strength
    problem_kwargs['ellipse_F_dist'] = ellipse_F_dist
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
    print_PoiseuilleFlow_info(**problem_kwargs)
    return True


def create_pipe_obj(**problem_kwargs):
    finite_pipe_length = problem_kwargs['finite_pipe_length']
    finite_pipe_cover = problem_kwargs['finite_pipe_cover']
    finite_pipe_ntheta = problem_kwargs['finite_pipe_ntheta']
    matrix_method = problem_kwargs['matrix_method']

    pipe_geo0 = tunnel_geo()
    pipe_geo0.create_deltatheta(2 * np.pi / finite_pipe_ntheta, 1, finite_pipe_length,
                                1, finite_pipe_cover, factor=1, left_hand=False)
    pipe_geo0.set_rigid_velocity(np.array((0, 0, 0, 0, 0, 0)))
    pipe_obj = sf.obj_dic[matrix_method]()
    pipe_obj.set_data(pipe_geo0, pipe_geo0, name='finite_pipe')
    return pipe_obj


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

    ellipse_geo0 = ellipse_geo()
    ellipse_geo0.create_delta(ds, rs1, rs2)
    ellipse_geo0.node_rotation(rot_norm, rot_theta)
    ellipse_geo0.move(ellipse_center)
    ellipse_geo0.set_rigid_velocity(np.zeros(6))
    ellipse_obj = sf.PointForceObj()
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
    print_case_info(**problem_kwargs)
    fileHandle = problem_kwargs['fileHandle']
    max_iter = problem_kwargs['max_iter']
    eval_dt = problem_kwargs['eval_dt']

    if not problem_kwargs['restart']:
        # create obj
        ellipse_obj = create_ellipse_obj(**problem_kwargs)
        ellipse_geo0 = ellipse_obj.get_re_velocity()
        pipe_obj = create_pipe_obj(**problem_kwargs)
        F_ellipse = ellipse_obj.get_point_force_list()[0][1]
        givenF = np.hstack((F_ellipse, np.zeros(3)))
        ecoli_comp = sf.GivenForceComposite(center=ellipse_geo0.get_center(), name='ecoli_0', givenF=givenF)
        ecoli_comp.add_obj(obj=ellipse_obj, rel_U=np.zeros(6))
        ecoli_comp.set_update_para(fix_x=False, fix_y=False, fix_z=True)

        # prepare problem
        problem = sf.GivenForcePoiseuilleFlowProblem(**problem_kwargs)
        problem.add_given_flow_obj(pipe_obj)
        for obj in [ecoli_comp, pipe_obj]:
            problem.add_obj(obj)
        problem.print_info()
        problem.create_matrix()
        vtk_matname = problem_kwargs['vtk_matname']
        vtk_geo = geo()
        vtk_geo.mat_nodes(filename=vtk_matname, mat_handle='vtk_nodes')
        vtk_geo.mat_elmes(filename=vtk_matname, mat_handle='vtk_elems', elemtype='tetra')

        # evaluation loop
        t0 = time()
        for idx in range(max_iter):
            problem.solve()
            problem.update_location(eval_dt, print_handle='%d / %d' % (idx, max_iter))
            problem.updata_matrix(ellipse_obj, ellipse_obj)
            problem.updata_matrix(pipe_obj, ellipse_obj)
            problem.updata_matrix(ellipse_obj, pipe_obj)
            # Todo: check if ecoli out of pipe boundary.
            re_ecoli_F = ecoli_comp.get_total_force()
            PETSc.Sys.Print('    re_err_F', (re_ecoli_F - givenF) / givenF)
            problem.vtk_obj(fileHandle, idx)
            problem.vtk_tetra('%s_vtkU_%05d' % (fileHandle, idx), vtk_geo)
        t1 = time()
        PETSc.Sys.Print('%s: run %d loops using %f' % (fileHandle, max_iter, (t1 - t0)))
        # dbg
        PETSc.Sys.Print(ecoli_comp.get_center_hist())
        PETSc.Sys.Print(ellipse_obj.get_obj_norm_hist())
        PETSc.Sys.Print(ellipse_obj.get_force_norm_hist()[0])
        PETSc.Sys.Print(ecoli_comp.get_ref_U_hist())

        problem.destroy()
        comm = PETSc.COMM_WORLD.tompi4py()
        rank = comm.Get_rank()
        if rank == 0:
            savemat(fileHandle,
                    {'ecoli_center': np.vstack(ecoli_comp.get_center_hist()),
                     'ecoli_norm':   np.vstack(ellipse_obj.get_obj_norm_hist()),
                     'ecoli_Floc':   np.vstack(ellipse_obj.get_force_norm_hist()[0]),
                     'ecoli_U':      np.vstack(ecoli_comp.get_ref_U_hist())},
                    oned_as='column')
    else:
        pass
    return True


if __name__ == '__main__':
    main_fun()

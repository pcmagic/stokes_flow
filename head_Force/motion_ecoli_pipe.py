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
    fileHandle = OptDB.getString('f', 'ecoliInPipe_rs')
    OptDB.setValue('f', fileHandle)
    problem_kwargs = ec.get_problem_kwargs()
    problem_kwargs['fileHandle'] = fileHandle

    kwargs_list = (get_pipe_kwargs(), get_PoiseuilleFlow_kwargs(), main_kwargs,)
    for t_kwargs in kwargs_list:
        for key in t_kwargs:
            problem_kwargs[key] = t_kwargs[key]

    vtk_matname = OptDB.getString('vtk_matname', 'pipe_dbg')
    t_path = os.path.dirname(os.path.abspath(__file__))
    vtk_matname = os.path.normpath(os.path.join(t_path, vtk_matname))
    max_iter = OptDB.getInt('max_iter', 10)
    eval_dt = OptDB.getReal('eval_dt', 0.001)
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
    problem_kwargs['max_iter'] = max_iter
    problem_kwargs['eval_dt'] = eval_dt
    return problem_kwargs


def print_case_info(**problem_kwargs):
    caseIntro = '-->Ecoli in pipe case, force free case, finite pipe and use regularized Stokeslets. '
    ec.print_case_info(caseIntro, **problem_kwargs)
    ellipse_center = problem_kwargs['ellipse_center']
    ecoli_tail_strength = problem_kwargs['ecoli_tail_strength']
    ellipse_F_dist = problem_kwargs['ellipse_F_dist']
    PETSc.Sys.Print('    ellipse_center %s, ecoli_tail_strength %f, ellipse_F_dist %f' %
                    (str(ellipse_center), ecoli_tail_strength, ellipse_F_dist))

    max_iter = problem_kwargs['max_iter']
    eval_dt = problem_kwargs['eval_dt']
    PETSc.Sys.Print('Interation Loop: max_iter %f, eval_dt %f' % (max_iter, eval_dt))

    print_pipe_info(**problem_kwargs)
    print_PoiseuilleFlow_info(**problem_kwargs)
    return True


def create_pipe_obj(**problem_kwargs):
    finite_pipe_length = problem_kwargs['finite_pipe_length']
    finite_pipe_cover = problem_kwargs['finite_pipe_cover']
    finite_pipe_ntheta = problem_kwargs['finite_pipe_ntheta']
    matrix_method = problem_kwargs['matrix_method']

    pipe_ugeo = tunnel_geo()
    pipe_ugeo.create_deltatheta(2 * np.pi / finite_pipe_ntheta, 1, finite_pipe_length,
                                1, finite_pipe_cover, factor=1, left_hand=False)
    pipe_ugeo.set_rigid_velocity(np.array((0, 0, 0, 0, 0, 0)))
    pipe_obj = sf.obj_dic[matrix_method]()
    pipe_obj.set_data(pipe_ugeo, pipe_ugeo, name='finite_pipe')
    return pipe_obj



# @profile
def main_fun(**main_kwargs):
    assert 1 == 2
    main_kwargs['rot_norm'] = np.array((0, 1, 0))
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    print_case_info(**problem_kwargs)
    fileHandle = problem_kwargs['fileHandle']
    max_iter = problem_kwargs['max_iter']
    eval_dt = problem_kwargs['eval_dt']

    if not problem_kwargs['restart']:
        # create obj
        ellipse_obj = create_ellipse_obj(**problem_kwargs)
        pipe_obj = create_pipe_obj(**problem_kwargs)
        ellipse_ugeo = ellipse_obj.get_u_geo()
        F_ellipse = ellipse_obj.get_point_force_list()[0][1]
        givenF = np.hstack((F_ellipse, np.zeros(3)))
        ecoli_comp = sf.givenForceComposite(center=ellipse_ugeo.get_center(), name='ecoli_0', givenF=givenF)
        ecoli_comp.add_obj(obj=ellipse_obj, rel_U=np.zeros(6))

        # prepare problem
        ecoli_center_list = []
        ecoli_norm_list = []
        ecoli_Floc_list = []
        ecoli_U_list = []
        ecoli_reset_z = 0   # each time step move the ecoli to the pipe center to improve the accuracy.
        problem = sf.GivenForcePoiseuilleFlowProblem(**problem_kwargs)
        problem.add_given_flow_obj(pipe_obj)
        for obj in [ecoli_comp, pipe_obj]:
            problem.add_obj(obj)
        problem.print_info()
        vtk_matname = problem_kwargs['vtk_matname']
        vtk_geo = geo()
        vtk_geo.mat_nodes(filename=vtk_matname, mat_handle='vtk_nodes')
        vtk_geo.mat_elmes(filename=vtk_matname, mat_handle='vtk_elems', elemtype='tetra')

        # evaluation loop
        t0 = time()
        def append2list():
            ecoli_center_list.append(ecoli_comp.get_center() + [0, 0, ecoli_reset_z])
            ecoli_norm_list.append(ellipse_ugeo.get_geo_norm())
            ecoli_Floc_list.append(ellipse_obj.get_point_force_list()[0][0] + [0, 0, ecoli_reset_z])
            ecoli_U_list.append(ecoli_comp.get_ref_U())
        append2list()
        for idx in range(max_iter):
            problem.create_matrix()
            problem.solve()
            re_ecoli_U = ecoli_comp.get_ref_U()
            ecoli_move = re_ecoli_U[:3] * eval_dt * [1, 1, 0] # each time step move the ecoli to the pipe center to improve the accuracy.
            ecoli_reset_z = ecoli_reset_z + re_ecoli_U[2] * eval_dt
            ecoli_rot = re_ecoli_U[3:] * eval_dt
            ecoli_comp.move(ecoli_move)
            ecoli_comp.node_rotation(norm=np.array([1, 0, 0]), theta=ecoli_rot[0])
            ecoli_comp.node_rotation(norm=np.array([0, 1, 0]), theta=ecoli_rot[1])
            ecoli_comp.node_rotation(norm=np.array([0, 0, 1]), theta=ecoli_rot[2])
            # Todo: check if ecoli out of pipe boundary.
            append2list()
            PETSc.Sys.Print('re_ecoli_U', re_ecoli_U)
            re_ecoli_F = ecoli_comp.get_total_force()
            PETSc.Sys.Print('re_err_F', (re_ecoli_F - givenF) / givenF)
            problem.vtk_obj(fileHandle, idx)
            problem.vtk_tetra('%s_vtkU_%5d'% (fileHandle, idx), vtk_geo)
        t1 = time()
        PETSc.Sys.Print('%s: run %d loops using %f' % (fileHandle, max_iter, (t1 - t0)))
        # dbg
        PETSc.Sys.Print(ecoli_center_list)
        PETSc.Sys.Print(ecoli_norm_list)
        PETSc.Sys.Print(ecoli_Floc_list)
        PETSc.Sys.Print(ecoli_U_list)

        problem.destroy()
        comm = PETSc.COMM_WORLD.tompi4py()
        rank = comm.Get_rank()
        if rank == 0:
            savemat(fileHandle,
                    {'ecoli_center': np.vstack(ecoli_center_list),
                     'ecoli_norm':   np.vstack(ecoli_norm_list),
                     'ecoli_Floc':   np.vstack(ecoli_Floc_list),
                     'ecoli_U':      np.vstack(ecoli_U_list)},
                    oned_as='column')
    else:
        pass
    return True


if __name__ == '__main__':
    main_fun()

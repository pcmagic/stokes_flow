# coding=utf-8

import sys
import petsc4py

petsc4py.init(sys.argv)

import numpy as np
# from time import time
# from scipy.io import loadmat
# from src.stokes_flow import problem_dic, obj_dic
from src.geo import *
from petsc4py import PETSc
from src import stokes_flow as sf
from src.myio import *
# from src.support_class import *
from src.objComposite import createEcoliComp_tunnel
from src.myvtk import save_singleEcoli_vtk
import ecoli_in_pipe.ecoli_common as ec


def get_problem_kwargs(**main_kwargs):
    problem_kwargs = ec.get_problem_kwargs()

    kwargs_list = (get_pipe_kwargs(), get_shearFlow_kwargs(), main_kwargs,)
    for t_kwargs in kwargs_list:
        for key in t_kwargs:
            problem_kwargs[key] = t_kwargs[key]
    return problem_kwargs


def print_case_info(**problem_kwargs):
    caseIntro = '-->Ecoli in pipe case, force free case, finite pipe and use regularized Stokeslets. '
    ec.print_case_info(caseIntro, **problem_kwargs)
    print_pipe_info(**problem_kwargs)
    print_shearFlow_info(**problem_kwargs)
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


def create_ellipse_obj(**problem_kwargs):
    ds = problem_kwargs['ds']
    rs1 = problem_kwargs['rs1']
    rs2 = problem_kwargs['rs2']
    rot_norm = problem_kwargs['rot_norm']
    rot_theta = problem_kwargs['rot_theta']
    matrix_method = problem_kwargs['matrix_method']

    ellipse_ugeo = ellipse_geo()
    ellipse_ugeo.create_delta(ds, rs1, rs2)
    ellipse_ugeo.node_rotation(rot_norm, rot_theta)
    ellipse_ugeo.set_rigid_velocity(np.zeros(6))
    ellipse_obj = sf.obj_dic[matrix_method]()
    ellipse_obj.set_data(ellipse_ugeo, ellipse_ugeo, name='head_ellipse')
    return ellipse_obj


# @profile
def main_fun(**main_kwargs):
    OptDB = PETSc.Options()
    fileHandle = OptDB.getString('f', 'ecoliInPipe_rs')
    OptDB.setValue('f', fileHandle)
    main_kwargs['fileHandle'] = fileHandle
    main_kwargs['rot_norm'] = np.random.sample(3)
    main_kwargs['rot_theta'] = np.pi * np.random.sample(1)
    main_kwargs['matrix_method'] = 'lg_rs'
    OptDB.setValue('sm', 'lg_rs')
    t1 = np.random.sample(3).reshape((1, 3))
    t1[0, 2] = 0
    main_kwargs['planeShearRate'] = t1
    main_kwargs['ds'] = 0.2
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    print_case_info(**problem_kwargs)

    if not problem_kwargs['restart']:
        ecoli_U = np.random.sample(6)
        # ecoli_U = np.array([0.57413863, 0.2361188, 0.93489046, 0.55094027, 0.55477833, 0.16346175])
        # ecoli_rot_norm = np.array((1, 1, 1))
        # ecoli_rot_theta = np.pi / 7
        ellipse_obj = create_ellipse_obj(**problem_kwargs)
        ellipse_ugeo = ellipse_obj.get_u_geo()
        ellipse_ugeo.set_rigid_velocity(ecoli_U)
        # ellipse_obj.show_u_nodes()

        # problem = sf.StokesFlowProblem(**problem_kwargs)
        problem = sf.ShearFlowProblem(**problem_kwargs)
        problem.do_solve_process([ellipse_obj, ])
        F_ellipse = ellipse_obj.get_total_force()
        PETSc.Sys.Print('ecoli_U', ecoli_U)
        PETSc.Sys.Print('F_ellipse', F_ellipse)
        # assert  1 == 2

        ecoli_comp = sf.GivenForceComposite(center=ellipse_ugeo.get_center(),
                                            name='ecoli_0', givenF=F_ellipse)
        ecoli_comp.add_obj(obj=ellipse_obj, rel_U=np.zeros(6))
        # problem = sf.GivenForceProblem(**problem_kwargs)
        problem = sf.GivenForceShearFlowProblem(**problem_kwargs)
        problem.do_solve_process(ecoli_comp)
        re_ecoli_U = ecoli_comp.get_ref_U()
        re_ecoli_F = ecoli_comp.get_total_force()
        PETSc.Sys.Print('re_ecoli_U', re_ecoli_U)
        PETSc.Sys.Print('re_ecoli_F', re_ecoli_F)
        PETSc.Sys.Print('re_error  ', (re_ecoli_U - ecoli_U) / ecoli_U)
        PETSc.Sys.Print('re_error  ', (re_ecoli_F - F_ellipse) / F_ellipse)
    else:
        ecoli_U = None
    return ecoli_U


if __name__ == '__main__':
    main_fun()

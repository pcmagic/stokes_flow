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
from src.StokesFlowMethod import light_stokeslets_matrix_3d
# from src.support_class import *
from src.objComposite import createEcoliComp_tunnel
from src.myvtk import save_singleEcoli_vtk
import ecoli_in_pipe.ecoli_common as ec
import os


# import import_my_lib


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
    ellipse_centerx = OptDB.getReal('ellipse_centerx', 0)
    ellipse_centery = OptDB.getReal('ellipse_centery', 0)
    ellipse_centerz = OptDB.getReal('ellipse_centerz', 0)
    ellipse_center = np.array((ellipse_centerx, ellipse_centery, ellipse_centerz))  # center of ecoli
    ecoli_tail_strength = OptDB.getReal('ecoli_tail_strength', 1)
    problem_kwargs['vtk_matname'] = vtk_matname
    problem_kwargs['ellipse_center'] = ellipse_center
    problem_kwargs['ecoli_tail_strength'] = ecoli_tail_strength
    return problem_kwargs


def print_case_info(**problem_kwargs):
    caseIntro = '-->Ecoli in pipe case, force free case, finite pipe and use regularized Stokeslets. '
    ec.print_case_info(caseIntro, **problem_kwargs)
    ellipse_center = problem_kwargs['ellipse_center']
    ecoli_tail_strength = problem_kwargs['ecoli_tail_strength']
    PETSc.Sys.Print('    ellipse_center %s, ecoli_tail_strength %f' % (str(ellipse_center), ecoli_tail_strength))

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


def create_ellipse_obj(**problem_kwargs):
    ds = problem_kwargs['ds']
    rs1 = problem_kwargs['rs1']
    rs2 = problem_kwargs['rs2']
    rot_norm = problem_kwargs['rot_norm']
    rot_theta = problem_kwargs['rot_theta']
    ellipse_center = problem_kwargs['ellipse_center']
    # matrix_method = problem_kwargs['matrix_method']
    ecoli_tail_strength = problem_kwargs['ecoli_tail_strength']

    ellipse_ugeo = ellipse_geo()
    ellipse_ugeo.create_delta(ds, rs1, rs2)
    ellipse_ugeo.node_rotation(rot_norm, rot_theta)
    ellipse_ugeo.move(ellipse_center)
    ellipse_ugeo.set_rigid_velocity(np.zeros(6))
    ellipse_obj = sf.PointForceObj()
    ellipse_obj.set_data(ellipse_ugeo, ellipse_ugeo, name='head_ellipse')
    F_ellipse = -1 * ellipse_ugeo.get_geo_norm() * ecoli_tail_strength
    ellipse_obj.add_point_force(location=np.array((0, 0, 1)), force=F_ellipse,
                                StokesletsHandle=light_stokeslets_matrix_3d)
    return ellipse_obj


# @profile
def main_fun(**main_kwargs):
    # # dbg
    # OptDB = PETSc.Options()
    # main_kwargs['matrix_method'] = 'lg_rs'
    # OptDB.setValue('sm', 'lg_rs')
    # # main_kwargs['PoiseuilleStrength'] = np.random.sample(1)
    # main_kwargs['PoiseuilleStrength'] = 0
    # # kwargs for ellipse.
    # main_kwargs['rs1'] = 0.3
    # main_kwargs['rs2'] = 0.1
    # main_kwargs['ds'] = 0.02
    # # main_kwargs['rot_norm'] = np.random.sample(3)
    # # main_kwargs['rot_theta'] = np.pi * np.random.sample(1)
    # main_kwargs['rot_norm'] = np.ones(3)
    # main_kwargs['rot_theta'] = np.pi * 0.34324242
    # # kwargs for pipe
    # main_kwargs['finite_pipe_length'] = 10
    # main_kwargs['finite_pipe_cover'] = 1
    # main_kwargs['finite_pipe_ntheta'] = 120

    main_kwargs['rot_norm'] = np.array((0, 1, 0))
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    print_case_info(**problem_kwargs)
    fileHandle = problem_kwargs['fileHandle']
    # assert 1 == 2
    # user_paths = os.environ['PYTHONPATH'].split(os.pathsep)
    # sys.path.append(os.path.join(os.path.dirname(__file__), "lib"))
    # user_paths = os.environ['PYTHONPATH'].split(os.pathsep)

    if not problem_kwargs['restart']:
        # ecoli_U = np.random.sample(6)
        # ecoli_U = np.array([0.96835616, 0.84817981, 0.21891314, 0.67224158, 0.87721622, 0.30945817])
        # ellipse_obj = create_ellipse_obj(**problem_kwargs)
        # ellipse_ugeo = ellipse_obj.get_u_geo()
        # ellipse_ugeo.set_rigid_velocity(ecoli_U)
        # pipe_obj = create_pipe_obj(**problem_kwargs)

        # problem = sf.PoiseuilleFlowProblem(**problem_kwargs)
        # problem.do_solve_process([ellipse_obj, pipe_obj])
        # F_ellipse = ellipse_obj.get_total_force()
        # PETSc.Sys.Print('ecoli_U', ecoli_U)
        # PETSc.Sys.Print('F_ellipse', F_ellipse)

        ellipse_obj = create_ellipse_obj(**problem_kwargs)
        pipe_obj = create_pipe_obj(**problem_kwargs)
        # # dbg
        # rs1 = problem_kwargs['rs1']
        # print(ellipse_obj.get_n_u_node(), pipe_obj.get_n_u_node())
        # print(ellipse_obj.get_u_geo().get_deltaLength(), pipe_obj.get_u_geo().get_deltaLength(),
        #       1 - (ellipse_obj.get_u_geo().get_deltaLength() + pipe_obj.get_u_geo().get_deltaLength()) * 3 - 0.75 - rs1)
        # assert 1 == 2

        ellipse_ugeo = ellipse_obj.get_u_geo()
        F_ellipse = ellipse_obj.get_point_force_list()[0][1]
        givenF = np.hstack((F_ellipse, np.zeros(3)))
        ecoli_comp = sf.givenForceComposite(center=ellipse_ugeo.get_center(), name='ecoli_0', givenF=givenF)
        ecoli_comp.add_obj(obj=ellipse_obj, rel_U=np.zeros(6))
        problem = sf.GivenForcePoiseuilleFlowProblem(**problem_kwargs)
        problem.add_given_flow_obj(pipe_obj)
        # pipe_obj.show_velocity()
        problem.do_solve_process([ecoli_comp, pipe_obj])
        re_ecoli_U = ecoli_comp.get_ref_U()
        re_ecoli_F = ecoli_comp.get_total_force()
        PETSc.Sys.Print('re_ecoli_U', re_ecoli_U)
        PETSc.Sys.Print('F_ellipse', givenF)
        PETSc.Sys.Print('re_ecoli_F', re_ecoli_F)
        PETSc.Sys.Print('re_err_F', (re_ecoli_F - givenF) / givenF)

        vtk_matname = problem_kwargs['vtk_matname']
        vtk_geo = geo()
        vtk_geo.mat_nodes(filename=vtk_matname, mat_handle='vtk_nodes')
        vtk_geo.mat_elmes(filename=vtk_matname, mat_handle='vtk_elems', elemtype='tetra')
        problem.vtk_obj(fileHandle)
        problem.vtk_tetra(fileHandle + '_vtkU', vtk_geo)
    else:
        re_ecoli_U = None
    return re_ecoli_U


if __name__ == '__main__':
    main_fun()

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

    kwargs_list = (get_pipe_kwargs(), main_kwargs,)
    for t_kwargs in kwargs_list:
        for key in t_kwargs:
            problem_kwargs[key] = t_kwargs[key]
    return problem_kwargs


def print_case_info(**problem_kwargs):
    caseIntro = '-->Ecoli in pipe case, force free case, finite pipe and use regularized Stokeslets. '
    ec.print_case_info(caseIntro, **problem_kwargs)
    print_pipe_info(**problem_kwargs)
    return True


def create_pipe_obj(**problem_kwargs):
    finite_pipe_length = problem_kwargs['finite_pipe_length']
    finite_pipe_cover = problem_kwargs['finite_pipe_cover']
    # finite_pipe_epsilon = problem_kwargs['finite_pipe_epsilon']
    finite_pipe_ntheta = problem_kwargs['finite_pipe_ntheta']
    matrix_method = problem_kwargs['matrix_method']

    pipe_ugeo = tunnel_geo()
    pipe_ugeo.create_deltatheta(2 * np.pi / finite_pipe_ntheta, 1, finite_pipe_length,
                                0, finite_pipe_cover, factor=1, left_hand=False)
    pipe_ugeo.set_rigid_velocity(np.array((0, 0, 0, 0, 0, 0)))
    pipe_obj = sf.obj_dic[matrix_method]()
    pipe_obj.set_data(pipe_ugeo, pipe_ugeo, name='finite_pipe')
    return pipe_obj


# @profile
def main_fun(**main_kwargs):
    OptDB = PETSc.Options()
    fileHandle = OptDB.getString('f', 'ecoliInPipe_rs')
    OptDB.setValue('f', fileHandle)
    main_kwargs['fileHandle'] = fileHandle
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    print_case_info(**problem_kwargs)

    if not problem_kwargs['restart']:
        ecoli_comp = createEcoliComp_tunnel(name='ecoli_0', **problem_kwargs)

        pipe_obj = create_pipe_obj(**problem_kwargs)

        problem = sf.forcefreeProblem(**problem_kwargs)
        problem.add_obj(ecoli_comp)
        problem.add_obj(pipe_obj)
        # problem.show_velocity()
        problem.print_info()
        problem.create_matrix()
        problem.solve()
        # problem.show_re_velocity()

        # post process
        head_U, tail_U = print_single_ecoli_forcefree_result(ecoli_comp, **problem_kwargs)
        ecoli_U = ecoli_comp.get_ref_U()
        save_singleEcoli_vtk(problem, createHandle=createEcoliComp_tunnel)
    else:
        head_U, tail_U, ecoli_U = [None, None, None]
    return head_U, tail_U, ecoli_U


if __name__ == '__main__':
    main_fun()

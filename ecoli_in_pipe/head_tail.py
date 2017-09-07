import sys

import petsc4py

petsc4py.init(sys.argv)

import numpy as np
import pickle
# from time import time
# from scipy.io import loadmat
# from src.stokes_flow import problem_dic, obj_dic
# from src.geo import *
from petsc4py import PETSc
from src import stokes_flow as sf
from src.myio import *
from src.objComposite import createEcoliComp_tunnel
from src.myvtk import save_singleEcoli_vtk


def get_problem_kwargs(**main_kwargs):
    problem_kwargs = get_solver_kwargs()
    OptDB = PETSc.Options()
    fileHeadle = OptDB.getString('f', '...')
    err_msg = 'specify the fileHeadle. '
    assert fileHeadle != '...', err_msg
    problem_kwargs['fileHeadle'] = fileHeadle

    import os
    t_name = os.path.basename(__file__)
    need_args = ['head_U', 'tail_U', ]
    for key in need_args:
        if key not in main_kwargs:
            err_msg = 'information about ' + key + ' is necessary for %s . ' % t_name
            raise ValueError(err_msg)

    kwargs_list = (main_kwargs,)
    for t_kwargs in kwargs_list:
        for key in t_kwargs:
            problem_kwargs[key] = t_kwargs[key]
    return problem_kwargs


def print_case_info(**problem_kwargs):
    pass
    return True


def main_fun(**main_kwargs):
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    fileHeadle = problem_kwargs['fileHeadle']
    head_U = problem_kwargs['head_U']
    tail_U = problem_kwargs['tail_U']

    with open(fileHeadle + '_pick.bin', 'rb') as myinput:
        unpick = pickle.Unpickler(myinput)
        problem = unpick.load()
        problem.unpickmyself()
    kwargs = problem.get_kwargs()

    newProb = sf.stokesFlowProblem(**kwargs)
    t_obj_list = [problem.get_obj_list()[0].get_obj_list()[0], ]
    new_obj_list = [t_obj.copy() for t_obj in t_obj_list]
    for new_obj in new_obj_list:
        new_obj.set_rigid_velocity(head_U)
        newProb.add_obj(new_obj)
    newProb.create_F_U()
    new_M = newProb.create_empty_M()
    for t_obj, new_obj in zip(t_obj_list, new_obj_list):
        problem.create_part_matrix(t_obj, t_obj, new_obj, new_obj, new_M)
    newProb.solve()
    # newProb.show_velocity(length_factor=0.01)
    t_force = newProb.get_total_force()
    PETSc.Sys.Print('---->>>head resultant is', t_force / 6 / np.pi)

    newProb = sf.stokesFlowProblem(**kwargs)
    t_obj_list = problem.get_obj_list()[0].get_obj_list()[1:]
    new_obj_list = [t_obj.copy() for t_obj in t_obj_list]
    for new_obj in new_obj_list:
        new_obj.set_rigid_velocity(tail_U)
        newProb.add_obj(new_obj)
    # newProb.show_velocity(length_factor=0.01)
    newProb.create_F_U()
    new_M = newProb.create_empty_M()
    for t_obj, new_obj in zip(t_obj_list, new_obj_list):
        problem.create_part_matrix(t_obj, t_obj, new_obj, new_obj, new_M)
    newProb.solve()
    t_force = newProb.get_total_force()
    PETSc.Sys.Print('---->>>tail resultant is', t_force / 6 / np.pi)


if __name__ == '__main__':
    main_fun()

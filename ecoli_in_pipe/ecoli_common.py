import sys

import petsc4py

petsc4py.init(sys.argv)

from petsc4py import PETSc
import pickle
from src.support_class import *
from src.myio import *
from src.myvtk import save_singleEcoli_vtk
from src.objComposite import *

__all__ = ['get_problem_kwargs', 'print_case_info', 'ecoli_restart']


def get_problem_kwargs(**main_kwargs):
    problem_kwargs = get_solver_kwargs()

    kwargs_list = (get_vtk_tetra_kwargs(), get_ecoli_kwargs(), get_forceFree_kwargs(), main_kwargs,)
    for t_kwargs in kwargs_list:
        for key in t_kwargs:
            problem_kwargs[key] = t_kwargs[key]
    return problem_kwargs


def print_case_info(**problem_kwargs):
    fileHeadle = problem_kwargs['fileHeadle']
    PETSc.Sys.Print('-->Ecoli in pipe case, force free case.')
    print_solver_info(**problem_kwargs)
    print_forceFree_info(**problem_kwargs)
    print_ecoli_info(fileHeadle, **problem_kwargs)
    return True


def ecoli_restart(**main_kwargs):
    err_msg = 'keyword fileHeadle is necessary. '
    assert 'fileHeadle' in main_kwargs.keys(), err_msg
    new_kwargs = get_problem_kwargs(**main_kwargs)
    fileHeadle = new_kwargs['fileHeadle']
    t_name = check_file_extension(fileHeadle, '_pick.bin')
    with open(t_name, 'rb') as myinput:
        unpick = pickle.Unpickler(myinput)
        problem = unpick.load()
    problem.unpickmyself()
    ecoli_comp = problem.get_obj_list()[0]

    old_kwargs = problem.get_kwargs()
    old_kwargs['matname'] = new_kwargs['matname']
    old_kwargs['bnodesHeadle'] = new_kwargs['bnodesHeadle']
    old_kwargs['belemsHeadle'] = new_kwargs['belemsHeadle']
    old_kwargs['ffweightx'] = new_kwargs['ffweightx']
    old_kwargs['ffweighty'] = new_kwargs['ffweighty']
    old_kwargs['ffweightz'] = new_kwargs['ffweightz']
    old_kwargs['ffweightT'] = new_kwargs['ffweightT']

    problem.set_kwargs(**old_kwargs)
    print_case_info(**old_kwargs)
    problem.print_info()
    problem.set_force_free()
    problem.solve()

    # post process
    head_U, tail_U = print_single_ecoli_forceFree_result(ecoli_comp, **old_kwargs)
    ecoli_U = ecoli_comp.get_ref_U()
    save_singleEcoli_vtk(problem, createHandle=createEcoliComp_tunnel)
    return head_U, tail_U, ecoli_U

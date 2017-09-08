# coding=utf-8
# main codes, call functions at stokes_flow.py
# Zhang Ji, 20170518

import sys
import petsc4py

petsc4py.init(sys.argv)

# from scipy.io import savemat, loadmat
# from src.ref_solution import *
# import warnings
# from memory_profiler import profile
# from time import time
import pickle
import numpy as np
from src import stokes_flow as sf
from src.stokes_flow import problem_dic, obj_dic
from petsc4py import PETSc
from src.geo import *
from src.myio import *
from src.objComposite import *
from src.myvtk import *


def print_case_info(**problem_kwargs):
    fileHeadle = problem_kwargs['fileHeadle']
    print_solver_info(**problem_kwargs)
    print_shearFlow_info(**problem_kwargs)
    print_sphere_info(fileHeadle, **problem_kwargs)
    return True


def get_problem_kwargs(**main_kwargs):
    problem_kwargs = get_solver_kwargs()
    OptDB = PETSc.Options()
    fileHeadle = OptDB.getString('f', 'sphereShearFlow')
    OptDB.setValue('f', fileHeadle)
    problem_kwargs['fileHeadle'] = fileHeadle

    kwargs_list = (main_kwargs, get_vtk_tetra_kwargs(),
                   get_sphere_kwargs(), get_shearFlow_kwargs())
    for t_kwargs in kwargs_list:
        for key in t_kwargs:
            problem_kwargs[key] = t_kwargs[key]
    return problem_kwargs


def main_fun(**main_kwargs):
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    print_case_info(**problem_kwargs)
    rs = problem_kwargs['rs']

    obj_sphere = create_sphere(**problem_kwargs)[0]
    problem = sf.ShearFlowProblem(**problem_kwargs)
    problem.do_solve_process((obj_sphere, ))

    force_sphere = obj_sphere.get_total_force()
    temp_F = (6 * rs, 6 * rs, 6 * rs, 8 * rs ** 3, 8 * rs ** 3, 8 * rs ** 3)
    force_sphere = force_sphere / temp_F / np.pi
    PETSc.Sys.Print('---->>>%s: Resultant is %s' % (str(problem), str(force_sphere)))

    save_grid_sphere_vtk(problem, create_sphere)

    return True

if __name__ == '__main__':
    main_fun()


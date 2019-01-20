import sys
import petsc4py

petsc4py.init(sys.argv)

import numpy as np
from petsc4py import PETSc
from src import stokes_flow as sf
from src.myio import *
from src.support_class import *
from src.objComposite import *
from matplotlib import pyplot as plt


def get_problem_kwargs(**main_kwargs):
    problem_kwargs = get_solver_kwargs()
    OptDB = PETSc.Options()
    fileHandle = OptDB.getString('f', 'infhelixPro')
    OptDB.setValue('f', fileHandle)
    problem_kwargs['fileHandle'] = fileHandle
    n_helix = OptDB.getReal('n_helix', 2)
    OptDB.setValue('n_helix', n_helix)
    problem_kwargs['n_helix'] = n_helix

    kwargs_list = (main_kwargs, get_helix_kwargs(), get_givenForce_kwargs())
    for t_kwargs in kwargs_list:
        for key in t_kwargs:
            problem_kwargs[key] = t_kwargs[key]
    return problem_kwargs


def print_case_info(obj_name, **problem_kwargs):
    fileHandle = problem_kwargs['fileHandle']
    n_helix = problem_kwargs['n_helix']
    print_solver_info(**problem_kwargs)
    print_helix_info(obj_name, **problem_kwargs)
    PETSc.Sys.Print('  given unite spin wz, # helix %d. ' % n_helix)
    return True


# @profile
def main_fun(**main_kwargs):
    main_kwargs['matrix_method'] = 'pf_infhelix'
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    objname = 'infhelix'
    print_case_info(objname, **problem_kwargs)

    # helix obj
    helix_list = create_infHelix(objname, **problem_kwargs)
    nSegment = helix_list[0].get_u_geo().get_nSegment()
    maxtheta = helix_list[0].get_u_geo().get_maxlength()

    # create problem, given velocity
    problem = sf.StokesFlowProblem(**problem_kwargs)
    for tobj in helix_list:
        problem.add_obj(tobj)
    problem.create_matrix()

    # case 1, translation
    for tobj in helix_list:
        tobj.set_rigid_velocity((0, 0, 1, 0, 0, 0))
    problem.create_F_U()
    problem.solve()
    # problem.show_velocity(length_factor=0.003)
    # problem.show_force(length_factor=0.5)
    helix_force = np.sum([tobj.get_total_force() for tobj in helix_list], axis=0)
    norm_force = helix_force * nSegment / (2 * maxtheta / (2 * np.pi))   # total force / helix arc length
    PETSc.Sys.Print('Translation, helix forces and torques', norm_force)

    # case 2, rotation
    for tobj in helix_list:
        tobj.set_rigid_velocity((0, 0, 0, 0, 0, 1))
    problem.create_F_U()
    problem.solve()
    # problem.show_velocity(length_factor=0.003)
    # problem.show_force(length_factor=0.5)
    helix_force = np.sum([tobj.get_total_force() for tobj in helix_list], axis=0)
    norm_force = helix_force * nSegment / (2 * maxtheta / (2 * np.pi))   # total force / helix arc length
    PETSc.Sys.Print('Rotation, helix forces and torques', norm_force)

    PETSc.Sys.Print(problem_kwargs)
    return True


if __name__ == '__main__':
    main_fun()

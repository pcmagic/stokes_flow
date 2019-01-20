import sys
import petsc4py

petsc4py.init(sys.argv)

import numpy as np
from petsc4py import PETSc
from src import stokes_flow as sf
from src.myio import *
from src.geo import *
from src.objComposite import *
from matplotlib import pyplot as plt


def get_problem_kwargs(**main_kwargs):
    problem_kwargs = get_solver_kwargs()
    OptDB = PETSc.Options()
    fileHandle = OptDB.getString('f', 'HelixInPipe')
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
    print_forcefree_info(**problem_kwargs)
    print_helix_info(obj_name, **problem_kwargs)
    PETSc.Sys.Print('  given unite spin wz, # helix %d. ' % n_helix)
    return True


# @profile
def main_fun(**main_kwargs):
    main_kwargs['matrix_method'] = 'pf_infhelix'
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    ph = problem_kwargs['ph']
    ch = problem_kwargs['ch']
    eh = problem_kwargs['eh']
    nth = problem_kwargs['nth']
    rh1 = problem_kwargs['rh1']
    rh2 = problem_kwargs['rh2']
    # helix_theta = np.arctan(2 * np.pi * rh1 / ph)
    # R = (rh1 + rh2 * np.cos(helix_theta)) / problem_kwargs['zoom_factor']
    # R = rh1 / problem_kwargs['zoom_factor']
    R = 1
    objname = 'infhelix'
    print_case_info(objname, **problem_kwargs)

    # helix obj
    helix_list = create_infHelix(namehandle=objname, normalize=True, **problem_kwargs)
    nSegment = helix_list[0].get_u_geo().get_nSegment()
    maxtheta = helix_list[0].get_u_geo().get_maxlength()

    # pipe obj
    infPipe_ugeo = infPipe(ch * ph)
    # dbg
    OptDB = PETSc.Options()
    factor = OptDB.getReal('dbg_theta_factor', 0)
    PETSc.Sys.Print('--------------------> DBG: dbg_theta_factor = %f' % factor)
    infPipe_ugeo.create_n(R, nth * (R / rh2), factor * np.pi)
    infPipe_ugeo.set_rigid_velocity((0, 0, 0, 0, 0, 0))
    infPipe_fgeo = infPipe_ugeo.create_fgeo(epsilon=-eh)
    infPipe_obj = sf.StokesFlowObj()
    infPipe_obj.set_data(f_geo=infPipe_fgeo, u_geo=infPipe_ugeo, name='InfPipeObj')

    # create problem, given velocity
    problem = sf.StokesFlowProblem(**problem_kwargs)
    for tobj in helix_list:
        problem.add_obj(tobj)
    problem.add_obj(infPipe_obj)
    problem.print_info()
    problem.create_matrix()

    # case 1, translation
    for tobj in helix_list:
        tobj.set_rigid_velocity((0, 0, 1, 0, 0, 0))
    infPipe_ugeo.set_rigid_velocity((0, 0, 0, 0, 0, 0))
    problem.create_F_U()
    problem.solve()
    # problem.show_force(length_factor=0.1)
    helix_force = np.sum([tobj.get_total_force() for tobj in helix_list], axis=0)
    helix_force = helix_force * nSegment / (2 * maxtheta / (2 * np.pi))  # total force / ch
    pipe_force = infPipe_obj.get_total_force()
    pipe_force = pipe_force * nSegment / (2 * maxtheta / (2 * np.pi))  # total force / ch
    PETSc.Sys.Print('Translation, helix forces and torques', helix_force)
    PETSc.Sys.Print('Translation, pipe forces and torques', pipe_force)
    PETSc.Sys.Print('Translation, total forces and torques', helix_force + pipe_force)

    # case 2, rotation
    for tobj in helix_list:
        tobj.set_rigid_velocity((0, 0, 0, 0, 0, 1))
    infPipe_ugeo.set_rigid_velocity((0, 0, 0, 0, 0, 0))
    problem.create_F_U()
    problem.solve()
    # problem.show_force(length_factor=0.1)
    helix_force = np.sum([tobj.get_total_force() for tobj in helix_list], axis=0)
    helix_force = helix_force * nSegment / (2 * maxtheta / (2 * np.pi) * ph)   # total force / helix arc length
    pipe_force = infPipe_obj.get_total_force()
    pipe_force = pipe_force * nSegment / (2 * maxtheta / (2 * np.pi) * ph)   # total force / helix arc length
    PETSc.Sys.Print('Rotation, helix forces and torques', helix_force)
    PETSc.Sys.Print('Rotation, pipe forces and torques', pipe_force)
    PETSc.Sys.Print('Rotation, total forces and torques', helix_force + pipe_force)

    PETSc.Sys.Print(problem_kwargs)
    return True


if __name__ == '__main__':
    main_fun()

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
    n_tail = OptDB.getReal('n_tail', 2)
    OptDB.setValue('n_tail', n_tail)
    problem_kwargs['n_tail'] = n_tail

    kwargs_list = (main_kwargs, get_helix_kwargs(), get_givenForce_kwargs())
    for t_kwargs in kwargs_list:
        for key in t_kwargs:
            problem_kwargs[key] = t_kwargs[key]
    return problem_kwargs


def print_case_info(obj_name, **problem_kwargs):
    fileHandle = problem_kwargs['fileHandle']
    n_tail = problem_kwargs['n_tail']
    print_solver_info(**problem_kwargs)
    print_forcefree_info(**problem_kwargs)
    print_helix_info(obj_name, **problem_kwargs)
    PETSc.Sys.Print('  given unite spin wz, # helix %d. ' % n_tail)
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
    zoom_factor = problem_kwargs['zoom_factor']
    R = 1
    objname = 'helixInPipe_U'
    print_case_info(objname, **problem_kwargs)

    # helix obj
    helix_list = create_infHelix(namehandle=objname, normalize=True, **problem_kwargs)
    nSegment = helix_list[0].get_u_geo().get_nSegment()
    # ch = helix_list[0].get_u_geo().get_max_period()

    # pipe obj
    infPipe_ugeo = infPipe()
    infPipe_ugeo.create_n(R, ph, ch, nth * (R / rh2))
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
    helix_force = helix_force * nSegment  # total force per period
    pipe_force = infPipe_obj.get_total_force()
    pipe_force = pipe_force * nSegment  # total force per period
    fct = np.array([zoom_factor ** 1, zoom_factor ** 1, zoom_factor ** 1,
                    zoom_factor ** 2, zoom_factor ** 2, zoom_factor ** 2])
    norm_force = helix_force / fct
    PETSc.Sys.Print('Translation, helix forces and torques', norm_force)
    PETSc.Sys.Print('Translation, pipe forces and torques', pipe_force / fct)
    PETSc.Sys.Print('Translation, total forces and torques', (helix_force + pipe_force) / fct)
    PETSc.Sys.Print('Calculated force free forward speed is ', (-norm_force[5] / norm_force[2]))

    # case 2, rotation
    for tobj in helix_list:
        tobj.set_rigid_velocity((0, 0, 0, 0, 0, 1))
    infPipe_ugeo.set_rigid_velocity((0, 0, 0, 0, 0, 0))
    problem.create_F_U()
    problem.solve()
    # problem.show_force(length_factor=0.1)
    helix_force = np.sum([tobj.get_total_force() for tobj in helix_list], axis=0)
    helix_force = helix_force * nSegment  # total force per period
    pipe_force = infPipe_obj.get_total_force()
    pipe_force = pipe_force * nSegment  # total force per period
    fct = np.array([zoom_factor ** 2, zoom_factor ** 2, zoom_factor ** 2,
                    zoom_factor ** 3, zoom_factor ** 3, zoom_factor ** 3])
    norm_force = helix_force / fct
    PETSc.Sys.Print('Rotation, helix forces and torques', norm_force)
    PETSc.Sys.Print('Rotation, pipe forces and torques', pipe_force / fct)
    PETSc.Sys.Print('Rotation, total forces and torques', (helix_force + pipe_force) / fct)

    # case 3, force free problem, iterate method
    problem_ff = sf.GivenTorqueIterateVelocity1DProblem(axis='z', **problem_kwargs)
    for tobj in helix_list:
        problem_ff.add_obj(tobj)
    problem_ff.add_obj(infPipe_obj)
    problem_ff.set_iterate_obj(helix_list)
    # problem_ff.print_info()
    problem_ff.set_matrix(problem.get_M_petsc())
    u0, tol = problem_ff.do_iterate(tolerate=1e-3)
    PETSc.Sys.Print('---->>>helix force relative tolerate', tol)
    helixU = np.array((0, 0, u0, 0, 0, 1))
    PETSc.Sys.Print('---->>>helix velocity is', helixU)
    PETSc.Sys.Print('---->>>Norm forward helix velocity is', helixU[2] / (helixU[5] * rh1 * zoom_factor))
    PETSc.Sys.Print(problem_kwargs)
    return True


if __name__ == '__main__':
    main_fun()

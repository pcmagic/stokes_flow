import sys
import petsc4py

petsc4py.init(sys.argv)

import numpy as np
from petsc4py import PETSc
from src import stokes_flow as sf
from src.myio import *
from src.geo import *


def get_problem_kwargs(**main_kwargs):
    problem_kwargs = get_solver_kwargs()
    OptDB = PETSc.Options()
    fileHeadle = OptDB.getString('f', 'HelixInPipe')
    OptDB.setValue('f', fileHeadle)
    problem_kwargs['fileHeadle'] = fileHeadle

    kwargs_list = (main_kwargs,)
    for t_kwargs in kwargs_list:
        for key in t_kwargs:
            problem_kwargs[key] = t_kwargs[key]
    return problem_kwargs


def print_case_info(obj_name, **problem_kwargs):
    fileHeadle = problem_kwargs['fileHeadle']
    PETSc.Sys.Print('-->Infinite helix case, given unite spin wz.')
    print_solver_info(**problem_kwargs)
    print_infhelix_info(obj_name, **problem_kwargs)
    return True


# @profile
def main_fun(**main_kwargs):
    OptDB = PETSc.Options()
    main_kwargs['matrix_method'] = 'pf_infhelix'
    main_kwargs['infhelix_maxtheta'] = OptDB.getReal('infhelix_maxtheta', 15)
    main_kwargs['infhelix_ntheta'] = OptDB.getReal('infhelix_ntheta', 13)
    main_kwargs['infhelix_nnode'] = OptDB.getReal('infhelix_nnode', 10)
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    objname = 'infHelix'
    print_case_info(objname, **problem_kwargs)
    maxtheta = main_kwargs['infhelix_maxtheta']
    ntheta = main_kwargs['infhelix_ntheta']
    n_node = main_kwargs['infhelix_nnode']
    n_helix = OptDB.getReal('n_helix', 2)

    # helix obj
    rh1 = 1
    rh2 = 0.08
    ph = 2 * np.pi
    helix_list = []
    for i0, theta0 in enumerate(np.linspace(0, 2 * np.pi, n_helix, endpoint=False)):
        infhelix_ugeo = infHelix(maxtheta, ntheta)
        infhelix_ugeo.create_n(rh1, rh2, ph, n_node, theta0=theta0)
        infhelix_fgeo = infhelix_ugeo.create_fgeo(epsilon=-1)
        infhelix_obj = sf.stokesFlowObj()
        infhelix_obj.set_data(f_geo=infhelix_fgeo, u_geo=infhelix_ugeo, name=objname + '%02d' % i0)
        helix_list.append(infhelix_obj)

    # pipe obj
    R = 1 / 0.7
    infPipe_ugeo = infPipe(maxtheta / (2 * np.pi) * ph, ntheta)
    infPipe_ugeo.create_n(R, n_node * (R / rh2))
    infPipe_fgeo = infPipe_ugeo.create_fgeo(epsilon=1)
    infPipe_obj = sf.stokesFlowObj()
    infPipe_obj.set_data(f_geo=infPipe_fgeo, u_geo=infPipe_ugeo, name='InfPipeObj')
    # ppl = 1e20
    # infPipe_ugeo = infHelix(maxtheta * ph / ppl, ntheta)
    # infPipe_ugeo.create_n(0, R, ppl, n_node * (R / rh2), theta0=0)
    # infPipe_fgeo = infPipe_ugeo.create_fgeo(epsilon=-1)
    # infPipe_obj = sf.stokesFlowObj()
    # infPipe_obj.set_data(f_geo=infPipe_fgeo, u_geo=infPipe_ugeo, name='infHelixApx')

    # create problem
    problem = sf.stokesFlowProblem(**problem_kwargs)
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
    helix_force = helix_force * ntheta / (2 * maxtheta / (2 * np.pi) * ph)   # total force / helix arc length
    PETSc.Sys.Print('Translation, helix forces and torques', helix_force)
    pipe_force = infPipe_obj.get_total_force()
    pipe_force = pipe_force * ntheta / (2 * maxtheta / (2 * np.pi) * ph)   # total force / helix arc length
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
    helix_force = helix_force * ntheta / (2 * maxtheta / (2 * np.pi) * ph)   # total force / helix arc length
    PETSc.Sys.Print('Rotation, helix forces and torques', helix_force)
    pipe_force = infPipe_obj.get_total_force()
    pipe_force = pipe_force * ntheta / (2 * maxtheta / (2 * np.pi) * ph)   # total force / helix arc length
    PETSc.Sys.Print('Rotation, pipe forces and torques', pipe_force)
    PETSc.Sys.Print('Rotation, total forces and torques', helix_force + pipe_force)

    PETSc.Sys.Print(problem_kwargs)
    return True


if __name__ == '__main__':
    main_fun()

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
    print_helix_info(obj_name, **problem_kwargs)
    PETSc.Sys.Print('  given unite spin wz, # helix %d. ' % n_tail)
    return True


# @profile
def main_fun(**main_kwargs):
    main_kwargs['matrix_method'] = 'pf_infhelix'
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    ph = problem_kwargs['ph']
    ch = problem_kwargs['ch']
    rh1 = problem_kwargs['rh1']
    rh2 = problem_kwargs['rh2']
    nth = problem_kwargs['nth']
    objname = 'infhelix'
    print_case_info(objname, **problem_kwargs)

    # helix obj
    helix_list = create_infHelix(objname, **problem_kwargs)

    # # create problem, given velocity
    # problem = sf.StokesFlowProblem(**problem_kwargs)
    # for tobj in helix_list:
    #     problem.add_obj(tobj)
    # problem.create_matrix()
    #
    # # case 1, translation
    # for tobj in helix_list:
    #     tobj.set_rigid_velocity((0, 0, 1, 0, 0, 0))
    # problem.create_F_U()
    # problem.solve()
    # # problem.show_velocity(length_factor=0.003)
    # # problem.show_force(length_factor=0.5)
    # helix_force = np.sum([tobj.get_total_force() for tobj in helix_list], axis=0)
    # norm_force = helix_force * ntheta / (2 * (maxtheta / (2 * np.pi)) * ph)   # total force / helix arc length
    # PETSc.Sys.Print('Translation, helix forces and torques', norm_force)
    #
    # # case 2, rotation
    # for tobj in helix_list:
    #     tobj.set_rigid_velocity((0, 0, 0, 0, 0, 1))
    # problem.create_F_U()
    # problem.solve()
    # # problem.show_velocity(length_factor=0.003)
    # # problem.show_force(length_factor=0.5)
    # helix_force = np.sum([tobj.get_total_force() for tobj in helix_list], axis=0)
    # norm_force = helix_force * ntheta / (2 * (maxtheta / (2 * np.pi)) * ph)   # total force / helix arc length
    # PETSc.Sys.Print('Rotation, helix forces and torques', norm_force)

    # # case 3, locomotion
    # for tobj in helix_list:
    #     tobj.set_rigid_velocity((0, 0, -0.02252446, 0, 0, 1))
    # problem.create_F_U()
    # problem.solve()
    # # problem.show_velocity(length_factor=0.003)
    # # problem.show_force(length_factor=0.5)
    # helix_force = np.sum([tobj.get_total_force() for tobj in helix_list], axis=0)
    # norm_force = helix_force * ntheta  # total force per period
    # PETSc.Sys.Print(norm_force)

    # # case 3, create problem, given force and torque
    # fct = (2 * (maxtheta / (2 * np.pi)) * ph) / ntheta  # rescale factor
    # # fct = 1
    # # helix_givenF = norm_force
    # helix_givenF = np.array((0, 0, 0, 0, 0, 1))
    # helix_rel_U = np.array((0, 0, 0, 0, 0, 0))
    # helix_composite = sf.GivenForce1DInfComposite(name='helix_composite', givenF=helix_givenF * fct)
    # for tobj in helix_list:
    #     helix_composite.add_obj(tobj, rel_U=helix_rel_U)
    # problem = sf.givenForce1DInfPoblem(axis='z', **problem_kwargs)
    # problem.add_obj(helix_composite)
    # problem.print_info()
    # problem.create_matrix()
    # problem.solve()
    # # problem.show_velocity(length_factor=0.00000001)
    # # problem.show_force(length_factor=0.000003)
    # helix_force = helix_composite.get_total_force() / fct
    # helixU = helix_composite.get_ref_U() + helix_rel_U
    # PETSc.Sys.Print('External force per period', helix_givenF)
    # PETSc.Sys.Print('helix force per period', helix_force)
    # PETSc.Sys.Print('---->>>Resultant err is',
    #                 np.linalg.norm(helix_force[[2, 5]] - helix_givenF[[2, 5]]) / np.linalg.norm(helix_givenF[[2, 5]]))
    # PETSc.Sys.Print('---->>>helix velocity is', helixU)
    # PETSc.Sys.Print('---->>>Norm forward helix velocity is', helixU[2] / (helixU[5] * rh1))

    # case 4, create problem, given force and torque, iterate method
    helix_composite = sf.ForceFreeComposite(center=np.zeros(3), norm=np.array((0, 0, 1)), name='helix_composite')
    problem_kwargs['givenF'] = 0
    problem = sf.GivenTorqueIterateVelocity1DProblem(axis='z', **problem_kwargs)
    for tobj in helix_list:
        helix_composite.add_obj(tobj, rel_U=np.zeros(6))
        problem.add_obj(tobj)
    problem.set_iterate_obj(helix_list)
    problem.print_info()
    problem.create_matrix()
    u0, tol = problem.do_iterate(tolerate=1e-3)
    PETSc.Sys.Print('---->>>helix force relative tolerate', tol)
    helixU = np.array((0, 0, u0, 0, 0, 1))
    PETSc.Sys.Print('---->>>helix velocity is', helixU)
    PETSc.Sys.Print('---->>>Norm forward helix velocity is', helixU[2] / (helixU[5] * rh1))
    # problem.show_force(length_factor=0.1)
    # for tobj in helix_list:
    #     tobj.set_rigid_velocity(helixU)
    # problem.solve()
    # problem.create_F_U()
    # helix_force = np.sum([tobj.get_total_force() for tobj in helix_list], axis=0)
    # PETSc.Sys.Print('helix force per period', helix_force)

    # comm = PETSc.COMM_WORLD.tompi4py()
    # rank = comm.Get_rank()
    # if rank == 0:
    #     helix_obj = helix_composite.get_obj_list()[0]
    #     helixForce = helix_obj.get_force().reshape((-1, 3))
    #     T, N, B = helix_obj.get_f_geo().Frenetframe(0)
    #     helixForce = coordinate_transformation.vector_rotation(helixForce, np.vstack((N, B, T)).T)
    #     helixTheta = helix_obj.get_f_geo().get_phi()
    #     fig = plt.figure(figsize=(16, 12))
    #     ax = fig.subplots(nrows=1, ncols=1)
    #     fig.patch.set_facecolor('white')
    #     ax.plot(helixTheta, helixForce[:, 0], label='fx')
    #     ax.plot(helixTheta, helixForce[:, 1], label='fy')
    #     ax.plot(helixTheta, helixForce[:, 2], label='fz')
    #     ax.legend()
    #     plt.show()

    PETSc.Sys.Print(problem_kwargs)
    return True


if __name__ == '__main__':
    main_fun()

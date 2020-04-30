import sys
import petsc4py
# import matplotlib
petsc4py.init(sys.argv)
# matplotlib.use('Agg')

import numpy as np
from tqdm import tqdm
# import pickle
# from time import time
# from scipy.io import loadmat
# from src.stokes_flow import problem_dic, obj_dic
# from src.geo import *
from petsc4py import PETSc
from src import stokes_flow as sf
from src.myio import *
from src.objComposite import createEcoliComp_ellipse
from src.myvtk import save_singleEcoli_vtk


def get_problem_kwargs(**main_kwargs):
    problem_kwargs = get_solver_kwargs()
    OptDB = PETSc.Options()
    fileHandle = OptDB.getString('f', 'ecoliShearFlow')
    problem_kwargs['fileHandle'] = fileHandle

    kwargs_list = (main_kwargs, get_vtk_tetra_kwargs(), get_ecoli_kwargs(),
                   get_forcefree_kwargs(), get_shearFlow_kwargs())
    for t_kwargs in kwargs_list:
        for key in t_kwargs:
            problem_kwargs[key] = t_kwargs[key]
    return problem_kwargs


def print_case_info(**problem_kwargs):
    fileHandle = problem_kwargs['fileHandle']
    print_solver_info(**problem_kwargs)
    print_forcefree_info(**problem_kwargs)
    print_shearFlow_info(**problem_kwargs)
    print_ecoli_info(fileHandle, **problem_kwargs)
    return True


# @profile
def main_fun(**main_kwargs):
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    fileHandle = problem_kwargs['fileHandle']
    rot_theta = problem_kwargs['rot_theta']
    fileHandle = '%s_rotTh%f' % (fileHandle, rot_theta)
    problem_kwargs['fileHandle'] = fileHandle

    if not problem_kwargs['restart']:
        print_case_info(**problem_kwargs)
        ecoli_comp = createEcoliComp_ellipse(name='ecoli_0', **problem_kwargs)
        problem = sf.ShearFlowForceFreeProblem(**problem_kwargs)
        problem.do_solve_process((ecoli_comp,), pick_M=False)
        head_U, tail_U = print_single_ecoli_forcefree_result(ecoli_comp, **problem_kwargs)
        ecoli_U = ecoli_comp.get_ref_U()
        # save_singleEcoli_vtk(problem, createHandle=createEcoliComp_ellipse)
        ecoli_comp.png_u_nodes(fileHandle)
        problem.destroy()
    else:
        pass
        # with open(fileHandle + '_pick.bin', 'rb') as input:
        #     unpick = pickle.Unpickler(input)
        #     problem = unpick.load( )
        #     problem.unpick_myself( )
        # problem_kwargs = problem.get_kwargs( )
        # forcepipe = problem_kwargs['forcepipe']
        # rh1 = problem_kwargs['rh1']
        # zoom_factor = problem_kwargs['zoom_factor']
        # rel_Us = problem_kwargs['rel_Us']
        # rel_Uh = problem_kwargs['rel_Uh']
        # prb_index = problem_kwargs['prb_index']
        # with_T_geo = len(problem.get_obj_list( )[0].get_obj_list( )) == 4
        # ecoli_comp = problem.get_obj_list( )[0]
        # if with_T_geo:
        #     vsobj, vhobj0, vhobj1, vTobj = ecoli_comp.get_obj_list( )
        # else:
        #     vsobj, vhobj0, vhobj1 = ecoli_comp.get_obj_list( )
        #
        # problem_kwargs1 = get_problem_kwargs(**main_kwargs)
        # problem_kwargs['matname'] = problem_kwargs1['matname']
        # problem_kwargs['bnodesHeadle'] = problem_kwargs1['bnodesHeadle']
        # problem_kwargs['belemsHeadle'] = problem_kwargs1['belemsHeadle']
        # problem_kwargs['ffweight'] = problem_kwargs1['ffweight']
        # problem.set_kwargs(**problem_kwargs)
        # print_case_info(**problem_kwargs)
        # problem.print_info( )
        #
        # OptDB = PETSc.Options( )
        # if OptDB.getBool('check_MPISIZE', True):
        #     err_msg = 'problem was picked with MPI size %d, current MPI size %d is wrong. ' % (
        #         problem_kwargs['MPISIZE'], problem_kwargs1['MPISIZE'],)
        #     assert problem_kwargs['MPISIZE'] == problem_kwargs1['MPISIZE'], err_msg
        #
        # problem.set_force_free( )
        # problem.solve( )
        # print_single_ecoli_forcefree_result(ecoli_comp, **problem_kwargs)
        #
        # # save_singleEcoli_vtk(problem)
    return head_U, tail_U, ecoli_U


def job_script():
    def save_contourf(figname, U):
        fig0 = plt.figure()
        ax0 = fig0.gca()
        cf = ax0.contourf(rot_theta, planeShearRatey, U)
        fig0.colorbar(cf, ax=ax0)
        ax0.set_title(figTitHeadle + '_' + figname)
        ax0.set_xlabel('rot_theta')
        ax0.set_ylabel('planeShearRate_y')
        fig0.set_size_inches(18.5, 10.5)
        fig0.savefig('%s_%s.png' % (fileHandle, figname), dpi=100)
        plt.close()

    comm = PETSc.COMM_WORLD.tompi4py()
    rank = comm.Get_rank()
    OptDB = PETSc.Options()
    n_rot_theta = OptDB.getInt('n_rot_theta', 2)
    n_planeShearRatey = OptDB.getInt('n_planeShearRatey', 2)

    problem_kwargs = get_problem_kwargs()
    rel_Us = problem_kwargs['rel_Us']
    rel_Uh = problem_kwargs['rel_Uh']
    rh1 = problem_kwargs['rh1']
    zoom_factor = problem_kwargs['zoom_factor']
    t_nondim = np.sqrt(np.sum((rel_Uh[-3:] + rel_Us[-3:]) ** 2)) * \
               np.array((zoom_factor * rh1, zoom_factor * rh1, zoom_factor * rh1, 1, 1, 1))
    fileHandle = problem_kwargs['fileHandle']

    rot_theta, planeShearRatey = np.meshgrid(np.linspace(0, 2, n_rot_theta),
                                             np.linspace(0, 0.2, n_planeShearRatey), )
    ecoli_U = []
    for i0, (t_rot_theta, t_planeShearRatey) in enumerate(
            tqdm(zip(rot_theta.flatten(), planeShearRatey.flatten()), desc=fileHandle)):
        OptDB.setValue('rot_theta', t_rot_theta)
        OptDB.setValue('planeShearRatey', t_planeShearRatey)
        ecoli_U.append(main_fun()[2] / t_nondim)
        PETSc.Sys.Print(' ')
        PETSc.Sys.Print(' ')
    ecoli_U = np.vstack(ecoli_U)
    ecoli_ux = ecoli_U[:, 0].reshape((n_rot_theta, n_planeShearRatey))
    ecoli_uy = ecoli_U[:, 1].reshape((n_rot_theta, n_planeShearRatey))
    ecoli_uz = ecoli_U[:, 2].reshape((n_rot_theta, n_planeShearRatey))
    ecoli_wx = ecoli_U[:, 3].reshape((n_rot_theta, n_planeShearRatey))
    ecoli_wy = ecoli_U[:, 4].reshape((n_rot_theta, n_planeShearRatey))
    ecoli_wz = ecoli_U[:, 5].reshape((n_rot_theta, n_planeShearRatey))
    ecoli_uall = np.sqrt(ecoli_ux ** 2 + ecoli_uy ** 2 + ecoli_uz ** 2)
    ecoli_wall = np.sqrt(ecoli_wx ** 2 + ecoli_wy ** 2 + ecoli_wz ** 2)

    # plot and save figures.
    import matplotlib.pyplot as plt
    import matplotlib
    font = {'size': 40}
    matplotlib.rc('font', **font)
    figTitHeadle = ''
    save_contourf('ux', ecoli_ux)
    save_contourf('uy', ecoli_uy)
    save_contourf('uz', ecoli_uz)
    save_contourf('uall', ecoli_uall)
    save_contourf('wx', ecoli_wx)
    save_contourf('wy', ecoli_wy)
    save_contourf('wz', ecoli_wz)
    save_contourf('wall', ecoli_wall)


if __name__ == '__main__':
    main_fun()
    # job_script()

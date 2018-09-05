# coding=utf-8
# 1. generate velocity and force nodes of sphere using MATLAB,
# 2. for each force node, get b, solve surrounding velocity boundary condition (pipe and cover, named boundary velocity) using formula from Liron's paper, save .mat file
# 3. read .mat file, for each boundary velocity, solve associated boundary force.
# 4. solve sphere M matrix using boundary force.
# 5. solve problem and check.

import sys
import petsc4py

petsc4py.init(sys.argv)
# import sys
# print(sys.path)
# exit()

import numpy as np
from tqdm import tqdm
# import pickle
# from time import time
from scipy.io import savemat, loadmat
# from src.stokes_flow import problem_dic, obj_dic
# from src.stokes_flow import problem_dic, obj_dic
from src.objComposite import *
from src.myvtk import *
from petsc4py import PETSc
from src import stokes_flow as sf
from src.myio import *
from src.support_class import *


def get_problem_kwargs(**main_kwargs):
    problem_kwargs = get_solver_kwargs()
    OptDB = PETSc.Options()
    fileHeadle = OptDB.getString('f', 'singleRodPlane')
    problem_kwargs['fileHeadle'] = fileHeadle

    kwargs_list = (main_kwargs, get_rod_kwargs(), get_givenForce_kwargs())
    for t_kwargs in kwargs_list:
        for key in t_kwargs:
            problem_kwargs[key] = t_kwargs[key]

    rRod = problem_kwargs['rRod']
    ntRod = problem_kwargs['ntRod']
    eRod = problem_kwargs['eRod']
    dth = 2 * np.pi / ntRod
    problem_kwargs['delta'] = eRod * 2 * np.sin(dth / 2) * rRod

    return problem_kwargs


def print_case_info(**problem_kwargs):
    fileHeadle = problem_kwargs['fileHeadle']
    print_solver_info(**problem_kwargs)
    print_givenForce_info(**problem_kwargs)
    print_Rod_info(fileHeadle, **problem_kwargs)
    return True


# @profile
def main_fun(**main_kwargs):
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    fileHeadle = problem_kwargs['fileHeadle']
    rel_URod = problem_kwargs['rel_URod']
    RodThe = problem_kwargs['RodThe']
    RodPhi = problem_kwargs['RodPhi']
    givenF = problem_kwargs['givenF']
    rRod = problem_kwargs['rRod']
    lRod = problem_kwargs['lRod']
    fileHeadle = '%s_phi%f_th%f' % (fileHeadle, RodPhi, RodThe)
    problem_kwargs['fileHeadle'] = fileHeadle

    if not problem_kwargs['restart']:
        rod_comp_list = create_rod(namehandle='rod_comp', **problem_kwargs)
        # rod_comp.show_givenF()
        # # dbg
        # rod_comp_list[0].get_obj_list()[0].get_u_geo().show_nodes()
        # rod_U = np.random.sample(6)
        rod_comp = rod_comp_list[0]
        rod_geo = rod_comp.get_obj_list()[0].get_u_geo()
        min_z = rod_geo.get_nodes()[:, 2].min()
        if min_z > 0:
            print_case_info(**problem_kwargs)
            problem = sf.givenForceProblem(**problem_kwargs)
            problem.do_solve_process(rod_comp_list)

            rod_U = rod_comp.get_ref_U() + rel_URod
            rod_F = rod_comp.get_total_force()
            PETSc.Sys.Print('---->>>Resultant err is', np.sqrt(np.sum((rod_F - givenF) ** 2)))
            PETSc.Sys.Print('---->>>Rod velocity is', rod_U)

            arrowFactor = np.max((1.5 * rRod, lRod / 2))
            finename = check_file_extension(fileHeadle, '.png')
            rod_comp.png_givenF(finename=finename, arrowFactor=arrowFactor)
            save_singleRod_vtk(problem)
            problem.destroy()
        else:
            PETSc.Sys.Print('%s: min_z = %f, ignore. ' % (fileHeadle, min_z))
            rod_U = np.full(6, np.nan)
    else:
        rod_U = np.full(6, np.nan)
    return rod_U


def job_script():
    comm = PETSc.COMM_WORLD.tompi4py()
    rank = comm.Get_rank()
    OptDB = PETSc.Options()
    rod_torque = OptDB.getReal('rod_torque', 1)  # given torque amplitude of the rod
    n_RodThe = OptDB.getInt('n_RodThe', 10)
    n_RodPhi = OptDB.getInt('n_RodPhi', 10)

    problem_kwargs = get_problem_kwargs()
    lRod = problem_kwargs['lRod']
    fileHeadle = problem_kwargs['fileHeadle']
    RodCenter = problem_kwargs['RodCenter']
    zoom_factor = problem_kwargs['zoom_factor']

    RodThe, RodPhi = np.meshgrid(np.linspace(0, np.pi / 2, n_RodThe),
                                 np.linspace(0, np.pi / 2, n_RodPhi))
    rod_U = []
    for i0, (t_RodThe, t_RodPhi) in enumerate(tqdm(zip(RodThe.flatten(), RodPhi.flatten()), desc=fileHeadle)):
        OptDB.setValue('RodThe', t_RodThe)
        OptDB.setValue('RodPhi', t_RodPhi)
        OptDB.setValue('givenTy', rod_torque * np.sin(t_RodPhi))
        OptDB.setValue('givenTz', rod_torque * np.cos(t_RodPhi))
        rod_U.append(main_fun())
        PETSc.Sys.Print(' ')
        PETSc.Sys.Print(' ')
    rod_U = np.vstack(rod_U)
    # # dbg
    # rod_U = np.random.sample((RodThe.size, 6))
    rod_u_x = rod_U[:, 0].reshape((n_RodThe, n_RodPhi))
    rod_u_y = rod_U[:, 1].reshape((n_RodThe, n_RodPhi))
    rod_u_z = rod_U[:, 2].reshape((n_RodThe, n_RodPhi))
    rod_u_xy = np.sqrt(rod_u_x ** 2 + rod_u_y ** 2)
    rod_w_x = rod_U[:, 3].reshape((n_RodThe, n_RodPhi))
    rod_w_y = rod_U[:, 4].reshape((n_RodThe, n_RodPhi))
    rod_w_z = rod_U[:, 5].reshape((n_RodThe, n_RodPhi))
    rod_w_all = np.sqrt(rod_w_x ** 2 + rod_w_y ** 2 + rod_w_z ** 2)
    norm_fct = rod_w_all * lRod / 2
    norm_rod_u_x = rod_u_x / norm_fct
    norm_rod_u_y = rod_u_y / norm_fct
    norm_rod_u_z = rod_u_z / norm_fct
    norm_rod_u_xy = rod_u_xy / norm_fct

    # plot and save figures.
    # rod_u_xy
    import matplotlib.pyplot as plt
    import matplotlib
    font = {'size': 40}
    matplotlib.rc('font', **font)
    figTitHeadle = 'l%.2f_z%.2f_' % (lRod, RodCenter[-1])
    fig0 = plt.figure()
    ax0 = fig0.gca()
    cf = ax0.contourf(RodThe, RodPhi, norm_rod_u_xy)
    fig0.colorbar(cf, ax=ax0)
    ax0.set_title(figTitHeadle + 'uxy/(w*0.5*l)')
    ax0.set_xlabel('theta')
    ax0.set_ylabel('phi')
    fig0.set_size_inches(18.5, 10.5)
    fig0.savefig('%s_uxy.png' % fileHeadle, dpi=100)
    plt.close()

    fig0 = plt.figure()
    ax0 = fig0.gca()
    cf = ax0.contourf(RodThe, RodPhi, norm_rod_u_x)
    fig0.colorbar(cf, ax=ax0)
    ax0.set_title(figTitHeadle + 'ux/(w*0.5*l)')
    ax0.set_xlabel('theta')
    ax0.set_ylabel('phi')
    fig0.set_size_inches(18.5, 10.5)
    fig0.savefig('%s_ux.png' % fileHeadle, dpi=100)
    plt.close()

    fig0 = plt.figure()
    ax0 = fig0.gca()
    cf = ax0.contourf(RodThe, RodPhi, norm_rod_u_y)
    fig0.colorbar(cf, ax=ax0)
    ax0.set_title(figTitHeadle + 'uy/(w*0.5*l)')
    ax0.set_xlabel('theta')
    ax0.set_ylabel('phi')
    fig0.set_size_inches(18.5, 10.5)
    fig0.savefig('%s_uy.png' % fileHeadle, dpi=100)
    plt.close()

    fig0 = plt.figure()
    ax0 = fig0.gca()
    cf = ax0.contourf(RodThe, RodPhi, norm_rod_u_y)
    fig0.colorbar(cf, ax=ax0)
    ax0.set_title(figTitHeadle + 'uy/(w*0.5*l)')
    ax0.set_xlabel('theta')
    ax0.set_ylabel('phi')
    fig0.set_size_inches(18.5, 10.5)
    fig0.savefig('%s_uy.png' % fileHeadle, dpi=100)
    plt.close()

    fig0 = plt.figure()
    ax0 = fig0.gca()
    ax0.plot(np.mean(RodPhi, axis=1), np.mean(norm_rod_u_x, axis=1))
    ax0.set_title(figTitHeadle + 'RodPhi vs rod_u_x')
    ax0.set_xlabel('RodPhi')
    ax0.set_ylabel('rod_u_x')
    fig0.set_size_inches(18.5, 10.5)
    fig0.savefig('%s_RodPhi_ux.png' % fileHeadle, dpi=100)
    plt.close()

    fig0 = plt.figure()
    ax0 = fig0.gca()
    ax0.plot(np.mean(RodPhi, axis=1), np.mean(norm_rod_u_y, axis=1))
    ax0.set_title(figTitHeadle + 'RodPhi vs rod_u_y')
    ax0.set_xlabel('RodPhi')
    ax0.set_ylabel('rod_u_y')
    fig0.set_size_inches(18.5, 10.5)
    fig0.savefig('%s_RodPhi_uy.png' % fileHeadle, dpi=100)
    plt.close()

    fig0 = plt.figure()
    ax0 = fig0.gca()
    ax0.plot(np.mean(RodPhi, axis=1), np.mean(norm_rod_u_xy, axis=1))
    ax0.set_title(figTitHeadle + 'RodPhi vs rod_u_xy')
    ax0.set_xlabel('RodPhi')
    ax0.set_ylabel('rod_u_xy')
    fig0.set_size_inches(18.5, 10.5)
    fig0.savefig('%s_RodPhi_uxy.png' % fileHeadle, dpi=100)
    plt.close()

    mat_name = check_file_extension(fileHeadle, '.mat')
    if rank == 0:
        savemat(mat_name,
                {'lRod':        lRod,
                 'RodThe':      RodThe,
                 'RodPhi':      RodPhi,
                 'RodCenter':   RodCenter,
                 'zoom_factor': zoom_factor,
                 'rod_u_x':     rod_u_x,
                 'rod_u_y':     rod_u_y,
                 'rod_u_z':     rod_u_z,
                 'rod_w_x':     rod_w_x,
                 'rod_w_y':     rod_w_y,
                 'rod_w_z':     rod_w_z, },
                oned_as='column')


if __name__ == '__main__':
    main_fun()
    # job_script()

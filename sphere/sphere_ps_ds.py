# coding=utf-8
# main codes, call functions at stokes_flow.py
# Zhang Ji, 20160410

import sys

import petsc4py

petsc4py.init(sys.argv)

import numpy as np
from src import stokes_flow as sf
from src.stokes_flow import problem_dic, obj_dic
from petsc4py import PETSc
from src.geo import *
from time import time
import pickle
from scipy.io import savemat
from src.ref_solution import *
from scipy.io import loadmat
import warnings
from memory_profiler import profile


# @profile
def view_matrix(m, **kwargs):
    args = {
        'vmin':  None,
        'vmax':  None,
        'title': ' ',
        'cmap':  None
    }
    for key, value in args.items():
        if key in kwargs:
            args[key] = kwargs[key]

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    cax = ax.matshow(m,
                     origin='lower',
                     vmin=args['vmin'],
                     vmax=args['vmax'],
                     cmap=plt.get_cmap(args['cmap']))
    fig.colorbar(cax)
    plt.title(args['title'])
    plt.show()


def save_vtk(problem: sf.StokesFlowProblem):
    t0 = time()
    ref_slt = sphere_slt(problem)
    comm = PETSc.COMM_WORLD.tompi4py()
    rank = comm.Get_rank()
    problem_kwargs = problem.get_kwargs()
    fileHandle = problem_kwargs['fileHandle']
    radius = problem_kwargs['radius']
    u = problem_kwargs['u']

    problem.vtk_obj(fileHandle)
    problem.vtk_velocity('%s_Velocity' % fileHandle)

    theta = np.pi / 2
    sphere_check = sf.StokesFlowObj()
    sphere_geo_check = sphere_geo()  # force geo

    if not 'r_factor' in main_kwargs:
        r_factor = np.ones(1)
    else:
        r_factor = main_kwargs['r_factor']
    sphere_err = r_factor.copy()
    for i0, d0 in enumerate(r_factor):
        sphere_geo_check.create_n(2000, radius * d0)
        sphere_geo_check.set_rigid_velocity([u, 0, 0, 0, 0, 0])
        sphere_geo_check.node_rotation(norm=np.array([0, 1, 0]), theta=theta)
        sphere_check.set_data(sphere_geo_check, sphere_geo_check)
        sphere_err[i0] = problem.vtk_check('%s_Check_%f' % (fileHandle, (radius * d0)), sphere_check, ref_slt)[0]

    t1 = time()
    PETSc.Sys.Print('%s: write vtk files use: %fs' % (str(problem), (t1 - t0)))

    return sphere_err


# @profile
def main_fun(**main_kwargs):
    OptDB = PETSc.Options()
    radius = OptDB.getReal('r', 1)
    deltaLength = OptDB.getReal('d', 0.2)
    epsilon = OptDB.getReal('e', 1)
    u = OptDB.getReal('u', 1)
    fileHandle = OptDB.getString('f', 'sphere')
    ps_ds_para = OptDB.getReal('ps_ds_para', 1)  # weight factor of dipole for ps_ds method
    pf_ds_para = OptDB.getReal('pf_ds_para', 1)  # weight factor of dipole for pf_ds method
    solve_method = OptDB.getString('s', 'gmres')
    precondition_method = OptDB.getString('g', 'none')
    plot = OptDB.getBool('plot', False)
    matrix_method = OptDB.getString('sm', 'ps_ds')
    restart = OptDB.getBool('restart', False)
    twoPara_n = OptDB.getInt('tp_n', 1)
    legendre_m = OptDB.getInt('legendre_m', 3)
    legendre_k = OptDB.getInt('legendre_k', 2)
    getConvergenceHistory = OptDB.getBool('getConvergenceHistory', False)
    pickProblem = OptDB.getBool('pickProblem', False)

    field_range = np.array([[-3, -3, -3], [3, 3, 3]])
    n_grid = np.array([1, 1, 1]) * 30
    comm = PETSc.COMM_WORLD.tompi4py()
    rank = comm.Get_rank()
    size = comm.Get_size()

    if not restart:
        t0 = time()
        n = int(16 * radius * radius / deltaLength / deltaLength)
        sphere_f_geo = sphere_geo()  # pf, force geo
        sphere_f_geo.create_n(n, radius - deltaLength * epsilon)
        sphere_f_geo.set_rigid_velocity([u, 0, 0, 0, 0, 0])
        sphere_ini_u_geo = sphere_geo()  # pf, velocity geo
        sphere_ini_u_geo.create_n(n, radius)
        sphere_ini_u_geo.set_rigid_velocity([u, 0, 0, 0, 0, 0])
        sphere_u_geo = sphere_geo()  # pf_ds, velocity geo
        sphere_u_geo.create_n(n * 2, radius)
        sphere_u_geo.set_rigid_velocity([u, 0, 0, 0, 0, 0])

        problem_kwargs = {
            'name':                  'spherePrb',
            'matrix_method':         matrix_method,
            'delta':                 deltaLength * epsilon,  # for rs method
            'd_radia':               deltaLength / 2,  # for sf method
            'solve_method':          solve_method,
            'precondition_method':   precondition_method,
            'ps_ds_para':            ps_ds_para,
            'pf_ds_para':            pf_ds_para,
            'field_range':           field_range,
            'n_grid':                n_grid,
            'fileHandle':            fileHandle,
            'region_type':           'rectangle',
            'radius':                radius,
            'u':                     u,
            'twoPara_n':             twoPara_n,
            'legendre_m':            legendre_m,
            'legendre_k':            legendre_k,
            'getConvergenceHistory': getConvergenceHistory,
            'pickProblem':           pickProblem,

        }

        PETSc.Sys.Print('sphere radius: %f, delta length: %f, velocity: %f' % (radius, deltaLength, u))
        err_msg = "Only 'ps_ds' method is accept for this main code. "
        assert matrix_method in ('ps_ds'), err_msg
        PETSc.Sys.Print('create matrix method: %s' % matrix_method)
        PETSc.Sys.Print('Number of force and velocity nodes are %d and %d, respectively.'
              % (sphere_f_geo.get_n_nodes(), sphere_ini_u_geo.get_n_nodes()))
        PETSc.Sys.Print('solve method: %s, precondition method: %s'
              % (solve_method, precondition_method))
        PETSc.Sys.Print('output file headle: ' + fileHandle)
        PETSc.Sys.Print('MPI size: %d' % size)

        # Todo: write bc class to handle boundary condition.
        problem = problem_dic[matrix_method](**problem_kwargs)
        problem.pickmyself(fileHandle)  # not save anything really, just check if the path is correct, to avoid this error after long time calculation.
        obj_sphere = obj_dic[matrix_method]()
        obj_sphere_kwargs = {
            'name':        'sphereObj',
            'pf_geo':      sphere_ini_u_geo,
            'pf_velocity': sphere_ini_u_geo.get_velocity()
        }
        obj_sphere.set_data(sphere_f_geo, sphere_u_geo, **obj_sphere_kwargs)
        problem.add_obj(obj_sphere)
        problem.create_matrix()
        t1 = time()
        PETSc.Sys.Print('%s: create problem use: %fs' % (str(problem), (t1 - t0)))

        ini_guess, ini_residualNorm, ini_problem = problem.ini_guess()
        # ini_guess, ini_residualNorm= None, 0
        # residualNorm = problem.solve(solve_method, precondition_method, ini_guess=ini_guess, Tolerances={'max_it':100000})
        residualNorm = problem.solve(ini_guess=ini_guess)

        problem.pickmyself(fileHandle)
    else:
        # Todo: unpick geo and ini_problem.
        with open(fileHandle + '_pick.bin', 'rb') as input:
            unpick = pickle.Unpickler(input)
            problem = unpick.load()
            problem.unpickmyself()
            obj_sphere = problem.get_obj_list()[-1]
            ini_problem = problem.get_ini_problem()
            residualNorm = problem.get_residualNorm()
            ini_residualNorm = ini_problem.get_residualNorm()
            sphere_f_geo = obj_sphere.get_f_geo()
            sphere_u_geo = obj_sphere.get_u_geo()
            if rank == 0:
                PETSc.Sys.Print('---->>>unpick the problem from file %s.pickle' % fileHandle)

    # Todo: let geo and obj classes do plot stuff.
    if plot:
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.scatter(sphere_f_geo.get_nodes_x(), sphere_f_geo.get_nodes_y(), sphere_f_geo.get_nodes_z(), c='b', marker='o')
        ax.scatter(sphere_u_geo.get_nodes_x(), sphere_u_geo.get_nodes_y(), sphere_u_geo.get_nodes_z(), c='r', marker='.')
        ax.quiver(sphere_u_geo.get_nodes_x(), sphere_u_geo.get_nodes_y(), sphere_u_geo.get_nodes_z(),
                  sphere_u_geo.get_velocity_x(), sphere_u_geo.get_velocity_y(), sphere_u_geo.get_velocity_z(),
                  color='r', length=deltaLength * 2)
        ax.set_aspect('equal')
        plt.grid()
        plt.get_current_fig_manager().window.showMaximized()
        plt.show()

    ini_sphere_err = save_vtk(ini_problem)
    sphere_err = save_vtk(problem, **main_kwargs)
    force_sphere = obj_sphere.get_force_x()
    PETSc.Sys.Print('---->>>%s: Resultant at x axis is %s' % (str(problem), str(np.sum(force_sphere))))

    return problem, sphere_err, ini_sphere_err, residualNorm, ini_residualNorm


if __name__ == '__main__':
    main_fun()
    # r_factor = 3 ** (np.arange(0, 1.2, 0.2) ** 2)
    # deltaLength = 0.05 ** np.arange(0.25, 1.05, 0.1)
    # epsilon = np.arange(0.1, 2, 0.2)
    # # r_factor = np.array((1, 1))
    # # deltaLength = np.array((0.25))
    # # epsilon = np.array((0.1))
    # deltaLength, epsilon = np.meshgrid(deltaLength, epsilon)
    # deltaLength = deltaLength.flatten()
    # epsilon = epsilon.flatten()
    # sphere_err = np.zeros((epsilon.size, r_factor.size))
    # ini_sphere_err = sphere_err.copy()
    # residualNorm = epsilon.copy()
    # ini_residualNorm = epsilon.copy()
    # main_kwargs = {'r_factor': r_factor}
    # OptDB = PETSc.Options()
    # for i0 in range(epsilon.size):
    #     d = deltaLength[i0]
    #     e = epsilon[i0]
    #     fileHandle = 'sphere_%d_%f_%f' % (i0, d, e)
    #     OptDB.setValue('d', d)
    #     OptDB.setValue('e', e)
    #     OptDB.setValue('f', fileHandle)
    #     _, sphere_err[i0, :], ini_sphere_err[i0, :], residualNorm[i0], ini_residualNorm[i0] = main_fun(**main_kwargs)
    # comm = PETSc.COMM_WORLD.tompi4py()
    # rank = comm.Get_rank()
    # if rank == 0:
    #     savemat('sphere_err.mat',
    #             {'deltaLength':      deltaLength,
    #              'epsilon':          epsilon,
    #              'sphere_err':       sphere_err,
    #              'ini_sphere_err':   ini_sphere_err,
    #              'residualNorm':     residualNorm,
    #              'ini_residualNorm': ini_residualNorm,
    #              'r_factor':         r_factor},
    #             oned_as='column')

        # OptDB = PETSc.Options()
        # OptDB.setValue('sm', 'rs')
        # m_rs = main_fun()
        # OptDB.setValue('sm', 'sf')
        # m_sf = main_fun()
        # delta_m = np.abs(m_rs - m_sf)
        # # view_matrix(np.log10(delta_m), 'rs_m - sf_m')
        # percentage = delta_m / (np.maximum(np.abs(m_rs), np.abs(m_sf)) + 1e-100)
        #
        # view_args = {'vmin': -10,
        #              'vmax': 0,
        #              'title': 'log10_abs_rs',
        #              'cmap': 'gray'}
        # view_matrix(np.log10(np.abs(m_rs) + 1e-100), **view_args)
        #
        # view_args = {'vmin': -10,
        #              'vmax': 0,
        #              'title': 'log10_abs_sf',
        #              'cmap': 'gray'}
        # view_matrix(np.log10(np.abs(m_sf) + 1e-100), **view_args)
        #
        # view_args = {'vmin': 0,
        #              'vmax': 1,
        #              'title': 'percentage',
        #              'cmap': 'gray'}
        # view_matrix(percentage, **view_args)
        #
        # view_args = {'vmin': 0,
        #              'vmax': -10,
        #              'title': 'log10_percentage',
        #              'cmap': 'gray'}
        # view_matrix(np.log10(percentage + 1e-100), **view_args)

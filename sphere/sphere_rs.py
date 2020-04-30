# coding=utf-8
# main codes, call functions at stokes_flow.py
# Zhang Ji, 20160410

import sys
import petsc4py

petsc4py.init(sys.argv)
# import warnings
# from memory_profiler import profile
import numpy as np
from src import stokes_flow as sf
# import stokes_flow as sf
from src.stokes_flow import problem_dic, obj_dic
from petsc4py import PETSc
from src.geo import *
from time import time
import pickle
from scipy.io import savemat, loadmat
from src.ref_solution import *


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
    sphere_err = 0

    # problem.vtk_obj(fileHandle)
    # problem.vtk_velocity('%s_Velocity' % fileHandle)
    # problem.vtk_self(fileHandle)

    theta = np.pi / 2
    sphere_check = sf.StokesFlowObj()
    sphere_geo_check = sphere_geo()  # force geo

    if not 'r_factor' in problem_kwargs:
        r_factor = np.ones(1)
    else:
        r_factor = problem_kwargs['r_factor']
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


def get_problem_kwargs(**main_kwargs):
    OptDB = PETSc.Options()
    radius = OptDB.getReal('r', 1)
    deltaLength = OptDB.getReal('d', 0.3)
    epsilon = OptDB.getReal('e', 0.3)
    u = OptDB.getReal('u', 1)
    fileHandle = OptDB.getString('f', 'sphere')
    solve_method = OptDB.getString('s', 'gmres')
    precondition_method = OptDB.getString('g', 'none')
    plot_geo = OptDB.getBool('plot_geo', False)
    debug_mode = OptDB.getBool('debug', False)
    matrix_method = OptDB.getString('sm', 'rs')
    restart = OptDB.getBool('restart', False)
    twoPara_n = OptDB.getInt('tp_n', 1)
    legendre_m = OptDB.getInt('legendre_m', 3)
    legendre_k = OptDB.getInt('legendre_k', 2)
    n_sphere_check = OptDB.getInt('n_sphere_check', 2000)
    n_node_threshold = OptDB.getInt('n_threshold', 10000)
    random_velocity = OptDB.getBool('random_velocity', False)
    getConvergenceHistory = OptDB.getBool('getConvergenceHistory', False)
    pickProblem = OptDB.getBool('pickProblem', False)
    prb_index = OptDB.getInt('prb_index', -1)

    n_obj = OptDB.getInt('n', 1)
    n_obj_x = OptDB.getInt('nx', n_obj)
    n_obj_y = OptDB.getInt('ny', n_obj)
    distance = OptDB.getReal('dist', 3)
    distance_x = OptDB.getReal('distx', distance)
    distance_y = OptDB.getReal('disty', distance)
    move_delta = np.array([distance_x, distance_y, 1])
    # field_range: describe a sector area.
    field_range = np.array([[-3, -3, -3], [n_obj_x - 1, n_obj_y - 1, 0] * move_delta + [3, 3, 3]])
    n_grid = np.array([n_obj_x, n_obj_y, 1]) * 20

    problem_kwargs = {
        'name':                  'spherePrb',
        'matrix_method':         matrix_method,
        'deltaLength':           deltaLength,
        'epsilon':               epsilon,
        'delta':                 deltaLength * epsilon,  # for rs method
        'd_radia':               deltaLength / 2,  # for sf method
        'solve_method':          solve_method,
        'precondition_method':   precondition_method,
        'field_range':           field_range,
        'n_grid':                n_grid,
        'plot_geo':              plot_geo,
        'debug_mode':            debug_mode,
        'fileHandle':            fileHandle,
        'region_type':           'rectangle',
        'twoPara_n':             twoPara_n,
        'legendre_m':            legendre_m,
        'legendre_k':            legendre_k,
        'radius':                radius,
        'u':                     u,
        'random_velocity':       random_velocity,
        'n_obj_x':               n_obj_x,
        'n_obj_y':               n_obj_y,
        'move_delta':            move_delta,
        'restart':               restart,
        'n_sphere_check':        n_sphere_check,
        'n_node_threshold':      n_node_threshold,
        'getConvergenceHistory': getConvergenceHistory,
        'pickProblem':           pickProblem,
        'prb_index':             prb_index,
    }

    for key in main_kwargs:
        problem_kwargs[key] = main_kwargs[key]
    return problem_kwargs


def print_case_info(**problem_kwargs):
    comm = PETSc.COMM_WORLD.tompi4py()
    rank = comm.Get_rank()
    size = comm.Get_size()

    fileHandle = problem_kwargs['fileHandle']
    radius = problem_kwargs['radius']
    deltaLength = problem_kwargs['deltaLength']
    matrix_method = problem_kwargs['matrix_method']
    u = problem_kwargs['u']

    PETSc.Sys.Print('sphere radius: %f, delta length: %f, velocity: %f' % (radius, deltaLength, u))

    err_msg = "Only 'pf', 'rs', 'tp_rs', and 'lg_rs' methods are accept for this main code. "
    assert matrix_method in (
        'rs', 'rs_plane', 'tp_rs', 'lg_rs', 'rs_precondition', 'tp_rs_precondition', 'lg_rs_precondition',
        'pf'), err_msg
    epsilon = problem_kwargs['epsilon']
    if matrix_method in ('rs', 'rs_plane', 'rs_precondition', 'pf'):
        PETSc.Sys.Print('create matrix method: %s, epsilon: %f'
                        % (matrix_method, epsilon))
    elif matrix_method in ('tp_rs', 'tp_rs_precondition'):
        twoPara_n = problem_kwargs['twoPara_n']
        PETSc.Sys.Print('create matrix method: %s, epsilon: %f, order: %d'
                        % (matrix_method, epsilon, twoPara_n))
    elif matrix_method in ('lg_rs', 'lg_rs_precondition'):
        legendre_m = problem_kwargs['legendre_m']
        legendre_k = problem_kwargs['legendre_k']
        PETSc.Sys.Print('create matrix method: %s, epsilon: %f, m: %d, k: %d, p: %d'
                        % (matrix_method, epsilon, legendre_m, legendre_k, (legendre_m + 2 * legendre_k + 1)))

    solve_method = problem_kwargs['solve_method']
    precondition_method = problem_kwargs['precondition_method']
    PETSc.Sys.Print('solve method: %s, precondition method: %s'
                    % (solve_method, precondition_method))
    PETSc.Sys.Print('output file headle: ' + fileHandle)
    PETSc.Sys.Print('MPI size: %d' % size)


# @profile
def main_fun(**main_kwargs):
    comm = PETSc.COMM_WORLD.tompi4py()
    rank = comm.Get_rank()
    problem_kwargs = get_problem_kwargs(**main_kwargs)

    restart = problem_kwargs['restart']
    fileHandle = problem_kwargs['fileHandle']
    radius = problem_kwargs['radius']
    deltaLength = problem_kwargs['deltaLength']
    epsilon = problem_kwargs['epsilon']
    u = problem_kwargs['u']
    matrix_method = problem_kwargs['matrix_method']
    n_obj_x = problem_kwargs['n_obj_x']
    n_obj_y = problem_kwargs['n_obj_y']
    move_delta = problem_kwargs['move_delta']
    random_velocity = problem_kwargs['random_velocity']
    getConvergenceHistory = problem_kwargs['getConvergenceHistory']
    pickProblem = problem_kwargs['pickProblem']

    if not restart:
        print_case_info(**problem_kwargs)

        sphere_geo0 = sphere_geo()  # force geo
        sphere_geo0.create_delta(deltaLength, radius)
        # # DBG
        # nodes = ((0.17389, 0.2938, 0.37454),
        #          (0.76774, 0.87325, 0.50809),
        #          (0.17557, 0.82348, 0.7485),
        #          (0.50734, 0.99882, 0.39992))
        # sphere_geo0.set_nodes(nodes=nodes, deltalength=deltaLength)
        if random_velocity:
            sphere_velocity = np.random.sample(6) * u
        else:
            sphere_velocity = np.array([0, u, 0, 0, 0, 0])
        sphere_geo0.set_rigid_velocity(sphere_velocity)

        problem = problem_dic[matrix_method](**problem_kwargs)
        if pickProblem:
            problem.pickmyself(fileHandle,
                               ifcheck=True)  # not save anything really, just check if the path is correct, to avoid this error after long time calculation.
        obj_sphere = obj_dic[matrix_method]()
        obj_sphere_kwargs = {'name': 'sphereObj_0_0'}
        sphere_geo1 = sphere_geo0.copy()
        if matrix_method in ('pf',):
            sphere_geo1.node_zoom((radius + deltaLength * epsilon) / radius)
        obj_sphere.set_data(sphere_geo1, sphere_geo0, **obj_sphere_kwargs)
        obj_sphere.move((0, 0, 0))
        for i in range(n_obj_x * n_obj_y):
            ix = i // n_obj_x
            iy = i % n_obj_x
            obj2 = obj_sphere.copy()
            obj2.set_name('sphereObj_%d_%d' % (ix, iy))
            move_dist = np.array([ix, iy, 0]) * move_delta
            obj2.move(move_dist)
            if random_velocity:
                sphere_velocity = np.random.sample(6) * u
                obj2.get_u_geo().set_rigid_velocity(sphere_velocity)
            problem.add_obj(obj2)

        problem.print_info()
        problem.create_matrix()
        residualNorm = problem.solve()
        fp = problem.get_force_petsc()
        if getConvergenceHistory:
            convergenceHistory = problem.get_convergenceHistory()
        if pickProblem:
            problem.pickmyself(fileHandle)
    else:
        with open(fileHandle + '_pick.bin', 'rb') as input:
            unpick = pickle.Unpickler(input)
            problem = unpick.load()
            problem.unpick_myself()
            residualNorm = problem.get_residualNorm()
            obj_sphere = problem.get_obj_list()[0]
            PETSc.Sys.Print('---->>>unpick the problem from file %s.pickle' % (fileHandle))

    sphere_err = 0
    # sphere_err = save_vtk(problem, **main_kwargs)
    force_sphere = obj2.get_total_force()
    PETSc.Sys.Print('---->>>Resultant is', force_sphere / 6 / np.pi / radius / u)

    return problem, sphere_err


# @profile
def two_step_main_fun(**main_kwargs):
    comm = PETSc.COMM_WORLD.tompi4py()
    rank = comm.Get_rank()
    problem_kwargs = get_problem_kwargs(**main_kwargs)

    restart = problem_kwargs['restart']
    fileHandle = problem_kwargs['fileHandle']
    radius = problem_kwargs['radius']
    deltaLength = problem_kwargs['deltaLength']
    u = problem_kwargs['u']
    matrix_method = problem_kwargs['matrix_method']

    if not restart:
        n = int(16 * radius * radius / deltaLength / deltaLength)
        sphere_geo0 = sphere_geo()  # force geo
        sphere_geo0.create_n(n, radius)
        sphere_geo0.set_rigid_velocity([u, 0, 0, 0, 0, 0])

        print_case_info(**problem_kwargs)

        problem = problem_dic[matrix_method](**problem_kwargs)
        problem.pickmyself(
                fileHandle)  # not save anything really, just check if the path is correct, to avoid this error after long time calculation.
        obj_sphere = obj_dic[matrix_method]()
        obj_sphere_kwargs = {'name': 'sphereObj'}
        obj_sphere.set_data(sphere_geo0, sphere_geo0, **obj_sphere_kwargs)
        problem.add_obj(obj_sphere)
        problem.print_info()
        problem.create_matrix()

        residualNorm = problem.solve()
        # problem.pickmyself(fileHandle)
    else:
        with open(fileHandle + '_pick.bin', 'rb') as input:
            unpick = pickle.Unpickler(input)
            problem = unpick.load()
            problem.unpick_myself()
            residualNorm = problem.get_residualNorm()
            obj_sphere = problem.get_obj_list()[-1]
            PETSc.Sys.Print('---->>>unpick the problem from file %s.pickle' % (fileHandle))

    sphere_err = 0
    # sphere_err = save_vtk(problem, **main_kwargs)
    factor = 10
    obj_sphere1 = obj_sphere.copy()
    obj_sphere1.zoom(factor)
    ref_slt = sphere_slt(problem)
    problem.vtk_check('%s_Check_%f' % (fileHandle, (radius * d0)), obj_sphere1)

    sphere_geo_check = sphere_geo()
    sphere_geo_check.create_n(2000, radius)
    sphere_geo_check.set_rigid_velocity([u, 0, 0, 0, 0, 0])
    theta = np.pi / 2
    sphere_geo_check.node_rotation(norm=np.array([0, 1, 0]), theta=theta)
    sphere_check = sf.StokesFlowObj()
    sphere_check.set_data(sphere_geo_check, sphere_geo_check)
    sphere_err0 = problem.vtk_check('%s_Check_%f' % (fileHandle, (radius)), sphere_check)[0]

    t0 = time()
    problem_kwargs['delta'] = deltaLength * epsilon * d0
    problem_kwargs['name'] = 'spherePrb1'
    problem1 = problem_dic[matrix_method](**problem_kwargs)
    problem1.add_obj(obj_sphere1)
    problem1.create_matrix()
    t1 = time()
    PETSc.Sys.Print('%s: create problem use: %fs' % (str(problem), (t1 - t0)))
    residualNorm1 = problem1.solve()
    sphere_err1 = problem1.vtk_check('%s_Check_%f' % (fileHandle, (radius * d0)), sphere_check)

    force_sphere = obj_sphere.get_force_x()
    PETSc.Sys.Print('sphere_err0=%f, sphere_err1=%f' % (sphere_err0, sphere_err1))
    PETSc.Sys.Print('---->>>Resultant at x axis is %f' % (np.sum(force_sphere)))

    return problem, sphere_err, residualNorm


def tp_rs_wrapper():
    # r_factor = np.array((1, 1))
    # deltaLength = (0.5, 0.4)
    # epsilon = (0.1, 0.2)
    # N = np.array((1, 2))
    r_factor = 3 ** (np.arange(0, 1.2, 0.2) ** 2)
    deltaLength = 0.05 ** np.arange(0.25, 1.05, 0.1)
    epsilon = 0.1 ** np.arange(-1, 1.2, 0.2)
    N = np.array((1, 2, 10, 20))

    deltaLength, epsilon, N = np.meshgrid(deltaLength, epsilon, N)
    deltaLength = deltaLength.flatten()
    epsilon = epsilon.flatten()
    N = N.flatten()
    sphere_err = np.zeros((epsilon.size, r_factor.size))
    residualNorm = epsilon.copy()
    main_kwargs = {'r_factor': r_factor}
    OptDB = PETSc.Options()
    OptDB.setValue('sm', 'tp_rs')
    for i0 in range(epsilon.size):
        d = deltaLength[i0]
        e = epsilon[i0]
        n = N[i0]
        fileHandle = 'sphere_%05d_%6.4f_%4.2f_%d' % (i0, d, e, n)
        OptDB.setValue('d', d)
        OptDB.setValue('e', e)
        OptDB.setValue('tp_n', int(n))
        OptDB.setValue('f', fileHandle)
        _, sphere_err[i0, :], residualNorm[i0] = main_fun(**main_kwargs)
    comm = PETSc.COMM_WORLD.tompi4py()
    rank = comm.Get_rank()
    if rank == 0:
        savemat('sphere_err.mat',
                {'deltaLength':  deltaLength,
                 'epsilon':      epsilon,
                 'N':            N,
                 'sphere_err':   sphere_err,
                 'residualNorm': residualNorm,
                 'r_factor':     r_factor},
                oned_as='column')


def lg_rs_wrapper():
    """
    to determine best combination of m and n for this method.
    :return:
    """
    # r_factor = np.array((1, 1))
    # deltaLength = (0.5, 0.4)
    # epsilon = (0.1, 0.2)
    # mk_bank = np.array(((2, 0), (2, 1), (2, 2), (2, 3), (2, 4),
    #                     (3, 0), (3, 1), (3, 2), (3, 3),
    #                     (4, 0), (4, 1), (4, 2), (4, 3),
    #                     (5, 0), (5, 1), (5, 2)))
    OptDB = PETSc.Options()
    r_factor = 3 ** (np.arange(0, 1.2, 0.2) ** 2)
    deltaLength = 0.05 ** np.arange(0.25, 1.05, 0.1)
    epsilon = 0.1 ** np.arange(-1, 1.2, 0.2)
    mk_case = OptDB.getInt('mk_case', 0)
    mk_banks = {
        0:  np.array((2, 1)),
        1:  np.array(((2, 0), (2, 1), (2, 2), (2, 3), (2, 4))),
        2:  np.array(((2, 1), (3, 1), (4, 1), (5, 1))),
        3:  np.array(((2, 2), (3, 2), (4, 2), (5, 2))),
        10: np.array(((2, 0), (2, 1), (2, 2), (2, 3), (2, 4),
                      (3, 0), (3, 1), (3, 2), (3, 3),
                      (4, 0), (4, 1), (4, 2), (4, 3),
                      (5, 0), (5, 1), (5, 2)))
    }
    mk_bank = mk_banks[mk_case].reshape((-1, 2))

    deltaLength, epsilon, mk_index = np.meshgrid(deltaLength, epsilon, range(mk_bank.shape[0]))
    deltaLength = deltaLength.flatten()
    epsilon = epsilon.flatten()
    mk_index = mk_index.flatten()
    sphere_err = np.zeros((epsilon.size, r_factor.size))
    residualNorm = epsilon.copy()
    main_kwargs = {'r_factor': r_factor}
    OptDB.setValue('sm', 'lg_rs')
    for i0 in range(epsilon.size):
        d = deltaLength[i0]
        e = epsilon[i0]
        m = mk_bank[mk_index[i0], 0]
        k = mk_bank[mk_index[i0], 1]
        fileHandle = 'sphere_%05d_%6.4f_%4.2f_m=%d,k=%d' % (i0, d, e, m, k)
        OptDB.setValue('d', d)
        OptDB.setValue('e', e)
        OptDB.setValue('legendre_m', int(m))
        OptDB.setValue('legendre_k', int(k))
        OptDB.setValue('f', fileHandle)
        _, sphere_err[i0, :], residualNorm[i0] = main_fun(**main_kwargs)
    comm = PETSc.COMM_WORLD.tompi4py()
    rank = comm.Get_rank()
    if rank == 0:
        savemat('sphere_err.mat',
                {'deltaLength':  deltaLength,
                 'epsilon':      epsilon,
                 'mk_bank':      mk_bank,
                 'mk_index':     mk_index,
                 'sphere_err':   sphere_err,
                 'residualNorm': residualNorm,
                 'r_factor':     r_factor},
                oned_as='column')


def percondition_wrapper():
    """
    multi spheres with random velocities. to determine if the precondition method is work.
    :return:
    """
    # r_factor = np.array((1, 1))
    # deltaLength = (0.5, 0.4)
    # epsilon = (0.1, 0.2)
    # mk_bank = np.array(((2, 0), (2, 1), (2, 2), (2, 3), (2, 4),
    #                     (3, 0), (3, 1), (3, 2), (3, 3),
    #                     (4, 0), (4, 1), (4, 2), (4, 3),
    #                     (5, 0), (5, 1), (5, 2)))
    OptDB = PETSc.Options()
    OptDB.setValue('r', 1)
    OptDB.setValue('d', 0.2)
    OptDB.setValue('e', 0.25)
    OptDB.setValue('f', 'sphere')
    OptDB.setValue('sm', 'lg_rs')
    OptDB.setValue('random_velocity', True)
    OptDB.setValue('getConvergenceHistory', True)
    OptDB.setValue('ksp_rtol', 1e-8)
    n_max = OptDB.getInt('n_max', 2)

    sphere_err = np.zeros((n_max,), dtype=np.object)
    convergenceHistory = np.zeros((n_max,), dtype=np.object)
    for n in range(0, n_max):
        OptDB.setValue('n', n + 1)
        problem, sphere_err[n] = main_fun()
        convergenceHistory[n] = problem.get_convergenceHistory()
    comm = PETSc.COMM_WORLD.tompi4py()
    rank = comm.Get_rank()
    if rank == 0:
        savemat('sphere_err.mat',
                {'n':                  np.arange(n_max),
                 'convergenceHistory': convergenceHistory,
                 'sphere_err':         sphere_err},
                oned_as='column')


def two_step_wrapper():
    OptDB = PETSc.Options()
    # r_factor = 3 ** (np.arange(0, 1.2, 0.2) ** 2)
    r_factor = np.ones(1)
    deltaLength = 0.10573713
    epsilon = 3
    mk_bank = np.array((3, 2))

    sphere_err = np.zeros((r_factor.size))
    main_kwargs = {'r_factor': r_factor}
    OptDB.setValue('sm', 'lg_rs')
    fileHandle = 'sphere_%6.4f_%4.2f_m=%d,k=%d' % \
                 (deltaLength, epsilon, mk_bank[0], mk_bank[1])
    OptDB.setValue('d', deltaLength)
    OptDB.setValue('e', epsilon)
    OptDB.setValue('legendre_m', int(mk_bank[0]))
    OptDB.setValue('legendre_k', int(mk_bank[1]))
    OptDB.setValue('f', fileHandle)
    problem, sphere_err[:], residualNorm = two_step_main_fun(**main_kwargs)


if __name__ == '__main__':
    # lg_rs_wrapper()
    # tp_rs_wrapper()
    # percondition_wrapper()
    main_fun()
    pass

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

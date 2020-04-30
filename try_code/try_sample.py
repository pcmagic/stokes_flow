# coding=utf-8
# main codes, call functions at stokes_flow.py
# Zhang Ji, 20170518

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
        'rs', 'tp_rs', 'lg_rs', 'rs_precondition', 'tp_rs_precondition', 'lg_rs_precondition',
        'pf'), err_msg
    epsilon = problem_kwargs['epsilon']
    if matrix_method in ('rs', 'rs_precondition', 'pf'):
        PETSc.Sys.Print('create matrix method: %s, epsilon: %f' % (matrix_method, epsilon))
    elif matrix_method in ('tp_rs', 'tp_rs_precondition'):
        twoPara_n = problem_kwargs['twoPara_n']
        PETSc.Sys.Print('create matrix method: %s, epsilon: %f, order: %d' % (
            matrix_method, epsilon, twoPara_n))
    elif matrix_method in ('lg_rs', 'lg_rs_precondition'):
        legendre_m = problem_kwargs['legendre_m']
        legendre_k = problem_kwargs['legendre_k']
        PETSc.Sys.Print('create matrix method: %s, epsilon: %f, m: %d, k: %d, p: %d' % (
            matrix_method, epsilon, legendre_m, legendre_k, (legendre_m + 2 * legendre_k + 1)))

    solve_method = problem_kwargs['solve_method']
    precondition_method = problem_kwargs['precondition_method']
    PETSc.Sys.Print(
            'solve method: %s, precondition method: %s' % (solve_method, precondition_method))
    PETSc.Sys.Print('output file headle: ' + fileHandle)
    PETSc.Sys.Print('MPI size: %d' % size)


def get_problem_kwargs(**main_kwargs):
    OptDB = PETSc.Options()
    radius = OptDB.getReal('r', 1)
    deltaLength = OptDB.getReal('d', 0.3)
    epsilon = OptDB.getReal('e', -0.3)
    u = OptDB.getReal('u', 1)
    fileHandle = OptDB.getString('f', 'sphere')
    solve_method = OptDB.getString('s', 'gmres')
    precondition_method = OptDB.getString('g', 'none')
    plot = OptDB.getBool('plot', False)
    debug_mode = OptDB.getBool('debug', False)
    matrix_method = OptDB.getString('sm', 'pf')
    restart = OptDB.getBool('restart', False)
    twoPara_n = OptDB.getInt('tp_n', 1)
    legendre_m = OptDB.getInt('legendre_m', 3)
    legendre_k = OptDB.getInt('legendre_k', 2)
    n_sphere_check = OptDB.getInt('n_sphere_check', 2000)
    n_node_threshold = OptDB.getInt('n_threshold', 10000)
    random_velocity = OptDB.getBool('random_velocity', False)
    getConvergenceHistory = OptDB.getBool('getConvergenceHistory', False)
    pickProblem = OptDB.getBool('pickProblem', False)

    n_obj = OptDB.getInt('n', 1)
    n_obj_x = OptDB.getInt('nx', 2)
    n_obj_y = OptDB.getInt('ny', n_obj)
    distance = OptDB.getReal('dist', 3)
    distance_x = OptDB.getReal('distx', distance)
    distance_y = OptDB.getReal('disty', distance)
    move_delta = np.array([distance_x, distance_y, 1])
    # field_range: describe a sector area.
    field_range = np.array([[-3, -3, -3], [n_obj_x - 1, n_obj_y - 1, 0] * move_delta + [3, 3, 3]])
    n_grid = np.array([n_obj_x, n_obj_y, 1]) * 20

    problem_kwargs = {'name':                  'spherePrb',
                      'matrix_method':         matrix_method,
                      'deltaLength':           deltaLength,
                      'epsilon':               epsilon,
                      'delta':                 deltaLength * epsilon,  # for rs method
                      'd_radia':               deltaLength / 2,  # for sf method
                      'solve_method':          solve_method,
                      'precondition_method':   precondition_method,
                      'field_range':           field_range,
                      'n_grid':                n_grid,
                      'plot':                  plot,
                      'debug_mode':            debug_mode,
                      'fileHandle':            fileHandle,
                      'region_type':           'rectangle',
                      'twoPara_n':             twoPara_n,
                      'legendre_m':            legendre_m,
                      'legendre_k':            legendre_k,
                      'radius':                radius, 'u': u,
                      'random_velocity':       random_velocity,
                      'n_obj_x':               n_obj_x,
                      'n_obj_y':               n_obj_y,
                      'move_delta':            move_delta,
                      'restart':               restart,
                      'n_sphere_check':        n_sphere_check,
                      'n_node_threshold':      n_node_threshold,
                      'getConvergenceHistory': getConvergenceHistory,
                      'pickProblem':           pickProblem,
                      'plot_geo':              False, }

    for key in main_kwargs:
        problem_kwargs[key] = main_kwargs[key]
    return problem_kwargs


def main_fun(**main_kwargs):
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    print_case_info(**problem_kwargs)

    fileHandle = problem_kwargs['fileHandle']
    radius = problem_kwargs['radius']
    deltaLength = problem_kwargs['deltaLength']
    matrix_method = problem_kwargs['matrix_method']
    n_obj_x = problem_kwargs['n_obj_x']
    n_obj_y = problem_kwargs['n_obj_y']
    move_delta = problem_kwargs['move_delta']
    random_velocity = problem_kwargs['random_velocity']
    pickProblem = problem_kwargs['pickProblem']
    epsilon = problem_kwargs['epsilon']
    delta = problem_kwargs['delta']
    u = problem_kwargs['u']
    method_dict = ('pf', 'rs')

    n = int(16 * radius * radius / deltaLength / deltaLength)
    sphere_geo0 = sphere_geo()  # force geo
    sphere_geo0.create_n(n, radius)
    sphere_velocity = np.array((u, 0, 0, 0, 0, 0))
    if random_velocity:
        sphere_velocity = np.random.sample(6) * u
    else:
        sphere_geo0.set_rigid_velocity(sphere_velocity)

    problem = problem_dic[matrix_method](**problem_kwargs)
    obj_sphere = obj_dic[matrix_method]()
    sphere_geo1 = sphere_geo0.copy()
    if matrix_method in ('pf',):
        sphere_geo1.create_n(n, radius + deltaLength * epsilon)
    obj_sphere_kwargs = {'matrix_method': matrix_method,
                         'epsilon':       epsilon,
                         'delta':         delta, }
    obj_sphere.set_data(sphere_geo1, sphere_geo0, name='sphereObj_0_0', **obj_sphere_kwargs)
    for i in range(n_obj_x * n_obj_y):
        ix = i // n_obj_x
        iy = i % n_obj_x
        move_dist = np.array([ix, iy, 0]) * move_delta
        obj2 = obj_sphere.copy()
        move_dist = np.array([ix, iy, 0]) * move_delta
        obj2.move(move_dist)
        if random_velocity:
            sphere_velocity = np.random.sample(6) * u
        obj2.get_u_geo().set_rigid_velocity(sphere_velocity)
        obj2.set_name('sphereObj_%d_%d' % (ix, iy))
        obj2_matrix_method = method_dict[i % len(method_dict)]
        obj2_kwargs = {'matrix_method': obj2_matrix_method,
                       'epsilon':       epsilon,
                       'delta':         delta, }
        obj2.set_matrix_method(**obj2_kwargs)
        problem.add_obj(obj2)

    # problem.show_f_nodes()
    problem.print_info()
    problem.create_matrix()
    residualNorm = problem.solve()

    geo_check = sphere_geo()  # force geo
    geo_check.create_n(n * 3, radius)
    geo_check.set_rigid_velocity(sphere_velocity)
    obj_check = obj_dic[matrix_method]()
    obj_check.set_data(geo_check, geo_check, name='checkObj', **obj_sphere_kwargs)
    problem.vtk_check(fileHandle + '_check', obj_check)

    problem.vtk_self(fileHandle)
    obj2.vtk(fileHandle)
    force_sphere = obj2.get_force_x()
    PETSc.Sys.Print('---->>>%s: Resultant at x axis is %f' % (
        str(problem), force_sphere.sum() / (6 * np.pi * radius)))

    return True


if __name__ == '__main__':
    main_fun()

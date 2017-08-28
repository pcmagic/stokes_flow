# coding=utf-8
# main codes, call functions at stokes_flow.py
# Zhang Ji, 20170518

import sys
import petsc4py

petsc4py.init(sys.argv)
from os import path as ospath

t_path = sys.path[0]
t_path = ospath.dirname(t_path)
if ospath.isdir(t_path):
    sys.path = [t_path] + sys.path
else:
    err_msg = "can not add path father path"
    raise ValueError(err_msg)

import numpy as np
# from src import stokes_flow as sf
from src.stokes_flow import problem_dic, obj_dic
from petsc4py import PETSc
from src.geo import *

# from time import time
import pickle


# from scipy.io import savemat
# from src.ref_solution import *
# from scipy.io import loadmat
# import warnings
# from memory_profiler import profile


def print_case_info(**problem_kwargs):
    fileHeadle = problem_kwargs['fileHeadle']
    radius = problem_kwargs['radius']
    deltaLength = problem_kwargs['deltaLength']
    matrix_method = problem_kwargs['matrix_method']
    twoPlateHeight = problem_kwargs['twoPlateHeight']
    movez = problem_kwargs['movez']
    sphere_velocity = problem_kwargs['sphere_velocity']

    PETSc.Sys.Print('sphere radius: %f, delta length: %f' % (radius, deltaLength))
    PETSc.Sys.Print('sphere velocity: %s' % str(sphere_velocity))

    err_msg = "Only 'pf', 'pf_stokesletsTwoPlate', 'rs', 'tp_rs', and 'lg_rs' methods are accept for this main code. "
    assert matrix_method in ('rs', 'tp_rs', 'lg_rs', 'rs_precondition', 'tp_rs_precondition', 'lg_rs_precondition',
                             'pf', 'pf_stokesletsTwoPlate'), err_msg
    epsilon = problem_kwargs['epsilon']
    if matrix_method in ('rs', 'rs_precondition', 'pf', 'pf_stokesletsTwoPlate'):
        PETSc.Sys.Print('create matrix method: %s, epsilon: %f' % (matrix_method, epsilon))
    elif matrix_method in ('tp_rs', 'tp_rs_precondition'):
        twoPara_n = problem_kwargs['twoPara_n']
        PETSc.Sys.Print('create matrix method: %s, epsilon: %f, order: %d' % (matrix_method, epsilon, twoPara_n))
    elif matrix_method in ('lg_rs', 'lg_rs_precondition'):
        legendre_m = problem_kwargs['legendre_m']
        legendre_k = problem_kwargs['legendre_k']
        PETSc.Sys.Print('create matrix method: %s, epsilon: %f, m: %d, k: %d, p: %d' % (
            matrix_method, epsilon, legendre_m, legendre_k, (legendre_m + 2 * legendre_k + 1)))

    solve_method = problem_kwargs['solve_method']
    precondition_method = problem_kwargs['precondition_method']
    PETSc.Sys.Print('solve method: %s, precondition method: %s' % (solve_method, precondition_method))
    PETSc.Sys.Print('output file headle: ' + fileHeadle)
    PETSc.Sys.Print('twoPlateHeight: %f, movez: %f' % (twoPlateHeight, movez))


def get_problem_kwargs(**main_kwargs):
    OptDB = PETSc.Options()
    radius = OptDB.getReal('r', 1)
    deltaLength = OptDB.getReal('d', 2.7)
    epsilon = OptDB.getReal('e', -0.1)
    ux = OptDB.getReal('ux', 1)
    uy = OptDB.getReal('uy', 1)
    uz = OptDB.getReal('uz', 1)
    wx = OptDB.getReal('wx', 0)
    wy = OptDB.getReal('wy', 0)
    wz = OptDB.getReal('wz', 0)
    sphere_velocity = np.array((ux, uy, uz, wx, wy, wz))
    fileHeadle = OptDB.getString('f', 'sphere')
    solve_method = OptDB.getString('s', 'gmres')
    precondition_method = OptDB.getString('g', 'none')
    debug_mode = OptDB.getBool('debug', False)
    matrix_method = OptDB.getString('sm', 'pf_stokesletsTwoPlate')
    restart = OptDB.getBool('restart', False)
    twoPara_n = OptDB.getInt('tp_n', 1)
    legendre_m = OptDB.getInt('legendre_m', 3)
    legendre_k = OptDB.getInt('legendre_k', 2)
    n_sphere_check = OptDB.getInt('n_sphere_check', 2000)
    n_node_threshold = OptDB.getInt('n_threshold', 10000)
    getConvergenceHistory = OptDB.getBool('getConvergenceHistory', False)
    pickProblem = OptDB.getBool('pickProblem', False)
    twoPlateHeight = OptDB.getReal('twoPlateHeight', 1)
    movez = OptDB.getReal('movez', 0.5)
    plot_geo = OptDB.getBool('plot_geo', False)

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
                      'debug_mode':            debug_mode,
                      'fileHeadle':            fileHeadle,
                      'region_type':           'rectangle',
                      'twoPara_n':             twoPara_n,
                      'legendre_m':            legendre_m,
                      'legendre_k':            legendre_k,
                      'radius':                radius,
                      'sphere_velocity':       sphere_velocity,
                      'n_obj_x':               n_obj_x,
                      'n_obj_y':               n_obj_y,
                      'move_delta':            move_delta,
                      'restart':               restart,
                      'n_sphere_check':        n_sphere_check,
                      'n_node_threshold':      n_node_threshold,
                      'getConvergenceHistory': getConvergenceHistory,
                      'pickProblem':           pickProblem,
                      'twoPlateHeight':        twoPlateHeight,
                      'movez':                 movez,
                      'plot_geo':              plot_geo,
                      }

    for key in main_kwargs:
        problem_kwargs[key] = main_kwargs[key]
    return problem_kwargs


def main_fun(**main_kwargs):
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    print_case_info(**problem_kwargs)

    fileHeadle = problem_kwargs['fileHeadle']
    radius = problem_kwargs['radius']
    deltaLength = problem_kwargs['deltaLength']
    matrix_method = problem_kwargs['matrix_method']
    movez = problem_kwargs['movez']
    pickProblem = problem_kwargs['pickProblem']
    epsilon = problem_kwargs['epsilon']
    sphere_velocity = problem_kwargs['sphere_velocity']

    sphere_geo0 = sphere_geo()  # force geo
    sphere_geo0.create_delta(deltaLength=deltaLength, radius=radius)
    sphere_geo0.set_rigid_velocity(sphere_velocity)

    obj_sphere = obj_dic[matrix_method]()
    sphere_geo1 = sphere_geo0.copy()
    if matrix_method in ('pf', 'pf_stokesletsTwoPlate',):
        sphere_geo1.node_zoom((radius + deltaLength * epsilon) / radius)
    obj_sphere.set_data(sphere_geo1, sphere_geo0, name='sphereObj_0_0')
    obj_sphere.move((0, 0, movez))

    problem = problem_dic[matrix_method](**problem_kwargs)
    problem.add_obj(obj_sphere)
    problem.print_info()
    problem.show_f_u_nodes()
    if pickProblem:
        problem.pickmyself(fileHeadle, check=True)
    problem.create_matrix()
    problem.solve()
    if problem_kwargs['pickProblem']:
        problem.pickmyself(fileHeadle, pick_M=True)

    problem.vtk_self(fileHeadle)
    force_sphere = obj_sphere.get_total_force()
    force_sphere = force_sphere / (6 * radius, 6 * radius, 6 * radius, 8 * radius, 8 * radius, 8 * radius) / np.pi
    PETSc.Sys.Print('---->>>%s: Resultant is %s' % (str(problem), str(force_sphere)))

    geo_check = sphere_geo()  # force geo
    n = int(16 * radius * radius / deltaLength / deltaLength)
    geo_check.create_n(n, radius)
    geo_check.set_rigid_velocity(sphere_velocity)
    geo_check.move((0, 0, movez))
    obj_check = obj_dic[matrix_method]()
    obj_check.set_data(geo_check, geo_check, name='sphereObj_check')
    err = problem.vtk_check(fileHeadle + '_check', obj_check)
    PETSc.Sys.Print('err: ', err)

    return True


def create_M(**main_kwargs):
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    print_case_info(**problem_kwargs)

    fileHeadle = problem_kwargs['fileHeadle']
    radius = problem_kwargs['radius']
    deltaLength = problem_kwargs['deltaLength']
    matrix_method = problem_kwargs['matrix_method']
    movez = problem_kwargs['movez']
    epsilon = problem_kwargs['epsilon']
    sphere_velocity = problem_kwargs['sphere_velocity']

    sphere_geo0 = sphere_geo()  # force geo
    sphere_geo0.create_delta(deltaLength=deltaLength, radius=radius)
    sphere_geo0.set_rigid_velocity(sphere_velocity)

    obj_sphere = obj_dic[matrix_method]()
    sphere_geo1 = sphere_geo0.copy()
    if matrix_method in ('pf', 'pf_stokesletsTwoPlate',):
        sphere_geo1.node_zoom((radius + deltaLength * epsilon) / radius)
    obj_sphere.set_data(sphere_geo1, sphere_geo0, name='sphereObj_0_0')
    obj_sphere.move((0, 0, movez))

    problem = problem_dic[matrix_method](**problem_kwargs)
    problem.add_obj(obj_sphere)
    problem.print_info()
    # problem.show_f_u_nodes()
    problem.pickmyself(fileHeadle, check=True)
    problem.create_matrix()
    problem.solve()
    problem.pickmyself(fileHeadle, pick_M=True)
    return True


def restart_solve(**main_kwargs):
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    fileHeadle = problem_kwargs['fileHeadle']
    sphere_velocity = problem_kwargs['sphere_velocity']
    PETSc.Sys.Print('sphere velocity: %s' % str(sphere_velocity))
    with open(fileHeadle + '_pick.bin', 'rb') as input:
        unpick = pickle.Unpickler(input)
        problem = unpick.load()
        problem.unpickmyself()

    problem_kwargs_old = problem.get_kwargs()
    radius = problem_kwargs_old['radius']
    deltaLength = problem_kwargs_old['deltaLength']
    movez = problem_kwargs_old['movez']
    matrix_method = problem_kwargs_old['matrix_method']

    obj_sphere = problem.get_obj_list()[0]
    sphere_geo0 = obj_sphere.get_u_geo()
    sphere_geo0.set_rigid_velocity(sphere_velocity)
    problem.create_F_U()
    problem.print_info()
    problem.solve()

    problem.vtk_self(fileHeadle)
    force_sphere = obj_sphere.get_total_force()
    force_sphere = force_sphere / np.pi / \
                   (6 * radius, 6 * radius, 6 * radius, 8 * radius ** 3, 8 * radius ** 3, 8 * radius ** 3)
    PETSc.Sys.Print('---->>>%s: Resultant is %s' % (str(problem), str(force_sphere)))

    geo_check = sphere_geo()  # force geo
    n = int(16 * radius * radius / deltaLength / deltaLength)
    geo_check.create_n(n, radius)
    geo_check.set_rigid_velocity(sphere_velocity)
    geo_check.move((0, 0, movez))
    obj_check = obj_dic[matrix_method]()
    obj_check.set_data(geo_check, geo_check, name='sphereObj_check')
    err = problem.vtk_check(fileHeadle + '_check', obj_check)
    PETSc.Sys.Print('err: ', err)


def export_data(**main_kwargs):
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    fileHeadle = problem_kwargs['fileHeadle']
    with open(fileHeadle + '_pick.bin', 'rb') as input:
        unpick = pickle.Unpickler(input)
        problem = unpick.load()
        problem.unpickmyself()

    obj_sphere = problem.get_obj_list()[0]
    obj_sphere.save_mat()
    # problem.saveM_mat(fileHeadle + '_M')
    problem.saveM_ASCII(fileHeadle + '_M')
    # problem.saveM_HDF5(fileHeadle + '_M')
    PETSc.Sys.Print('finish')


if __name__ == '__main__':
    OptDB = PETSc.Options()
    mode = OptDB.getString('mode', 'main')
    if mode == 'main':
        main_fun()
    elif mode == 'createM':
        create_M()
    elif mode == 'resolve':
        restart_solve()
    elif mode == 'export':
        export_data()
    else:
        err_msg = 'wrong program mode'
        raise err_msg

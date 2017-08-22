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
from src.stokes_flow import problem_dic, obj_dic
import src.stokes_flow as sf
from petsc4py import PETSc
from src.geo import *
import pickle


# def print_case_info(**problem_kwargs):
#     comm = PETSc.COMM_WORLD.tompi4py()
#     rank = comm.Get_rank()
#     size = comm.Get_size()
#
#     fileHeadle = problem_kwargs['fileHeadle']
#     radius = problem_kwargs['radius']
#     deltaLength = problem_kwargs['deltaLength']
#     matrix_method = problem_kwargs['matrix_method']
#     u = problem_kwargs['u']
#
#     PETSc.Sys.Print('sphere radius: %f, delta length: %f, velocity: %f' % (radius, deltaLength, u))
#
#     err_msg = "Only 'pf', 'rs', 'tp_rs', and 'lg_rs' methods are accept for this main code. "
#     assert matrix_method in ('rs', 'tp_rs', 'lg_rs', 'rs_precondition', 'tp_rs_precondition', 'lg_rs_precondition', 'pf'), err_msg
#     epsilon = problem_kwargs['epsilon']
#     if matrix_method in ('rs', 'rs_precondition', 'pf'):
#         PETSc.Sys.Print('create matrix method: %s, epsilon: %f'
#                         % (matrix_method, epsilon))
#     elif matrix_method in ('tp_rs', 'tp_rs_precondition'):
#         twoPara_n = problem_kwargs['twoPara_n']
#         PETSc.Sys.Print('create matrix method: %s, epsilon: %f, order: %d'
#                         % (matrix_method, epsilon, twoPara_n))
#     elif matrix_method in ('lg_rs', 'lg_rs_precondition'):
#         legendre_m = problem_kwargs['legendre_m']
#         legendre_k = problem_kwargs['legendre_k']
#         PETSc.Sys.Print('create matrix method: %s, epsilon: %f, m: %d, k: %d, p: %d'
#                         % (matrix_method, epsilon, legendre_m, legendre_k, (legendre_m + 2 * legendre_k + 1)))
#
#     solve_method = problem_kwargs['solve_method']
#     precondition_method = problem_kwargs['precondition_method']
#     PETSc.Sys.Print('solve method: %s, precondition method: %s'
#                     % (solve_method, precondition_method))
#     PETSc.Sys.Print('output file headle: ' + fileHeadle)
#     PETSc.Sys.Print('MPI size: %d' % size)


def get_problem_kwargs(**main_kwargs):
    OptDB = PETSc.Options()
    radius = OptDB.getReal('r', 1)
    n = OptDB.getInt('n', 200)
    deltaLength = np.sqrt(4 * np.pi * radius * radius / n)
    epsilon = OptDB.getReal('e', -1)
    rel_omega = OptDB.getReal('rel_omega', 1)
    rel_u = OptDB.getReal('rel_u', 1)
    fileHeadle = OptDB.getString('f', 'try_forcefree')
    solve_method = OptDB.getString('s', 'gmres')
    precondition_method = OptDB.getString('g', 'none')
    plot = OptDB.getBool('plot', False)
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
        'solve_method':          solve_method,
        'precondition_method':   precondition_method,
        'field_range':           field_range,
        'n_grid':                n_grid,
        'plot':                  plot,
        'fileHeadle':            fileHeadle,
        'region_type':           'rectangle',
        'twoPara_n':             twoPara_n,
        'legendre_m':            legendre_m,
        'legendre_k':            legendre_k,
        'radius':                radius,
        'rel_omega':             rel_omega,
        'rel_u':                 rel_u,
        'random_velocity':       random_velocity,
        'n_obj_x':               n_obj_x,
        'n_obj_y':               n_obj_y,
        'move_delta':            move_delta,
        'restart':               restart,
        'n_sphere_check':        n_sphere_check,
        'n_node_threshold':      n_node_threshold,
        'getConvergenceHistory': getConvergenceHistory,
        'pickProblem':           pickProblem
    }

    for key in main_kwargs:
        problem_kwargs[key] = main_kwargs[key]
    return problem_kwargs


def main_fun(**main_kwargs):
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    # print_case_info(**problem_kwargs)

    fileHeadle = problem_kwargs['fileHeadle']
    radius = problem_kwargs['radius']
    deltaLength = problem_kwargs['deltaLength']
    matrix_method = problem_kwargs['matrix_method']
    n_obj_x = problem_kwargs['n_obj_x']
    n_obj_y = problem_kwargs['n_obj_y']
    move_delta = problem_kwargs['move_delta']
    random_velocity = problem_kwargs['random_velocity']
    getConvergenceHistory = problem_kwargs['getConvergenceHistory']
    pickProblem = problem_kwargs['pickProblem']
    epsilon = problem_kwargs['epsilon']
    rel_u = problem_kwargs['rel_u']
    rel_omega = problem_kwargs['rel_omega']
    norm_rel_u = np.array((1, 1, 1))
    norm_rel_omega = np.array((0, 0, 0))
    rel_U = np.hstack((rel_u * norm_rel_u, rel_omega * norm_rel_omega))
    center = np.zeros(3)

    n = int(16 * radius * radius / deltaLength / deltaLength)
    sphere_geo0 = sphere_geo()  # force geo
    sphere_geo0.create_n(n, radius)
    sphere_geo0.node_rotation(theta=np.pi / 5)
    sphere_geo1 = sphere_geo0.copy()
    if matrix_method in ('pf',):
        sphere_geo1.create_n(n, radius + deltaLength * epsilon)

    obj_sphere = obj_dic[matrix_method]()
    obj_sphere_kwargs = {'name': 'sphereObj_0_0'}
    obj_sphere.set_data(sphere_geo1, sphere_geo0, **obj_sphere_kwargs)

    name = 'obj_composite1'
    obj_composite = sf.forceFreeComposite(center, name)
    obj_composite.add_obj(obj_sphere, rel_U=rel_U)

    problem = sf.stokesletsInPipeForceFreeProblem(**problem_kwargs)
    problem.add_obj(obj_composite)
    problem.create_matrix()

    problem.print_info()
    problem.solve()

    problem.pickmyself(fileHeadle)
    with open(fileHeadle + '_pick.bin', 'rb') as input:
        unpick = pickle.Unpickler(input)
        problem = unpick.load()
        problem.unpickmyself()
    problem.create_matrix()
    problem.print_info()
    problem.solve()
    problem.pickmyself(fileHeadle)

    PETSc.Sys.Print(obj_composite.get_ref_U())
    PETSc.Sys.Print(obj_composite.get_re_sum())

    problem.vtk_obj(fileHeadle)
    geo_check = sphere_geo()  # force geo
    geo_check.create_n(n * 2, radius)
    geo_check.set_rigid_velocity(rel_U + obj_composite.get_ref_U())
    obj_check = obj_dic[matrix_method]()
    obj_check.set_data(geo_check, geo_check, **obj_sphere_kwargs)
    problem.vtk_check(fileHeadle + '_check', obj_check)

    problem.vtk_self(fileHeadle)
    obj_sphere.vtk(fileHeadle)
    force_sphere = obj_sphere.get_force_x()
    PETSc.Sys().Print('---->>>%s: Resultant at x axis is %f' % (str(problem), force_sphere.sum()/(6*np.pi*radius)))

    return True


if __name__ == '__main__':
    main_fun()

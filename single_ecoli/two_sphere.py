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
    radius1 = OptDB.getReal('r1', 1)
    radius2 = OptDB.getReal('r2', 1.5)
    deltaLength = OptDB.getReal('d', 1)  # delta length
    epsilon = OptDB.getReal('e', -2)
    rel_omega = OptDB.getReal('rel_omega', 10)
    rel_u = OptDB.getReal('rel_u', 1)
    fileHeadle = OptDB.getString('f', 'two_sphere')
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
    plot_geo = OptDB.getBool('plot_geo', False)
    zoom_factor = OptDB.getReal('zoom_factor', 1)
    prb_index = OptDB.getInt('prb_index', -1)

    # n_obj = OptDB.getInt('n', 1)
    # n_obj_x = OptDB.getInt('nx', n_obj)
    # n_obj_y = OptDB.getInt('ny', n_obj)
    # distance = OptDB.getReal('dist', 3)
    # distance_x = OptDB.getReal('distx', distance)
    # distance_y = OptDB.getReal('disty', distance)
    # move_delta = np.array([distance_x, distance_y, 1])
    # # field_range: describe a sector area.
    # field_range = np.array([[-3, -3, -3], [n_obj_x - 1, n_obj_y - 1, 0] * move_delta + [3, 3, 3]])
    # n_grid = np.array([n_obj_x, n_obj_y, 1]) * 20

    problem_kwargs = {
        'name':                  'two_sphere',
        'matrix_method':         matrix_method,
        'deltaLength':           deltaLength,
        'epsilon':               epsilon,
        'delta':                 deltaLength * epsilon,  # for rs method
        'solve_method':          solve_method,
        'precondition_method':   precondition_method,
        # 'field_range':           field_range,
        # 'n_grid':                n_grid,
        'plot':                  plot,
        'fileHeadle':            fileHeadle,
        'region_type':           'rectangle',
        'twoPara_n':             twoPara_n,
        'legendre_m':            legendre_m,
        'legendre_k':            legendre_k,
        'r1':                    radius1,
        'r2':                    radius2,
        'rel_omega':             rel_omega,
        'rel_u':                 rel_u,
        'random_velocity':       random_velocity,
        # 'n_obj_x':               n_obj_x,
        # 'n_obj_y':               n_obj_y,
        # 'move_delta':            move_delta,
        'restart':               restart,
        'n_sphere_check':        n_sphere_check,
        'n_node_threshold':      n_node_threshold,
        'getConvergenceHistory': getConvergenceHistory,
        'pickProblem':           pickProblem,
        'plot_geo':              plot_geo,
        'zoom_factor':           zoom_factor,
        'prb_index':             prb_index,
    }

    for key in main_kwargs:
        problem_kwargs[key] = main_kwargs[key]
    return problem_kwargs


def two_sphere_geo():
    problem_kwargs = get_problem_kwargs()
    r1 = problem_kwargs['r1']
    r2 = problem_kwargs['r2']
    deltaLength = problem_kwargs['deltaLength']
    epsilon = problem_kwargs['epsilon']
    matrix_method = problem_kwargs['matrix_method']
    zoom_factor = problem_kwargs['zoom_factor']

    # obj_sphere0
    sphere_geo0 = sphere_geo()  # force geo
    sphere_geo0.create_delta(deltaLength, r1)
    sphere_geo0.node_rotation(theta=np.pi / 5)
    sphere_geo1 = sphere_geo0.copy()
    if matrix_method in ('pf',):
        sphere_geo1.node_zoom((r1 + deltaLength * epsilon) / r1)
    obj_sphere0 = obj_dic[matrix_method]()
    obj_sphere_kwargs = {'name': 'sphereObj_0_0'}
    obj_sphere0.set_data(sphere_geo1, sphere_geo0, **obj_sphere_kwargs)
    obj_sphere0.zoom(zoom_factor)
    # obj_sphere1
    sphere_geo2 = sphere_geo()  # force geo
    sphere_geo2.create_delta(deltaLength, r2)
    sphere_geo2.node_rotation(theta=np.pi / 5)
    sphere_geo3 = sphere_geo2.copy()
    if matrix_method in ('pf',):
        sphere_geo3.node_zoom((r2 + deltaLength * epsilon) / r2)
    obj_sphere1 = obj_dic[matrix_method]()
    obj_sphere_kwargs = {'name': 'sphereObj_0_1'}
    obj_sphere1.set_data(sphere_geo3, sphere_geo2, **obj_sphere_kwargs)
    obj_sphere1.zoom(zoom_factor)
    obj_sphere1.move(np.array((1.5 * (r1 + r2), 0, 0)) * zoom_factor)
    return obj_sphere0, obj_sphere1


def main_fun(**main_kwargs):
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    # print_case_info(**problem_kwargs)

    fileHeadle = problem_kwargs['fileHeadle']
    deltaLength = problem_kwargs['deltaLength']
    matrix_method = problem_kwargs['matrix_method']
    random_velocity = problem_kwargs['random_velocity']
    getConvergenceHistory = problem_kwargs['getConvergenceHistory']
    pickProblem = problem_kwargs['pickProblem']
    rel_u = problem_kwargs['rel_u']
    rel_omega = problem_kwargs['rel_omega']
    zoom_factor = problem_kwargs['zoom_factor']
    prb_index = problem_kwargs['prb_index']
    r1 = problem_kwargs['r1']
    r2 = problem_kwargs['r2']
    norm_rel_u = np.array((0, 0, 0))
    norm_rel_omega = np.array((4, 5, 6))
    rel_U = np.hstack((rel_u * norm_rel_u, rel_omega * norm_rel_omega))
    center = np.zeros(3)

    obj_sphere0, obj_sphere1 = two_sphere_geo()
    name = 'obj_composite1'
    obj_composite = sf.forceFreeComposite(center, name)
    obj_composite.add_obj(obj_sphere0, rel_U=rel_U)
    obj_composite.add_obj(obj_sphere1, rel_U=np.zeros(6))
    if problem_kwargs['plot_geo']:
        obj_composite.show_f_u_nodes()

    problem = sf.forceFreeProblem(**problem_kwargs)
    problem.add_obj(obj_composite)
    problem.create_matrix()

    problem.print_info()
    residualNorm = problem.solve()

    temp_f = 0.5 * (np.abs(obj_sphere0.get_force().reshape((-1, 3)).sum(axis=0)) + np.abs(obj_sphere1.get_force().reshape((-1, 3)).sum(axis=0)))
    temp_F = np.hstack((temp_f, temp_f * zoom_factor))
    non_dim_F = obj_composite.get_re_sum() / temp_F
    non_dim_U = obj_composite.get_ref_U() / np.array((zoom_factor, zoom_factor, zoom_factor, 1, 1, 1))
    PETSc.Sys.Print(non_dim_U)
    PETSc.Sys.Print(non_dim_F)

    problem.vtk_obj(fileHeadle)
    OptDB = PETSc.Options()
    OptDB.setValue('d', deltaLength * 0.8)
    obj_sphere0_check, obj_sphere1_check = two_sphere_geo()
    obj_sphere0_check.set_rigid_velocity(rel_U + obj_composite.get_ref_U())
    obj_sphere1_check.set_rigid_velocity(np.zeros(6) + obj_composite.get_ref_U())
    check_obj = sf.stokesFlowObj()
    check_obj.combine([obj_sphere0_check, obj_sphere1_check], set_re_u=True, set_force=True)
    if problem_kwargs['plot_geo']:
        check_obj.show_f_u_nodes()
    vel_err = problem.vtk_check(fileHeadle + '_check', check_obj)
    vel_err1 = problem.vtk_check(fileHeadle + '_check1', obj_sphere0_check)
    vel_err2 = problem.vtk_check(fileHeadle + '_check2', obj_sphere1_check)
    PETSc.Sys.Print('total velocity error is: %f' % vel_err[0])
    PETSc.Sys.Print('velocity error of sphere1 and sphere2 aer %f and %f, respectively. ' % (vel_err1[0], vel_err2[0]))

    with open("caseInfo.txt", "a") as outfile:
        outline = np.hstack((prb_index, zoom_factor, non_dim_U, non_dim_F, vel_err1, vel_err2, vel_err))
        outfile.write(' '.join('%e'%i for i in outline))
        outfile.write('\n')

    problem.vtk_self(fileHeadle)
    obj_sphere0.vtk(fileHeadle)
    return True


if __name__ == '__main__':
    main_fun()

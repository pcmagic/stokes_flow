# coding=utf-8
# main codes, call functions at stokes_flow.py
# Assuming u=u1+u2, u1 is velocity filed due to a stokeslet. u2=-u1 at boundary of a pip.
# Thus, u==0, no-slip boundary condition at the pip.
# Zhang Ji, 20170320

import sys
from typing import Any, Union

import petsc4py

petsc4py.init(sys.argv)
import numpy as np
from src import stokes_flow as sf
from src.stokes_flow import problem_dic, obj_dic
from petsc4py import PETSc
from src.geo import *
from time import time
from scipy.io import savemat
import pickle


def save_vtk(problem: sf.StokesFlowProblem):
    t0 = time()
    comm = PETSc.COMM_WORLD.tompi4py()
    rank = comm.Get_rank()
    problem_kwargs = problem.get_kwargs()
    fileHandle = problem_kwargs['fileHandle']
    tunnel_radius = problem_kwargs['tunnel_radius']
    length = problem_kwargs['length']
    n_tunnel_check = problem_kwargs['n_tunnel_check']

    problem.vtk_obj(fileHandle)
    problem.vtk_velocity('%s_Velocity' % fileHandle)

    obj_check = sf.StokesFlowObj()
    tunnel_geo_check = tunnel_geo()  # pf, force geo
    tunnel_geo_check.create_n(n_tunnel_check, length / 3, tunnel_radius)
    tunnel_geo_check.set_rigid_velocity(np.array((0, 0, 0, 0, 0, 0)))
    obj_check.set_data(tunnel_geo_check, tunnel_geo_check)
    problem.vtk_check(fileHandle + '_Check_tunnel', obj_check)

    t1 = time()
    PETSc.Sys.Print('%s: write vtk files use: %fs' % (str(problem), (t1 - t0)))

    return True


def get_problem_kwargs(**main_kwargs):
    OptDB = PETSc.Options()
    length = OptDB.getReal('l', 3)
    n_tunnel_parts = OptDB.getInt('tunnel_parts', 1)
    tunnel_radius = OptDB.getReal('tunnel_radius', 1)
    deltaLength = OptDB.getReal('d', 0.3)
    epsilon = OptDB.getReal('e', 1)
    fx = OptDB.getReal('fx', 1)
    fy = OptDB.getReal('fy', 0)
    fz = OptDB.getReal('fz', 0)
    stokeslets_f = np.array((fx, fy, fz))
    stokeslets_b = OptDB.getReal('stokeslets_b', 0)
    fileHandle = OptDB.getString('f', 'stokeletInPipe')
    solve_method = OptDB.getString('s', 'gmres')
    precondition_method = OptDB.getString('g', 'none')
    plot = OptDB.getBool('plot', False)
    debug_mode = OptDB.getBool('debug', False)
    matrix_method = OptDB.getString('sm', 'rs_stokeslets')
    restart = OptDB.getBool('restart', False)
    grid_para = OptDB.getInt('ngrid', 20)
    twoPara_n = OptDB.getInt('tp_n', 1)
    legendre_m = OptDB.getInt('legendre_m', 3)
    legendre_k = OptDB.getInt('legendre_k', 2)
    n_tunnel_check = OptDB.getInt('n_tunnel_check', 30000)
    n_node_threshold = OptDB.getInt('n_threshold', 10000)
    getConvergenceHistory = OptDB.getBool('getConvergenceHistory', False)
    pickProblem = OptDB.getBool('pickProblem', False)
    xRange1 = OptDB.getReal('xRange1', 0)
    xRange2 = OptDB.getReal('xRange2', 1)
    xfactor = OptDB.getReal('xfactor', 1)

    stokeslets_post = np.hstack((0, stokeslets_b, 0))
    # field_range: describe a sector area.
    region_type = 'sector'
    theta = np.pi * 2
    field_range = np.array([[xRange1, 0, -theta / 2 + np.pi / 2], [xRange2, tunnel_radius, theta / 2 + np.pi / 2]])
    # field_range = np.array([[0, 0, -theta / 2 + np.pi / 2], [length / 20, tunnel_radius, theta / 2 + np.pi / 2]])
    temp = np.abs(field_range[0] - field_range[1]) * (xfactor, 1, tunnel_radius / 2) * grid_para
    n_grid = np.array(temp, dtype='int')

    problem_kwargs = {
        'name':                  'stokeletInPipePrb',
        'matrix_method':         matrix_method,
        'tunnel_radius':         tunnel_radius,
        'length':                length,
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
        'region_type':           region_type,
        'twoPara_n':             twoPara_n,
        'legendre_m':            legendre_m,
        'legendre_k':            legendre_k,
        'stokeslets_f':          stokeslets_f,
        'stokeslets_b':          stokeslets_b,
        'stokeslets_post':       stokeslets_post,
        'n_tunnel_parts':        n_tunnel_parts,
        'restart':               restart,
        'n_tunnel_check':        n_tunnel_check,
        'n_node_threshold':  n_node_threshold,
        'getConvergenceHistory': getConvergenceHistory,
        'pickProblem':           pickProblem
    }

    for key in main_kwargs:
        problem_kwargs[key] = main_kwargs[key]
    return problem_kwargs


def print_case_info(**problem_kwargs):
    comm = PETSc.COMM_WORLD.tompi4py()
    rank = comm.Get_rank()
    size = comm.Get_size()

    fileHandle = problem_kwargs['fileHandle']
    tunnel_radius = problem_kwargs['tunnel_radius']
    deltaLength = problem_kwargs['deltaLength']
    matrix_method = problem_kwargs['matrix_method']
    stokeslets_f = problem_kwargs['stokeslets_f']
    length = problem_kwargs['length']
    stokeslets_b = problem_kwargs['stokeslets_b']

    PETSc.Sys.Print('tunnel length: %f, tunnel radius: %f, delta length: %f, velocity: %f'
          % (length, tunnel_radius, deltaLength, 0))
    PETSc.Sys.Print('stokeslets: %s, stokeslets position: %f' % (str(stokeslets_f), stokeslets_b))

    err_msg = "Only 'rs_stokeslets', 'tp_rs_stokeslets', 'lg_rs_stokeslets', and 'ps_stokeslets' methods are accept for this main code. "
    acceptType = ('rs_stokeslets', 'tp_rs_stokeslets', 'lg_rs_stokeslets', 'rs_stokeslets_precondition',
                  'tp_rs_stokeslets_precondition', 'lg_rs_stokeslets_precondition',
                  'pf')
    assert matrix_method in acceptType, err_msg
    epsilon = problem_kwargs['epsilon']
    if matrix_method in ('rs_stokeslets', 'rs_stokeslets_precondition'):
        PETSc.Sys.Print('create matrix method: %s, epsilon: %f'
              % (matrix_method, epsilon))
    elif matrix_method in ('tp_rs_stokeslets', 'tp_rs_stokeslets_precondition'):
        twoPara_n = problem_kwargs['twoPara_n']
        PETSc.Sys.Print('create matrix method: %s, epsilon: %f, order: %d'
              % (matrix_method, epsilon, twoPara_n))
    elif matrix_method in ('lg_rs_stokeslets', 'lg_rs_stokeslets_precondition'):
        legendre_m = problem_kwargs['legendre_m']
        legendre_k = problem_kwargs['legendre_k']
        PETSc.Sys.Print('create matrix method: %s, epsilon: %f, m: %d, k: %d, p: %d'
              % (matrix_method, epsilon, legendre_m, legendre_k, (legendre_m + 2 * legendre_k + 1)))
    elif matrix_method in 'pf':
        PETSc.Sys.Print('create matrix method: %s, epsilon: %f' % (matrix_method, epsilon))
    else:
        raise Exception('set how to print matrix method please. ')

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
    fileHandle = problem_kwargs['fileHandle']
    stokeslets_post = problem_kwargs['stokeslets_post']
    stokeslets_f = problem_kwargs['stokeslets_f']
    tunnel_radius = problem_kwargs['tunnel_radius']

    if not problem_kwargs['restart']:
        tunnel_radius = problem_kwargs['tunnel_radius']
        deltaLength = problem_kwargs['deltaLength']
        matrix_method = problem_kwargs['matrix_method']
        length = problem_kwargs['length']
        n_tunnel_parts = problem_kwargs['n_tunnel_parts']
        print_case_info(**problem_kwargs)
        problem = problem_dic[matrix_method](**problem_kwargs)
        if problem_kwargs['pickProblem']:
            # do NOT save anything really, just check if the path is correct, to avoid this error after long time calculation.
            problem.pickmyself(fileHandle, ifcheck=True)

        # The tunnel is divided into n objects having a similar length.
        tunnel_geo_u = stokeslets_tunnel_geo()
        part_length = length / n_tunnel_parts - (1 - 1 / n_tunnel_parts) * deltaLength
        err_msg = 'length of each part of object >= deltaLength.'
        assert part_length > deltaLength, err_msg
        tunnel_geo_u.create_deltaz(deltaLength, part_length, tunnel_radius)
        tunnel_geo.node_rotation(norm=np.array((0, 1, 0)), theta=np.pi/2)
        move_dist = np.array([-(length - part_length) / 2, 0, 0])
        tunnel_geo_u.move(move_dist)
        tunnel_geo_u.stokeslets_velocity(problem)
        tunnel_geo_f = tunnel_geo_u.copy()
        if matrix_method in sf.two_geo_method_list:
            epsilon = problem_kwargs['epsilon']
            tunnel_geo_f.node_zoom_radius((tunnel_radius + deltaLength * epsilon) / tunnel_radius)
        obj_tunnel = obj_dic[matrix_method]()
        obj_tunnel_kwargs = {'name':            'tunnelObj_0',
                             'stokeslets_post': stokeslets_post,
                             'stokeslets_f':    stokeslets_f}
        obj_tunnel.set_data(tunnel_geo_f, tunnel_geo_u, **obj_tunnel_kwargs)
        problem.add_obj(obj_tunnel)
        for i in np.arange(1, n_tunnel_parts):
            move_dist = np.array([part_length + deltaLength, 0, 0]) * i
            obj2 = obj_tunnel.copy()
            obj2.move(move_dist)
            tunnel_geo_u = obj2.get_u_geo()
            tunnel_geo_f = obj2.get_f_geo()
            tunnel_geo_u.stokeslets_velocity(problem)
            obj2_kwargs = {'name': 'tunnelObj_%d' % (i)}
            obj2.set_data(tunnel_geo_f, tunnel_geo_u, **obj2_kwargs)
            problem.add_obj(obj2)

        problem.print_info()
        problem.create_matrix()

        residualNorm = problem.solve()
        save_vtk(problem)
        if problem_kwargs['pickProblem']:
            problem.pickmyself(fileHandle)
    else:
        with open(fileHandle + '_pick.bin', 'rb') as input:
            unpick = pickle.Unpickler(input)
            problem = unpick.load()
            problem.unpick_myself()
            residualNorm = problem.get_residualNorm()
            PETSc.Sys.Print('---->>>unpick the problem from file %s.pickle' % (fileHandle))

            problem_kwargs = get_problem_kwargs(**main_kwargs)
            problem.set_kwargs(**problem_kwargs)
            save_vtk(problem)

    return problem, residualNorm


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


def casebank():
    ksp_rtol = 1e-6
    n_tunnel_check = 30000
    pickProblem = 'True'
    xRange1 = -0.5
    xRange2 = 0.5
    ngrid = 20
    xfactor = 10
    ksp_max_it = 1e4

    # stokeslets_list = ((0, 1, 0, 0),  # (b, fx, fy, fz)
    #                    (0, 0, 0, 0),
    #                    (0.5, 1, 0, 0),
    #                    (0.5, 0, 1, 0),
    #                    (0.5, 0, 0, 1))
    # fileHeadle1_list = ('b0_100',
    #                     'b0_010',
    #                     'b05_100',
    #                     'b05_010',
    #                     'b05_001')
    stokeslets_list = ((0.2, 1, 0, 0),  # (b, fx, fy, fz)
                       (0.2, 0, 1, 0),
                       (0.2, 0, 0, 1),
                       (0.9, 1, 0, 0),
                       (0.9, 0, 1, 0),
                       (0.9, 0, 0, 1),)
    fileHeadle1_list = ('b02_100',
                        'b02_010',
                        'b02_001',
                        'b09_100',
                        'b09_010',
                        'b09_001',)
    length_list = (70,)
    fileHeadle2_list = ['_l%d' % l for l in length_list]
    # sm_e_d_tube = (('rs_stokeslets', 0.25, 0.15),
    #                ('rs_stokeslets', 0.3, 0.15),
    #                ('lg_rs_stokeslets', 1, 0.15),
    #                ('lg_rs_stokeslets', 3, 0.15),
    #                ('lg_rs_stokeslets', 6, 0.15),
    #                ('pf', 1, 0.2),
    #                ('pf', 1.5, 0.2),
    #                ('pf', 2, 0.2))
    # fileHeadle3_list = ('_rs_e025',
    #                     '_rs_e03',
    #                     '_lg_e1',
    #                     '_lg_e3',
    #                     '_lg_e6',
    #                     '_pf_e1',
    #                     '_pf_e1.5',
    #                     '_pf_e2',)
    sm_e_d_tube = (('lg_rs_stokeslets', 6, 0.15),
                   ('pf', 2, 0.15))
    fileHeadle3_list = ('_lg_e6',
                        '_pf_e2',)

    for i0 in range(len(fileHeadle1_list)):
        fileHeadle1 = fileHeadle1_list[i0]
        b, fx, fy, fz = stokeslets_list[i0]
        for i1 in range(len(fileHeadle2_list)):
            fileHeadle2 = fileHeadle2_list[i1]
            l = length_list[i1]
            for i2 in range(len(fileHeadle3_list)):
                fileHeadle3 = fileHeadle3_list[i2]
                sm, e, d = sm_e_d_tube[i2]
                fileHandle = fileHeadle1 + fileHeadle2 + fileHeadle3
                kwargs = '-l %f -d %f -e %f -fx %f -fy %f -fz %f -stokeslets_b %f -sm %s -f %s ' \
                         '-n_tunnel_check %d -pickProblem %s -ksp_rtol %f -ksp_max_it %d ' \
                         '-xRange1 %f -xRange2 %f -ngrid %d -xfactor %d ' % \
                         (l, d, e, fx, fy, fz, b, sm, fileHandle,
                          n_tunnel_check, pickProblem, ksp_rtol, ksp_max_it,
                          xRange1, xRange2, ngrid, xfactor)
                PETSc.Sys.Print('echo \'-------------------------------------------->>>>>>%s\'; '
                      'mpirun -n 24 python ../../StokesletInPipe.py %s > %s.txt' %
                      (fileHandle, kwargs, fileHandle))

                # fileHeadle1_list = ('b0_100',
                #                     'b0_010',
                #                     'b05_100',
                #                     'b05_010',
                #                     'b05_001')
                # length_list = (50, 60, 70)
                # fileHeadle2_list = ['_l%d' % l for l in length_list]
                # fileHeadle3_list = ('_rs_e025',
                #                     '_rs_e03',
                #                     '_lg_e1',
                #                     '_lg_e3',
                #                     '_lg_e6',
                #                     '_pf_e1',
                #                     '_pf_e1.5',
                #                     '_pf_e2',)
                # for i0 in range(len(fileHeadle1_list)):
                #     fileHeadle1 = fileHeadle1_list[i0]
                #     for i1 in range(len(fileHeadle2_list)):
                #         fileHeadle2 = fileHeadle2_list[i1]
                #         for i2 in range(len(fileHeadle3_list)):
                #             fileHeadle3 = fileHeadle3_list[i2]
                #             fileHandle = fileHeadle1 + fileHeadle2 + fileHeadle3
                #             PETSc.Sys.Print(fileHandle)


if __name__ == '__main__':
    # casebank()
    main_fun()

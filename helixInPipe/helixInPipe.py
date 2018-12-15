# coding=utf-8
# 1. generate velocity and force nodes of sphere using MATLAB,
# 2. for each force node, get b, solve surrounding velocity boundary condition (pipe and cover, named boundary velocity) using formula from Liron's paper, save .mat file
# 3. read .mat file, for each boundary velocity, solve associated boundary force.
# 4. solve sphere M matrix using boundary force.
# 5. solve problem and check.

import sys

import petsc4py

petsc4py.init(sys.argv)

import numpy as np
# from scipy.io import loadmat
from src import stokes_flow as sf
from src.stokes_flow import problem_dic, obj_dic
from petsc4py import PETSc
from src.geo import *
from time import time
import pickle


def save_vtk(problem: sf.StokesFlowProblem):
    t0 = time()
    comm = PETSc.COMM_WORLD.tompi4py()
    rank = comm.Get_rank()
    problem_kwargs = problem.get_kwargs()
    fileHandle = problem_kwargs['fileHandle']
    nth = problem_kwargs['nth']
    nh = problem_kwargs['nh']
    ch = problem_kwargs['ch']
    rh1 = problem_kwargs['rh1']
    rh2 = problem_kwargs['rh2']
    ph = problem_kwargs['ph']
    rU = problem_kwargs['rU']
    n_helix_check = problem_kwargs['n_helix_check']
    velocity_err = 0

    # bgeo = geo()
    # bnodesHeadle = problem_kwargs['bnodesHeadle']
    # matname = problem_kwargs['matname']
    # bgeo.mat_nodes(filename=matname, mat_handle=bnodesHeadle)
    # belemsHeadle = problem_kwargs['belemsHeadle']
    # bgeo.mat_elmes(filename=matname, mat_handle=belemsHeadle, elemtype='tetra')
    # problem.vtk_tetra(fileHandle + '_Velocity', bgeo)

    problem.vtk_obj(fileHandle)

    helix_geo_check = supHelix()  # force geo
    # dth = 2 * np.pi / nth * 0.7
    # dth = 2 * np.pi * 0.061 * np.sqrt(ch)
    dth = np.max((2 * np.pi / nth * 0.8, 2 * np.pi * 0.061 * np.sqrt(ch)))
    B = ph / (2 * np.pi)
    helix_geo_check.create_deltatheta(dth=dth, radius=rh2, R=rh1, B=B, n_c=ch, epsilon=0, with_cover=1)
    helix_check = sf.StokesFlowObj()
    helix_check.set_data(helix_geo_check, helix_geo_check)

    ang_helix = 2 * np.pi / nh  # the angle of two nearest helixes.
    norm = np.array((0, 0, 1))
    for i0 in range(nh):
        t_obj = helix_check.copy()
        theta = i0 * ang_helix
        t_obj.node_rotation(norm=norm, theta=theta)
        t_obj.set_velocity(np.ones_like(t_obj.get_u_nodes()))
        t_obj.set_rigid_velocity(np.array((0, 0, 0, 0, 0, rU)))
        t_obj.set_name('helix_Check_%d' % i0)
        velocity_err = velocity_err + problem.vtk_check('%s_Check_%d' % (fileHandle, i0), t_obj)[0]
    velocity_err = velocity_err / nh

    t1 = time()
    PETSc.Sys.Print('velocity error is: %f' % velocity_err)
    PETSc.Sys.Print('%s: write vtk files use: %fs' % (str(problem), (t1 - t0)))

    return True


def get_problem_kwargs(**main_kwargs):
    OptDB = PETSc.Options()
    fileHandle = OptDB.getString('f', 'helixInPipe')
    prbHeadle = OptDB.getString('prbHeadle', 'construct07')
    nth = OptDB.getInt('nth', 2)  # amount of helix nodes
    nh = OptDB.getInt('nh', 1)  # total of helixes
    hfct = OptDB.getReal('hfct', 1)  # helix axis line factor, put more nodes near both tops
    eh = OptDB.getReal('eh', -0.5)  # epsilon of helix
    ch = OptDB.getReal('ch', 0.1)  # cycles of helix
    rh1 = OptDB.getReal('rh1', 0.6)  # radius of helix
    rh2 = OptDB.getReal('rh2', 0.1)  # radius of helix
    ph = OptDB.getReal('ph', 0.2 * np.pi)  # helix pitch
    rU = OptDB.getReal('rU', 1)  # rotation velocity
    matname = OptDB.getString('mat', 'body1')
    bnodesHeadle = OptDB.getString('bnodes', 'bnodes')  # body nodes, for vtu output
    belemsHeadle = OptDB.getString('belems', 'belems')  # body tetrahedron mesh, for vtu output
    solve_method = OptDB.getString('s', 'gmres')
    precondition_method = OptDB.getString('g', 'none')
    plot = OptDB.getBool('plot', False)
    matrix_method = OptDB.getString('sm', 'pf_stokesletsInPipe')
    restart = OptDB.getBool('restart', False)
    twoPara_n = OptDB.getInt('tp_n', 1)
    legendre_m = OptDB.getInt('legendre_m', 3)
    legendre_k = OptDB.getInt('legendre_k', 2)
    n_helix_check = OptDB.getInt('n_helix_check', 2000)
    n_node_threshold = OptDB.getInt('n_threshold', 10000)
    getConvergenceHistory = OptDB.getBool('getConvergenceHistory', False)
    pickProblem = OptDB.getBool('pickProblem', False)
    plot_geo = OptDB.getBool('plot_geo', False)

    if prbHeadle[-9:] != '_pick.bin':
        prbHeadle = prbHeadle + '_pick.bin'

    problem_kwargs = {
        'name':                  'helixInPipe',
        'matrix_method':         matrix_method,
        'nth':                   nth,
        'nh':                    nh,
        'hfct':                  hfct,
        'eh':                    eh,
        'ch':                    ch,
        'rh1':                   rh1,
        'rh2':                   rh2,
        'ph':                    ph,
        'rU':                    rU,
        'matname':               matname,
        'bnodesHeadle':          bnodesHeadle,
        'belemsHeadle':          belemsHeadle,
        'solve_method':          solve_method,
        'precondition_method':   precondition_method,
        'plot':                  plot,
        'fileHandle':            fileHandle,
        'prbHeadle':             prbHeadle,
        'twoPara_n':             twoPara_n,
        'legendre_m':            legendre_m,
        'legendre_k':            legendre_k,
        'restart':               restart,
        'n_helix_check':         n_helix_check,
        'n_node_threshold':      n_node_threshold,
        'getConvergenceHistory': getConvergenceHistory,
        'pickProblem':           pickProblem,
        'plot_geo':              plot_geo,
    }

    for key in main_kwargs:
        problem_kwargs[key] = main_kwargs[key]
    return problem_kwargs


def print_case_info(**problem_kwargs):
    comm = PETSc.COMM_WORLD.tompi4py()
    rank = comm.Get_rank()
    size = comm.Get_size()

    fileHandle = problem_kwargs['fileHandle']
    prbHeadle = problem_kwargs['prbHeadle']
    matrix_method = problem_kwargs['matrix_method']
    nth = problem_kwargs['nth']
    nh = problem_kwargs['nh']
    hfct = problem_kwargs['hfct']
    eh = problem_kwargs['eh']
    ch = problem_kwargs['ch']
    rh1 = problem_kwargs['rh1']
    rh2 = problem_kwargs['rh2']
    ph = problem_kwargs['ph']

    if rank == 0:
        PETSc.Sys.Print('Case information: ')
        # PETSc.Sys.Print('  pipe length: %f, pipe radius: %f' % (lp, rp))
        PETSc.Sys.Print('  helix radius: %f and %f, helix pitch: %f, helix cycle: %f' % (rh1, rh2, ph, ch))
        PETSc.Sys.Print('  nth, nh, hfct and epsilon of helix are %d, %d, %f and %f, ' % (nth, nh, hfct, eh))

        err_msg = "Only 'rs_stokesletsInPipe', 'tp_rs_stokesletsInPipe', 'lg_rs_stokesletsInPipe', and 'pf_stokesletsInPipe' methods are accept for this main code. "
        acceptType = ('rs_stokesletsInPipe', 'tp_rs_stokesletsInPipe', 'lg_rs_stokesletsInPipe', 'pf_stokesletsInPipe')
        assert matrix_method in acceptType, err_msg
        if matrix_method in 'rs_stokesletsInPipe':
            PETSc.Sys.Print('  create matrix method: %s, ' % matrix_method)
        elif matrix_method in 'tp_rs_stokesletsInPipe':
            twoPara_n = problem_kwargs['twoPara_n']
            PETSc.Sys.Print('  create matrix method: %s, order: %d'
                            % (matrix_method, twoPara_n))
        elif matrix_method in 'lg_rs_stokesletsInPipe':
            legendre_m = problem_kwargs['legendre_m']
            legendre_k = problem_kwargs['legendre_k']
            PETSc.Sys.Print('  create matrix method: %s, m: %d, k: %d, p: %d'
                            % (matrix_method, legendre_m, legendre_k, (legendre_m + 2 * legendre_k + 1)))
        elif matrix_method in 'pf_stokesletsInPipe':
            PETSc.Sys.Print('  create matrix method: %s ' % matrix_method)
        else:
            raise Exception('set how to print matrix method please. ')

        solve_method = problem_kwargs['solve_method']
        precondition_method = problem_kwargs['precondition_method']
        PETSc.Sys.Print('  solve method: %s, precondition method: %s'
                        % (solve_method, precondition_method))
        t_headle = '_pick.bin'
        prbHeadle = prbHeadle if prbHeadle[-len(t_headle):] == t_headle else prbHeadle + t_headle
        PETSc.Sys.Print('  read problem from: ' + prbHeadle)
        PETSc.Sys.Print('  output file headle: ' + fileHandle)
        PETSc.Sys.Print('MPI size: %d' % size)


# @profile
def main_fun(**main_kwargs):
    comm = PETSc.COMM_WORLD.tompi4py()
    rank = comm.Get_rank()
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    fileHandle = problem_kwargs['fileHandle']
    prbHeadle = problem_kwargs['prbHeadle']

    if not problem_kwargs['restart']:
        print_case_info(**problem_kwargs)
        matrix_method = problem_kwargs['matrix_method']

        # create helix
        nth = problem_kwargs['nth']
        nh = problem_kwargs['nh']
        hfct = problem_kwargs['hfct']
        eh = problem_kwargs['eh']
        ch = problem_kwargs['ch']
        rh1 = problem_kwargs['rh1']
        rh2 = problem_kwargs['rh2']
        ph = problem_kwargs['ph']
        rU = problem_kwargs['rU']
        B = ph / (2 * np.pi)
        vhgeo = supHelix()  # velocity node geo of helix
        dth = 2 * np.pi / nth
        fhgeo = vhgeo.create_deltatheta(dth=dth, radius=rh2, R=rh1, B=B, n_c=ch, epsilon=eh, with_cover=1, factor=hfct)
        # vhgeo.show_nodes()
        # vhgeo.show_velocity(length_factor=0.01)

        vhobj = obj_dic[matrix_method]()
        vhobj_kwargs = {'name': 'helix_0', }
        vhobj.set_data(fhgeo, vhgeo, **vhobj_kwargs)

        # load problem, solved force at (or outside) the pipe prepared.
        t_headle = '_pick.bin'
        prbHeadle = prbHeadle if prbHeadle[-len(t_headle):] == t_headle else prbHeadle + t_headle
        with open(prbHeadle, 'rb') as input:
            unpick = pickle.Unpickler(input)
            problem = unpick.load()
            problem.unpickmyself()
        problem.set_kwargs(**problem_kwargs)

        ang_helix = 2 * np.pi / nh  # the angle of two nearest helixes.
        norm = np.array((0, 0, 1))
        for i0 in range(nh):
            t_obj = vhobj.copy()
            theta = i0 * ang_helix
            t_obj.node_rotation(norm=norm, theta=theta)
            t_obj.set_velocity(np.ones_like(t_obj.get_u_nodes()))
            t_obj.set_rigid_velocity(np.array((0, 0, 0, 0, 0, rU)))
            t_obj.set_name('helix_%d' % i0)
            problem.add_obj(t_obj)

        problem.print_info()
        if problem_kwargs['plot_geo']:
            problem.show_f_u_nodes()
            problem.show_velocity(length_factor=0.001)
        problem.create_matrix()
        residualNorm = problem.solve()
        # # debug
        # problem.saveM_ASCII('%s_M.txt' % fileHandle)

        if problem_kwargs['pickProblem']:
            problem.pickmyself(fileHandle)
        force_helix = vhobj.get_force_z()
        PETSc.Sys.Print('---->>>Resultant at z axis is %f' % (np.sum(force_helix) / (6 * np.pi * rh1)))
        save_vtk(problem)
    else:
        t_headle = '_pick.bin'
        fileHandle = fileHandle if fileHandle[-len(t_headle):] == fileHandle else fileHandle + t_headle
        with open(fileHandle, 'rb') as input:
            unpick = pickle.Unpickler(input)
            problem = unpick.load()
            problem.unpickmyself()

        residualNorm = problem.get_residualNorm()
        problem_kwargs1 = get_problem_kwargs(**main_kwargs)
        problem_kwargs = problem.get_kwargs()
        problem_kwargs['matname'] = problem_kwargs1['matname']
        problem_kwargs['bnodesHeadle'] = problem_kwargs1['bnodesHeadle']
        problem_kwargs['belemsHeadle'] = problem_kwargs1['belemsHeadle']
        problem.set_kwargs(**problem_kwargs)
        print_case_info(**problem_kwargs)
        problem.print_info()
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


if __name__ == '__main__':
    main_fun()

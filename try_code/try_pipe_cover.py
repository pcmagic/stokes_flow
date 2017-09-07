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
from src import stokes_flow as sf
from src.stokes_flow import problem_dic, obj_dic
from petsc4py import PETSc
from src.geo import *
from time import time
from scipy.io import loadmat
import pickle


def save_vtk(problem: sf.stokesFlowProblem):
    t0 = time()
    comm = PETSc.COMM_WORLD.tompi4py()
    rank = comm.Get_rank()
    problem_kwargs = problem.get_kwargs()
    fileHeadle = problem_kwargs['fileHeadle']
    rs = problem_kwargs['rs']
    U = problem_kwargs['U']
    n_sphere_check = problem_kwargs['n_sphere_check']
    n_tunnel_check = problem_kwargs['n_tunnel_check']
    lp = problem_kwargs['lp']
    rp = problem_kwargs['rp']

    # bgeo = geo()
    # bnodesHeadle = problem_kwargs['bnodesHeadle']
    # matname = problem_kwargs['matname']
    # bgeo.mat_nodes(filename=matname, mat_handle=bnodesHeadle)
    # belemsHeadle = problem_kwargs['belemsHeadle']
    # bgeo.mat_elmes(filename=matname, mat_handle=belemsHeadle, elemtype='tetra')
    # problem.vtk_tetra(fileHeadle + '_Velocity', bgeo)

    # velocity_err = -1
    obj_check = sf.stokesFlowObj()
    tunnel_geo_check = tunnel_geo()  # pf, force geo
    tunnel_geo_check.create_n(n_tunnel_check, lp/3*2, rp)
    tunnel_geo_check.set_rigid_velocity(np.array((0, 0, 0, 0, 0, 0)))
    obj_check.set_data(tunnel_geo_check, tunnel_geo_check)
    velocity_err = problem.vtk_check(fileHeadle + '_Check_tunnel', obj_check)[0]

    t1 = time()
    PETSc.Sys.Print('velocity error is: %f' % velocity_err)
    PETSc.Sys.Print('%s: write vtk files use: %fs' % (str(problem), (t1 - t0)))
    # PETSc.Sys.Print('err_cf1=%f' % err_cf1)
    # PETSc.Sys.Print('err_cf2=%f' % err_cf2)
    # PETSc.Sys.Print('err_cf3=%f' % err_cf3)

    return True


def get_problem_kwargs(**main_kwargs):
    OptDB = PETSc.Options()
    fileHeadle = OptDB.getString('f', 'sphereInPipe')
    dp = OptDB.getReal('dp', 0.05)  # delta length of pipe
    ds = OptDB.getReal('ds', 0.1)  # delta length of sphere
    ep = OptDB.getReal('ep', 0.5)  # epsilon of pipe
    es = OptDB.getReal('es', -2)  # epsilon of shpere
    lp = OptDB.getReal('lp', 2)  # length of pipe
    rs = OptDB.getReal('rs', 0.5)  # radius of sphere
    th = OptDB.getInt('th', 30)  # threshold
    U = OptDB.getReal('U', 1)  # sphere velocity
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
    n_sphere_check = OptDB.getInt('n_sphere_check', 2000)
    n_tunnel_check = OptDB.getInt('n_tunnel_check', 10000)
    n_node_threshold = OptDB.getInt('n_threshold', 10000)
    getConvergenceHistory = OptDB.getBool('getConvergenceHistory', False)
    pickProblem = OptDB.getBool('pickProblem', False)

    problem_kwargs = {
        'name':                  'sphereInPipe',
        'matrix_method':         matrix_method,
        'dp':                    dp,
        'ds':                    ds,
        'ep':                    ep,
        'es':                    es,
        'lp':                    lp,
        'rp':                    1,
        'rs':                    rs,
        'th':                    th,
        'U':                     U,
        'matname':               matname,
        'bnodesHeadle':          bnodesHeadle,
        'belemsHeadle':          belemsHeadle,
        'solve_method':          solve_method,
        'precondition_method':   precondition_method,
        'plot':                  plot,
        'fileHeadle':            fileHeadle,
        'twoPara_n':             twoPara_n,
        'legendre_m':            legendre_m,
        'legendre_k':            legendre_k,
        'restart':               restart,
        'n_sphere_check':        n_sphere_check,
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

    fileHeadle = problem_kwargs['fileHeadle']
    matrix_method = problem_kwargs['matrix_method']
    dp = problem_kwargs['dp']
    ds = problem_kwargs['ds']
    rp = problem_kwargs['rp']
    rs = problem_kwargs['rs']
    lp = problem_kwargs['lp']
    th = problem_kwargs['th']
    ep = problem_kwargs['ep']
    es = problem_kwargs['es']

    PETSc.Sys.Print('Case information: ')
    PETSc.Sys.Print('  pipe length: %f, pipe radius: %f, sphere radius: %f' % (lp, rp, rs))
    PETSc.Sys.Print('  delta length of pipe and sphere are %f and %f, respectively. ' % (dp, ds))
    PETSc.Sys.Print('  threshold of seriers is %d' % th)
    PETSc.Sys.Print('  epsilon of pipe and sphere are %f and %f, respectively. ' % (ep, es))

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
    PETSc.Sys.Print('  output file headle: ' + fileHeadle)
    PETSc.Sys.Print('MPI size: %d' % size)


# @profile
def main_fun(**main_kwargs):
    comm = PETSc.COMM_WORLD.tompi4py()
    rank = comm.Get_rank()
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    fileHeadle = problem_kwargs['fileHeadle']

    if not problem_kwargs['restart']:
        print_case_info(**problem_kwargs)
        matrix_method = problem_kwargs['matrix_method']

        # create sphere
        ds = problem_kwargs['ds']
        rs = problem_kwargs['rs']
        es = problem_kwargs['es']
        vsgeo = sphere_geo()  # velocity node geo of sphere
        vsgeo.create_delta(ds, rs)
        vsgeo.node_rotation(norm=np.array((0,1,0)), theta=np.pi/2)
        # vsgeo.show_nodes()
        U = problem_kwargs['U']
        vsgeo.set_rigid_velocity(np.array((0, 0, U, 0, 0, 0)))
        fsgeo = vsgeo.copy()  # force node geo of sphere
        fsgeo.node_zoom(1 + ds * es)

        # debug
        dbg_nodes = np.array((0.81, 0, 0), order='F').reshape((-1, 3))
        vsgeo.set_nodes(dbg_nodes, deltalength=0)
        vsgeo.set_rigid_velocity(np.array((0, 0, U, 0, 0, 0)))
        dbg_nodes = np.array((0.8, 0, 0), order='F').reshape((-1, 3))
        fsgeo.set_nodes(dbg_nodes, deltalength=0)

        vsobj = obj_dic[matrix_method]()
        vsobj_kwargs = {'name': 'sphere_0', }
        vsobj.set_data(fsgeo, vsgeo, **vsobj_kwargs)

        # create problem
        problem = problem_dic[matrix_method](**problem_kwargs)
        problem.add_obj(vsobj)
        problem.solve_prepare(**problem_kwargs)
        if problem_kwargs['pickProblem']:
            # do NOT save anything really, just check if the path is correct, to avoid this error after long time calculation.
            problem.pickmyself(fileHeadle, check=True)

        problem.create_matrix()
        Tolerances = {'max_it': 3000}
        residualNorm = problem.solve(Tolerances=Tolerances)
        save_vtk(problem)

        if problem_kwargs['pickProblem']:
            problem.pickmyself(fileHeadle)
    else:
        with open(fileHeadle + '_pick.bin', 'rb') as input:
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

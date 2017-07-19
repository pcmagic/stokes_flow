# coding=utf-8
# main codes, call functions at stokes_flow.py
# Liron, N., and R. Shahar. "Stokes flow due to a Stokeslet in a pipe." Journal of Fluid Mechanics 86.04 (1978): 727-744.
# the convergence of the series solution for the problem is bad around the stokeslet, i.e. z->0.
# thus, a numerical solution is given to fill this flaw.
# boundary conditions at the pip are no-split.
# inlet and outlet flow within the pip are given by the series solution, whose converge is fast when z is not small.
# Zhang Ji, 20170410

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
    matname = problem_kwargs['matname']
    obj_check = sf.stokesFlowObj()
    geo_check = geo()

    problem.vtk_obj(fileHeadle)

    ucbHeadle = problem_kwargs['ucbHeadle']
    cbnodesHeadle = problem_kwargs['cbnodesHeadle']
    geo_check.mat_nodes(filename=matname, mat_handle=cbnodesHeadle)
    geo_check.mat_velocity(filename=matname, mat_handle=ucbHeadle)
    obj_check.set_data(geo_check, geo_check)
    err_cb = problem.vtk_check(fileHeadle + '_cbnodes', obj_check)[0]

    # ucf1Headle = problem_kwargs['ucf1Headle']
    # cf1nodesHeadle = problem_kwargs['cf1nodesHeadle']
    # geo_check.mat_nodes(filename=matname, mat_handle=cf1nodesHeadle)
    # geo_check.mat_velocity(filename=matname, mat_handle=ucf1Headle)
    # obj_check.set_data(geo_check, geo_check)
    # err_cf1 = problem.vtk_check(fileHeadle + '_cf1nodes', obj_check)[0]
    #
    # ucf2Headle = problem_kwargs['ucf2Headle']
    # cf2nodesHeadle = problem_kwargs['cf2nodesHeadle']
    # geo_check.mat_nodes(filename=matname, mat_handle=cf2nodesHeadle)
    # geo_check.mat_velocity(filename=matname, mat_handle=ucf2Headle)
    # obj_check.set_data(geo_check, geo_check)
    # err_cf2 = problem.vtk_check(fileHeadle + '_cf2nodes', obj_check)[0]
    #
    # ucf3Headle = problem_kwargs['ucf3Headle']
    # cf3nodesHeadle = problem_kwargs['cf3nodesHeadle']
    # geo_check.mat_nodes(filename=matname, mat_handle=cf3nodesHeadle)
    # geo_check.mat_velocity(filename=matname, mat_handle=ucf3Headle)
    # obj_check.set_data(geo_check, geo_check)
    # err_cf3 = problem.vtk_check(fileHeadle + '_cf3nodes', obj_check)[0]

    bgeo = geo()
    bnodesHeadle = problem_kwargs['bnodesHeadle']
    bgeo.mat_nodes(filename=matname, mat_handle=bnodesHeadle)
    belemsHeadle = problem_kwargs['belemsHeadle']
    bgeo.mat_elmes(filename=matname, mat_handle=belemsHeadle, elemtype='tetra')
    problem.vtk_tetra(fileHeadle + '_Velocity', bgeo)

    # Todo wrapper print: print0 (print at rank_0).
    t1 = time()
    PETSc.Sys.Print('%s: write vtk files use: %fs' % (str(problem), (t1 - t0)))
    PETSc.Sys.Print('err_cb=%f' % err_cb)
    # PETSc.Sys.Print('err_cf1=%f' % err_cf1)
    # PETSc.Sys.Print('err_cf2=%f' % err_cf2)
    # PETSc.Sys.Print('err_cf3=%f' % err_cf3)

    return True


def get_problem_kwargs(**main_kwargs):
    OptDB = PETSc.Options()
    matname = OptDB.getString('mat', 'body1')
    fileHeadle = OptDB.getString('f', 'stokeletInPipe')
    fnodesHeadle = OptDB.getString('fnodes', 'fnodes')  # force nodes, for solver
    vnodesHeadle = OptDB.getString('vnodes', 'vnodes')  # velocity nodes, for solver
    cbnodesHeadle = OptDB.getString('cbnodes', 'cbnodes')  # check nodes, for solver
    cf1nodesHeadle = OptDB.getString('cf1nodes', 'cf1nodes')  # check nodes, for solver
    cf2nodesHeadle = OptDB.getString('cf2nodes', 'cf2nodes')  # check nodes, for solver
    cf3nodesHeadle = OptDB.getString('cf3nodes', 'cf3nodes')  # check nodes, for solver
    uHeadle = OptDB.getString('U', 'U11')  # boundary condition at velocity nodes, for solver
    ucbHeadle = OptDB.getString('Ucb', 'Ucb11')  # boundary condition at velocity nodes, for solver
    ucf1Headle = OptDB.getString('Ucf1', 'Ucf111')  # check velocity file
    ucf2Headle = OptDB.getString('Ucf2', 'Ucf211')  # check velocity file
    ucf3Headle = OptDB.getString('Ucf3', 'Ucf311')  # check velocity file
    bnodesHeadle = OptDB.getString('bnodes', 'bnodes')  # body nodes, for vtu output
    belemsHeadle = OptDB.getString('belems', 'belems')  # body tetrahedron mesh, for vtu output
    bHeadle = OptDB.getString('stokeslets_b', 'b')
    fHeadle = OptDB.getString('stokeslets_f', 'f1')
    dsHeadle = OptDB.getString('ds', 'ds')
    epsilon = OptDB.getReal('e', 1)  # for rs method family
    epsilonHeadle = OptDB.getReal('eHeadle', 'epsilon')  # for pf method family
    solve_method = OptDB.getString('s', 'gmres')
    precondition_method = OptDB.getString('g', 'none')
    plot = OptDB.getBool('plot', False)
    matrix_method = OptDB.getString('sm', 'rs_stokeslets')
    restart = OptDB.getBool('restart', False)
    twoPara_n = OptDB.getInt('tp_n', 1)
    legendre_m = OptDB.getInt('legendre_m', 3)
    legendre_k = OptDB.getInt('legendre_k', 2)
    n_tunnel_check = OptDB.getInt('n_tunnel_check', 30000)
    n_node_threshold = OptDB.getInt('n_threshold', 10000)
    getConvergenceHistory = OptDB.getBool('getConvergenceHistory', False)
    pickProblem = OptDB.getBool('pickProblem', False)

    if matname[-4:] != '.mat':
        matname = matname + '.mat'
    mat_contents = loadmat(matname)
    deltaLength = mat_contents[dsHeadle][0]
    stokeslets_b = mat_contents[bHeadle][0]
    stokeslets_post = np.hstack((stokeslets_b, 0, 0))
    stokeslets_f = mat_contents[fHeadle].reshape((3,))

    problem_kwargs = {
        'name':                  'problem reconstruction',
        'matname':               matname,
        'matrix_method':         matrix_method,
        'deltaLength':           deltaLength,
        'epsilon':               epsilon,
        'delta':                 deltaLength * epsilon,  # for rs method
        'd_radia':               deltaLength / 2,  # for sf method
        'fnodesHeadle':          fnodesHeadle,
        'vnodesHeadle':          vnodesHeadle,
        'cbnodesHeadle':         cbnodesHeadle,
        'cf1nodesHeadle':        cf1nodesHeadle,
        'cf2nodesHeadle':        cf2nodesHeadle,
        'cf3nodesHeadle':        cf3nodesHeadle,
        'uHeadle':               uHeadle,
        'ucbHeadle':             ucbHeadle,
        'ucf1Headle':            ucf1Headle,
        'ucf2Headle':            ucf2Headle,
        'ucf3Headle':            ucf3Headle,
        'bnodesHeadle':          bnodesHeadle,
        'belemsHeadle':          belemsHeadle,
        'epsilonHeadle':         epsilonHeadle,
        'solve_method':          solve_method,
        'precondition_method':   precondition_method,
        'plot':                  plot,
        'stokeslets_f':          stokeslets_f,
        'stokeslets_b':          stokeslets_b,
        'stokeslets_post':       stokeslets_post,
        'fileHeadle':            fileHeadle,
        'twoPara_n':             twoPara_n,
        'legendre_m':            legendre_m,
        'legendre_k':            legendre_k,
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

    fileHeadle = problem_kwargs['fileHeadle']
    fnodesHeadle = problem_kwargs['fnodesHeadle']
    vnodesHeadle = problem_kwargs['vnodesHeadle']
    cbnodesHeadle = problem_kwargs['cbnodesHeadle']
    uHeadle = problem_kwargs['uHeadle']
    bnodesHeadle = problem_kwargs['bnodesHeadle']
    belemsHeadle = problem_kwargs['belemsHeadle']
    matrix_method = problem_kwargs['matrix_method']
    matname = problem_kwargs['matname']

    if rank == 0:
        PETSc.Sys.Print('read information from %s. ' % matname)
        PETSc.Sys.Print('  read force node locations from variable %s.' % fnodesHeadle)
        PETSc.Sys.Print('  read velocity node locations and figures from variables %s and %s, respectively. ' % (vnodesHeadle, uHeadle))
        PETSc.Sys.Print('  read check node locations from variable %s.' % cbnodesHeadle)
        PETSc.Sys.Print('  read body node locations from variable %s. ' % bnodesHeadle)
        PETSc.Sys.Print('  read body tetrahedron mesh from variable %s. ' % belemsHeadle)

        err_msg = "Only 'rs_stokeslets', 'tp_rs_stokeslets', 'lg_rs_stokeslets', and 'ps_stokeslets' methods are accept for this main code. "
        acceptType = ('rs_stokeslets', 'tp_rs_stokeslets', 'lg_rs_stokeslets', 'rs_stokeslets_precondition',
                      'tp_rs_stokeslets_precondition', 'lg_rs_stokeslets_precondition',
                      'pf_stokeslets')
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
        elif matrix_method in 'pf_stokeslets':
            epsilonHeadle = problem_kwargs['epsilonHeadle']
            mat_contents = loadmat(matname)
            epsilon = mat_contents[epsilonHeadle][0]
            PETSc.Sys.Print('create matrix method: %s, epsilon: %f' % (matrix_method, epsilon))
        else:
            raise Exception('set how to print matrix method please. ')

        solve_method = problem_kwargs['solve_method']
        precondition_method = problem_kwargs['precondition_method']
        PETSc.Sys.Print('solve method: %s, precondition method: %s'
              % (solve_method, precondition_method))
        PETSc.Sys.Print('output file headle: ' + fileHeadle)
        PETSc.Sys.Print('MPI size: %d' % size)


# @profile
def main_fun(**main_kwargs):
    comm = PETSc.COMM_WORLD.tompi4py()
    rank = comm.Get_rank()
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    fileHeadle = problem_kwargs['fileHeadle']
    stokeslets_post = problem_kwargs['stokeslets_post']
    stokeslets_f = problem_kwargs['stokeslets_f']

    if not problem_kwargs['restart']:
        print_case_info(**problem_kwargs)
        matrix_method = problem_kwargs['matrix_method']

        problem = problem_dic[matrix_method](**problem_kwargs)
        if problem_kwargs['pickProblem']:
            # do NOT save anything really, just check if the path is correct, to avoid this error after long time calculation.
            problem.pickmyself(fileHeadle, check=True)

        # create problem
        matname = problem_kwargs['matname']
        fgeo = geo()
        fnodesHeadle = problem_kwargs['fnodesHeadle']
        fgeo.mat_nodes(filename=matname, mat_handle=fnodesHeadle)
        vgeo = geo()
        vnodesHeadle = problem_kwargs['vnodesHeadle']
        vgeo.mat_nodes(filename=matname, mat_handle=vnodesHeadle)
        uHeadle = problem_kwargs['uHeadle']
        vgeo.mat_velocity(filename=matname, mat_handle=uHeadle)

        err_msg = 'shape of variables contain force and velocity nodes are not same. '
        assert fgeo.get_n_nodes() == vgeo.get_n_nodes(), err_msg

        obj = obj_dic[matrix_method]()
        obj_kwargs = {'name':            'obj1',
                      'stokeslets_f':    stokeslets_f,
                      'stokeslets_post': stokeslets_post}
        obj.set_data(fgeo, vgeo, **obj_kwargs)
        problem.add_obj(obj)
        problem.print_info()
        problem.create_matrix()

        residualNorm = problem.solve()
        save_vtk(problem)
        if problem_kwargs['pickProblem']:
            problem.pickmyself(fileHeadle)
    else:
        with open(fileHeadle + '_pick.bin', 'rb') as input:
            unpick = pickle.Unpickler(input)
            problem = unpick.load()
            problem.unpickmyself()
            residualNorm = problem.get_residualNorm()
            PETSc.Sys.Print('---->>>unpick the problem from file %s_pick.bin' % (fileHeadle))

            problem_kwargs1 = get_problem_kwargs(**main_kwargs)
            problem_kwargs = problem.get_kwargs()
            problem_kwargs['ucbHeadle'] = problem_kwargs1['ucbHeadle']
            problem_kwargs['cbnodesHeadle'] = problem_kwargs1['cbnodesHeadle']
            problem_kwargs['ucf1Headle'] = problem_kwargs1['ucf1Headle']
            problem_kwargs['cf1nodesHeadle'] = problem_kwargs1['cf1nodesHeadle']
            problem_kwargs['ucf2Headle'] = problem_kwargs1['ucf2Headle']
            problem_kwargs['cf2nodesHeadle'] = problem_kwargs1['cf2nodesHeadle']
            problem_kwargs['ucf3Headle'] = problem_kwargs1['ucf3Headle']
            problem_kwargs['cf3nodesHeadle'] = problem_kwargs1['cf3nodesHeadle']
            problem_kwargs['bnodesHeadle'] = problem_kwargs1['bnodesHeadle']
            problem_kwargs['belemsHeadle'] = problem_kwargs1['belemsHeadle']
            problem.set_kwargs(**problem_kwargs)
            print_case_info(**problem_kwargs)
            problem.print_info()
            save_vtk(problem)

    return problem, residualNorm


if __name__ == '__main__':
    main_fun()

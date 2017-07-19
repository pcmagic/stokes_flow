# coding=utf-8
# 1. generate velocity and force nodes of sphere using MATLAB,
# 2. for each force node, get b, solve surrounding velocity boundary condition (pipe and cover, named boundary velocity) using formula from Liron's paper, save .mat file
# 3. read .mat file, for each boundary velocity, solve associated boundary force.
# 4. solve sphere M matrix using boundary force.
# 5. solve problem and check.

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
from src.stokes_flow import obj_dic
from petsc4py import PETSc
from src.geo import *
from time import time


def save_vtk(problem: sf.stokesFlowProblem):
    t0 = time()
    problem_kwargs = problem.get_kwargs()
    fileHeadle = problem_kwargs['fileHeadle']
    matrix_method = problem_kwargs['matrix_method']

    problem.vtk_obj(fileHeadle)
    problem.vtk_self(fileHeadle)


    # bgeo = geo()
    # bnodesHeadle = problem_kwargs['bnodesHeadle']
    # matname = problem_kwargs['matname']
    # bgeo.mat_nodes(filename=matname, mat_handle=bnodesHeadle)
    # belemsHeadle = problem_kwargs['belemsHeadle']
    # bgeo.mat_elmes(filename=matname, mat_handle=belemsHeadle, elemtype='tetra')
    # problem.vtk_tetra(fileHeadle + '_Velocity', bgeo)

    # create check obj
    check_kwargs = problem_kwargs.copy()
    check_kwargs['nth'] = problem_kwargs['nth'] + 1
    check_kwargs['ds'] = problem_kwargs['ds'] * 0.8
    check_kwargs['hfct'] = 1
    objtype = obj_dic[matrix_method]
    vsobj_check, vhobj0_check, vhobj1_check = createEcoli(objtype, **check_kwargs)
    # set boundary condition
    rel_Us = problem_kwargs['rel_Us']
    rel_Uh = problem_kwargs['rel_Uh']
    ecoli_comp = problem.get_obj_list()[0]
    vsobj_check.node_rotation(norm=(0, 1, 0), theta=np.pi / 2, rotation_origin=(0, 0, 0))
    vhobj0_check.node_rotation(norm=(0, 1, 0), theta=np.pi / 2, rotation_origin=(0, 0, 0))
    vhobj1_check.node_rotation(norm=(0, 1, 0), theta=np.pi / 2, rotation_origin=(0, 0, 0))
    ref_U = ecoli_comp.get_ref_U()
    vsobj_check.set_rigid_velocity(rel_Us + ref_U)
    vhobj0_check.set_rigid_velocity(rel_Uh + ref_U)
    vhobj1_check.set_rigid_velocity(rel_Uh + ref_U)
    # create ecoli
    ecoli_check_obj = sf.stokesFlowObj()
    ecoli_check_obj.combine([vsobj_check, vhobj0_check, vhobj1_check], set_re_u=True, set_force=True)
    # # dbg
    # # ecoli_check_obj.show_velocity(length_factor=0.0002)
    # show_obj = sf.stokesFlowObj()
    # import itertools
    # aaa = [item for item in itertools.chain(ecoli_comp.get_obj_list(), [ecoli_check_obj])]
    # show_obj.combine(aaa)
    # ecoli_comp.show_f_u_nodes(' ')
    # ecoli_check_obj.show_f_u_nodes(' ')
    # show_obj.show_f_u_nodes(' ')

    velocity_err = problem.vtk_check('%s_Check' % fileHeadle, ecoli_check_obj)[0]
    PETSc.Sys.Print('velocity error is: %f' % velocity_err)

    t1 = time()
    PETSc.Sys.Print('%s: write vtk files use: %fs' % (str(problem), (t1 - t0)))

    return True


def get_problem_kwargs(**main_kwargs):
    OptDB = PETSc.Options()
    fileHeadle = OptDB.getString('f', '...')
    nth = OptDB.getInt('nth', 8)  # amount of nodes on each cycle of helix
    hfct = OptDB.getReal('hfct', 1)  # helix axis line factor, put more nodes near both tops
    eh = OptDB.getReal('eh', -1)  # epsilon of helix
    ch = OptDB.getReal('ch', 1)  # cycles of helix
    rh1 = OptDB.getReal('rh1', 0.02)  # radius of helix
    rh2 = OptDB.getReal('rh2', 0.001)  # radius of helix
    ph = OptDB.getReal('ph', 0.3)  # helix pitch
    with_cover = OptDB.getBool('with_cover', True)
    left_hand = OptDB.getBool('left_hand', True)
    # rs = OptDB.getReal('rs', 0.5)  # radius of head
    rs1 = OptDB.getReal('rs1', 0.05)  # radius of head
    rs2 = OptDB.getReal('rs2', 0.05)  # radius of head
    ds = OptDB.getReal('ds', 0.1)  # delta length of sphere
    es = OptDB.getReal('es', -1)  # epsilon of shpere
    matname = OptDB.getString('mat', 'body1')
    bnodesHeadle = OptDB.getString('bnodes', 'bnodes')  # body nodes, for vtu output
    belemsHeadle = OptDB.getString('belems', 'belems')  # body tetrahedron mesh, for vtu output
    solve_method = OptDB.getString('s', 'gmres')
    precondition_method = OptDB.getString('g', 'none')
    plot = OptDB.getBool('plot', False)
    matrix_method = OptDB.getString('sm', 'pf')
    restart = OptDB.getBool('restart', False)
    twoPara_n = OptDB.getInt('tp_n', 1)
    legendre_m = OptDB.getInt('legendre_m', 3)
    legendre_k = OptDB.getInt('legendre_k', 2)
    n_helix_check = OptDB.getInt('n_helix_check', 2000)
    n_node_threshold = OptDB.getInt('n_threshold', 10000)
    getConvergenceHistory = OptDB.getBool('getConvergenceHistory', False)
    pickProblem = OptDB.getBool('pickProblem', False)
    plot_geo = OptDB.getBool('plot_geo', False)

    # rel_Usx = OptDB.getReal('rel_Usx', 0)
    # rel_Usy = OptDB.getReal('rel_Usy', 0)
    rel_Usz = OptDB.getReal('rel_Usz', 0)
    # rel_Uhx = OptDB.getReal('rel_Uhx', 0)
    # rel_Uhy = OptDB.getReal('rel_Uhy', 0)
    rel_Uhz = OptDB.getReal('rel_Uhz', 200)
    rel_Us = np.array((0, 0, 0, rel_Usz, 0, 0))  # relative omega of sphere
    rel_Uh = np.array((0, 0, 0, rel_Uhz, 0, 0))  # relative omega of helix
    dist_hs = OptDB.getReal('dist_hs', 0.2)  # distance between head and tail
    lh = ph * ch  # length of helix
    movesz = -rs2
    movehz = - dist_hs - lh / 2
    moves = np.array((0, 0, movesz))  # move distance of sphere
    moveh = np.array((0, 0, -movehz))  # move distance of helix
    centerx = OptDB.getReal('centerx', 0)
    centery = OptDB.getReal('centery', 0)
    centerz = OptDB.getReal('centerz', 0)
    center = np.array((centerx, centery, centerz))  # center of ecoli

    problem_kwargs = {
        'name':                  'try_singleEcoliPro',
        'matrix_method':         matrix_method,
        'nth':                   nth,
        'hfct':                  hfct,
        'eh':                    eh,
        'ch':                    ch,
        'rh1':                   rh1,
        'rh2':                   rh2,
        'ph':                    ph,
        'rs1':                   rs1,
        'rs2':                   rs2,
        'ds':                    ds,
        'es':                    es,
        'rel_Us':                rel_Us,
        'rel_Uh':                rel_Uh,
        'moves':                 moves,
        'moveh':                 moveh,
        'dist_hs':               dist_hs,
        'center':                center,
        'left_hand':             left_hand,
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
        'n_helix_check':         n_helix_check,
        'n_node_threshold':      n_node_threshold,
        'getConvergenceHistory': getConvergenceHistory,
        'pickProblem':           pickProblem,
        'plot_geo':              plot_geo,
        'with_cover':            with_cover,
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
    nth = problem_kwargs['nth']
    hfct = problem_kwargs['hfct']
    eh = problem_kwargs['eh']
    ch = problem_kwargs['ch']
    rh1 = problem_kwargs['rh1']
    rh2 = problem_kwargs['rh2']
    ph = problem_kwargs['ph']
    ds = problem_kwargs['ds']
    rs1 = problem_kwargs['rs1']
    rs2 = problem_kwargs['rs2']
    es = problem_kwargs['es']
    center = problem_kwargs['center']
    rel_Us = problem_kwargs['rel_Us']
    rel_Uh = problem_kwargs['rel_Uh']
    dist_hs = problem_kwargs['dist_hs']

    PETSc.Sys.Print('Case information: ')
    PETSc.Sys.Print('  helix radius: %f and %f, helix pitch: %f, helix cycle: %f' % (rh1, rh2, ph, ch))
    PETSc.Sys.Print('  nth, hfct and epsilon of helix are %d, %f and %f, ' % (nth, hfct, eh))
    PETSc.Sys.Print('  sphere/ellipse radius: %f and %f, delta length: %f, epsilon: %f' % (rs1, rs2, ds, es))
    PETSc.Sys.Print('  ecoli center: %s, distance between head and tail is %f' % (str(center), dist_hs))
    PETSc.Sys.Print('  relative velocity of head and tail are %s and %s' % (str(rel_Us), str(rel_Uh)))

    err_msg = "Only 'rs', 'tp_rs', 'lg_rs', and 'pf' methods are accept for this main code. "
    acceptType = ('rs', 'tp_rs', 'lg_rs', 'pf')
    assert matrix_method in acceptType, err_msg
    if matrix_method in 'rs':
        PETSc.Sys.Print('  create matrix method: %s, ' % matrix_method)
    elif matrix_method in 'tp_rs':
        twoPara_n = problem_kwargs['twoPara_n']
        PETSc.Sys.Print('  create matrix method: %s, order: %d'
                        % (matrix_method, twoPara_n))
    elif matrix_method in 'lg_rs':
        legendre_m = problem_kwargs['legendre_m']
        legendre_k = problem_kwargs['legendre_k']
        PETSc.Sys.Print('  create matrix method: %s, m: %d, k: %d, p: %d'
                        % (matrix_method, legendre_m, legendre_k, (legendre_m + 2 * legendre_k + 1)))
    elif matrix_method in 'pf':
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
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    matname = 'reduced10'
    problem_kwargs['fileHeadle'] = matname
    fileHeadle = problem_kwargs['fileHeadle']
    # print_case_info(**problem_kwargs)
    matrix_method = problem_kwargs['matrix_method']

    # create ecoli
    vsgeo = geo()
    vsgeo.mat_nodes(matname, 'vsgeo')
    fsgeo = geo()
    fsgeo.mat_nodes(matname, 'fsgeo')
    vhgeo = geo()
    vhgeo.mat_nodes(matname, 'vhgeo')
    fhgeo = geo()
    fhgeo.mat_nodes(matname, 'fhgeo')
    objtype = obj_dic[matrix_method]
    vsobj = objtype()
    vsobj.set_name('vsobj')
    vsobj.set_data(fsgeo, vsgeo)
    vhobj = objtype()
    vhobj.set_name('vhobj')
    vhobj.set_data(fhgeo, vhgeo)
    center = problem_kwargs['center']
    rel_Us = problem_kwargs['rel_Us']
    rel_Uh = problem_kwargs['rel_Uh']
    ecoli_comp = sf.forceFreeComposite(center, 'ecoli_0')
    ecoli_comp.add_obj(vsobj, rel_U=rel_Us)
    ecoli_comp.add_obj(vhobj, rel_U=rel_Uh)

    problem = sf.forceFreeProblem(**problem_kwargs)
    if problem_kwargs['pickProblem']:
        problem.pickmyself(fileHeadle, check=True)
    problem.add_obj(ecoli_comp)
    problem.print_info()
    if problem_kwargs['plot_geo']:
        vsobj.show_f_u_nodes(' ')
        vhobj.show_f_u_nodes(' ')
        ecoli_comp.show_f_u_nodes(' ')
    # save_vtk(problem)

    problem.create_matrix()
    residualNorm = problem.solve()
    PETSc.Sys.Print(ecoli_comp.get_ref_U())
    PETSc.Sys.Print(ecoli_comp.get_re_sum())
    # # debug
    # problem.saveM_ASCII('%s_M.txt' % fileHeadle)
    # dbg_M = problem.get_M()

    if problem_kwargs['pickProblem']:
        problem.pickmyself(fileHeadle)
    save_vtk(problem)

    force_helix = vhobj.get_force_z()
    rh1 = problem_kwargs['rh1']
    PETSc.Sys.Print('---->>>Resultant at z axis is %f' % (np.sum(force_helix) / (6 * np.pi * rh1)))

    return True


if __name__ == '__main__':
    main_fun()

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
from src.StokesFlowMethod import *
from src.stokes_flow import problem_dic, obj_dic
from petsc4py import PETSc
from src.geo import *
from scipy.io import loadmat, savemat
import pickle
import matplotlib.pyplot as plt
from src.stokesletsInPipe import detail
from src.support_class import *


def get_problem_kwargs(**main_kwargs):
    OptDB = PETSc.Options()
    fileHeadle = OptDB.getString('f', 'dbg')
    dp = OptDB.getReal('dp', 0.5)  # delta length of pipe
    ep = OptDB.getReal('ep', 2)  # epsilon of pipe
    lp = OptDB.getReal('lp', 1)  # length of pipe
    b0 = OptDB.getReal('b0', 1e-4)  # (b0, b1) the range of b, location of force
    b1 = OptDB.getReal('b1', 0.9)
    nb = OptDB.getInt('nb', 2)  # amount of b
    th = OptDB.getInt('th', 30)  # threshold
    stokesletsInPipe_pipeFactor = OptDB.getReal('stokesletsInPipe_pipeFactor',
                                                5)  # geometrical parameter, control the distribution of nodes of the pipe.
    solve_method = OptDB.getString('s', 'gmres')
    precondition_method = OptDB.getString('g', 'none')
    matrix_method = OptDB.getString('sm', 'pf_stokesletsInPipe')
    check_acc = OptDB.getBool('check_acc', False)  # Accuracy check
    plot_geo = OptDB.getBool('plot_geo', False)  # show pipe
    comm = PETSc.COMM_WORLD.tompi4py()
    MPISIZE = comm.Get_size()

    problem_kwargs = {
        'name':                        'force_pipe',
        'matrix_method':               matrix_method,
        'dp':                          dp,
        'ep':                          ep,
        'lp':                          lp,
        'rp':                          1,
        'b0':                          b0,
        'b1':                          b1,
        'nb':                          nb,
        'th':                          th,
        'stokesletsInPipe_pipeFactor': stokesletsInPipe_pipeFactor,
        'solve_method':                solve_method,
        'precondition_method':         precondition_method,
        'fileHeadle':                  fileHeadle,
        'check_acc':                   check_acc,
        'plot_geo':                    plot_geo,
        'MPISIZE':                     MPISIZE,
        'ffweightx':                   1,
        'ffweighty':                   1,
        'ffweightz':                   1,
        'ffweightT':                   1,
        'zoom_factor':                 1,
    }

    for key in main_kwargs:
        problem_kwargs[key] = main_kwargs[key]
    return problem_kwargs


def print_case_info(**problem_kwargs):
    comm = PETSc.COMM_WORLD.tompi4py()
    size = comm.Get_size()

    fileHeadle = problem_kwargs['fileHeadle']
    matrix_method = problem_kwargs['matrix_method']
    dp = problem_kwargs['dp']
    ep = problem_kwargs['ep']
    lp = problem_kwargs['lp']
    rp = problem_kwargs['rp']
    th = problem_kwargs['th']
    stokesletsInPipe_pipeFactor = problem_kwargs['stokesletsInPipe_pipeFactor']
    b0 = problem_kwargs['b0']
    b1 = problem_kwargs['b1']
    nb = problem_kwargs['nb']
    check_acc = problem_kwargs['check_acc']

    PETSc.Sys.Print('Case information: ')
    PETSc.Sys.Print('  pipe length: %f, pipe radius: %f' % (lp, rp))
    PETSc.Sys.Print(
            '  delta length, epsilon and factor of pipe are %f, %f and %f' % (dp, ep, stokesletsInPipe_pipeFactor))
    PETSc.Sys.Print('  threshold of series is %d' % th)
    PETSc.Sys.Print('  b: %d numbers are evenly distributed within the range [%f, %f]' % (nb, b0, b1))
    PETSc.Sys.Print('  check accuracy of forces: %s' % check_acc)

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
    print_case_info(**problem_kwargs)
    check_acc = problem_kwargs['check_acc']
    dp = problem_kwargs['dp']
    rp = problem_kwargs['rp']
    lp = problem_kwargs['lp']
    ep = problem_kwargs['ep']
    th = problem_kwargs['th']
    stokesletsInPipe_pipeFactor = problem_kwargs['stokesletsInPipe_pipeFactor']

    # # debug
    # vsgeo = geo()  # velocity node geo of sphere
    # dbg_nodes = np.array((0.5, 0, 0), order='F').reshape((-1, 3))
    # vsgeo.set_nodes(dbg_nodes, deltalength=0)
    # vsgeo.set_rigid_velocity(np.array((0, 0, 1, 0, 0, 0)))
    # fsgeo = geo()  # force node geo of sphere
    # dbg_nodes = np.array((0.4, 0, 0), order='F').reshape((-1, 3))
    # fsgeo.set_nodes(dbg_nodes, deltalength=0)
    # vsobj = sf.stokesFlowObj()
    # vsobj_kwargs = {'name': 'sphere_0', }
    # vsobj.set_data(fsgeo, vsgeo, **vsobj_kwargs)

    # # create problem
    # problem = problem_dic[matrix_method](**problem_kwargs)
    problem = sf.stokesletsInPipeProblem(**problem_kwargs)
    problem.solve_prepare()
    # problem.pickmyself(fileHeadle)
    b = np.array(problem.get_b_list())
    residualNorm = np.array(problem.get_residualNorm_list())
    err = np.array(problem.get_err_list())
    # do_show_err(fileHeadle, b, residualNorm, err)
    f1_list, f2_list, f3_list = problem.get_f_list()
    do_export_mat(fileHeadle, b, f1_list, f2_list, f3_list, residualNorm, err, dp, ep, lp, rp, th,
                  stokesletsInPipe_pipeFactor)

    PETSc.Sys().Print('                b -- residualNorm      ')
    PETSc.Sys().Print(np.hstack((b.reshape((-1, 1)), residualNorm)))
    if check_acc:
        PETSc.Sys().Print('                b -- err          ')
        PETSc.Sys().Print(np.hstack((b.reshape((-1, 1)), err)))

    return True


def do_show_err(fileHeadle, b, residualNorm, err):
    comm = PETSc.COMM_WORLD.tompi4py()
    rank = comm.Get_rank()
    if rank == 0:
        b0 = np.min(b)
        b1 = np.max(b)
        fig1 = plt.figure()
        ax1 = fig1.gca()
        # ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        line1 = ax1.plot(b, residualNorm, marker='o')
        line1[0].set_label('f1')
        line1[1].set_label('f2')
        line1[2].set_label('f3')
        ax1.legend(handles=line1, loc=2)
        ax1.set_xlabel('b')
        ax1.set_ylabel('residualNorm')
        ax1.set_xlim(b0, b1)
        # ax1.set_ylim(0, 0.03)
        fig1.savefig('%s_rN.png' % fileHeadle)

    if err.size > 0 and rank == 0:
        fig2 = plt.figure()
        ax2 = fig2.gca()
        line2 = ax2.plot(b, err, marker='o')
        line2[0].set_label('f1')
        line2[1].set_label('f2')
        line2[2].set_label('f3')
        ax2.legend(handles=line1, loc=2)
        ax2.set_xlabel('b')
        ax2.set_ylabel('velocity err')
        ax2.set_xlim(b0, b1)
        # ax2.set_ylim(0, 0.02)
        fig2.savefig('%s_err.png' % fileHeadle)
        # plt.show()
    PETSc.Sys().Print('export figles to %s_rN.png and %s_err.png' % (fileHeadle, fileHeadle))
    return True


def do_export_mat(fileHeadle, b, f1_list, f2_list, f3_list, residualNorm, err, dp, ep, lp, rp, th,
                  stokesletsInPipe_pipeFactor):
    comm = PETSc.COMM_WORLD.tompi4py()
    rank = comm.Get_rank()
    fileHeadle = check_file_extension(fileHeadle, extension='_force_pipe.mat')
    if rank == 0:
        savemat(fileHeadle,
                {'b':                           b,
                 'f1_list':                     f1_list,
                 'f2_list':                     f2_list,
                 'f3_list':                     f3_list,
                 'residualNorm':                residualNorm,
                 'err':                         err,
                 'dp':                          dp,
                 'ep':                          ep,
                 'lp':                          lp,
                 'rp':                          rp,
                 'th':                          th,
                 'stokesletsInPipe_pipeFactor': stokesletsInPipe_pipeFactor, },
                oned_as='column')
    PETSc.Sys().Print('export mat file to %s ' % fileHeadle)
    pass


def cart2pol(x, y):
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return rho, phi


def show_err():
    problem_kwargs = get_problem_kwargs()
    fileHeadle = problem_kwargs['fileHeadle']
    with open(fileHeadle + '_pick.bin', 'rb') as input:
        unpick = pickle.Unpickler(input)
        problem = unpick.load()
        problem.unpickmyself()
    assert isinstance(problem, sf.stokesletsInPipeProblem)
    b = np.array(problem.get_b_list())
    residualNorm = np.array(problem.get_residualNorm_list())
    err = np.array(problem.get_err_list())
    do_show_err(fileHeadle, b, residualNorm, err)
    return True


def export_mat():
    problem_kwargs = get_problem_kwargs()
    fileHeadle = problem_kwargs['fileHeadle']
    filePick = check_file_extension(fileHeadle, extension='_pick.bin')
    with open(filePick, 'rb') as myinput:
        unpick = pickle.Unpickler(myinput)
        problem = unpick.load()
        problem.unpickmyself()

    problem_kwargs = problem.get_kwargs()
    dp = problem_kwargs['dp']
    rp = problem_kwargs['rp']
    lp = problem_kwargs['lp']
    ep = problem_kwargs['ep']
    th = problem_kwargs['th']
    stokesletsInPipe_pipeFactor = problem_kwargs['stokesletsInPipe_pipeFactor']

    assert isinstance(problem, sf.stokesletsInPipeProblem)
    b = np.array(problem.get_b_list())
    residualNorm = np.array(problem.get_residualNorm_list())
    err = np.array(problem.get_err_list())
    f1_list, f2_list, f3_list = problem.get_f_list()
    do_export_mat(fileHeadle, b, f1_list, f2_list, f3_list, residualNorm, err, dp, ep, lp, rp, th,
                  stokesletsInPipe_pipeFactor)
    return True


def construct(**main_kwargs):
    comm = PETSc.COMM_WORLD.tompi4py()
    rank = comm.Get_rank()
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    fileHeadle = problem_kwargs['fileHeadle']
    matrix_method = problem_kwargs['matrix_method']
    dp = problem_kwargs['dp']
    rp = problem_kwargs['rp']
    lp = problem_kwargs['lp']
    ep = problem_kwargs['ep']
    th = problem_kwargs['th']
    stokesletsInPipe_pipeFactor = problem_kwargs['stokesletsInPipe_pipeFactor']
    problem = problem_dic[matrix_method](**problem_kwargs)
    # problem = sf.stokesletsInPipeforcefreeProblem(**problem_kwargs)

    problem.set_prepare(fileHeadle)
    problem.pickmyself(fileHeadle)
    b = np.array(problem.get_b_list())
    residualNorm = np.array(problem.get_residualNorm_list())
    err = np.array(problem.get_err_list())
    do_show_err(fileHeadle, b, residualNorm, err)
    f1_list, f2_list, f3_list = problem.get_f_list()
    do_export_mat(fileHeadle, b, f1_list, f2_list, f3_list, residualNorm, err, dp, ep, lp, rp, th,
                  stokesletsInPipe_pipeFactor)

    PETSc.Sys().Print('                b -- residualNorm      ')
    PETSc.Sys().Print(np.hstack((b.reshape((-1, 1)), residualNorm)))
    if err.size > 0:
        PETSc.Sys().Print('                b -- err          ')
        PETSc.Sys().Print(np.hstack((b.reshape((-1, 1)), err)))


def debug_stokeslets_b(b, node):
    problem_kwargs = get_problem_kwargs()
    fileHeadle = problem_kwargs['fileHeadle']
    problem = sf.stokesletsInPipeforcefreeProblem(**problem_kwargs)
    # fileHeadle = 'construct07'
    problem.set_prepare(fileHeadle)

    # t_headle = '_pick.bin'
    # if fileHeadle[-len(t_headle):] != t_headle:
    #     fileHeadle = fileHeadle + t_headle
    # with open(fileHeadle, 'rb') as input:
    #     unpick = pickle.Unpickler(input)
    #     problem = unpick.load()
    #     problem.unpickmyself()
    # assert isinstance(problem, sf.stokesletsInPipeProblem)

    node = np.array(node).reshape((-1, 3))
    num_ans1, num_ans2, num_ans3 = problem.debug_solve_stokeslets_b(b=b, node=node)

    tR, tphi = cart2pol(node[:, 0], node[:, 1])
    greenFun = detail(threshold=10, b=b)
    greenFun.solve_prepare()
    any_ans1 = np.hstack([greenFun.solve_u1(R, phi, z) for R, phi, z in zip(tR, tphi, node[:, 2])])
    any_ans2 = np.hstack([greenFun.solve_u2(R, phi, z) for R, phi, z in zip(tR, tphi, node[:, 2])])
    any_ans3 = np.hstack([greenFun.solve_u3(R, phi, z) for R, phi, z in zip(tR, tphi, node[:, 2])])
    print('analitical, numerical, abs_err, relative_err')
    print('u1')
    print(np.vstack((any_ans1, num_ans1[:], num_ans1[:] - any_ans1,
                     (num_ans1[:] - any_ans1) / any_ans1)).T)
    print('u2')
    print(np.vstack((any_ans2, num_ans2[:], num_ans2[:] - any_ans2,
                     (num_ans2[:] - any_ans2) / any_ans2)).T)
    print('u3')
    print(np.vstack((any_ans3, num_ans3[:], num_ans3[:] - any_ans3,
                     (num_ans3[:] - any_ans3) / any_ans3)).T)
    print(np.sqrt(np.sum((num_ans1[:] - any_ans1) ** 2 +
                         (num_ans2[:] - any_ans2) ** 2 +
                         (num_ans3[:] - any_ans3) ** 2)
                  / np.sum(any_ans1 ** 2 + any_ans2 ** 2 + any_ans3 ** 2)))
    return True


def debug_num_speed(nnode=1000):
    # get the speed of the numerical method for the calculation of the Stokeslets.
    problem_kwargs = get_problem_kwargs()
    node = np.random.sample(nnode * 3).reshape((-1, 3))
    b = 0.5

    import os
    import glob
    from time import time

    PWD = os.getcwd()
    # PWD = '/home/zhangji/stokes_flow_master/sphereInPipe/test_L_ds/dbg'
    mat_name_list = glob.glob('%s/*_force_pipe.mat' % PWD)
    dt = np.zeros_like(mat_name_list, dtype=np.float)
    nnode_pipe = np.zeros_like(mat_name_list, dtype=np.float)
    for i0, mat_name in enumerate(mat_name_list):
        fileHeadle = mat_name[:-15]
        problem = sf.stokesletsInPipeProblem(**problem_kwargs)
        t0 = time()
        problem.set_prepare(fileHeadle, fullpath=True)
        problem.debug_solve_stokeslets_b(b=b, node=node)
        t1 = time()
        dt[i0] = t1 - t0
        nnode_pipe[i0] = problem.get_fpgeo().get_n_nodes()
        PETSc.Sys.Print('%s: solve stokeslets numerically use: %fs' % (os.path.basename(fileHeadle), dt[i0]))

    comm = PETSc.COMM_WORLD.tompi4py()
    rank = comm.Get_rank()
    if rank == 0:
        savemat('debug_num_speed.mat',
                {'mat_name_list': mat_name_list,
                 'dt':            dt,
                 'nnode_pipe':    nnode_pipe,
                 'node':          node, },
                oned_as='column')
    return True


def debug_ana_speed(nnode=1000):
    # get the speed of the series solution for the calculation of the Stokeslets.
    node = np.random.sample(nnode * 3).reshape((-1, 3))
    b = 0.5

    from time import time

    cth_list = np.arange(10, 1000, 10)
    dt = np.zeros_like(cth_list, dtype=np.float)
    for i0, cth in enumerate(cth_list):
        greenFun = detail(threshold=cth, b=b)
        t0 = time()
        greenFun.solve_prepare()
        greenFun.solve_uxyz(node)
        t1 = time()
        dt[i0] = t1 - t0
        PETSc.Sys.Print('cth=%d: solve stokeslets analytically use: %fs' % (cth, dt[i0]))

    comm = PETSc.COMM_WORLD.tompi4py()
    rank = comm.Get_rank()
    if rank == 0:
        savemat('debug_ana_speed.mat',
                {'cth':    cth_list,
                 'dt_ana': dt,
                 'node':   node, },
                oned_as='column')
    return True


def debug_solve_u_pipe(b, dp, lp):
    problem_kwargs = get_problem_kwargs()
    matrix_method = problem_kwargs['matrix_method']
    rp = 1
    greenFun = detail(threshold=50, b=b)
    greenFun.solve_prepare()

    problem = problem_dic[matrix_method](**problem_kwargs)
    assert isinstance(problem, sf.stokesletsInPipeProblem)

    outputHandle = ' '
    pgeo = tunnel_geo()  # velocity node geo of pipe
    dth = 2 * np.arcsin(dp / 2 / rp)
    pgeo.create_deltatheta(dth=dth, radius=rp, length=lp, epsilon=0, with_cover=True, factor=2.5)

    problem.debug_solve_u_pipe(pgeo, outputHandle, greenFun)
    return True


def debug_solve_stokeslets_fnode(fnode):
    from src.geo import tunnel_geo
    problem_kwargs = get_problem_kwargs()
    fileHeadle = problem_kwargs['fileHeadle']
    problem = sf.stokesletsInPipeforcefreeProblem(**problem_kwargs)
    problem.set_prepare(fileHeadle)
    fnode = np.array(fnode).reshape((1, 3))

    dp = 0.1
    rp = 1
    lp = 1
    stokesletsInPipe_pipeFactor = 1
    vpgeo = tunnel_geo()  # velocity node geo of pipe
    dth = 2 * np.arcsin(dp / 2 / rp)
    vpgeo.create_deltatheta(dth=dth, radius=rp, length=lp, epsilon=0, with_cover=True,
                            factor=stokesletsInPipe_pipeFactor)
    # vpgeo.show_nodes()

    problem.debug_solve_stokeslets_fnode(fnode, vpgeo)
    return True


def m2_err_z():
    # for the paper m2, test the relative error between numerical and analytical values of uz
    mat_contents = loadmat('convergence_z.mat')
    b = mat_contents['b'][0][0]
    z1 = mat_contents['z1'][0]
    IDX = z1 > 0.011
    z1 = z1[IDX]
    R1 = mat_contents['R1'][0][IDX]
    phi1 = mat_contents['phi1'][0][IDX]
    uz = mat_contents['u_struct'][0][-1][3][0][IDX]
    nodes = np.vstack((R1 * np.cos(phi1), R1 * np.sin(phi1), z1)).T

    # velocity (stokeslets) in infinite space
    m = light_stokeslets_matrix_3d(nodes, np.array((b, 0, 0)))
    uz_inf = np.dot(m, (0, 0, 1))[2::3]

    # velocity (stokeslets) in pipe
    problem_kwargs = get_problem_kwargs()
    fileHeadle = problem_kwargs['fileHeadle']
    problem = sf.stokesletsInPipeforcefreeProblem(**problem_kwargs)
    problem.set_prepare(fileHeadle)
    _, _, num_ans3 = problem.debug_solve_stokeslets_b(b=b, node=nodes)
    num_uz = num_ans3.getArray().reshape((-1, 3))[:, 2]
    comm = PETSc.COMM_WORLD.tompi4py()
    rank = comm.Get_rank()
    if rank == 0:
        savemat('num_uz.mat',
                {'z1_use': z1,
                 'uz_ana': uz,
                 'uz_inf': uz_inf,
                 'uz_num': num_uz, },
                oned_as='column')

    print('analytical, numerical, abs_err, relative_err')
    print(np.vstack((z1, uz, num_uz, num_uz - uz, (num_uz - uz) / uz)).T)
    # print(np.sqrt(np.sum((num_ans1[:] - any_ans1) ** 2 +
    #                      (num_ans2[:] - any_ans2) ** 2 +
    #                      (num_ans3[:] - any_ans3) ** 2)
    #               / np.sum(any_ans1 ** 2 + any_ans2 ** 2 + any_ans3 ** 2)))
    return True


if __name__ == '__main__':
    # main_fun()
    # show_err()
    # export_mat()
    # debug_stokeslets_b(5.81500000e-01, np.vstack((np.ones(10) * 0.5, np.ones(10) * 0.5, np.linspace(0.1, 1, 10))).T)
    # debug_stokeslets_b(0.5, np.array((0.5, 0, 1)))
    # debug_solve_u_pipe(0.5, 0.1, 0.5)
    # debug_solve_stokeslets_fnode((0.3/2**0.5, 0.3/2**0.5, 0))

    OptDB = PETSc.Options()
    # if OptDB.getBool('show_err', False):
    #     OptDB.setValue('main_fun', False)
    #     show_err()
    #
    # if OptDB.getBool('export_mat', False):
    #     OptDB.setValue('main_fun', False)
    #     export_mat()

    if OptDB.getBool('debug_num_speed', False):
        OptDB.setValue('main_fun', False)
        debug_num_speed(nnode=1000)

    if OptDB.getBool('debug_ana_speed', False):
        OptDB.setValue('main_fun', False)
        debug_ana_speed(nnode=1000)

    if OptDB.getBool('main_fun', True):
        main_fun()

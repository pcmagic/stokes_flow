# coding: utf-8
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 11:05:23 2017

@author: zhangji
"""

import matplotlib
import subprocess
import os

devnull = open(os.devnull, 'w')
latex_installed = not subprocess.call(['which', 'latex'],
                                      stdout=devnull, stderr=devnull)
matplotlib.use('agg')
font = {'size':   20,
        'family': 'sans-serif'}
matplotlib.rc('font', **font)
if latex_installed:
    matplotlib.rc('text', usetex=True)

import numpy as np
from scipy.io import loadmat
from scipy import interpolate, integrate, spatial, signal
from src import jeffery_model as jm
from src.objComposite import *
from src.support_class import *
from matplotlib import animation
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import colorbar
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, zoomed_inset_axes
import matplotlib.ticker as mtick
import importlib
import inspect
from tqdm import tqdm, tqdm_notebook
import glob
import natsort
from time import time
import pickle
import re
from codeStore import support_fun as spf
import shutil
import multiprocessing

markerstyle_list = ['^', 'v', 'o', 's', 'p', 'd', 'H',
                    '1', '2', '3', '4', '8', 'P', '*',
                    'h', '+', 'x', 'X', 'D', '|', '_', ]
PWD = os.getcwd()
if latex_installed:
    params = {'text.latex.preamble': [r'\usepackage{bm}', r'\usepackage{amsmath}']}
    plt.rcParams.update(params)


def read_data_lookup_table(psi_dir, tcenter):
    ecoli_U_list = []
    ecoli_norm_list = []
    ecoli_center_list = []
    ecoli_nodes_list = []
    ecoli_u_list = []
    ecoli_f_list = []
    ecoli_lateral_norm_list = []
    norm_phi_list = []
    norm_psi_list = []
    norm_theta_list = []
    planeShearRate = None
    file_handle = os.path.basename(psi_dir)
    mat_names = natsort.natsorted(glob.glob('%s/%s_*.mat' % (psi_dir, file_handle)))
    for mati in mat_names:
        mat_contents = loadmat(mati)
        ecoli_U = mat_contents['ecoli_U'].flatten()
        ecoli_norm = mat_contents['ecoli_norm'].flatten()
        ecoli_center = mat_contents['ecoli_center'].flatten()
        ecoli_nodes = mat_contents['ecoli_nodes']
        ecoli_u = mat_contents['ecoli_u']
        ecoli_f = mat_contents['ecoli_f']
        planeShearRate = mat_contents['planeShearRate'].flatten()
        norm_phi = mat_contents['norm_phi'].flatten()
        norm_psi = mat_contents['norm_psi'].flatten()
        norm_theta = mat_contents['norm_theta'].flatten()
        ecoli_U_list.append(ecoli_U)
        ecoli_norm_list.append(ecoli_norm)
        ecoli_center_list.append(ecoli_center)
        norm_phi_list.append(norm_phi)
        norm_psi_list.append(norm_psi)
        norm_theta_list.append(norm_theta)
        r0 = ecoli_nodes[-1] - ecoli_center
        n0 = np.dot(r0, ecoli_norm) * ecoli_norm / np.dot(ecoli_norm, ecoli_norm)
        t0 = r0 - n0
        ecoli_lateral_norm_list.append(t0 / np.linalg.norm(t0))

    ecoli_U = np.vstack(ecoli_U_list)
    ecoli_norm = np.vstack(ecoli_norm_list)
    ecoli_center = np.vstack(ecoli_center_list)
    ecoli_lateral_norm = np.vstack(ecoli_lateral_norm_list)
    norm_phi = np.hstack(norm_phi_list)
    norm_psi = np.hstack(norm_psi_list)
    norm_theta = np.hstack(norm_theta_list)
    norm_tpp = np.vstack((norm_theta, norm_phi, norm_psi)).T

    # calculate velocity u000(t,x,y,z) that the location initially at (0, 0, 0): u000(0, 0, 0, 0)
    n_u000 = -np.linalg.norm(ecoli_center[0] - tcenter) * ecoli_norm
    ecoli_u000 = ecoli_U[:, :3] + np.cross(ecoli_U[:, 3:], n_u000)
    # calculate center center000(t,x,y,z) that at initially at (0, 0, 0): center000(0, 0, 0, 0)
    ecoli_center000 = ecoli_center + n_u000
    using_U = ecoli_U
    omega_norm = np.array(
            [np.dot(t1, t2) * t2 / np.dot(t2, t2) for t1, t2 in zip(using_U[:, 3:], ecoli_norm)])
    omega_tang = using_U[:, 3:] - omega_norm

    return ecoli_U, ecoli_norm, ecoli_center, ecoli_lateral_norm, norm_tpp, \
           ecoli_u000, ecoli_center000, omega_norm, omega_tang, planeShearRate, file_handle


def get_ecoli_table(tnorm, lateral_norm, tcenter, max_iter, eval_dt=0.001, update_order=1,
                    planeShearRate=np.array((1, 0, 0))):
    ellipse_kwargs = {'name':         'ecoli_torque',
                      'center':       tcenter,
                      'norm':         tnorm / np.linalg.norm(tnorm),
                      'lateral_norm': lateral_norm / np.linalg.norm(lateral_norm),
                      'speed':        0,
                      'lbd':          np.nan,
                      'omega_tail':   193.66659814,
                      'table_name':   'planeShearRatex_1d', }
    fileHandle = 'ShearTableProblem'
    ellipse_obj = jm.TableEcoli(**ellipse_kwargs)
    ellipse_obj.set_update_para(fix_x=False, fix_y=False, fix_z=False, update_order=update_order)
    problem = jm.ShearTableProblem(name=fileHandle, planeShearRate=planeShearRate)
    problem.add_obj(ellipse_obj)
    t0 = time()
    for idx in range(1, max_iter + 1):
        problem.update_location(eval_dt, print_handle='%d / %d' % (idx, max_iter))
    t1 = time()
    Table_X = np.vstack(ellipse_obj.center_hist)
    Table_U = np.vstack(ellipse_obj.U_hist)
    Table_P = np.vstack(ellipse_obj.norm_hist)
    Table_t = np.arange(max_iter) * eval_dt + eval_dt
    Table_theta, Table_phi, Table_psi = ellipse_obj.theta_phi_psi
    t1U = np.array([np.dot(t1, t2) for t1, t2 in zip(Table_U[:, :3], Table_P)]).reshape((-1, 1))
    t1W = np.array([np.dot(t1, t2) for t1, t2 in zip(Table_U[:, 3:], Table_P)]).reshape((-1, 1))
    #     Table_U_horizon = np.hstack((Table_P * t1U, Table_P * t1W))
    #     Table_U_vertical = Table_U - Table_U_horizon
    omega = Table_U[:, 3:]
    dP = np.vstack([np.cross(t1, t2) for t1, t2 in zip(omega, Table_P)])
    Table_dtheta = -dP[:, 2] / np.sin(np.abs(Table_theta))
    Table_dphi = (dP[:, 1] * np.cos(Table_phi) - dP[:, 0] * np.sin(Table_phi)) / np.sin(Table_theta)
    Table_eta = np.arccos(np.sin(Table_theta) * np.sin(Table_phi))
    #     print('%s: run %d loops using %f' % (fileHandle, max_iter, (t1 - t0)))
    return Table_t, Table_theta, Table_phi, Table_psi, Table_eta, Table_dtheta, Table_dphi, \
           Table_X, Table_U, Table_P


def do_calculate_prepare(norm, ):
    importlib.reload(jm)
    norm = norm / np.linalg.norm(norm)
    planeShearRate = np.array((1, 0, 0))
    tcenter = np.zeros(3)

    # print('dbg do_calculate_prepare')
    tlateral_norm = np.array((np.pi, np.e, np.euler_gamma))
    # tlateral_norm = np.random.sample(3)
    tlateral_norm = tlateral_norm / np.linalg.norm(tlateral_norm)
    tlateral_norm = tlateral_norm - norm * np.dot(norm, tlateral_norm)
    tlateral_norm = tlateral_norm / np.linalg.norm(tlateral_norm)
    P0 = norm / np.linalg.norm(norm)
    P20 = tlateral_norm / np.linalg.norm(tlateral_norm)
    fileHandle = 'ShearTableProblem'
    problem = jm.ShearTableProblem(name=fileHandle, planeShearRate=planeShearRate)
    return P0, P20, tcenter, problem


def do_calculate(problem, obj, ini_t, max_t, update_fun, rtol, atol, eval_dt, save_every, tqdm_fun):
    obj.set_update_para(fix_x=False, fix_y=False, fix_z=False, update_fun=update_fun,
                        rtol=rtol, atol=atol, save_every=save_every, tqdm_fun=tqdm_fun)
    problem.add_obj(obj)
    Table_t, Table_dt, Table_X, Table_P, Table_P2 = \
        obj.update_self(t0=ini_t, t1=max_t, eval_dt=eval_dt)
    Table_theta, Table_phi, Table_psi = obj.theta_phi_psi
    Table_eta = np.arccos(np.sin(Table_theta) * np.sin(Table_phi))
    return Table_t, Table_dt, Table_X, Table_P, Table_P2, \
           Table_theta, Table_phi, Table_psi, Table_eta


def do_ellipse_kwargs(tcenter, P0, P20, ini_psi, table_name):
    ellipse_kwargs = {'name':         'ellipse',
                      'center':       tcenter,
                      'norm':         P0,
                      'lateral_norm': P20,
                      'speed':        0,
                      'lbd':          np.nan,
                      'ini_psi':      ini_psi,
                      'omega_tail':   0,
                      'table_name':   table_name, }
    return ellipse_kwargs


def do_ecoli_kwargs(tcenter, P0, P20, ini_psi, omega_tail, table_name):
    ecoli_kwargs = {'name':         'ecoli_torque',
                    'center':       tcenter,
                    'norm':         P0,
                    'lateral_norm': P20,
                    'speed':        0,
                    'lbd':          np.nan,
                    'ini_psi':      ini_psi,
                    'omega_tail':   omega_tail,
                    'table_name':   table_name, }
    return ecoli_kwargs


def do_ecoli_passive_kwargs(tcenter, P0, P20, ini_psi, table_name):
    ecoli_passive_kwargs = {'name':         'ecoli_passive',
                            'center':       tcenter,
                            'norm':         P0,
                            'lateral_norm': P20,
                            'speed':        0,
                            'lbd':          np.nan,
                            'ini_psi':      ini_psi,
                            'omega_tail':   0,
                            'table_name':   table_name, }
    return ecoli_passive_kwargs


def do_helix_kwargs(tcenter, P0, P20, ini_psi, table_name):
    helix_kwargs = {'name':         'helix',
                    'center':       tcenter,
                    'norm':         P0,
                    'lateral_norm': P20,
                    'speed':        0,
                    'lbd':          np.nan,
                    'ini_psi':      ini_psi,
                    'omega_tail':   0,
                    'table_name':   table_name, }
    return helix_kwargs


def do_calculate_helix_Petsc4n(norm, ini_psi, max_t, update_fun='3bs', rtol=1e-6, atol=1e-9,
                               eval_dt=0.001, ini_t=0,
                               save_every=1, table_name='hlxB01_tau1a', tqdm_fun=tqdm_notebook,
                               omega_tail=0):
    fun_name = inspect.stack()[0][3]
    err_msg = '%s: omega_tail NOT 0 (now omega_tail=%f)' % (fun_name, omega_tail)
    assert np.isclose(omega_tail, 0), err_msg
    P0, P20, tcenter, problem = do_calculate_prepare(norm)
    helix_kwargs = do_helix_kwargs(tcenter, P0, P20, ini_psi, table_name=table_name)
    helix_obj = jm.TablePetsc4nEcoli(**helix_kwargs)
    return do_calculate(problem, helix_obj, ini_t, max_t, update_fun, rtol, atol, eval_dt,
                        save_every, tqdm_fun)


def do_calculate_helix_AvrPetsc4n(norm, ini_psi, max_t, update_fun='3bs', rtol=1e-6, atol=1e-9,
                                  eval_dt=0.001, ini_t=0,
                                  save_every=1, table_name='hlxB01_tau1a_avr',
                                  tqdm_fun=tqdm_notebook, omega_tail=0):
    fun_name = inspect.stack()[0][3]
    err_msg = '%s: omega_tail NOT 0 (now omega_tail=%f)' % (fun_name, omega_tail)
    assert np.isclose(omega_tail, 0), err_msg
    P0, P20, tcenter, problem = do_calculate_prepare(norm)
    helix_kwargs = do_helix_kwargs(tcenter, P0, P20, ini_psi, table_name=table_name)
    helix_obj = jm.TableAvrPetsc4nEcoli(**helix_kwargs)
    return do_calculate(problem, helix_obj, ini_t, max_t, update_fun, rtol, atol, eval_dt,
                        save_every, tqdm_fun)


def do_calculate_ellipse_Petsc4n(norm, ini_psi, max_t, update_fun='3bs', rtol=1e-6, atol=1e-9,
                                 eval_dt=0.001, ini_t=0,
                                 save_every=1, table_name='ellipse_alpha3', tqdm_fun=tqdm_notebook,
                                 omega_tail=0):
    fun_name = inspect.stack()[0][3]
    err_msg = '%s: omega_tail NOT 0 (now omega_tail=%f)' % (fun_name, omega_tail)
    assert np.isclose(omega_tail, 0), err_msg
    P0, P20, tcenter, problem = do_calculate_prepare(norm)
    ellipse_kwargs = do_ellipse_kwargs(tcenter, P0, P20, ini_psi, table_name=table_name)
    ellipse_obj = jm.TablePetsc4nEcoli(**ellipse_kwargs)
    return do_calculate(problem, ellipse_obj, ini_t, max_t, update_fun, rtol, atol, eval_dt,
                        save_every, tqdm_fun)


def do_calculate_ellipse_AvrPetsc4n(norm, ini_psi, max_t, update_fun='3bs', rtol=1e-6, atol=1e-9,
                                    eval_dt=0.001, ini_t=0,
                                    save_every=1, table_name='ellipse_alpha3_avr',
                                    tqdm_fun=tqdm_notebook,
                                    omega_tail=0):
    fun_name = inspect.stack()[0][3]
    err_msg = '%s: omega_tail NOT 0 (now omega_tail=%f)' % (fun_name, omega_tail)
    assert np.isclose(omega_tail, 0), err_msg
    P0, P20, tcenter, problem = do_calculate_prepare(norm)
    ellipse_kwargs = do_ellipse_kwargs(tcenter, P0, P20, ini_psi, table_name=table_name)
    ellipse_obj = jm.TableAvrPetsc4nEcoli(**ellipse_kwargs)
    return do_calculate(problem, ellipse_obj, ini_t, max_t, update_fun, rtol, atol, eval_dt,
                        save_every, tqdm_fun)


def do_calculate_ecoli_Petsc4n(norm, ini_psi, max_t, update_fun='3bs', rtol=1e-6, atol=1e-9,
                               eval_dt=0.001, ini_t=0,
                               save_every=1, table_name='planeShearRatex_1d',
                               tqdm_fun=tqdm_notebook,
                               omega_tail=193.66659814):
    # fun_name = inspect.stack()[0][3]
    # err_msg = '%s: omega_tail IS 0 (now omega_tail=%f)' % (fun_name, omega_tail)
    # assert not np.isclose(omega_tail, 0), err_msg
    P0, P20, tcenter, problem = do_calculate_prepare(norm)
    ecoli_kwargs = do_ecoli_kwargs(tcenter, P0, P20, ini_psi, omega_tail, table_name)
    ecoli_obj = jm.TablePetsc4nEcoli(**ecoli_kwargs)
    return do_calculate(problem, ecoli_obj, ini_t, max_t, update_fun, rtol, atol, eval_dt,
                        save_every, tqdm_fun)


def do_calculate_ecoli_Petsc4nPsi(norm, ini_psi, max_t, update_fun='3bs', rtol=1e-6, atol=1e-9,
                                  eval_dt=0.001, ini_t=0,
                                  save_every=1, table_name='planeShearRatex_1d',
                                  tqdm_fun=tqdm_notebook,
                                  omega_tail=193.66659814):
    fun_name = inspect.stack()[0][3]
    # err_msg = '%s: omega_tail IS 0 (now omega_tail=%f)' % (fun_name, omega_tail)
    # assert not np.isclose(omega_tail, 0), err_msg
    P0, P20, tcenter, problem = do_calculate_prepare(norm)
    ecoli_kwargs = do_ecoli_kwargs(tcenter, P0, P20, ini_psi, omega_tail, table_name)
    ecoli_obj = jm.TablePetsc4nPsiEcoli(**ecoli_kwargs)

    obj = ecoli_obj
    obj.set_update_para(fix_x=False, fix_y=False, fix_z=False, update_fun=update_fun,
                        rtol=rtol, atol=atol, save_every=save_every, tqdm_fun=tqdm_fun)
    problem.add_obj(obj)
    Table_t, Table_dt, Table_X, Table_P, Table_P2, Table_psi = \
        obj.update_self(t0=ini_t, t1=max_t, eval_dt=eval_dt)
    Table_theta, Table_phi, Table_psib = obj.theta_phi_psi
    Table_eta = np.arccos(np.sin(Table_theta) * np.sin(Table_phi))
    # return Table_t, Table_dt, Table_X, Table_P, Table_P2, \
    #        Table_theta, Table_phi, Table_psib, Table_eta, Table_psi
    return Table_t, Table_dt, Table_X, Table_P, Table_P2, \
           Table_theta, Table_phi, Table_psi, Table_eta,


def do_calculate_ecoli_AvrPetsc4n(norm, ini_psi, max_t, update_fun='3bs', rtol=1e-6, atol=1e-9,
                                  eval_dt=0.001, ini_t=0,
                                  save_every=1, table_name='planeShearRatex_1d_avr',
                                  tqdm_fun=tqdm_notebook,
                                  omega_tail=193.66659814):
    fun_name = inspect.stack()[0][3]
    err_msg = '%s: omega_tail IS 0 (now omega_tail=%f)' % (fun_name, omega_tail)
    assert not np.isclose(omega_tail, 0), err_msg
    P0, P20, tcenter, problem = do_calculate_prepare(norm)
    ecoli_kwargs = do_ecoli_kwargs(tcenter, P0, P20, ini_psi, omega_tail, table_name)
    ecoli_obj = jm.TableAvrPetsc4nEcoli(**ecoli_kwargs)
    return do_calculate(problem, ecoli_obj, ini_t, max_t, update_fun, rtol, atol, eval_dt,
                        save_every, tqdm_fun)


def do_calculate_ecoli_passive_Petsc4n(norm, ini_psi, max_t, update_fun='3bs', rtol=1e-6, atol=1e-9,
                                       eval_dt=0.001, ini_t=0, save_every=1,
                                       table_name='planeShearRatex_1d_passive',
                                       tqdm_fun=tqdm_notebook,
                                       omega_tail=0):
    fun_name = inspect.stack()[0][3]
    err_msg = '%s: omega_tail NOT 0 (now omega_tail=%f)' % (fun_name, omega_tail)
    assert np.isclose(omega_tail, 0), err_msg
    P0, P20, tcenter, problem = do_calculate_prepare(norm)
    ecoli_passive_kwargs = do_ecoli_passive_kwargs(tcenter, P0, P20, ini_psi, table_name)
    ecoli_passive_obj = jm.TablePetsc4nEcoli(**ecoli_passive_kwargs)
    return do_calculate(problem, ecoli_passive_obj, ini_t, max_t, update_fun, rtol, atol, eval_dt,
                        save_every, tqdm_fun)


def do_calculate_ecoli_passive_AvrPetsc4n(norm, ini_psi, max_t, update_fun='3bs', rtol=1e-6,
                                          atol=1e-9, eval_dt=0.001, ini_t=0,
                                          save_every=1, table_name='planeShearRatex_1d_passive_avr',
                                          tqdm_fun=tqdm_notebook, omega_tail=0):
    fun_name = inspect.stack()[0][3]
    err_msg = '%s: omega_tail NOT 0 (now omega_tail=%f)' % (fun_name, omega_tail)
    assert np.isclose(omega_tail, 0), err_msg
    P0, P20, tcenter, problem = do_calculate_prepare(norm)
    ecoli_passive_kwargs = do_ecoli_passive_kwargs(tcenter, P0, P20, ini_psi, table_name)
    ecoli_passive_obj = jm.TableAvrPetsc4nEcoli(**ecoli_passive_kwargs)
    return do_calculate(problem, ecoli_passive_obj, ini_t, max_t, update_fun, rtol, atol, eval_dt,
                        save_every, tqdm_fun)


def core_show_table_result(Table_t, Table_dt, Table_X, Table_P, Table_P2,
                           Table_theta, Table_phi, Table_psi, Table_eta, move_z=False,
                           planeShearRate=np.array((1, 0, 0)), fig=None,
                           save_every=1, resampling=False, resampling_fct=2):
    fontsize = 40
    figsize = (20, 15)
    if move_z:
        z_mean = np.mean(Table_X[:, 2])
        Table_X[:, 2] = Table_X[:, 2] - z_mean
        ux_shear = z_mean * planeShearRate[0]
        Xz_mean = (Table_t - Table_t[0]) * ux_shear
        Table_X[:, 0] = Table_X[:, 0] - Xz_mean

    if resampling:
        Table_t, Table_dt, Table_X, Table_P, Table_P2, \
        Table_theta, Table_phi, Table_psi, Table_eta = \
            resampling_data(Table_t, Table_dt, Table_X, Table_P, Table_P2,
                            Table_theta, Table_phi, Table_psi, Table_eta, resampling_fct)

    # show table results.
    if fig is None:
        fig = plt.figure(figsize=figsize)
    else:
        fig.clf()
    fig.patch.set_facecolor('white')
    ax0 = plt.subplot2grid((7, 6), (0, 0), rowspan=3, colspan=3, polar=True)
    ax4 = plt.subplot2grid((7, 6), (3, 3), colspan=3)
    ax1 = plt.subplot2grid((7, 6), (0, 3), colspan=3, sharex=ax4)
    ax2 = plt.subplot2grid((7, 6), (1, 3), colspan=3, sharex=ax4)
    ax3 = plt.subplot2grid((7, 6), (2, 3), colspan=3, sharex=ax4)
    axdt = plt.subplot2grid((7, 6), (3, 0), colspan=3)
    axP = plt.subplot2grid((7, 6), (6, 0), colspan=2)
    axP2 = plt.subplot2grid((7, 6), (6, 2), colspan=2)
    axPdotP2 = plt.subplot2grid((7, 6), (6, 4), colspan=2)
    ax5 = plt.subplot2grid((7, 6), (4, 0), rowspan=2, colspan=2, sharex=axP)
    ax6 = plt.subplot2grid((7, 6), (4, 2), rowspan=2, colspan=2, sharex=axP2)
    ax7 = plt.subplot2grid((7, 6), (4, 4), rowspan=2, colspan=2, sharex=axPdotP2)
    # polar version
    norm = plt.Normalize(Table_t.min(), Table_t.max())
    cmap = plt.get_cmap('jet')
    ax0.plot(Table_phi, Table_theta, '-', alpha=0.2)
    ax0.plot(Table_phi[0], Table_theta[0], '*k')
    lc = ax0.scatter(Table_phi, Table_theta, c=Table_t, cmap=cmap, norm=norm, s=fontsize * 0.1)
    clb = fig.colorbar(lc, ax=ax0, orientation="vertical")
    clb.ax.tick_params(labelsize=fontsize * 0.5)
    clb.ax.set_title('time', size=fontsize * 0.5)
    # ax0.set_xlabel('$\\phi / \pi$', size=fontsize*0.7)
    # ax0.set_ylabel('$\\theta / \pi$', size=fontsize*0.7)
    ax0.set_ylim(0, np.pi)
    plt.sca(ax0)
    plt.xticks(fontsize=fontsize * 0.5)
    plt.yticks(fontsize=fontsize * 0.5)
    # # phase map version
    #     norm=plt.Normalize(Table_t.min(), Table_t.max())
    #     cmap=plt.get_cmap('jet')
    #     ax0.plot(Table_phi / np.pi, Table_theta / np.pi, ' ')
    #     lc = spf.colorline(Table_phi / np.pi, Table_theta / np.pi, Table_t,
    #                        ax=ax0, cmap=cmap, norm=norm, linewidth=3)
    #     clb = fig.colorbar(lc, ax=ax0, orientation="vertical")
    #     clb.ax.tick_params(labelsize=fontsize*0.5)
    #     clb.ax.set_title('time', size=fontsize*0.5)
    #     ax0.set_xlabel('$\\phi / \pi$', size=fontsize*0.7)
    #     ax0.set_ylabel('$\\theta / \pi$', size=fontsize*0.7)
    #     plt.sca(ax0)
    #     plt.xticks(fontsize=fontsize*0.5)
    #     plt.yticks(fontsize=fontsize*0.5)
    xticks = np.around(np.linspace(Table_t.min(), Table_t.max(), 21), decimals=2)[1::6]
    for axi, ty, axyi in zip((ax1, ax2, ax3, ax4, ax5, ax6, ax7, axdt, axP, axP2, axPdotP2),
                             (Table_theta / np.pi, Table_phi / np.pi, Table_psi / np.pi,
                              Table_eta / np.pi,
                              Table_X[:, 0], Table_X[:, 1], Table_X[:, 2], Table_dt,
                              np.linalg.norm(Table_P, axis=1),
                              np.linalg.norm(Table_P2, axis=1),
                              np.abs(np.einsum('ij,ij->i', Table_P, Table_P2))),
                             ('$\\theta / \pi$', '$\\phi / \pi$', '$\\psi / \pi$', '$\\eta / \pi$',
                              '$center_x$', '$center_y$', '$center_z$', 'dt',
                              '$\|P_1\|$', '$\|P_2\|$', '$\|P_1 \cdot P_2\|$')):
        plt.sca(axi)
        axi.plot(Table_t, ty, '-*', label='Table')
        #     axi.set_xlabel('t', size=fontsize)
        #     axi.legend()
        axi.set_ylabel('%s' % axyi, size=fontsize * 0.7)
        axi.set_xticks(xticks)
        axi.set_xticklabels(xticks)
        plt.xticks(fontsize=fontsize * 0.5)
        plt.yticks(fontsize=fontsize * 0.5)
    for axi in (ax4, axdt, axP, axP2, axPdotP2):
        axi.set_xlabel('t', size=fontsize * 0.7)
    for axi in (axP, axP2):
        axi.set_ylim(0.9, 1.1)
    axdt.axes.set_yscale('log')
    plt.tight_layout()
    return fig


def show_table_result(Table_t, Table_dt, Table_X, Table_P, Table_P2,
                      Table_theta, Table_phi, Table_psi, Table_eta, move_z=False,
                      planeShearRate=np.array((1, 0, 0)), fig=None,
                      save_every=1, resampling=False, resampling_fct=2):
    core_show_table_result(Table_t, Table_dt, Table_X, Table_P, Table_P2,
                           Table_theta, Table_phi, Table_psi, Table_eta, move_z,
                           planeShearRate, fig, save_every, resampling, resampling_fct)
    return True


def save_table_result(filename, Table_t, Table_dt, Table_X, Table_P, Table_P2,
                      Table_theta, Table_phi, Table_psi, Table_eta, move_z=False,
                      planeShearRate=np.array((1, 0, 0)), fig=None,
                      save_every=1, resampling=False, resampling_fct=2):
    fig = core_show_table_result(Table_t, Table_dt, Table_X, Table_P, Table_P2,
                                 Table_theta, Table_phi, Table_psi, Table_eta, move_z,
                                 planeShearRate, fig, save_every, resampling, resampling_fct)
    fig.savefig(filename, dpi=100)
    return fig


def core_show_theta_phi(Table_t, Table_dt, Table_X, Table_P, Table_P2,
                        Table_theta, Table_phi, Table_psi, Table_eta,
                        fig=None, show_back_direction=True):
    def add_axs_psi_theta(ax0, psi_list, theta_list, ax_size_fct=0.1, alpha=0.0):
        for tphi in psi_list:
            for ttheta in theta_list:
                tx = ttheta * np.cos(tphi)
                ty = ttheta * np.sin(tphi)
                bbox = (tx - ax_size_fct / 2 * np.pi, ty - ax_size_fct / 2 * np.pi,
                        ax_size_fct * np.pi, ax_size_fct * np.pi)
                axin = spf.add_inset(ax0, bbox, projection='3d')
                for spine in axin.spines.values():
                    spine.set_visible(False)
                axin.xaxis.set_major_locator(plt.NullLocator())
                axin.yaxis.set_major_locator(plt.NullLocator())
                axin.zaxis.set_major_locator(plt.NullLocator())
                axin.set_xlim(-1, 1)
                axin.set_ylim(-1, 1)
                axin.set_zlim(-1, 1)
                axin.patch.set_alpha(alpha)

                axin.quiver(0, 0, 0,
                            np.sin(ttheta) * np.cos(tphi),
                            np.sin(ttheta) * np.sin(tphi),
                            np.cos(ttheta),
                            arrow_length_ratio=0.5, colors='k', linewidth=fontsize * 0.1)

    # background
    fontsize = 30
    if fig is None:
        fig = plt.figure(figsize=(20, 20))
    else:
        fig.clf()
    fig.patch.set_facecolor('white')
    ax0 = fig.add_subplot(111)
    ax0.set_xlim(-np.pi * 1.1, np.pi * 1.1)
    ax0.set_ylim(-np.pi * 1.1, np.pi * 1.1)
    ax0.axis('off')
    cax0 = colorbar.make_axes(ax0, orientation='vertical', aspect=20, shrink=0.6)[0]
    ax0.set_aspect('equal')

    # norms of different directions
    if show_back_direction:
        # 1
        psi_list = (0,)
        theta_list = (0,)
        add_axs_psi_theta(ax0, psi_list, theta_list, ax_size_fct=0.2, alpha=0.3)
        # 2
        psi_list = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        theta_list = np.linspace(0.2 * np.pi, np.pi, 4)
        add_axs_psi_theta(ax0, psi_list, theta_list, ax_size_fct=0.2, alpha=0.3)
        # 3
        psi_list = np.linspace(0, 2 * np.pi, 16, endpoint=False)[1::2]
        theta_list = np.linspace(0.25 * np.pi, np.pi, 8)[1::2]
        add_axs_psi_theta(ax0, psi_list, theta_list, ax_size_fct=0.2, alpha=0.3)
        # 4
        psi_list = np.linspace(0, 2 * np.pi, 32, endpoint=False)[1::2]
        t1 = np.linspace(0.25 * np.pi, np.pi, 8)[1::2]
        theta_list = (np.mean((t1[2], t1[3])), np.mean((t1[1], t1[2])))
        add_axs_psi_theta(ax0, psi_list, theta_list, ax_size_fct=0.2, alpha=0.3)

    # polar version of theta-phi
    ax1 = fig.add_axes(ax0.get_position(), projection='polar')
    ax1.patch.set_alpha(0)
    plt.sca(ax1)
    ax1.set_ylim(0, np.pi)
    ax1.xaxis.set_ticklabels(['$\dfrac{%d}{8}2\pi$' % i0 for i0 in np.arange(8)])
    ax1.yaxis.set_ticklabels([])
    plt.xticks(fontsize=fontsize * 0.5)
    plt.yticks(fontsize=fontsize * 0.5)
    norm = plt.Normalize(Table_t.min(), Table_t.max())
    cmap = plt.get_cmap('jet')
    ax1.plot(Table_phi, Table_theta, '-', alpha=0.2)
    ax1.scatter(Table_phi[0], Table_theta[0], c='k', s=fontsize * 6, marker='*')
    lc = ax1.scatter(Table_phi, Table_theta, c=Table_t, cmap=cmap, norm=norm, s=fontsize * 0.2)
    clb = fig.colorbar(lc, cax=cax0, orientation="vertical")
    clb.ax.tick_params(labelsize=fontsize * 0.6)
    clb.ax.set_title('time', size=fontsize * 0.6)

    fig2 = plt.figure(figsize=(20, 20))
    fig2.patch.set_facecolor('white')
    ax0 = fig2.add_subplot(1, 1, 1, projection='3d')
    ax0.set_title('$P_1$', size=fontsize)
    cax0 = inset_axes(ax0, width="80%", height="5%", bbox_to_anchor=(0, 0.1, 1, 1),
                      loc=1, bbox_transform=ax0.transAxes, borderpad=0, )
    norm = plt.Normalize(Table_t.min(), Table_t.max())
    cmap = plt.get_cmap('jet')
    # Create the 3D-line collection object
    points = Table_P.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = Line3DCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(Table_t)
    ax0.add_collection3d(lc, zs=points[:, :, 2].flatten(), zdir='z')
    ax0.set_xlim(points[:, :, 0].min(), points[:, :, 0].max())
    ax0.set_ylim(points[:, :, 1].min(), points[:, :, 1].max())
    ax0.set_zlim(points[:, :, 2].min(), points[:, :, 2].max())
    spf.set_axes_equal(ax0)
    ax0.plot(np.ones_like(points[:, :, 0].flatten()) * ax0.get_xlim()[0], points[:, :, 1].flatten(),
             points[:, :, 2].flatten())
    ax0.plot(points[:, :, 0].flatten(), np.ones_like(points[:, :, 1].flatten()) * ax0.get_ylim()[1],
             points[:, :, 2].flatten())
    ax0.plot(points[:, :, 0].flatten(), points[:, :, 1].flatten(),
             np.ones_like(points[:, :, 2].flatten()) * ax0.get_zlim()[0])
    clb = fig2.colorbar(lc, cax=cax0, orientation="horizontal")
    clb.ax.tick_params(labelsize=fontsize)
    clb.ax.set_title('Sim, time', size=fontsize)
    plt.sca(ax0)
    ax0.set_xlabel('$x$', size=fontsize)
    ax0.set_ylabel('$y$', size=fontsize)
    ax0.set_zlabel('$z$', size=fontsize)
    plt.xticks(fontsize=fontsize * 0.8)
    plt.yticks(fontsize=fontsize * 0.8)
    for t in ax0.zaxis.get_major_ticks():
        t.label.set_fontsize(fontsize * 0.8)
    for spine in ax0.spines.values():
        spine.set_visible(False)
    plt.tight_layout()
    return fig, fig2


def show_theta_phi(Table_t, Table_dt, Table_X, Table_P, Table_P2,
                   Table_theta, Table_phi, Table_psi, Table_eta, fig=None,
                   show_back_direction=True):
    core_show_theta_phi(Table_t, Table_dt, Table_X, Table_P, Table_P2,
                        Table_theta, Table_phi, Table_psi, Table_eta, fig,
                        show_back_direction)
    return True


def save_theta_phi(filename, Table_t, Table_dt, Table_X, Table_P, Table_P2,
                   Table_theta, Table_phi, Table_psi, Table_eta, fig=None,
                   show_back_direction=True):
    fig, fig2 = core_show_theta_phi(Table_t, Table_dt, Table_X, Table_P, Table_P2,
                                    Table_theta, Table_phi, Table_psi, Table_eta, fig,
                                    show_back_direction)
    fig.savefig(filename + '_1', dpi=100)
    fig2.savefig(filename + '_2', dpi=100)
    return fig, fig2


def core_light_show_theta_phi(Table_t, Table_dt, Table_X, Table_P, Table_P2,
                              Table_theta, Table_phi, Table_psi, Table_eta,
                              fig=None):
    fontsize = 30
    if fig is None:
        fig = plt.figure(figsize=(5, 5), dpi=300)
    else:
        fig.clf()
    fig.patch.set_facecolor('white')
    ax0 = fig.add_subplot(111)
    ax0.set_xlim(-np.pi * 1.1, np.pi * 1.1)
    ax0.set_ylim(-np.pi * 1.1, np.pi * 1.1)
    ax0.axis('off')
    cax0 = colorbar.make_axes(ax0, orientation='vertical', aspect=20, shrink=0.6)[0]
    ax0.set_aspect('equal')

    # polar version of theta-phi
    ax1 = fig.add_axes(ax0.get_position(), projection='polar')
    ax1.patch.set_alpha(0)
    plt.sca(ax1)
    ax1.set_ylim(0, np.pi)
    ax1.xaxis.set_ticklabels(['$\dfrac{%d}{8}2\pi$' % i0 for i0 in np.arange(8)])
    ax1.yaxis.set_ticklabels([])
    plt.xticks(fontsize=fontsize * 0.5)
    plt.yticks(fontsize=fontsize * 0.5)
    norm = plt.Normalize(Table_t.min(), Table_t.max())
    cmap = plt.get_cmap('jet')
    ax1.plot(Table_phi, Table_theta, '-', alpha=0.2)
    ax1.scatter(Table_phi[0], Table_theta[0], c='k', s=fontsize * 6, marker='*')
    lc = ax1.scatter(Table_phi, Table_theta, c=Table_t, cmap=cmap, norm=norm, s=fontsize * 0.2)
    clb = fig.colorbar(lc, cax=cax0, orientation="vertical")
    clb.ax.tick_params(labelsize=fontsize * 0.6)
    clb.ax.set_title('time', size=fontsize * 0.6)
    # plt.sca(ax1)
    # plt.tight_layout()
    return fig


def light_show_theta_phi(Table_t, Table_dt, Table_X, Table_P, Table_P2,
                         Table_theta, Table_phi, Table_psi, Table_eta,
                         fig=None):
    core_light_show_theta_phi(Table_t, Table_dt, Table_X, Table_P, Table_P2,
                              Table_theta, Table_phi, Table_psi, Table_eta,
                              fig)
    return True


def light_save_theta_phi(filename, Table_t, Table_dt, Table_X, Table_P, Table_P2,
                         Table_theta, Table_phi, Table_psi, Table_eta,
                         fig=None):
    fig = core_light_show_theta_phi(Table_t, Table_dt, Table_X, Table_P, Table_P2,
                                    Table_theta, Table_phi, Table_psi, Table_eta,
                                    fig)
    fig.savefig(filename, dpi=300)
    return fig


def core_show_pickle_results(job_dir, theta, phi, table_name, fast_mode=0):
    tpick, _ = load_table_date_pickle(job_dir, theta, phi)
    Table_t = tpick['Table_t']
    Table_dt = tpick['Table_dt']
    Table_X = tpick['Table_X']
    Table_P = tpick['Table_P']
    Table_P2 = tpick['Table_P2']
    Table_theta = tpick['Table_theta']
    Table_phi = tpick['Table_phi']
    Table_psi = tpick['Table_psi']
    Table_eta = tpick['Table_eta']
    print('-ini_theta %f -ini_phi %f -ini_psi %f' %
          (tpick['Table_theta'][0], tpick['Table_phi'][0], tpick['Table_psi'][0]))

    freq_pk = get_major_fre(Table_t, Table_theta)
    idx = Table_t > Table_t.max() - 1 / freq_pk * 10

    if fast_mode == 0:
        show_theta_phi(Table_t[idx], Table_dt[idx], Table_X[idx], Table_P[idx],
                       Table_P2[idx],
                       Table_theta[idx], Table_phi[idx], Table_psi[idx], Table_eta[idx])
        show_theta_phi_psi_eta(Table_t[idx], Table_dt[idx], Table_X[idx], Table_P[idx],
                               Table_P2[idx],
                               Table_theta[idx], Table_phi[idx], Table_psi[idx],
                               Table_eta[idx])
        show_center_X(Table_t[idx], Table_dt[idx], Table_X[idx], Table_P[idx], Table_P2[idx],
                      Table_theta[idx], Table_phi[idx], Table_psi[idx], Table_eta[idx],
                      table_name=table_name)
    elif fast_mode == 1:
        show_table_result(Table_t[idx], Table_dt[idx], Table_X[idx], Table_P[idx], Table_P2[idx],
                          Table_theta[idx], Table_phi[idx], Table_psi[idx], Table_eta[idx],
                          save_every=1)
    elif fast_mode == 2:
        light_show_theta_phi(Table_t[idx], Table_dt[idx], Table_X[idx], Table_P[idx], Table_P2[idx],
                             Table_theta[idx], Table_phi[idx], Table_psi[idx], Table_eta[idx], )
    return True


def show_pickle_results(job_dir, theta, phi, table_name, fast_mode=0):
    core_show_pickle_results(job_dir, theta, phi, table_name, fast_mode=fast_mode)
    return True


def core_show_theta_phi_psi_eta(Table_t, Table_dt, Table_X, Table_P, Table_P2,
                                Table_theta, Table_phi, Table_psi, Table_eta,
                                fig=None, resampling_fct=2, fft_full_mode=False):
    fontsize = 40
    figsize = (20, 15)
    Table_t, Table_theta, Table_phi, Table_psi, Table_eta = \
        resampling_angle(Table_t, Table_theta, Table_phi, Table_psi, Table_eta, resampling_fct)

    if fig is None:
        fig = plt.figure(figsize=figsize)
    else:
        fig.clf()
    fig.patch.set_facecolor('white')
    axs = fig.subplots(nrows=4, ncols=2)
    for (ax0, ax1), ty1, ylab in zip(axs,
                                     (Table_theta, Table_phi, Table_psi, Table_eta),
                                     ('$\\theta / \pi$', '$\\phi / \pi$',
                                      '$\\psi / \pi$', '$\\eta / \pi$')):
        for i0, i1 in separate_angle_idx(ty1):
            ax0.plot(Table_t[i0:i1], ty1[i0:i1] / np.pi, '-', color='#1f77b4')
        ax0.set_ylabel(ylab, size=fontsize * 0.7)
        plt.sca(ax0)
        plt.xticks(fontsize=fontsize * 0.5)
        plt.yticks(fontsize=fontsize * 0.5)

        # find major frequrence and display
        idx = np.ones_like(Table_t, dtype=bool)
        if not fft_full_mode:
            idx[:-20000] = False
        tfft = np.fft.rfft(ty1[idx])
        tfft_abs = np.abs(tfft)
        # noinspection PyTypeChecker
        tfreq = np.fft.rfftfreq(Table_t[idx].size, np.mean(np.diff(Table_t[idx])))
        ax1.loglog(tfreq, tfft_abs, '.')
        tpk = signal.find_peaks(tfft_abs)[0]
        if tpk.size > 0:
            fft_abs_pk = tfft_abs[tpk]
            freq_pk = tfreq[tpk]
            tidx = np.argsort(fft_abs_pk)[-1]
            ax1.text(freq_pk[tidx], fft_abs_pk[tidx], '$%.5f$' % freq_pk[tidx],
                     fontsize=fontsize * 0.7)
            ax1.loglog(freq_pk[tidx], fft_abs_pk[tidx], '*', ms=fontsize * 0.5)
        plt.yticks(fontsize=fontsize * 0.5)
    axs[-1, 0].set_xlabel('$t$', size=fontsize * 0.7)
    axs[-1, 1].set_xlabel('$Hz$', size=fontsize * 0.7)
    plt.tight_layout()
    return fig


def show_theta_phi_psi_eta(Table_t, Table_dt, Table_X, Table_P, Table_P2,
                           Table_theta, Table_phi, Table_psi, Table_eta,
                           fig=None, resampling_fct=2, fft_full_mode=False):
    core_show_theta_phi_psi_eta(Table_t, Table_dt, Table_X, Table_P, Table_P2,
                                Table_theta, Table_phi, Table_psi, Table_eta,
                                fig, resampling_fct, fft_full_mode)
    return True


def save_theta_phi_psi_eta(filename, Table_t, Table_dt, Table_X, Table_P, Table_P2,
                           Table_theta, Table_phi, Table_psi, Table_eta,
                           fig=None, resampling_fct=2, fft_full_mode=False):
    fig = core_show_theta_phi_psi_eta(Table_t, Table_dt, Table_X, Table_P, Table_P2,
                                      Table_theta, Table_phi, Table_psi, Table_eta,
                                      fig, resampling_fct, fft_full_mode)
    fig.savefig(filename, dpi=100)
    return fig


def core_show_center_X(Table_t, Table_dt, Table_X, Table_P, Table_P2,
                       Table_theta, Table_phi, Table_psi, Table_eta, table_name,
                       move_z=False, planeShearRate=np.array((1, 0, 0)), fig=None,
                       resampling=False, resampling_fct=2):
    fontsize = 40
    figsize = (20, 15)
    if move_z:
        z_mean = np.mean(Table_X[:, 2])
        Table_X[:, 2] = Table_X[:, 2] - z_mean
        ux_shear = z_mean * planeShearRate[0]
        Xz_mean = (Table_t - Table_t[0]) * ux_shear
        Table_X[:, 0] = Table_X[:, 0] - Xz_mean
    if resampling:
        Table_t, Table_dt, Table_X, Table_P, Table_P2, \
        Table_theta, Table_phi, Table_psi, Table_eta = \
            resampling_data(Table_t, Table_dt, Table_X, Table_P, Table_P2,
                            Table_theta, Table_phi, Table_psi, Table_eta, resampling_fct)

    # get velocity from table
    norm = np.array((0, 0, 1))
    P0, P20, tcenter, problem = do_calculate_prepare(norm)
    tkwargs = do_ellipse_kwargs(tcenter=tcenter, P0=P0, P20=P20, ini_psi=0, table_name=table_name)
    tobj = jm.TableObj(**tkwargs)
    problem.add_obj(tobj)
    Table_dX_rel = []
    for X, theta, phi, psi in zip(Table_X, Table_theta, Table_phi, Table_psi):
        # ref_U = tobj.get_velocity_at(X, P, P2, check_orthogonality=False)
        ref_U = tobj.get_velocity_at3(X, theta, phi, psi)
        Ub = problem.flow_velocity(X)
        rel_U = ref_U - np.hstack((Ub, np.zeros(3)))
        Table_dX_rel.append(rel_U)
    Table_dX_rel = np.vstack(Table_dX_rel)
    # relative translational and rotational velocities at norm direction
    up_rel = np.array([np.dot(P, U[:3]) for (P, U) in zip(Table_P, Table_dX_rel)])
    wp_rel = np.array([np.dot(P, U[3:]) for (P, U) in zip(Table_P, Table_dX_rel)])

    if fig is None:
        fig = plt.figure(figsize=figsize)
    else:
        fig.clf()
    fig.patch.set_facecolor('white')
    axs = fig.subplots(nrows=5, ncols=1)
    # center and velocity
    for ax0, ty1, ty2, ylab1, ylab2 in zip(axs, Table_X.T, Table_dX_rel.T,
                                           ('$x$', '$y$', '$z$'),
                                           ('$u_x-u_{fx}$', '$u_y-u_{fy}$', '$u_z-u_{fz}$')):
        color = 'tab:red'
        ax0.plot(Table_t, ty1, '-', color=color)
        ax0.set_ylabel(ylab1, size=fontsize * 0.7, color=color)
        ax0.tick_params(axis='y', labelcolor=color)
        plt.sca(ax0)
        plt.xticks(fontsize=fontsize * 0.5)
        plt.yticks(fontsize=fontsize * 0.5)
        ax1 = ax0.twinx()
        color = 'tab:blue'
        ax1.plot(Table_t, ty2, '-', color=color)
        ax1.set_ylabel(ylab2, size=fontsize * 0.7, color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        plt.sca(ax1)
        plt.xticks(fontsize=fontsize * 0.5)
        plt.yticks(fontsize=fontsize * 0.5)
    # translational and rotational velocity at norm direction
    ax0 = axs[3]
    color = 'tab:red'
    ax0.plot(Table_t, up_rel, '-', color=color)
    ax0.set_ylabel('$\\bm{u}_p = \\bm{u} \\cdot \\bm{p}$', size=fontsize * 0.7, color=color)
    ax0.tick_params(axis='y', labelcolor=color)
    plt.sca(ax0)
    plt.xticks(fontsize=fontsize * 0.5)
    plt.yticks(fontsize=fontsize * 0.5)
    ax1 = ax0.twinx()
    color = 'tab:blue'
    ax1.plot(Table_t, wp_rel, '-', color=color)
    ax1.set_ylabel('$\\bm{\omega}_{bp} = \\bm{\omega}_b \\cdot \\bm{p}$',
                   size=fontsize * 0.7, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    plt.sca(ax1)
    plt.xticks(fontsize=fontsize * 0.5)
    plt.yticks(fontsize=fontsize * 0.5)
    ax0 = axs[4]
    ax0.plot(Table_t, wp_rel / up_rel, '.')
    ax0.set_ylabel('$\\bm{\omega}_{bp} / \\bm{u}_p$', size=fontsize * 0.7)
    ax0.set_yscale('symlog', linthreshy=0.01)
    t1 = np.max((1, ax0.get_yticks().size // 4))
    tticks = ax0.get_yticks()[::t1]
    ax0.set_yticks(tticks)
    ax0.set_yticklabels(tticks)
    ax0.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0e'))
    fig.tight_layout()
    ax0.set_xlabel('t', size=fontsize * 0.7)
    plt.sca(ax0)
    plt.xticks(fontsize=fontsize * 0.5)
    plt.yticks(fontsize=fontsize * 0.5)
    plt.tight_layout()
    return fig


def show_center_X(Table_t, Table_dt, Table_X, Table_P, Table_P2,
                  Table_theta, Table_phi, Table_psi, Table_eta, table_name,
                  move_z=False, planeShearRate=np.array((1, 0, 0)), fig=None,
                  resampling=False, resampling_fct=2):
    core_show_center_X(Table_t, Table_dt, Table_X, Table_P, Table_P2,
                       Table_theta, Table_phi, Table_psi, Table_eta, table_name,
                       move_z, planeShearRate, fig, resampling, resampling_fct)
    return True


def save_center_X(filename, Table_t, Table_dt, Table_X, Table_P, Table_P2,
                  Table_theta, Table_phi, Table_psi, Table_eta, table_name,
                  move_z=False, planeShearRate=np.array((1, 0, 0)), fig=None,
                  resampling=False, resampling_fct=2):
    fig = core_show_center_X(Table_t, Table_dt, Table_X, Table_P, Table_P2,
                             Table_theta, Table_phi, Table_psi, Table_eta, table_name,
                             move_z, planeShearRate, fig, resampling, resampling_fct)
    fig.savefig(filename, dpi=100)
    return fig


def get_continue_angle(tx, ty1, t_use=None):
    ty = ty1.copy()
    if t_use is None:
        t_use = np.linspace(tx.min(), tx.max(), 2 * tx.size)

    for i0, dt in enumerate(np.diff(ty)):
        if dt > np.pi:
            ty[i0 + 1:] = ty[i0 + 1:] - 2 * np.pi
        elif dt < -np.pi:
            ty[i0 + 1:] = ty[i0 + 1:] + 2 * np.pi
    intp_fun1d = interpolate.interp1d(tx, ty, kind='quadratic', copy=False, axis=0,
                                      bounds_error=True)
    return intp_fun1d(t_use) % (2 * np.pi)


def get_major_fre(tx, ty1, fft_full_mode=False):
    freq_pk = get_primary_fft_fre(tx, ty1, fft_full_mode=fft_full_mode)
    return freq_pk[-1]


def get_primary_fft_fre(tx, ty1, fft_full_mode=False):
    idx = np.ones_like(tx, dtype=bool)
    if not fft_full_mode:
        idx[:-20000] = False
    t_use = np.linspace(tx[idx].min(), tx[idx].max(), tx[idx].size)
    ty = get_continue_angle(tx[idx], ty1[idx], t_use)
    tfft = np.fft.rfft(ty)
    tfft_abs = np.abs(tfft)
    # noinspection PyTypeChecker
    tfreq = np.fft.rfftfreq(t_use.size, np.mean(np.diff(t_use)))
    tpk = signal.find_peaks(tfft_abs)[0]
    fft_abs_pk = tfft_abs[tpk]
    freq_pk = tfreq[tpk]
    tidx = np.argsort(fft_abs_pk)
    return freq_pk[tidx]


def separate_angle_idx(ty):
    # separate to small components to avoid the jump between 0 and 2pi.
    idx_list = []
    dty = np.diff(ty)
    idx_list.append(np.argwhere(dty > np.pi).flatten())
    idx_list.append(np.argwhere(dty < -np.pi).flatten())
    idx_list.append(-1)  # first idx is 0, but later will plus 1.
    idx_list.append(ty.size - 1)  # last idx is (size-1).
    t1 = np.sort(np.hstack(idx_list))
    return np.vstack((t1[:-1] + 1, t1[1:])).T


def resampling_data(Table_t, Table_dt, Table_X, Table_P, Table_P2,
                    Table_theta, Table_phi, Table_psi, Table_eta, resampling_fct=2):
    def intp_fun(ty):
        intp_fun1d = interpolate.interp1d(Table_t, ty, kind='quadratic', copy=False, axis=0,
                                          bounds_error=True)
        return intp_fun1d(t_use)

    # resampling the date to a uniform distance
    # noinspection PyTypeChecker
    t_use = np.linspace(Table_t.min(), Table_t.max(), np.around(Table_t.size * resampling_fct))
    Table_X = intp_fun(Table_X)
    Table_P = intp_fun(Table_P)
    Table_P2 = intp_fun(Table_P2)
    Table_dt = intp_fun(Table_dt)
    Table_theta = get_continue_angle(Table_t, Table_theta, t_use)
    Table_phi = get_continue_angle(Table_t, Table_phi, t_use)
    Table_psi = get_continue_angle(Table_t, Table_psi, t_use)
    Table_eta = np.arccos(np.sin(Table_theta) * np.sin(Table_phi))
    Table_t = t_use
    return Table_t, Table_dt, Table_X, Table_P, Table_P2, \
           Table_theta, Table_phi, Table_psi, Table_eta


def resampling_angle(Table_t, Table_theta, Table_phi, Table_psi, Table_eta, resampling_fct=2):
    # resampling the date to a uniform distance
    # noinspection PyTypeChecker
    t_use = np.linspace(Table_t.min(), Table_t.max(), np.around(Table_t.size * resampling_fct))
    Table_theta = get_continue_angle(Table_t, Table_theta, t_use)
    Table_phi = get_continue_angle(Table_t, Table_phi, t_use)
    Table_psi = get_continue_angle(Table_t, Table_psi, t_use)
    Table_eta = np.arccos(np.sin(Table_theta) * np.sin(Table_phi))
    Table_t = t_use
    return Table_t, Table_theta, Table_phi, Table_psi, Table_eta


def make_table_video(Table_t, Table_X, Table_P, Table_P2,
                     Table_theta, Table_phi, Table_psi, Table_eta,
                     zm_fct=1, stp=1, interval=50, trange=None, resampling_fct=2):
    fontsize = 35
    figsize = (25, 15)

    def update_fun(num, tl1, tl2, tl3, scs, Table_t, Table_X, Table_P, Table_P2,
                   Table_theta, Table_phi, Table_psi, Table_eta, zm_fct):
        num = num * stp
        tqdm_fun.update(1)
        # print('update_fun', num)

        # left, 3d trajection
        tX = Table_X[num]
        tP1 = Table_P[num]
        tP2 = Table_P2[num]
        tP1 = tP1 / np.linalg.norm(tP1) * zm_fct
        tP2 = tP2 / np.linalg.norm(tP2) * zm_fct
        tP3 = np.cross(tP1, tP2) / zm_fct
        t1 = np.vstack([tX, tX + tP1])
        tl1.set_data(t1[:, 0], t1[:, 1])
        tl1.set_3d_properties(t1[:, 2])
        t2 = np.vstack([tX, tX + tP2])
        tl2.set_data(t2[:, 0], t2[:, 1])
        tl2.set_3d_properties(t2[:, 2])
        t3 = np.vstack([tX, tX + tP3])
        tl3.set_data(t3[:, 0], t3[:, 1])
        tl3.set_3d_properties(t3[:, 2])

        # right, theta-phi
        scs[0].set_data(Table_phi[num], Table_theta[num])
        # right, other 2d plots
        for axi, ty, sci, in zip((ax3, ax4, ax5, ax6),
                                 (Table_psi / np.pi, Table_X[:, 0], Table_X[:, 1], Table_X[:, 2]),
                                 scs[1:]):
            sci.set_data(Table_t[num], ty[num])
        return tl1, tl2, tl3, scs

    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor('white')
    ax0 = plt.subplot2grid((6, 8), (0, 0), rowspan=6, colspan=6, projection='3d')
    ax6 = plt.subplot2grid((6, 8), (5, 6), colspan=2)  # Table_X[:, 2]
    # ax1 = plt.subplot2grid((6, 8), (0, 6), colspan=2, sharex=ax6) #Table_theta
    # ax2 = plt.subplot2grid((6, 8), (1, 6), colspan=2, sharex=ax6) #Table_phi
    axth_ph = plt.subplot2grid((6, 8), (0, 6), rowspan=2, colspan=2,
                               projection='polar')  # Table_theta-#Table_phi
    ax3 = plt.subplot2grid((6, 8), (2, 6), colspan=2, sharex=ax6)  # Table_psi
    ax4 = plt.subplot2grid((6, 8), (3, 6), colspan=2, sharex=ax6)  # Table_X[:, 0]
    ax5 = plt.subplot2grid((6, 8), (4, 6), colspan=2, sharex=ax6)  # Table_X[:, 1]
    for spine in ax0.spines.values():
        spine.set_visible(False)

    # left part, animate of axis (which represent the object, i.e. helix, ecoli...)
    tX = Table_X[0]
    tP1 = Table_P[0]
    tP2 = Table_P2[0]
    tP1 = tP1 / np.linalg.norm(tP1) * zm_fct
    tP2 = tP2 / np.linalg.norm(tP2) * zm_fct
    tP3 = np.cross(tP1, tP2) / zm_fct
    tmp_line1 = ax0.plot([tX[0], tX[0] + tP1[0]],
                         [tX[1], tX[1] + tP1[1]],
                         [tX[2], tX[2] + tP1[2]], color='k', lw=fontsize * 0.1)[0]
    tmp_line2 = ax0.plot([tX[0], tX[0] + tP2[0]],
                         [tX[1], tX[1] + tP2[1]],
                         [tX[2], tX[2] + tP2[2]], color='r')[0]
    tmp_line3 = ax0.plot([tX[0], tX[0] + tP3[0]],
                         [tX[1], tX[1] + tP3[1]],
                         [tX[2], tX[2] + tP3[2]], color='b')[0]
    if trange is None:
        trange = np.max(Table_X.max(axis=0) - Table_X.min(axis=0))
    print('trange=', trange)
    tmid = (Table_X.max(axis=0) + Table_X.min(axis=0)) / 2
    ax0.set_xlim3d([tmid[0] - trange, tmid[0] + trange])
    tticks = np.around(np.linspace(tmid[0] - trange, tmid[0] + trange, 21), decimals=2)[1::6]
    ax0.set_xticks(tticks)
    ax0.set_xticklabels(tticks)
    ax0.set_xlabel('X')
    ax0.set_ylim3d([tmid[1] - trange, tmid[1] + trange])
    tticks = np.around(np.linspace(tmid[1] - trange, tmid[1] + trange, 21), decimals=2)[1::6]
    ax0.set_xticks(tticks)
    ax0.set_xticklabels(tticks)
    ax0.set_ylabel('Y')
    ax0.set_zlim3d([tmid[2] - trange, tmid[2] + trange])
    tticks = np.around(np.linspace(tmid[2] - trange, tmid[2] + trange, 21), decimals=2)[1::6]
    ax0.set_xticks(tticks)
    ax0.set_xticklabels(tticks)
    ax0.set_zlabel('Z')

    # right part, standard part
    # theta-phi
    plt.sca(axth_ph)
    axth_ph.plot(Table_phi, Table_theta, '-.', alpha=0.5)
    axth_ph.set_ylim(0, np.pi)
    plt.xticks(fontsize=fontsize * 0.5)
    plt.yticks(fontsize=fontsize * 0.5)
    xticks = np.around(np.linspace(Table_t.min(), Table_t.max(), 21), decimals=2)[1::6]
    # xticks = np.linspace(Table_t.min(), Table_t.max(), 3)
    # other variables
    for axi, ty, axyi in zip((ax3, ax4, ax5, ax6),
                             (Table_psi / np.pi, Table_X[:, 0], Table_X[:, 1], Table_X[:, 2]),
                             ('$\\psi / \pi$', '$X$', '$Y$', '$Z$')):
        plt.sca(axi)
        axi.plot(Table_t, ty, '-.', label='Table')
        axi.set_ylabel('%s' % axyi, size=fontsize * 0.7)
        axi.set_xticks(xticks)
        axi.set_xticklabels(xticks)
        plt.xticks(fontsize=fontsize * 0.5)
        plt.yticks(fontsize=fontsize * 0.5)
    for axi in (ax6,):
        axi.set_xlabel('t', size=fontsize * 0.7)
    plt.tight_layout()

    # right part, point indicates the time.
    scs = []
    scs.append(axth_ph.plot(Table_phi[0], Table_theta[0], 'or', markersize=fontsize * 0.3)[0])
    for axi, ty, in zip((ax3, ax4, ax5, ax6),
                        (Table_psi / np.pi, Table_X[:, 0], Table_X[:, 1], Table_X[:, 2])):
        plt.sca(axi)
        scs.append(axi.plot(Table_t[0], ty[0], 'or', markersize=fontsize * 0.3)[0])

    Table_dt = np.hstack((np.diff(Table_t), 0))
    Table_t, Table_dt, Table_X, Table_P, Table_P2, Table_theta, Table_phi, Table_psi, Table_eta \
        = resampling_data(Table_t, Table_dt, Table_X, Table_P, Table_P2, Table_theta, Table_phi,
                          Table_psi, Table_eta, resampling_fct)
    video_length = Table_t.size // stp
    tqdm_fun = tqdm_notebook(total=video_length + 2)
    anim = animation.FuncAnimation(fig, update_fun, video_length, interval=interval, blit=False,
                                   fargs=(tmp_line1, tmp_line2, tmp_line3, scs,
                                          Table_t, Table_X, Table_P, Table_P2,
                                          Table_theta, Table_phi, Table_psi, Table_eta,
                                          zm_fct), )
    return anim


def make_table_video_geo(Table_t, Table_X, Table_P, Table_P2,
                         Table_theta, Table_phi, Table_psi, Table_eta, move_z=False,
                         zm_fct=1, stp=1, interval=50, trange_geo=None, trange_trj=None,
                         create_obj_at_fun=get_tail_nodes_split_at, resampling_fct=2,
                         **problem_kwargs):
    fontsize = 35
    figsize = (25, 15)
    if move_z:
        z_mean = np.mean(Table_X[:, 2])
        Table_X[:, 2] = Table_X[:, 2] - z_mean
        planeShearRate = problem_kwargs['planeShearRate'][0]
        ux_shear = z_mean * planeShearRate[0]
        Xz_mean = (Table_t - Table_t[0]) * ux_shear
        Table_X[:, 0] = Table_X[:, 0] - Xz_mean

    def update_fun(num, tmp_line1, tmp_line2, tmp_trj, scs, Table_t, Table_X, Table_P, Table_P2,
                   Table_theta, Table_phi, Table_psi, Table_eta, zm_fct):
        num = num * stp
        tqdm_fun.update(1)
        # print('update_fun', num)

        # left, 3d orientation
        ttheta = Table_theta[num]
        tphi = Table_phi[num]
        tpsi = Table_psi[num]
        tnode1, tnode2 = create_obj_at_fun(ttheta, tphi, tpsi, now_center=np.zeros(3),
                                           **problem_kwargs)
        tmp_line1.set_data(tnode1[:, 0], tnode1[:, 1])
        tmp_line1.set_3d_properties(tnode1[:, 2])
        tmp_line2.set_data(tnode2[:, 0], tnode2[:, 1])
        tmp_line2.set_3d_properties(tnode2[:, 2])
        # left, 3d trajectory
        tX = Table_X[num]
        tmp_trj.set_data(tX[0], tX[1])
        tmp_trj.set_3d_properties(tX[2])

        # right, theta-phi
        scs[0].set_data(Table_phi[num], Table_theta[num])
        # right, other 2d plots
        for axi, ty, sci, in zip((ax3, ax4, ax5, ax6),
                                 (Table_psi / np.pi, Table_X[:, 0], Table_X[:, 1], Table_X[:, 2]),
                                 scs[1:]):
            sci.set_data(Table_t[num], ty[num])
        # return tmp_line, tmp_trj, scs

    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor('white')
    ax0 = plt.subplot2grid((6, 8), (0, 0), rowspan=6, colspan=6, projection='3d')
    axtrj = fig.add_axes((0, 0.6, 0.25, 0.35), projection='3d')
    ax6 = plt.subplot2grid((6, 8), (5, 6), colspan=2)  # Table_X[:, 2]
    axth_ph = plt.subplot2grid((6, 8), (0, 6), rowspan=2, colspan=2,
                               projection='polar')  # Table_theta-#Table_phi
    ax3 = plt.subplot2grid((6, 8), (2, 6), colspan=2, sharex=ax6)  # Table_psi
    ax4 = plt.subplot2grid((6, 8), (3, 6), colspan=2, sharex=ax6)  # Table_X[:, 0]
    ax5 = plt.subplot2grid((6, 8), (4, 6), colspan=2, sharex=ax6)  # Table_X[:, 1]
    for spine in ax0.spines.values():
        spine.set_visible(False)
    for spine in axtrj.spines.values():
        spine.set_visible(False)
    axtrj.patch.set_alpha(0.2)

    # left part, animate of axis (which represent the object, i.e. helix, ecoli...)
    # object orientation
    ttheta = Table_theta[0]
    tphi = Table_phi[0]
    tpsi = Table_psi[0]
    tnode1, tnode2 = create_obj_at_fun(ttheta, tphi, tpsi, now_center=np.zeros(3), **problem_kwargs)
    tmp_line1 = ax0.plot(tnode1[:, 0], tnode1[:, 1], tnode1[:, 2])[0]
    tmp_line2 = ax0.plot(tnode2[:, 0], tnode2[:, 1], tnode2[:, 2])[0]
    if trange_geo is None:
        tnode = np.vstack((tnode1, tnode2))
        trange_geo = np.linalg.norm(tnode.max(axis=0) - tnode.min(axis=0))
    print('trange_geo=', trange_geo)
    tmid = np.zeros(3)
    ax0.set_xlim3d([tmid[0] - trange_geo, tmid[0] + trange_geo])
    tticks = np.around(np.linspace(tmid[0] - trange_geo, tmid[0] + trange_geo, 21),
                       decimals=2)[1::6]
    ax0.set_xticks(tticks)
    ax0.set_xticklabels(tticks)
    ax0.set_xlabel('X')
    ax0.set_ylim3d([tmid[1] - trange_geo, tmid[1] + trange_geo])
    tticks = np.around(np.linspace(tmid[1] - trange_geo, tmid[1] + trange_geo, 21),
                       decimals=2)[1::6]
    ax0.set_yticks(tticks)
    ax0.set_yticklabels(tticks)
    ax0.set_ylabel('Y')
    ax0.set_zlim3d([tmid[2] - trange_geo, tmid[2] + trange_geo])
    tticks = np.around(np.linspace(tmid[2] - trange_geo, tmid[2] + trange_geo, 21),
                       decimals=2)[1::6]
    ax0.set_zticks(tticks)
    ax0.set_zticklabels(tticks)
    ax0.set_zlabel('Z')
    # object trajectory
    tX = Table_X[0]
    axtrj.plot(Table_X[:, 0], Table_X[:, 1], Table_X[:, 2], '-.')  # stable part
    tmp_trj = axtrj.plot((tX[0],), (tX[1],), (tX[2],), 'or', markersize=fontsize * 0.3)[0]
    if trange_trj is None:
        trange_trj = np.max(Table_X.max(axis=0) - Table_X.min(axis=0))
    print('trange_trj=', trange_trj)
    tmid = (Table_X.max(axis=0) + Table_X.min(axis=0)) / 2
    axtrj.set_xlim3d([tmid[0] - trange_trj, tmid[0] + trange_trj])
    tticks = np.around(np.linspace(tmid[0] - trange_trj, tmid[0] + trange_trj, 8),
                       decimals=2)[[1, -2]]
    axtrj.set_xticks(tticks)
    axtrj.set_xticklabels(tticks)
    axtrj.set_xlabel('X')
    axtrj.set_ylim3d([tmid[1] - trange_trj, tmid[1] + trange_trj])
    tticks = np.around(np.linspace(tmid[1] - trange_trj, tmid[1] + trange_trj, 8),
                       decimals=2)[[1, -2]]
    axtrj.set_yticks(tticks)
    axtrj.set_yticklabels(tticks)
    axtrj.set_ylabel('Y')
    axtrj.set_zlim3d([tmid[2] - trange_trj, tmid[2] + trange_trj])
    tticks = np.around(np.linspace(tmid[2] - trange_trj, tmid[2] + trange_trj, 8),
                       decimals=2)[[1, -2]]
    axtrj.set_zticks(tticks)
    axtrj.set_zticklabels(tticks)
    axtrj.set_zlabel('Z')

    # right part, standard part
    # theta-phi
    plt.sca(axth_ph)
    axth_ph.plot(Table_phi, Table_theta, '-.', alpha=0.5)
    axth_ph.set_ylim(0, np.pi)
    plt.xticks(fontsize=fontsize * 0.5)
    plt.yticks(fontsize=fontsize * 0.5)
    xticks = np.around(np.linspace(Table_t.min(), Table_t.max(), 8), decimals=2)[1::6]
    # other variables
    for axi, ty, axyi in zip((ax3, ax4, ax5, ax6),
                             (Table_psi / np.pi, Table_X[:, 0], Table_X[:, 1], Table_X[:, 2]),
                             ('$\\psi / \pi$', '$X$', '$Y$', '$Z$')):
        plt.sca(axi)
        axi.plot(Table_t, ty, '-.', label='Table')
        axi.set_ylabel('%s' % axyi, size=fontsize * 0.7)
        axi.set_xticks(xticks)
        axi.set_xticklabels(xticks)
        plt.xticks(fontsize=fontsize * 0.5)
        plt.yticks(fontsize=fontsize * 0.5)
        for axi in (ax6,):
            axi.set_xlabel('t', size=fontsize * 0.7)
        plt.tight_layout()

    # right part, point indicates the time.
    scs = []
    scs.append(axth_ph.plot(Table_phi[0], Table_theta[0], 'or', markersize=fontsize * 0.3)[0])
    for axi, ty, in zip((ax3, ax4, ax5, ax6),
                        (Table_psi / np.pi, Table_X[:, 0], Table_X[:, 1], Table_X[:, 2])):
        plt.sca(axi)
        scs.append(axi.plot(Table_t[0], ty[0], 'or', markersize=fontsize * 0.3)[0])

    Table_dt = np.hstack((np.diff(Table_t), 0))
    Table_t, Table_dt, Table_X, Table_P, Table_P2, Table_theta, Table_phi, Table_psi, Table_eta \
        = resampling_data(Table_t, Table_dt, Table_X, Table_P, Table_P2, Table_theta, Table_phi,
                          Table_psi, Table_eta, resampling_fct)
    video_length = Table_t.size // stp
    tqdm_fun = tqdm_notebook(total=video_length + 2)
    fargs = (tmp_line1, tmp_line2, tmp_trj, scs, Table_t, Table_X, Table_P, Table_P2,
             Table_theta, Table_phi, Table_psi, Table_eta, zm_fct)
    anim = animation.FuncAnimation(fig, update_fun, video_length, interval=interval, blit=False,
                                   fargs=fargs, )
    return anim


def load_problem_kwargs(pickle_name):
    pickle_name = check_file_extension(pickle_name, extension='.pickle')
    t_path = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.normpath(t_path + '/' + pickle_name)
    with open(full_path, 'rb') as handle:
        problem_kwargs = pickle.load(handle)
    return problem_kwargs


def load_table_date_pickle(job_dir, theta, phi):
    t_headle = 'th%5.3f_ph%5.3f_(.*?).pickle' % (theta, phi)
    filename = [filename for filename in os.listdir(job_dir)
                if re.match(t_headle, filename) is not None][0]
    with open(os.path.join(PWD, job_dir, filename), 'rb') as handle:
        tpick = pickle.load(handle)
    if 'Table_dt' not in tpick.keys():
        Table_dt = np.hstack((np.diff(tpick['Table_t']), 0))
        tpick['Table_dt'] = Table_dt
    return tpick, filename


def _do_plot_process(args):
    job_dir, dirpath, filename, theta, phi, pick_fre = args
    pick_name = os.path.join(job_dir, filename)
    with open(pick_name, 'rb') as handle:
        tpick = pickle.load(handle)
    if 'Table_dt' not in tpick.keys():
        tpick['Table_dt'] = np.hstack((np.diff(tpick['Table_t']), 0))

    # print('%s, Fth=%.6f' % (filename, pick_fre))
    tname = os.path.splitext(os.path.basename(filename))[0]
    filename = os.path.join(dirpath, tname)
    tmin = tpick['Table_t'].max() - 1 / pick_fre * 10
    idx = tpick['Table_t'] > tmin
    fig0 = save_table_result('%s_1.jpg' % filename,
                             tpick['Table_t'][idx], tpick['Table_dt'][idx],
                             tpick['Table_X'][idx],
                             tpick['Table_P'][idx], tpick['Table_P2'][idx],
                             tpick['Table_theta'][idx], tpick['Table_phi'][idx],
                             tpick['Table_psi'][idx], tpick['Table_eta'][idx])
    fig1 = save_theta_phi_psi_eta('%s_2.jpg' % filename,
                                  tpick['Table_t'][idx], tpick['Table_dt'][idx],
                                  tpick['Table_X'][idx],
                                  tpick['Table_P'][idx], tpick['Table_P2'][idx],
                                  tpick['Table_theta'][idx], tpick['Table_phi'][idx],
                                  tpick['Table_psi'][idx], tpick['Table_eta'][idx])
    plt.close(fig0)
    plt.close(fig1)
    return True


def _save_separate_angle_fft(job_dir, dirpath, tfre, tidx):
    # clear dir
    if os.path.exists(dirpath) and os.path.isdir(dirpath):
        shutil.rmtree(dirpath)
        print('remove folder %s' % dirpath)
    os.makedirs(dirpath)
    print('make folder %s' % dirpath)

    pickle_info_list = []
    tfre_shape = tfre.values.shape
    tfre_idx_list = tfre.unstack().index.to_numpy().reshape(tfre_shape[1], tfre_shape[0])
    for phi, theta in tfre_idx_list[tidx]:
        t_headle = 'th%5.3f_ph%5.3f_(.*?).pickle' % (theta, phi)
        filenames = [filename for filename in os.listdir(job_dir)
                     if re.match(t_headle, filename) is not None]
        pick_fre = tfre.loc[theta].loc[phi]
        for filename in filenames:
            pickle_info_list.append((job_dir, dirpath, filename, theta, phi, pick_fre))

    # # multi process version, ignore becouse sometimes have unknow error.
    # pool = multiprocessing.Pool()
    # for _ in tqdm_notebook(pool.imap_unordered(_do_plot_process, pickle_info_list),
    #                        total=len(pickle_info_list)):
    #     pass

    # single process version
    for pickle_info in tqdm_notebook(pickle_info_list):
        # print(pickle_info)
        _do_plot_process(pickle_info)
    return True


def save_separate_angle_fft(job_dir, tfre, check_fre, atol_fre):
    use_idx = np.isclose(tfre, check_fre, rtol=0, atol=atol_fre).T
    fre_subdir = 'fre_%f' % check_fre
    dirpath = os.path.join(job_dir, 'fre_separate', fre_subdir)
    print('frequency in the range (%f, %f)' % (check_fre - atol_fre, check_fre + atol_fre))
    _save_separate_angle_fft(job_dir, dirpath, tfre, use_idx)
    return use_idx


def save_separate_angleList_fft(job_dir, tfre, check_fre_list, atol_fre_list):
    remaind_idx = np.ones_like(tfre, dtype=bool).T
    for check_fre, atol_fre in zip(check_fre_list, atol_fre_list):
        use_idx = save_separate_angle_fft(job_dir, tfre, check_fre, atol_fre)
        # use_idx = np.isclose(tfre, check_fre, rtol=0, atol=atol_fre).T
        remaind_idx[use_idx] = False

    # process the remainders
    if np.any(remaind_idx):
        dirpath = os.path.join(job_dir, 'fre_separate', 'remainders')
        _save_separate_angle_fft(job_dir, dirpath, tfre, remaind_idx)
    return True


def separate_fre_path(check_fre_list, atol_list, data0, pickle_path_list):
    for i0, (check_fre, atol) in enumerate(zip(check_fre_list, atol_list)):
        print('%dth frequence range: (%f, %f)' % (i0, check_fre - atol, check_fre + atol))

    case_path_list = [[] for ti in check_fre_list]
    for i0 in data0.index:
        datai = data0.loc[i0]
        tdata_idx = int(datai.data_idx)
        tmax_fre = datai.use_max_fre
        tpath = pickle_path_list[tdata_idx]

        n_match = 0
        for check_fre, atol, case_path in zip(check_fre_list, atol_list, case_path_list):
            if np.isclose(tmax_fre, check_fre, rtol=0, atol=atol):
                case_path.append(tpath)
                n_match = n_match + 1
        if not np.isclose(n_match, 1):
            print('tmax_fre=%f, n_match=%d' % (tmax_fre, n_match), tpath)
    return case_path_list


def draw_phase_map_theta(case_path, color, psi_lim, axs=None,
                         resampling=False, resampling_fct=2, thandle=''):
    fontsize = 40
    if axs is None:
        n_xticks = 32
        xticks = np.arange(n_xticks)
        fig = plt.figure(figsize=(20, 20))
        fig.patch.set_facecolor('white')
        ax0 = fig.add_subplot(221, polar=True)
        ax0.set_xticks(xticks / n_xticks * 2 * np.pi)
        ax0.set_xticklabels(['$\dfrac{%d}{%d}2\pi$' % (i0, n_xticks) for i0 in xticks])
        ax0.set_yticklabels([])
        ax0.set_ylim(0, np.pi)
        plt.tight_layout()
        axs = (ax0,)

    for tpath in tqdm_notebook(case_path[:], desc=thandle):
        with open(tpath, 'rb') as handle:
            tpick = pickle.load(handle)
        Table_t = tpick['Table_t']
        Table_theta = tpick['Table_theta']
        Table_phi = tpick['Table_phi']
        Table_psi = tpick['Table_psi']
        Table_eta = tpick['Table_eta']
        if resampling:
            Table_t, Table_theta, Table_phi, Table_psi, Table_eta = \
                resampling_angle(Table_t, Table_theta, Table_phi, Table_psi, Table_eta,
                                 resampling_fct)
        tidx = np.logical_and(Table_psi >= psi_lim[0], Table_psi < psi_lim[1])
        for ax0 in tube_flatten((axs,)):
            ax0.scatter(Table_phi[tidx], Table_theta[tidx], c=color, s=fontsize * 0.2)
    return axs


# show phase map of final trajectory in theta-phi space, using frequence.
def show_traj_phase_map_fre(tuse):
    fontsize = 40
    fig = plt.figure(figsize=(20, 12))
    fig.patch.set_facecolor('white')
    ax0 = fig.add_subplot(111, polar=True)
    n_xticks = 32
    xticks = np.arange(n_xticks)
    ax0.set_xticks(xticks / n_xticks * 2 * np.pi)
    ax0.set_xticklabels(['$\dfrac{%d}{%d}2\pi$' % (i0, n_xticks) for i0 in xticks])
    ax0.set_yticklabels([])
    ax0.set_ylim(0, np.pi)
    tdata = tuse.values
    im = ax0.pcolor(tuse.columns.values, tuse.index.values, tdata,
                    cmap=plt.get_cmap('Set2'))
    fig.colorbar(im, ax=ax0, orientation='vertical').ax.tick_params(labelsize=fontsize)
    return True


# show phase map of final trajectory in theta-phi space, using prepared type.
def show_traj_phase_map_type(tuse):
    fontsize = 40
    fig = plt.figure(figsize=(20, 12))
    fig.patch.set_facecolor('white')
    ax0 = fig.add_subplot(111, polar=True)
    n_xticks = 32
    xticks = np.arange(n_xticks)
    ax0.set_xticks(xticks / n_xticks * 2 * np.pi)
    ax0.set_xticklabels(['$\dfrac{%d}{%d}2\pi$' % (i0, n_xticks) for i0 in xticks])
    ax0.set_yticklabels([])
    ax0.set_ylim(0, np.pi)
    tdata = tuse.values
    im = ax0.pcolor(tuse.columns.values, tuse.index.values, tdata,
                    cmap=plt.get_cmap('tab20', np.nanmax(tdata) + 1),
                    vmin=np.nanmin(tdata) - .5, vmax=np.nanmax(tdata) + .5)
    ticks = np.arange(np.nanmin(tdata), np.nanmax(tdata) + 1)
    fig.colorbar(im, ax=ax0, orientation='vertical', ticks=ticks).ax.tick_params(labelsize=fontsize)
    return True

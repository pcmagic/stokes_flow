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
latex_installed = not subprocess.call(['which', 'latex'], stdout=devnull, stderr=devnull)
matplotlib.use('agg')
font = {'size':   20,
        'family': 'sans-serif'}
# matplotlib.rc('font', **font)
if latex_installed:
    matplotlib.rc('text', usetex=True)
# matplotlib.rc('text', usetex=True)

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy import interpolate, integrate, spatial, signal
from scipy.optimize import leastsq, curve_fit
from src import jeffery_model as jm
from src.objComposite import *
from src.support_class import *
from matplotlib import animation
from matplotlib import pyplot as plt
# from mpl_toolkits.axes_grid1 import colorbar
from matplotlib import colorbar
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, zoomed_inset_axes
import matplotlib.ticker as mtick
from matplotlib import colors as mcolors
import importlib
import inspect
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook
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


# params = {'text.latex.preamble': [r'\usepackage{bm}', r'\usepackage{amsmath}']}
# plt.rcParams.update(params)

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


def _do_calculate_prepare_v1(norm):
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


def _do_calculate_prepare_v2(norm):
    importlib.reload(jm)
    t_theta = np.arccos(norm[2] / np.linalg.norm(norm))
    t_phi = np.arctan2(norm[1], norm[0])
    tfct = 2 if t_phi < 0 else 0
    t_phi = t_phi + tfct * np.pi  # (-pi,pi) -> (0, 2pi)
    rotM = Rloc2glb(t_theta, t_phi, 0)
    P0 = rotM[:, 2]
    P20 = rotM[:, 1]

    planeShearRate = np.array((1, 0, 0))
    tcenter = np.zeros(3)
    fileHandle = 'ShearTableProblem'
    problem = jm.ShearTableProblem(name=fileHandle, planeShearRate=planeShearRate)
    return P0, P20, tcenter, problem


def do_calculate_prepare(norm):
    return _do_calculate_prepare_v2(norm)


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


def do_ecoli_kwargs(tcenter, P0, P20, ini_psi, omega_tail, table_name,
                    flow_strength=0, name='ecoli_torque'):
    ecoli_kwargs = {'name':          name,
                    'center':        tcenter,
                    'norm':          P0,
                    'lateral_norm':  P20,
                    'speed':         0,
                    'lbd':           np.nan,
                    'ini_psi':       ini_psi,
                    'omega_tail':    omega_tail,
                    'flow_strength': flow_strength,
                    'table_name':    table_name, }
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
    # fun_name = inspect.stack()[0][3]
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


def do_ShearFlowPetsc4nPsiObj(norm, ini_psi, max_t, table_name, update_fun='3bs',
                              rtol=1e-6, atol=1e-9, eval_dt=0.001, ini_t=0, save_every=1,
                              tqdm_fun=tqdm_notebook, omega_tail=0, flow_strength=0,
                              return_psi_body=False):
    P0, P20, tcenter, problem = do_calculate_prepare(norm)
    ecoli_kwargs = do_ecoli_kwargs(tcenter, P0, P20, ini_psi, omega_tail, table_name,
                                   flow_strength=flow_strength, name='ShearFlowPetsc4nPsi')
    obj = jm.ShearFlowPetsc4nPsiObj(**ecoli_kwargs)

    obj.set_update_para(fix_x=False, fix_y=False, fix_z=False, update_fun=update_fun,
                        rtol=rtol, atol=atol, save_every=save_every, tqdm_fun=tqdm_fun)
    problem.add_obj(obj)
    Table_t, Table_dt, Table_X, Table_P, Table_P2, Table_psi = \
        obj.update_self(t0=ini_t, t1=max_t, eval_dt=eval_dt)
    Table_theta, Table_phi, Table_psib = obj.theta_phi_psi
    Table_eta = np.arccos(np.sin(Table_theta) * np.sin(Table_phi))
    if return_psi_body:
        return Table_t, Table_dt, Table_X, Table_P, Table_P2, \
               Table_theta, Table_phi, Table_psi, Table_eta, Table_psib,
    else:
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


def core_show_table_theta_phi_list(theta_phi_list, job_dir, Table_t_range=(-np.inf, np.inf),
                                   figsize=np.array((20, 20)), dpi=100, fast_mode=0):
    cmap_list = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']

    def _get_ax():
        # fig = plt.figure(figsize=figsize, dpi=dpi)
        # fig.tight_layout(rect=[0, 0, 1, 0.8])
        # ax0 = fig.add_subplot(111)
        # ax0.set_xlim(-np.pi * 1.1, np.pi * 1.1)
        # ax0.set_ylim(-np.pi * 1.1, np.pi * 1.1)
        # ax0.axis('off')
        # ax0.set_aspect('equal')
        # fig.tight_layout(rect=[0, 0, 1, 0.8])
        # ax1 = fig.add_axes(ax0.get_position(), projection='polar')
        # ax1.patch.set_alpha(0)
        # plt.sca(ax1)
        # ax1.set_ylim(0, np.pi)
        # ax1.xaxis.set_ticklabels(['$\dfrac{%d}{8}2\pi$' % i0 for i0 in np.arange(8)])
        # ax1.yaxis.set_ticklabels([])
        fig, ax1 = plt.subplots(1, 1, figsize=np.ones(2) * np.min(figsize), dpi=dpi,
                                subplot_kw=dict(polar=True))
        plt.sca(ax1)
        ax1.set_ylim(0, np.pi)
        # ax1.xaxis.set_ticklabels(['$\dfrac{%d}{8}2\pi$' % i0 for i0 in np.arange(8)])
        ax1.yaxis.set_ticklabels([])
        return fig, ax1

    if fast_mode:
        fig, ax1 = _get_ax()
        fig2, ax2 = _get_ax()
        fig3, ax3 = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
        # fig3.patch.set_facecolor('white')
        for theta, phi in theta_phi_list:
            # print(theta, phi)
            tpick, _ = load_table_date_pickle(job_dir, theta, phi)
            Table_t = tpick['Table_t']
            # Table_dt = tpick['Table_dt']
            # Table_X = tpick['Table_X']
            # Table_P = tpick['Table_P']
            # Table_P2 = tpick['Table_P2']
            Table_theta = tpick['Table_theta']
            Table_phi = tpick['Table_phi']
            Table_psi = tpick['Table_psi']
            # Table_eta = tpick['Table_eta']

            idx = np.logical_and(Table_t >= Table_t_range[0], Table_t <= Table_t_range[1])
            if not np.any(idx):
                continue
            ax1.plot(Table_phi[idx], Table_theta[idx], '.', markersize=0.1)
            ax1.scatter(Table_phi[idx][0], Table_theta[idx][0], c='k', marker='*')
            ax2.plot(Table_psi[idx], Table_theta[idx], '.', markersize=0.1)
            ax2.scatter(Table_psi[idx][0], Table_theta[idx][0], c='k', marker='*')
            # tidx = Table_phi > 1.5 * np.pi
            tidx = Table_phi > 15 * np.pi
            t1 = Table_phi.copy()
            t1[tidx] = Table_phi[tidx] - 2 * np.pi
            ax3.plot(t1[idx] / np.pi, Table_psi[idx] / np.pi, '.', markersize=0.1)
            ax3.scatter(t1[idx][0] / np.pi, Table_psi[idx][0] / np.pi, c='k', marker='*')
        fig.suptitle('$\\theta - \\phi$')
        fig2.suptitle('$\\theta - \\psi$')
        ax3.set_xlabel('$\\phi / \\pi$')
        ax3.set_ylabel('$\\psi / \\pi$')
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig2.tight_layout(rect=[0, 0, 1, 0.95])
        fig3.tight_layout()
    else:
        fig, ax1 = _get_ax()
        for (theta, phi), cmap in zip(theta_phi_list, cmap_list):
            tpick, _ = load_table_date_pickle(job_dir, theta, phi)
            Table_t = tpick['Table_t']
            # Table_dt = tpick['Table_dt']
            # Table_X = tpick['Table_X']
            # Table_P = tpick['Table_P']
            # Table_P2 = tpick['Table_P2']
            Table_theta = tpick['Table_theta']
            Table_phi = tpick['Table_phi']
            # Table_psi = tpick['Table_psi']
            # Table_eta = tpick['Table_eta']

            idx = np.logical_and(Table_t >= Table_t_range[0], Table_t <= Table_t_range[1])
            t1 = Table_t[idx].max() - Table_t[idx].min()
            norm = plt.Normalize(Table_t[idx].min() - 0.3 * t1, Table_t[idx].max())
            ax1.scatter(Table_phi[idx][0], Table_theta[idx][0], c='k', marker='*')
            spf.colorline(Table_phi[idx], Table_theta[idx], z=Table_t[idx], cmap=plt.get_cmap(cmap),
                          norm=norm, linewidth=1, alpha=1.0, ax=ax1)
    return fig


def show_table_theta_phi_list(*args, **kwargs):
    core_show_table_theta_phi_list(*args, **kwargs)
    return True


def core_show_table_result_list(theta_phi_list, job_dir, label_list=None,
                                Table_t_range=(-np.inf, np.inf),
                                figsize=np.array((20, 20)), dpi=100):
    if label_list is None:
        label_list = [None] * len(theta_phi_list)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor('white')
    axs = fig.subplots(nrows=3, ncols=2)
    for (theta, phi), tlabel in zip(theta_phi_list, label_list):
        tpick, _ = load_table_date_pickle(job_dir, theta, phi)
        Table_t = tpick['Table_t']
        idx = np.logical_and(Table_t >= Table_t_range[0], Table_t <= Table_t_range[1])
        Table_t = tpick['Table_t'][idx]
        Table_X = tpick['Table_X'][idx]
        Table_theta = tpick['Table_theta'][idx]
        Table_phi = tpick['Table_phi'][idx]
        Table_psi = tpick['Table_psi'][idx]

        for _ in zip(axs,
                     (Table_X[:, 0], Table_X[:, 1], Table_X[:, 2]),
                     (Table_theta, Table_phi, Table_psi),
                     ('$x - x_{mean}$', '$y - y_{mean}$', '$z - z_{mean}$'),
                     ('$\\theta / \pi$', '$\\phi / \pi$', '$\\psi / \pi$'), ):
            (ax1, ax2), ty1, ty2, ylab1, ylab2 = _
            if tlabel is None:
                ax1.plot(Table_t, ty1 - np.mean(ty1), '-')
                ax2.plot(Table_t, ty2 / np.pi, '-')
            else:
                ax1.plot(Table_t, ty1 - np.mean(ty1), '-', label=tlabel)
                ax2.plot(Table_t, ty2 / np.pi, '-', label=tlabel)
                ax1.legend()
                ax2.legend()
            ax1.set_ylabel(ylab1)
            ax2.set_ylabel(ylab2)
        axs[0, 0].xaxis.set_ticklabels([])
        axs[0, 1].xaxis.set_ticklabels([])
        axs[1, 0].xaxis.set_ticklabels([])
        axs[1, 1].xaxis.set_ticklabels([])
        axs[2, 0].set_xlabel('$t$')
        axs[2, 1].set_xlabel('$t$')
    plt.tight_layout()
    return fig


def show_table_result_list(*args, **kwargs):
    core_show_table_result_list(*args, **kwargs)
    return True


def core_show_table_theta_phi_psi_fft_list(theta_phi_list, job_dir, label_list,
                                           figsize=np.array((20, 20)), dpi=100,
                                           resampling_fct=2, use_welch=False):
    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor('white')

    for (theta, phi), tlabel in zip(theta_phi_list, label_list):
        tpick, _ = load_table_date_pickle(job_dir, theta, phi)
        Table_t = tpick['Table_t']
        Table_t = Table_t
        # Table_dt = tpick['Table_dt']
        # Table_X = tpick['Table_X']
        # Table_P = tpick['Table_P']
        # Table_P2 = tpick['Table_P2']
        Table_theta = tpick['Table_theta']
        Table_phi = tpick['Table_phi']
        Table_psi = tpick['Table_psi']
        Table_eta = tpick['Table_eta']
        Table_t, Table_theta, Table_phi, Table_psi, Table_eta = \
            resampling_angle(Table_t, Table_theta, Table_phi, Table_psi, Table_eta, resampling_fct)

        for (ax1, ax2), ty1, ylab in zip(axs,
                                         (Table_theta, Table_phi, Table_psi),
                                         ('\\theta', '\\phi', '\\psi')):
            # find major frequence and display
            tmin = np.max((0, Table_t.max() - 1000))
            idx = Table_t > tmin
            freq_pk = get_major_fre(Table_t[idx], np.cos(Table_theta[idx]))
            idx = Table_t > (Table_t.max() - 1 / freq_pk * 10)

            if use_welch:
                fs = ty1[idx].size / (Table_t[idx].max() - Table_t[idx].min())
                nperseg = fs / freq_pk * 8
                tfreq, tfft = signal.welch(np.cos(ty1)[idx], fs=fs, nperseg=nperseg)
            else:
                tfft = np.fft.rfft(np.cos(ty1[idx]))
                # noinspection PyTypeChecker
                tfreq = np.fft.rfftfreq(Table_t[idx].size, np.mean(np.diff(Table_t[idx])))

            tfft_abs = np.abs(tfft)
            ax1.semilogx(tfreq[:], tfft_abs[:], '-', label=tlabel)
            ax2.loglog(tfreq[:], tfft_abs[:], '-', label=tlabel)
            ax1.set_title('FFT of $\\cos %s$' % ylab)
            ax2.set_title('FFT of $\\cos %s$' % ylab)
            ax1.legend()
        axs[0, 0].xaxis.set_ticklabels([])
        axs[0, 1].xaxis.set_ticklabels([])
        axs[1, 0].xaxis.set_ticklabels([])
        axs[1, 1].xaxis.set_ticklabels([])
        axs[2, 0].set_xlabel('$Hz$')
        axs[2, 1].set_xlabel('$Hz$')
    # fig.tight_layout()
    return fig


def show_table_theta_phi_psi_fft_list(*args, **kwargs):
    core_show_table_theta_phi_psi_fft_list(*args, **kwargs)
    return True


def core_show_table_result(Table_t, Table_dt, Table_X, Table_P, Table_P2,
                           Table_theta, Table_phi, Table_psi, Table_eta, move_z=False,
                           planeShearRate=np.array((1, 0, 0)), fig=None,
                           save_every=1, resampling=False, resampling_fct=2):
    fontsize = 40
    figsize = np.array((20, 15))
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
                              '$center_x$', '$center_y$', '$center_z$', '$dt$',
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
        axi.set_xlabel('$t$', size=fontsize * 0.7)
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


def core_show_table_result_v2(Table_t, Table_dt, Table_X, Table_P, Table_P2,
                              Table_theta, Table_phi, Table_psi, Table_eta, move_z=False,
                              planeShearRate=np.array((1, 0, 0)), fig=None,
                              save_every=1, resampling=False, resampling_fct=2,
                              figsize=np.array((16, 9)) * 1.5, dpi=100):
    markersize = 10
    fontsize = 10
    norm = plt.Normalize(Table_t.min(), Table_t.max())
    cmap = plt.get_cmap('jet')

    def _plot_polar(ax0, Table_angle, title):
        # polar version
        ax0.plot(Table_angle, Table_theta, '-', alpha=0.2)
        ax0.plot(Table_angle[0], Table_theta[0], '*k', markersize=markersize * 1.5)
        ax0.scatter(Table_angle, Table_theta, c=Table_t, cmap=cmap, norm=norm, s=markersize)
        ax0.set_ylim(0, np.pi)
        ax0.set_title(title, size=fontsize * 0.8)
        plt.sca(ax0)
        plt.xticks(fontsize=fontsize * 0.8)
        plt.yticks(fontsize=fontsize * 0.8)
        return True

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
        fig = plt.figure(figsize=figsize, dpi=dpi)
    else:
        fig.clf()
    fig.patch.set_facecolor('white')
    cax = plt.subplot2grid((19, 32), (0, 0), rowspan=18, colspan=1)
    ax0 = plt.subplot2grid((19, 32), (0, 2), rowspan=8, colspan=8, polar=True)
    ax1 = plt.subplot2grid((19, 32), (10, 2), rowspan=8, colspan=8, polar=True)
    ax2 = plt.subplot2grid((19, 32), (0, 11), rowspan=8, colspan=8)
    ax3 = plt.subplot2grid((19, 32), (10, 11), rowspan=8, colspan=8, projection='3d')
    ax9 = plt.subplot2grid((19, 32), (15, 21), rowspan=3, colspan=12)
    ax4 = plt.subplot2grid((19, 32), (0, 21), rowspan=3, colspan=12)
    ax5 = plt.subplot2grid((19, 32), (3, 21), rowspan=3, colspan=12)
    ax6 = plt.subplot2grid((19, 32), (6, 21), rowspan=3, colspan=12)
    ax7 = plt.subplot2grid((19, 32), (9, 21), rowspan=3, colspan=12)
    ax8 = plt.subplot2grid((19, 32), (12, 21), rowspan=3, colspan=12)

    _plot_polar(ax0, Table_phi, '$\\theta - \\phi$')
    _plot_polar(ax1, Table_psi, '$\\theta - \\psi$')

    ax2.plot(Table_phi / np.pi, Table_psi / np.pi, '-', alpha=0.2)
    ax2.plot(Table_phi[0] / np.pi, Table_psi[0] / np.pi, '*k', markersize=markersize * 1.5)
    ax2.scatter(Table_phi / np.pi, Table_psi / np.pi, c=Table_t, cmap=cmap, norm=norm, s=markersize)
    ax2.set_xlabel('$\\phi / \\pi$')
    ax2.set_ylabel('$\\psi / \\pi$')
    plt.sca(ax2)
    plt.xticks(fontsize=fontsize * 0.8)
    plt.yticks(fontsize=fontsize * 0.8)

    ax3.set_title('$P_1$', size=fontsize)
    points = Table_P.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = Line3DCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(Table_t)
    ax3.add_collection3d(lc, zs=points[:, :, 2].flatten(), zdir='z')
    ax3.set_xlim(points[:, :, 0].min(), points[:, :, 0].max())
    ax3.set_ylim(points[:, :, 1].min(), points[:, :, 1].max())
    ax3.set_zlim(points[:, :, 2].min(), points[:, :, 2].max())
    spf.set_axes_equal(ax3)
    ax3.plot(np.ones_like(points[:, :, 0].flatten()) * ax3.get_xlim()[0], points[:, :, 1].flatten(),
             points[:, :, 2].flatten())
    ax3.plot(points[:, :, 0].flatten(), np.ones_like(points[:, :, 1].flatten()) * ax3.get_ylim()[1],
             points[:, :, 2].flatten())
    ax3.plot(points[:, :, 0].flatten(), points[:, :, 1].flatten(),
             np.ones_like(points[:, :, 2].flatten()) * ax3.get_zlim()[0])
    plt.sca(ax3)
    ax3.set_xlabel('$x$', size=fontsize)
    ax3.set_ylabel('$y$', size=fontsize)
    ax3.set_zlabel('$z$', size=fontsize)
    plt.xticks(fontsize=fontsize * 0.8)
    plt.yticks(fontsize=fontsize * 0.8)
    for t in ax3.zaxis.get_major_ticks():
        t.label.set_fontsize(fontsize * 0.8)
    for spine in ax3.spines.values():
        spine.set_visible(False)

    clb = fig.colorbar(lc, cax=cax)
    clb.ax.tick_params(labelsize=fontsize)
    clb.ax.set_title('time', size=fontsize)

    for _ in zip(((ax4, ax7), (ax5, ax8), (ax6, ax9)),
                 (Table_X[:, 0], Table_X[:, 1], Table_X[:, 2]),
                 (Table_theta, Table_phi, Table_psi),
                 ('$x - x_{mean}$', '$y - y_{mean}$', '$z - z_{mean}$'),
                 ('x_{mean}', 'y_{mean}', 'z_{mean}'),
                 ('$\\theta / \pi$', '$\\phi / \pi$', '$\\psi / \pi$'), ):
        (ax1, ax2), ty1, ty2, ylab1, txt1, ylab2 = _
        ax1.plot(Table_t, ty1 - np.mean(ty1), '-')
        t1 = '$%s = %.2e$' % (txt1, np.mean(ty1))
        ax1.text(Table_t.min(), (ty1 - np.mean(ty1)).max() / 2, t1, fontsize=fontsize)
        for i0, i1 in separate_angle_idx(ty2):
            ax2.plot(Table_t[i0:i1], ty2[i0:i1] / np.pi, '-', color='#1f77b4')
            # ax2.plot(Table_t, ty2 / np.pi, '-')
        ax1.set_ylabel(ylab1)
        ax2.set_ylabel(ylab2)
    for axi in (ax4, ax5, ax6, ax7, ax8):
        axi.set_xticklabels([])
    plt.sca(ax9)
    ax9.set_xlabel('$t$')
    plt.xticks(fontsize=fontsize * 0.8)
    plt.yticks(fontsize=fontsize * 0.8)

    plt.tight_layout()
    return fig


def show_table_result_v2(*args, **kwargs):
    core_show_table_result_v2(*args, **kwargs)
    return True


def save_table_result_v2(filename, *args, dpi=100, **kwargs):
    fig = core_show_table_result_v2(*args, **kwargs)
    fig.savefig(fname=filename, dpi=dpi)
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
        fig = plt.figure(figsize=(10, 10), dpi=200)
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
        show_theta_phi(Table_t[idx], Table_dt[idx], Table_X[idx], Table_P[idx], Table_P2[idx],
                       Table_theta[idx], Table_phi[idx], Table_psi[idx], Table_eta[idx],
                       show_back_direction=False)
        show_theta_phi_psi_eta(Table_t[idx], Table_dt[idx], Table_X[idx],
                               Table_P[idx], Table_P2[idx],
                               Table_theta[idx], Table_phi[idx], Table_psi[idx], Table_eta[idx])
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
                                fig=None, resampling_fct=2, fft_full_mode=False,
                                show_prim_freq=3, dpi=100):
    fontsize = 40
    figsize = (20, 15)
    Table_t, Table_theta, Table_phi, Table_psi, Table_eta = \
        resampling_angle(Table_t, Table_theta, Table_phi, Table_psi, Table_eta, resampling_fct)

    if fig is None:
        fig = plt.figure(figsize=figsize, dpi=dpi)
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
        tfft = np.fft.rfft(np.cos(ty1[idx]))
        # tfft = signal.stft(np.cos(ty1[idx]))
        tfft_abs = np.abs(tfft)
        # noinspection PyTypeChecker
        tfreq = np.fft.rfftfreq(Table_t[idx].size, np.mean(np.diff(Table_t[idx])))
        ax1.loglog(tfreq, tfft_abs, '.')
        tpk = signal.find_peaks(tfft_abs)[0]
        if tpk.size > 0:
            fft_abs_pk = tfft_abs[tpk]
            freq_pk = tfreq[tpk]
            tidx = np.argsort(fft_abs_pk)[-show_prim_freq:]
            # ax1.text(freq_pk[tidx] / 5, fft_abs_pk[tidx], '$%.5f$' % freq_pk[tidx],
            #          fontsize=fontsize * 0.7)
            ax1.loglog(freq_pk[tidx], fft_abs_pk[tidx], '*', ms=fontsize * 0.5)
            t1 = 'starred freq: \n' + '\n'.join(['$%.5f$' % freq_pk[ti] for ti in tidx])
            ax1.text(ax1.get_xlim()[0] * 1.1, ax1.get_ylim()[0] * 1.1,
                     t1, fontsize=fontsize * 0.5)
        plt.yticks(fontsize=fontsize * 0.5)
    axs[-1, 0].set_xlabel('$t$', size=fontsize * 0.7)
    axs[-1, 1].set_xlabel('$Hz$', size=fontsize * 0.7)
    plt.tight_layout()
    return fig


def show_theta_phi_psi_eta(Table_t, Table_dt, Table_X, Table_P, Table_P2,
                           Table_theta, Table_phi, Table_psi, Table_eta,
                           fig=None, resampling_fct=2, fft_full_mode=False,
                           show_prim_freq=3, dpi=100):
    core_show_theta_phi_psi_eta(Table_t, Table_dt, Table_X, Table_P, Table_P2,
                                Table_theta, Table_phi, Table_psi, Table_eta,
                                fig, resampling_fct, fft_full_mode,
                                show_prim_freq, dpi)
    return True


def save_theta_phi_psi_eta(filename, Table_t, Table_dt, Table_X, Table_P, Table_P2,
                           Table_theta, Table_phi, Table_psi, Table_eta,
                           fig=None, resampling_fct=2, fft_full_mode=False,
                           show_prim_freq=3, dpi=100):
    fig = core_show_theta_phi_psi_eta(Table_t, Table_dt, Table_X, Table_P, Table_P2,
                                      Table_theta, Table_phi, Table_psi, Table_eta,
                                      fig, resampling_fct, fft_full_mode,
                                      show_prim_freq, dpi)
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
    if np.array(t_use).size == 1:
        t_use = np.linspace(tx.min(), tx.max(), t_use * tx.size)

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
    tidx = np.isfinite(Table_t)
    if Table_t[1] - Table_t[0] <= 0:
        tidx[0] = False
    if Table_t[-1] - Table_t[-2] <= 0:
        tidx[-1] = False
    Table_theta = get_continue_angle(Table_t[tidx], Table_theta[tidx], t_use)
    Table_phi = get_continue_angle(Table_t[tidx], Table_phi[tidx], t_use)
    Table_psi = get_continue_angle(Table_t[tidx], Table_psi[tidx], t_use)
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


def load_table_data_pickle_dir(t_dir, t_headle='(.*?).pickle'):
    t_path = os.listdir(t_dir)
    filename_list = [filename for filename in t_path if re.match(t_headle, filename) is not None]
    ini_theta_list = []
    ini_phi_list = []
    lst_eta_list = []
    theta_max_fre_list = []
    phi_max_fre_list = []
    psi_max_fre_list = []
    eta_max_fre_list = []
    pickle_path_list = []
    idx_list = []
    for i0, tname in enumerate(tqdm_notebook(filename_list[:])):
        tpath = os.path.join(t_dir, tname)
        with open(tpath, 'rb') as handle:
            tpick = pickle.load(handle)
        ini_theta_list.append(tpick['ini_theta'])
        ini_phi_list.append(tpick['ini_phi'])
        lst_eta_list.append(tpick['Table_eta'][-1])
        pickle_path_list.append(tpath)
        idx_list.append(i0)

        # fft rule
        tx = tpick['Table_t']
        tmin = np.max((0, tx.max() - 1000))
        idx = tx > tmin
        freq_pk = get_major_fre(tx[idx], tpick['Table_theta'][idx])
        idx = tx > (tx.max() - 1 / freq_pk * 10)
        theta_max_fre_list.append(get_major_fre(tx[idx], tpick['Table_theta'][idx]))
        phi_max_fre_list.append(get_major_fre(tx[idx], tpick['Table_phi'][idx]))
        psi_max_fre_list.append(get_major_fre(tx[idx], tpick['Table_psi'][idx]))
        eta_max_fre_list.append(get_major_fre(tx[idx], tpick['Table_eta'][idx]))

    data0 = pd.DataFrame({'ini_theta':     np.around(ini_theta_list, 3),
                          'ini_phi':       np.around(ini_phi_list, 3),
                          'lst_eta':       np.around(lst_eta_list, 3),
                          'theta_max_fre': theta_max_fre_list,
                          'phi_max_fre':   phi_max_fre_list,
                          'psi_max_fre':   psi_max_fre_list,
                          'eta_max_fre':   eta_max_fre_list,
                          'data_idx':      idx_list})
    data = data0.pivot_table(index=['ini_theta'], columns=['ini_phi'])
    # lst_eta = data.lst_eta
    # theta_max_fre = data.theta_max_fre
    # phi_max_fre = data.phi_max_fre
    # psi_max_fre = data.psi_max_fre
    # eta_max_fre = data.eta_max_fre
    # data_idx = data.data_idx.fillna(-1).astype(int)
    return data


def load_rand_data_pickle_dir(t_dir, t_headle='(.*?).pickle', n_load=None, rand_mode=False):
    t_path = os.listdir(t_dir)
    filename_list = [filename for filename in t_path if re.match(t_headle, filename) is not None]
    ini_theta_list = []
    ini_phi_list = []
    ini_psi_list = []
    theta_max_fre_list = []
    phi_max_fre_list = []
    psi_max_fre_list = []
    pickle_path_list = []

    n_load = len(filename_list) if n_load is None else n_load
    assert n_load <= len(filename_list)
    if rand_mode:
        tidx = np.random.choice(len(filename_list), n_load, replace=False)
    else:
        tidx = np.arange(n_load)
    use_filename_list = np.array(filename_list)[tidx]
    for i0, tname in enumerate(tqdm_notebook(use_filename_list)):
        tpath = os.path.join(t_dir, tname)
        with open(tpath, 'rb') as handle:
            tpick = pickle.load(handle)
        ini_theta_list.append(tpick['ini_theta'])
        ini_phi_list.append(tpick['ini_phi'])
        ini_psi_list.append(tpick['ini_psi'])
        pickle_path_list.append(tpath)

        # fft rule
        tx = tpick['Table_t']
        tmin = np.max((0, tx.max() - 1000))
        idx = tx > tmin
        freq_pk = get_major_fre(tx[idx], tpick['Table_theta'][idx])
        idx = tx > (tx.max() - 1 / freq_pk * 10)
        theta_max_fre_list.append(get_major_fre(tx[idx], tpick['Table_theta'][idx]))
        phi_max_fre_list.append(get_major_fre(tx[idx], tpick['Table_phi'][idx]))
        psi_max_fre_list.append(get_major_fre(tx[idx], tpick['Table_psi'][idx]))

    return ini_theta_list, ini_phi_list, ini_psi_list, \
           theta_max_fre_list, phi_max_fre_list, psi_max_fre_list, \
           pickle_path_list


def load_lookup_table_pickle(pickle_name):
    with open('%s.pickle' % pickle_name, 'rb') as handle:
        pickle_data = pickle.load(handle)
    ttheta_all, tphi_all = pickle_data[0][1][0][:2]
    if tphi_all[-1] < (2 * np.pi):
        tphi_all = np.hstack((tphi_all, 2 * np.pi))
    if ttheta_all[-1] < (np.pi):
        ttheta_all = np.hstack((ttheta_all, np.pi))
    tpsi_all = np.array([ti[0] for ti in pickle_data])
    U_all = [[] for i in range(6)]
    for _, table_psi_data in pickle_data:
        for (ttheta, tphi, tU), Ui in zip(table_psi_data, U_all):
            if tphi[-1] < (2 * np.pi):
                tU[2 * np.pi] = tU[0]
            if ttheta[-1] < (np.pi):
                tU = tU.append(tU.loc[0].rename(np.pi))
            Ui.append(tU)
    return U_all, ttheta_all, tphi_all, tpsi_all


def phase_map_show_idx(type_fre, tipical_th_ph_list, iidx, job_dir, table_name, fast_mode=0):
    theta = type_fre.index.values[iidx[0][0]]
    phi = type_fre.columns.values[iidx[1][0]]
    print('-ini_theta %f -ini_phi %f' % (theta, phi))
    tipical_th_ph_list.append((theta, phi))
    show_pickle_results(job_dir, theta, phi, table_name, fast_mode=fast_mode)
    return tipical_th_ph_list


def phase_map_show_idx_list(type_fre, iidx, job_dir, nshow=5, Table_t_range1=np.array((0, np.inf)),
                            Table_t_range2=np.array((0, np.inf)), fast_mode=0,
                            figsize=np.array((16, 9)) * 0.5, dpi=200):
    nshow = int(np.min((nshow, iidx[0].size)))
    tidx = np.random.choice(iidx[0].size, nshow, replace=False)
    theta = type_fre.index.values[iidx[0][tidx]]
    phi = type_fre.columns.values[iidx[1][tidx]]
    theta_phi_list = np.vstack((theta, phi)).T
    show_table_theta_phi_list(theta_phi_list, job_dir, Table_t_range=Table_t_range1,
                              figsize=figsize, dpi=dpi, fast_mode=fast_mode)
    show_table_result_list(theta_phi_list, job_dir, Table_t_range=Table_t_range2,
                           figsize=figsize, dpi=dpi)
    return True


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
    # color = np.array(color)
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

    if np.array(case_path).size > 0:
        th_all = []
        ph_all = []
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
            th_all.append(Table_theta[tidx])
            ph_all.append(Table_phi[tidx])
        for ax0 in tube_flatten((axs,)):
            ax0.scatter(np.hstack(ph_all), np.hstack(th_all), c=color, s=fontsize * 0.2)
    return axs


def draw_phase_map_theta_bck(case_path, color, psi_lim, axs=None,
                             resampling=False, resampling_fct=2, thandle=''):
    fontsize = 40
    # color = np.array(color)
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
                    cmap=plt.get_cmap('tab20', int(np.nanmax(tdata)) + 1),
                    vmin=np.nanmin(tdata) - .5, vmax=np.nanmax(tdata) + .5)
    ticks = np.arange(np.nanmin(tdata), np.nanmax(tdata) + 1)
    fig.colorbar(im, ax=ax0, orientation='vertical', ticks=ticks).ax.tick_params(labelsize=fontsize)
    return True


# The following code are used to do 2D FFT an 2D IFFT of \omega(\theta, \phi, psi)
#   of microswimmer in shear flow along \theta and \psi.
def do_fft_major(tw, tktl_list):
    # do FFT of velocity component and pick major frequence, then IFFT.
    tw_fft = np.fft.fft2(tw)
    ntk, ntl = tw_fft.shape
    idx = np.ones_like(tw_fft) * 1e-30
    for tk1, tl1 in tktl_list:
        tk2 = ntk - tk1 if tk1 > 0 else tk1
        tl2 = ntl - tl1 if tl1 > 0 else tl1
        idx[tk1, tl1] = 1
        idx[tk2, tl2] = 1
        tf1 = tw_fft[tk1, tl1]
        tf2 = tw_fft[tk2, tl2]
        if tk1 > 0 or tl1 > 0:
            print('use frequence pairs %f%+fi and %f%+fi at (%d, %d) and (%d, %d)' % (
                tf1.real, tf1.imag, tf2.real, tf2.imag, tk1, tl1, tk2, tl2))
        else:
            print('use frequence %f%+fi at (%d, %d)' % (tf1.real, tf1.imag, tk1, tl1))

    tw_fft2 = tw_fft * idx
    tw2 = np.fft.ifft2(tw_fft2)
    print('absolute abs of imag part is', np.abs(tw2.imag).max())
    return tw_fft, tw2.real, tw_fft2


def factor_wpi_kl(tw, tktl):
    # see decouplingIdea.tex for detail.
    # \omega_{pi}^{kl}(\theta, \phi, \psi) = \dfrac{2}{n_\theta n_\phi}
    #     \left(\Re(\Omega_{pi}(k,l, \psi)) \cos(2k\theta + l\phi) -
    #           \Im(\Omega_{pi}(k,l, \psi)) \sin(2k\theta + l\phi) \right)
    # \omega_{pi}^{kl}(\theta, \phi, \psi) = \dfrac{2}{n_\theta n_\phi}
    #     \norm{\Omega_{pi}(k,l, \psi)} \sin(\alpha_0 + 2k\theta + l\phi)
    # Amp_use = \dfrac{2}{n_\theta n_\phi}\norm{\Omega_{pi}(k,l, \psi)}
    # w_th_use = 2k
    # w_ph_use = l
    # alpha_use = \alpha_0

    tk1, tl1 = tktl
    nth, nph = tw.shape
    tw_fft = np.fft.fft2(tw)
    Akl1 = tw_fft[tk1, tl1]
    Aklr = Akl1.real
    Akli = Akl1.imag

    k_sign = 1 if tk1 < (nth / 2) else -1
    l_sign = 1 if tl1 < (nph / 2) else -1
    Amp_use = 2 * np.abs(Akl1) / tw.size * k_sign * l_sign
    w_th_use = 2 * tk1 if tk1 < (nth / 2) else -2 * (nth - tk1)
    w_ph_use = tl1 if tl1 < (nph / 2) else -1 * (nph - tl1)
    alpha_use = -np.arctan(Aklr / Akli)
    return Akl1, Amp_use, w_th_use, w_ph_use, alpha_use


def show_fft_major(tw, tktl_list, ttheta, tphi):
    tw_fft, tw2, tw_fft2 = do_fft_major(tw, tktl_list)
    th_freq, ph_freq = np.meshgrid(np.fft.fftshift(np.fft.fftfreq(ttheta.size, 1 / ttheta.size)),
                                   np.fft.fftshift(np.fft.fftfreq(tphi.size, 1 / tphi.size)),
                                   indexing='ij')
    tw_fft = np.fft.fftshift(tw_fft)
    tw_fft2 = np.fft.fftshift(tw_fft2)

    fig = plt.figure(figsize=(13, 11), dpi=300)
    fig.patch.set_facecolor('white')
    axs = fig.subplots(nrows=2, ncols=2)
    twmax = np.max(np.abs(tw)) * 1.2
    tw_levels = np.linspace(-twmax, twmax, 10)
    fft_max = np.max(np.abs(tw_fft))
    log_fft_max = np.ceil(np.log10(fft_max))
    log_fft_step = 3
    log_fft_min = log_fft_max - log_fft_step
    fft_ticks = 10 ** np.linspace(log_fft_min, log_fft_max, log_fft_step + 1)
    fft_formatter = mtick.LogFormatter(10, labelOnlyBase=False)

    ax = axs[0, 0]
    im = ax.contourf(tphi / np.pi, ttheta / np.pi, tw, tw_levels, cmap=plt.get_cmap('RdBu'))
    fig.colorbar(im, ax=ax, orientation='vertical')
    ax.set_title('original data')
    ax.set_xlabel('$\\phi / \pi$')
    ax.set_ylabel('$\\theta / \pi$')

    ax = axs[0, 1]
    im = ax.pcolor(ph_freq, th_freq, np.abs(tw_fft), cmap=plt.get_cmap('Greys'),
                   norm=mcolors.LogNorm(vmin=10 ** log_fft_min, vmax=10 ** log_fft_max))
    fig.colorbar(im, ax=ax, orientation='vertical', ticks=fft_ticks, format=fft_formatter)
    ax.set_title('original frequence')
    ax.set_xlabel('$f_\\phi$')
    ax.set_ylabel('$f_\\theta$')
    # ax.set_xlim(0, ax.get_xlim()[1])
    # ax.set_ylim(0, ax.get_ylim()[1])

    ax = axs[1, 0]
    im = ax.contourf(tphi / np.pi, ttheta / np.pi, tw2, tw_levels, cmap=plt.get_cmap('RdBu'))
    fig.colorbar(im, ax=ax, orientation='vertical')
    ax.set_title('after filter data')
    ax.set_xlabel('$\\phi / \pi$')
    ax.set_ylabel('$\\theta / \pi$')

    ax = axs[1, 1]
    im = ax.pcolor(ph_freq, th_freq, np.abs(tw_fft2), cmap=plt.get_cmap('Greys'),
                   norm=mcolors.LogNorm(vmin=10 ** log_fft_min, vmax=10 ** log_fft_max))
    fig.colorbar(im, ax=ax, orientation='vertical', ticks=fft_ticks, format=fft_formatter)
    ax.set_title('after filter frequence')
    ax.set_xlabel('$f_\\phi$')
    ax.set_ylabel('$f_\\theta$')
    # ax.set_xlim(0, ax.get_xlim()[1])
    # ax.set_ylim(0, ax.get_ylim()[1])
    plt.tight_layout()
    return True


def show_fft_fit(tw, tktl, ttheta, tphi):
    def fit_fun(tx, Amp, w_th, w_ph, alpha):
        theta, phi = tx
        return Amp * np.sin(w_th * theta + w_ph * phi + alpha)

    # analitical from IFFT. The input index includes and only includes a pair of conjugate frequencies.
    tk1, tl1 = tktl
    tw_fft, tw2, tw_fft2 = do_fft_major(tw, ((tk1, tl1),))
    ntk, ntl = tw_fft.shape
    Akl1 = tw_fft[tk1, tl1]
    tk2 = ntk - tk1 if tk1 > 0 else tk1
    tl2 = ntl - tl1 if tl1 > 0 else tl1
    Akl2 = tw_fft[tk2, tl2]
    Aklr = Akl1.real
    Akli = Akl1.imag
    th_freq, ph_freq = np.meshgrid(np.fft.fftshift(np.fft.fftfreq(ttheta.size, 1 / ttheta.size)),
                                   np.fft.fftshift(np.fft.fftfreq(tphi.size, 1 / tphi.size)),
                                   indexing='ij')
    theta_all, phi_all = np.meshgrid(ttheta, tphi, indexing='ij')
    tw_fft = np.fft.fftshift(tw_fft)
    tw_fft2 = np.fft.fftshift(tw_fft2)

    # fit
    Amp_ini = 0
    w_th_ini = 2 * tk1 if tk1 < (ttheta.size / 2) else -2 * (ttheta.size - tk1)
    w_ph_ini = tl1 if tl1 < (tphi.size / 2) else -1 * (tphi.size - tl1)
    alpha_ini = 0
    p0 = (Amp_ini, w_th_ini, w_ph_ini, alpha_ini)
    popt, pcov = curve_fit(fit_fun, (theta_all.ravel(), phi_all.ravel()), tw2.ravel(), p0=p0)
    tw_fit = fit_fun((theta_all, phi_all), *popt)

    # analitical solution
    k_sign = 1 if tk1 < (ttheta.size / 2) else -1
    l_sign = 1 if tl1 < (tphi.size / 2) else -1
    Amp_use = (np.abs(Akl1) + np.abs(Akl2)) / tw.size * k_sign * l_sign
    w_th_use = w_th_ini
    w_ph_use = w_ph_ini
    alpha_use = np.arctan(Aklr / -Akli)
    tw_ana = fit_fun((theta_all, phi_all), Amp_use, w_th_use, w_ph_use, alpha_use)

    fig = plt.figure(figsize=(13, 11), dpi=300)
    fig.patch.set_facecolor('white')
    axs = fig.subplots(nrows=2, ncols=2)
    twmax = np.max(np.abs(tw)) * 1.2
    tw_levels = np.linspace(-twmax, twmax, 10)
    fft_max = np.max(np.abs(tw_fft))
    log_fft_max = np.ceil(np.log10(fft_max))
    log_fft_step = 3
    log_fft_min = log_fft_max - log_fft_step
    fft_ticks = 10 ** np.linspace(log_fft_min, log_fft_max, log_fft_step + 1)
    fft_formatter = mtick.LogFormatter(10, labelOnlyBase=False)

    ax = axs[0, 0]
    im = ax.contourf(tphi / np.pi, ttheta / np.pi, tw2, tw_levels, cmap=plt.get_cmap('RdBu'))
    fig.colorbar(im, ax=ax, orientation='vertical')
    ax.set_title('after filter data')
    ax.set_xlabel('$\\phi / \pi$')
    ax.set_ylabel('$\\theta / \pi$')

    ax = axs[0, 1]
    im = ax.pcolor(ph_freq, th_freq, np.abs(tw_fft2), cmap=plt.get_cmap('Greys'),
                   norm=mcolors.LogNorm(vmin=10 ** log_fft_min, vmax=10 ** log_fft_max))
    fig.colorbar(im, ax=ax, orientation='vertical', ticks=fft_ticks, format=fft_formatter)
    ax.set_title('after filter frequence')
    ax.set_xlabel('$f_\\phi$')
    ax.set_ylabel('$f_\\theta$')
    # ax.set_xlim(0, ax.get_xlim()[1])
    # ax.set_ylim(0, ax.get_ylim()[1])
    ax.text(tphi.size * -0.4, ttheta.size * +0.3,
            '$A(%d, %d) = %f %+fi$' % (tk1, tl1, Akl1.real, Akl1.imag), fontsize='x-small')
    ax.text(tphi.size * -0.4, ttheta.size * -0.3,
            '$A(%d, %d) = %f %+fi$' % (tk2, tl2, Akl2.real, Akl2.imag), fontsize='x-small')

    ax = axs[1, 0]
    im = ax.contourf(tphi / np.pi, ttheta / np.pi, tw_fit, tw_levels, cmap=plt.get_cmap('RdBu'))
    fig.colorbar(im, ax=ax, orientation='vertical')
    ax.set_title('after filter and fit data')
    ax.set_xlabel('$\\phi / \pi$')
    ax.set_ylabel('$\\theta / \pi$')
    ax.text(0.1, 0.8, '$%5.3f \sin(%5.3f \\theta %+5.3f \\phi %+5.3f)$' % (
        popt[0], popt[1], popt[2], popt[3]), fontsize='x-small')

    ax = axs[1, 1]
    im = ax.contourf(tphi / np.pi, ttheta / np.pi, tw_ana, tw_levels, cmap=plt.get_cmap('RdBu'))
    fig.colorbar(im, ax=ax, orientation='vertical')
    ax.set_title('analitical solution')
    ax.set_xlabel('$\\phi / \pi$')
    ax.set_ylabel('$\\theta / \pi$')
    ax.text(0.1, 0.8, '$%5.3f \sin(%5.3f \\theta %+5.3f \\phi %+5.3f)$' % (
        Amp_use, w_th_use, w_ph_use, alpha_use), fontsize='x-small')
    plt.tight_layout()
    return True


# The following code are used to do 3D FFT an 3D IFFT of \omega(\theta, \phi, psi)
#   of microswimmer in shear flow.
def do_3dfft_major(tw, tktltj_list, print_info=True):
    # do FFT of velocity component and pick major frequence, then IFFT.
    tw_fft = np.fft.fftn(tw)
    ntk, ntl, ntj = tw_fft.shape
    idx = np.ones_like(tw_fft) * 1e-30
    for tk1, tl1, tj1 in tktltj_list:
        tk2 = ntk - tk1 if tk1 > 0 else tk1
        tl2 = ntl - tl1 if tl1 > 0 else tl1
        tj2 = ntj - tj1 if tj1 > 0 else tj1
        idx[tk1, tl1, tj1] = 1
        idx[tk2, tl2, tj2] = 1
        tf1 = tw_fft[tk1, tl1, tj1]
        tf2 = tw_fft[tk2, tl2, tj2]
        if print_info:
            if tk1 > 0 or tl1 > 0 or tj1 > 0:
                print('use frequence pairs %f%+fi and %f%+fi at (%d, %d, %d) and (%d, %d, %d)' % (
                    tf1.real, tf1.imag, tf2.real, tf2.imag, tk1, tl1, tj1, tk2, tl2, tj2))
            else:
                print('use frequence %f%+fi at (%d, %d, %d)' % (tf1.real, tf1.imag, tk1, tl1, tj1))

    tw_fft2 = tw_fft * idx
    tw2 = np.fft.ifftn(tw_fft2)
    print('absolute abs of imag part is', np.abs(tw2.imag).max())
    return tw_fft, tw2.real, tw_fft2


def do_3dfft_major_conj(tw, tktltj_list, print_info=True):
    # do FFT of velocity component and pick major frequence, then IFFT.
    tw_fft = np.fft.fftn(tw)
    tM, tN, tO = tw_fft.shape

    tw2 = np.zeros_like(tw)
    tm, tn, to = np.meshgrid(np.arange(tM), np.arange(tN), np.arange(tO), indexing='ij')
    ttheta = tm / tM * np.pi
    tphi = tn / tN * 2 * np.pi
    tpsi = to / tO * 2 * np.pi
    idx = np.ones_like(tw_fft) * 1e-30
    for tk1, tl1, tj1 in tktltj_list:
        tk2 = tM - tk1 if tk1 > 0 else tk1
        tl2 = tN - tl1 if tl1 > 0 else tl1
        tj2 = tO - tj1 if tj1 > 0 else tj1
        idx[tk1, tl1, tj1] = 1
        idx[tk2, tl2, tj2] = 1
        tf1 = tw_fft[tk1, tl1, tj1]
        tf2 = tw_fft[tk2, tl2, tj2]
        if print_info:
            if tk1 > 0 or tl1 > 0 or tj1 > 0:
                print('use frequence pairs %f%+fi and %f%+fi at (%d, %d, %d) and (%d, %d, %d)' % (
                    tf1.real, tf1.imag, tf2.real, tf2.imag, tk1, tl1, tj1, tk2, tl2, tj2))
            else:
                print('use frequence %f%+fi at (%d, %d, %d)' % (tf1.real, tf1.imag, tk1, tl1, tj1))
        tfct = 1 if np.allclose(np.array((tk1, tl1, tj1)), np.zeros(3)) else 2
        tw2 = tw2 + tfct / (tM * tN * tO) * \
              (np.real(tf1) * np.cos(2 * tk1 * ttheta + tl1 * tphi + tj1 * tpsi) -
               np.imag(tf1) * np.sin(2 * tk1 * ttheta + tl1 * tphi + tj1 * tpsi))
    tw_fft2 = tw_fft * idx
    return tw_fft, tw2, tw_fft2


def factor_wpi_klj(tw, tktltj):
    # see decouplingIdea.tex for detail.
    # \omega_{pi}^{kl}(\theta, \phi, \psi) = \dfrac{2}{n_\theta n_\phi}
    #     \left(\Re(\Omega_{pi}(k,l, \psi)) \cos(2k\theta + l\phi) -
    #           \Im(\Omega_{pi}(k,l, \psi)) \sin(2k\theta + l\phi) \right)
    # \omega_{pi}^{kl}(\theta, \phi, \psi) = \dfrac{2}{n_\theta n_\phi}
    #     \norm{\Omega_{pi}(k,l, \psi)} \sin(\alpha_0 + 2k\theta + l\phi)
    # Amp_use = \dfrac{2}{n_\theta n_\phi}\norm{\Omega_{pi}(k,l, \psi)}
    # w_th_use = 2k
    # w_ph_use = l
    # alpha_use = \alpha_0

    err_msg = 'do NOT test yet. '
    assert 1 == 2, err_msg

    tk1, tl1, tj1 = tktltj
    nth, nph, nps = tw.shape
    tw_fft = np.fft.fftn(tw)
    Akl1 = tw_fft[tk1, tl1, tj1]
    Aklr = Akl1.real
    Akli = Akl1.imag

    k_sign = 1 if tk1 < (nth / 2) else -1
    l_sign = 1 if tl1 < (nph / 2) else -1
    j_sing = 1 if tl1 < (nps / 2) else -1
    Amp_use = 2 * np.abs(Akl1) / tw.size * k_sign * l_sign * j_sing
    w_th_use = 2 * tk1 if tk1 < (nth / 2) else -2 * (nth - tk1)
    w_ph_use = tl1 if tl1 < (nph / 2) else -1 * (nph - tl1)
    w_ps_use = tj1 if tj1 < (nps / 2) else -1 * (nps - tj1)
    alpha_use = -np.arctan(Aklr / Akli)
    return Akl1, Amp_use, w_th_use, w_ph_use, w_ps_use, alpha_use


def fill_Ui(ttheta, tphi, use_U):
    if tphi[-1] < (2 * np.pi):
        tphi = np.hstack((tphi, 2 * np.pi))
        use_U = np.vstack((use_U.T, use_U[:, -1])).T
    if ttheta[-1] < (np.pi):
        ttheta = np.hstack((ttheta, np.pi))
        use_U = np.vstack((use_U, use_U[-1]))
    return ttheta, tphi, use_U


def _get_fig_axs_ui_psi(tw, dpi=100, polar=False):
    if tw.shape[-1] == 15:
        fig = plt.figure(figsize=np.array((16, 9)) * 2, dpi=dpi)
        fig.patch.set_facecolor('white')
        axs = fig.subplots(nrows=3, ncols=5, subplot_kw=dict(polar=polar))
    elif tw.shape[-1] == 16:
        fig = plt.figure(figsize=np.array((16, 9)) * 2, dpi=dpi)
        fig.patch.set_facecolor('white')
        axs = fig.subplots(nrows=4, ncols=4, subplot_kw=dict(polar=polar))
    elif tw.shape[-1] == 2:
        fig = plt.figure(figsize=np.array((16, 9)) * 2, dpi=dpi)
        fig.patch.set_facecolor('white')
        axs = np.array(fig.subplots(nrows=1, ncols=1, subplot_kw=dict(polar=polar))).reshape((1, 1))
    else:
        raise ValueError("currently, amount of psi is either 15 or 16. ")
    return fig, axs


def core_show_ui_psi(tw, ttheta0, tphi0, tpsi, dpi=100, polar=False):
    fig, axs = _get_fig_axs_ui_psi(tw, dpi=dpi, polar=polar)
    cmap = plt.get_cmap('RdBu')
    t1 = np.nanmax(np.abs(tw))
    n_polar_xticks = 8
    # noinspection PyTypeChecker
    levels = np.linspace(-t1, t1, 10)
    for i0, ax0 in zip(range(tw.shape[-1]), axs.flatten()):
        ttheta, tphi, use_U = fill_Ui(ttheta0.copy(), tphi0.copy(), tw[..., i0])
        if polar:
            im = ax0.contourf(tphi, ttheta, use_U, levels, cmap=cmap)
            xticks = np.arange(n_polar_xticks)
            ax0.set_xticks(xticks / n_polar_xticks * 2 * np.pi)
            ax0.set_xticklabels(['$\dfrac{%d}{%d}2\pi$' % (i0, n_polar_xticks) for i0 in xticks])
            ax0.set_yticklabels([])
            ax0.set_ylim(0, np.pi)
        else:
            im = ax0.contourf(tphi / np.pi, ttheta / np.pi, use_U, levels, cmap=cmap)
            ax0.set_xlabel('$\\phi / \pi$')
            ax0.set_ylabel('$\\theta / \pi$')
        ax0.set_title('$\\psi=%f \pi$' % (tpsi[i0] / np.pi))
        fig.colorbar(im, ax=ax0, orientation='vertical')
    plt.tight_layout()
    return fig


def show_ui_psi(tw, ttheta, tphi, tpsi, dpi=100, polar=False):
    core_show_ui_psi(tw, ttheta, tphi, tpsi, dpi=dpi, polar=polar)
    return True


def show_3dfft_major(tw, tktltj_list, ttheta, tphi, tpsi, dpi=100, polar=False):
    tw_fft, tw2, tw_fft2 = do_3dfft_major(tw, tktltj_list)
    core_show_ui_psi(tw, ttheta, tphi, tpsi, dpi=dpi, polar=polar)
    core_show_ui_psi(tw2, ttheta, tphi, tpsi, dpi=dpi, polar=polar)
    return True


def Rloc2glb(theta, phi, psi):
    Rloc2glb = np.array(
            ((np.cos(phi) * np.cos(psi) * np.cos(theta) - np.sin(phi) * np.sin(psi),
              -(np.cos(psi) * np.sin(phi)) - np.cos(phi) * np.cos(theta) * np.sin(psi),
              np.cos(phi) * np.sin(theta)),
             (np.cos(psi) * np.cos(theta) * np.sin(phi) + np.cos(phi) * np.sin(psi),
              np.cos(phi) * np.cos(psi) - np.cos(theta) * np.sin(phi) * np.sin(psi),
              np.sin(phi) * np.sin(theta)),
             (-(np.cos(psi) * np.sin(theta)),
              np.sin(psi) * np.sin(theta),
              np.cos(theta))))
    return Rloc2glb


def Eij_loc(theta, phi, psi):
    Eij_loc = np.array(
            ((np.cos(psi) * (-(np.cos(phi) * np.cos(psi) * np.cos(theta)) +
                             np.sin(phi) * np.sin(psi)) * np.sin(theta),
              (2 * np.cos(2 * psi) * np.sin(phi) * np.sin(theta) +
               np.cos(phi) * np.sin(2 * psi) * np.sin(2 * theta)) / 4.,
              (np.cos(phi) * np.cos(psi) * np.cos(2 * theta) -
               np.cos(theta) * np.sin(phi) * np.sin(psi)) / 2.),
             ((2 * np.cos(2 * psi) * np.sin(phi) * np.sin(theta) +
               np.cos(phi) * np.sin(2 * psi) * np.sin(2 * theta)) / 4.,
              -(np.sin(psi) * (np.cos(psi) * np.sin(phi) +
                               np.cos(phi) * np.cos(theta) * np.sin(psi)) * np.sin(theta)),
              (-(np.cos(psi) * np.cos(theta) * np.sin(phi)) -
               np.cos(phi) * np.cos(2 * theta) * np.sin(psi)) / 2.),
             ((np.cos(phi) * np.cos(psi) * np.cos(2 * theta) -
               np.cos(theta) * np.sin(phi) * np.sin(psi)) / 2.,
              (-(np.cos(psi) * np.cos(theta) * np.sin(phi)) -
               np.cos(phi) * np.cos(2 * theta) * np.sin(psi)) / 2.,
              np.cos(phi) * np.cos(theta) * np.sin(theta))))
    return Eij_loc


def Sij_loc(theta, phi, psi):
    Sij_loc = np.array(
            ((0,
              -(np.sin(phi) * np.sin(theta)) / 2.,
              (np.cos(phi) * np.cos(psi) - np.cos(theta) * np.sin(phi) * np.sin(psi)) / 2.),
             ((np.sin(phi) * np.sin(theta)) / 2.,
              0,
              (-(np.cos(psi) * np.cos(theta) * np.sin(phi)) - np.cos(phi) * np.sin(psi)) / 2.),
             ((-(np.cos(phi) * np.cos(psi)) + np.cos(theta) * np.sin(phi) * np.sin(psi)) / 2.,
              (np.cos(psi) * np.cos(theta) * np.sin(phi) + np.cos(phi) * np.sin(psi)) / 2.,
              0)))
    return Sij_loc

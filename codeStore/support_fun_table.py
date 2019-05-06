# coding: utf-8
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 11:05:23 2017

@author: zhangji
"""

from matplotlib import pyplot as plt

plt.rcParams['figure.figsize'] = (18.5, 10.5)
fontsize = 40

import numpy as np
import pandas as pd
import scipy as sp
from scipy.optimize import curve_fit
from scipy import interpolate, integrate
from scipy.io import loadmat, savemat
from src import jeffery_model as jm

import matplotlib
import matplotlib.colors as colors
from matplotlib import animation, rc
import matplotlib.ticker as mtick
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, zoomed_inset_axes
from mpl_toolkits.mplot3d import Axes3D, axes3d
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib import cm

import re
from scanf import scanf
import natsort
import importlib
from tqdm import tqdm_notebook
import os
import glob
import natsort
import pickle
from time import time
from src.support_class import *
from codeStore import support_fun as spf

PWD = os.getcwd()
font = {'size': 20}
matplotlib.rc('font', **font)
np.set_printoptions(linewidth=90, precision=5)

markerstyle_list = ['^', 'v', 'o', 's', 'p', 'd', 'H',
                    '1', '2', '3', '4', '8', 'P', '*',
                    'h', '+', 'x', 'X', 'D', '|', '_', ]


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
    omega_norm = np.array([np.dot(t1, t2) * t2 / np.dot(t2, t2) for t1, t2 in zip(using_U[:, 3:], ecoli_norm)])
    omega_tang = using_U[:, 3:] - omega_norm

    return ecoli_U, ecoli_norm, ecoli_center, ecoli_lateral_norm, norm_tpp, \
           ecoli_u000, ecoli_center000, omega_norm, omega_tang, planeShearRate, file_handle


def get_ecoli_table(tnorm, lateral_norm, tcenter, max_iter, eval_dt=0.001, update_order=1,
                    planeShearRate=np.array((1, 0, 0))):
    from time import time
    ellipse_kwargs = {'name':         'ecoli_torque',
                      'center':       tcenter,
                      'norm':         tnorm / np.linalg.norm(tnorm),
                      'lateral_norm': lateral_norm / np.linalg.norm(lateral_norm),
                      'speed':        0,
                      'lbd':          np.nan,
                      'omega_tail':   193.66659814,
                      'table_name':   'planeShearRatex_1c', }
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


def do_calculate_helix_Petsc(norm, ini_psi, max_t, update_fun='3bs', rtol=1e-6, atol=1e-9, eval_dt=0.001):
    importlib.reload(jm)
    norm = norm / np.linalg.norm(norm)
    planeShearRate = np.array((1, 0, 0))
    tcenter = np.zeros(3)
    tlateral_norm = np.random.sample(3)
    tlateral_norm = tlateral_norm / np.linalg.norm(tlateral_norm)
    tlateral_norm = tlateral_norm - norm * np.dot(norm, tlateral_norm)
    tlateral_norm = tlateral_norm / np.linalg.norm(tlateral_norm)
    P0 = norm / np.linalg.norm(norm)
    P20 = tlateral_norm / np.linalg.norm(tlateral_norm)
    helix_kwargs = {'name':         'helix',
                    'center':       tcenter,
                    'norm':         P0,
                    'lateral_norm': P20,
                    'speed':        0,
                    'lbd':          np.nan,
                    'ini_psi':      ini_psi,
                    'table_name':   'hlxB01_tau1a', }
    fileHandle = 'ShearTableProblem'
    helix_obj = jm.TablePetscObj(**helix_kwargs)
    helix_obj.set_update_para(fix_x=False, fix_y=False, fix_z=False,
                              update_fun=update_fun, rtol=rtol, atol=atol)
    problem = jm.ShearTableProblem(name=fileHandle, planeShearRate=planeShearRate)
    problem.add_obj(helix_obj)
    Table_t, Table_X, Table_P, Table_P2 = helix_obj.update_self(t1=max_t, eval_dt=eval_dt)
    Table_theta, Table_phi, Table_psi = helix_obj.theta_phi_psi
    Table_eta = np.arccos(np.sin(Table_theta) * np.sin(Table_phi))
    return Table_t, Table_X, Table_P, Table_P2, Table_theta, Table_phi, Table_psi, Table_eta


def do_calculate_helix_RK(norm, ini_psi, max_t, update_fun=integrate.RK45, rtol=1e-6, atol=1e-9):
    importlib.reload(jm)
    norm = norm / np.linalg.norm(norm)
    planeShearRate = np.array((1, 0, 0))
    tcenter = np.zeros(3)
    tlateral_norm = np.random.sample(3)
    tlateral_norm = tlateral_norm / np.linalg.norm(tlateral_norm)
    tlateral_norm = tlateral_norm - norm * np.dot(norm, tlateral_norm)
    tlateral_norm = tlateral_norm / np.linalg.norm(tlateral_norm)
    P0 = norm / np.linalg.norm(norm)
    P20 = tlateral_norm / np.linalg.norm(tlateral_norm)
    helix_kwargs = {'name':         'helix',
                    'center':       tcenter,
                    'norm':         P0,
                    'lateral_norm': P20,
                    'speed':        0,
                    'lbd':          np.nan,
                    'ini_psi':      ini_psi,
                    'table_name':   'hlxB01_tau1a', }
    fileHandle = 'ShearTableProblem'
    helix_obj = jm.TableRtObj(**helix_kwargs)
    helix_obj.set_update_para(fix_x=False, fix_y=False, fix_z=False,
                              update_fun=update_fun, rtol=rtol, atol=atol)
    problem = jm.ShearTableProblem(name=fileHandle, planeShearRate=planeShearRate)
    problem.add_obj(helix_obj)
    Table_t, Table_X, Table_P, Table_P2 = helix_obj.update_self(t1=max_t)
    Table_theta, Table_phi, Table_psi = helix_obj.theta_phi_psi
    Table_eta = np.arccos(np.sin(Table_theta) * np.sin(Table_phi))
    return Table_t, Table_X, Table_P, Table_P2, Table_theta, Table_phi, Table_psi, Table_eta


def do_calculate_ellipse_RK(norm, ini_psi, max_t, update_fun=integrate.RK45, rtol=1e-6, atol=1e-9):
    importlib.reload(jm)
    norm = norm / np.linalg.norm(norm)
    planeShearRate = np.array((1, 0, 0))
    tcenter = np.zeros(3)
    tlateral_norm = np.random.sample(3)
    tlateral_norm = tlateral_norm / np.linalg.norm(tlateral_norm)
    tlateral_norm = tlateral_norm - norm * np.dot(norm, tlateral_norm)
    tlateral_norm = tlateral_norm / np.linalg.norm(tlateral_norm)
    P0 = norm / np.linalg.norm(norm)
    P20 = tlateral_norm / np.linalg.norm(tlateral_norm)
    ellipse_kwargs = {'name':         'ellipse',
                      'center':       tcenter,
                      'norm':         P0,
                      'lateral_norm': P20,
                      'speed':        0,
                      'lbd':          np.nan,
                      'ini_psi':      ini_psi,
                      'table_name':   'ellipse_alpha3', }
    fileHandle = 'ShearTableProblem'
    helix_obj = jm.TableRtObj(**ellipse_kwargs)
    helix_obj.set_update_para(fix_x=False, fix_y=False, fix_z=False,
                              update_fun=update_fun, rtol=rtol, atol=atol)
    problem = jm.ShearTableProblem(name=fileHandle, planeShearRate=planeShearRate)
    problem.add_obj(helix_obj)
    Table_t, Table_X, Table_P, Table_P2 = helix_obj.update_self(t1=max_t)
    Table_theta, Table_phi, Table_psi = helix_obj.theta_phi_psi
    Table_eta = np.arccos(np.sin(Table_theta) * np.sin(Table_phi))
    return Table_t, Table_X, Table_P, Table_P2, Table_theta, Table_phi, Table_psi, Table_eta


def do_calculate_ecoli_Petsc(norm, ini_psi, max_t, update_fun='3bs', rtol=1e-6, atol=1e-9,
                             omega_tail=193.66659814, table_name='planeShearRatex_1d',
                             eval_dt=0.001, save_every=1):
    importlib.reload(jm)
    norm = norm / np.linalg.norm(norm)
    planeShearRate = np.array((1, 0, 0))
    tcenter = np.zeros(3)
    tlateral_norm = np.random.sample(3)
    tlateral_norm = tlateral_norm / np.linalg.norm(tlateral_norm)
    tlateral_norm = tlateral_norm - norm * np.dot(norm, tlateral_norm)
    tlateral_norm = tlateral_norm / np.linalg.norm(tlateral_norm)
    P0 = norm / np.linalg.norm(norm)
    P20 = tlateral_norm / np.linalg.norm(tlateral_norm)
    ecoli_kwargs = {'name':         'ecoli_torque',
                    'center':       tcenter,
                    'norm':         P0,
                    'lateral_norm': P20,
                    'speed':        0,
                    'lbd':          np.nan,
                    'ini_psi':      ini_psi,
                    'omega_tail':   omega_tail,
                    'table_name':   table_name, }
    fileHandle = 'ShearTableProblem'
    helix_obj = jm.TablePetscEcoli(**ecoli_kwargs)
    helix_obj.set_update_para(fix_x=False, fix_y=False, fix_z=False,
                              update_fun=update_fun, rtol=rtol, atol=atol, save_every=save_every)
    problem = jm.ShearTableProblem(name=fileHandle, planeShearRate=planeShearRate)
    problem.add_obj(helix_obj)
    Table_t, Table_X, Table_P, Table_P2 = helix_obj.update_self(t1=max_t, eval_dt=eval_dt)
    Table_theta, Table_phi, Table_psi = helix_obj.theta_phi_psi
    Table_eta = np.arccos(np.sin(Table_theta) * np.sin(Table_phi))
    return Table_t, Table_X, Table_P, Table_P2, Table_theta, Table_phi, Table_psi, Table_eta


def do_calculate_ecoli_RK(norm, ini_psi, max_t, update_fun=integrate.RK45, rtol=1e-6, atol=1e-9,
                          omega_tail=193.66659814, table_name='planeShearRatex_1d'):
    importlib.reload(jm)
    norm = norm / np.linalg.norm(norm)
    planeShearRate = np.array((1, 0, 0))
    tcenter = np.zeros(3)
    tlateral_norm = np.random.sample(3)
    tlateral_norm = tlateral_norm / np.linalg.norm(tlateral_norm)
    tlateral_norm = tlateral_norm - norm * np.dot(norm, tlateral_norm)
    tlateral_norm = tlateral_norm / np.linalg.norm(tlateral_norm)
    P0 = norm / np.linalg.norm(norm)
    P20 = tlateral_norm / np.linalg.norm(tlateral_norm)
    ecoli_kwargs = {'name':         'ecoli_torque',
                    'center':       tcenter,
                    'norm':         P0,
                    'lateral_norm': P20,
                    'speed':        0,
                    'lbd':          np.nan,
                    'ini_psi':      ini_psi,
                    'omega_tail':   omega_tail,
                    'table_name':   table_name, }
    fileHandle = 'ShearTableProblem'
    helix_obj = jm.TableRtEcoli(**ecoli_kwargs)
    helix_obj.set_update_para(fix_x=False, fix_y=False, fix_z=False,
                              update_fun=update_fun, rtol=rtol, atol=atol)
    problem = jm.ShearTableProblem(name=fileHandle, planeShearRate=planeShearRate)
    problem.add_obj(helix_obj)
    Table_t, Table_X, Table_P, Table_P2 = helix_obj.update_self(t1=max_t)
    Table_theta, Table_phi, Table_psi = helix_obj.theta_phi_psi
    Table_eta = np.arccos(np.sin(Table_theta) * np.sin(Table_phi))
    return Table_t, Table_X, Table_P, Table_P2, Table_theta, Table_phi, Table_psi, Table_eta


def do_calculate_ecoli(norm, ini_psi, eval_dt=0.1, max_iter=1000,
                       omega_tail=193.66659814, table_name='planeShearRatex_1d'):
    importlib.reload(jm)
    norm = norm / np.linalg.norm(norm)
    planeShearRate = np.array((1, 0, 0))
    tcenter = np.zeros(3)
    tlateral_norm = np.random.sample(3)
    tlateral_norm = tlateral_norm / np.linalg.norm(tlateral_norm)
    tlateral_norm = tlateral_norm - norm * np.dot(norm, tlateral_norm)
    tlateral_norm = tlateral_norm / np.linalg.norm(tlateral_norm)
    P0 = norm / np.linalg.norm(norm)
    P20 = tlateral_norm / np.linalg.norm(tlateral_norm)
    ecoli_kwargs = {'name':         'ecoli_torque',
                    'center':       tcenter,
                    'norm':         P0,
                    'lateral_norm': P20,
                    'speed':        0,
                    'lbd':          np.nan,
                    'ini_psi':      ini_psi,
                    'omega_tail':   omega_tail,
                    'table_name':   table_name, }
    fileHandle = 'ShearTableProblem'
    helix_obj = jm.TableEcoli(**ecoli_kwargs)
    helix_obj.set_update_para(fix_x=False, fix_y=False, fix_z=False, update_order=1)
    problem = jm.ShearTableProblem(name=fileHandle, planeShearRate=planeShearRate)
    problem.add_obj(helix_obj)
    for idx in tqdm_notebook(range(1, max_iter + 1)):
        problem.update_location(eval_dt, print_handle='%d / %d' % (idx, max_iter))
    Table_X = np.vstack(helix_obj.center_hist)
    Table_U = np.vstack(helix_obj.U_hist)
    Table_P = np.vstack(helix_obj.norm_hist)
    Table_P2 = np.vstack(helix_obj.lateral_norm_hist)
    Table_t = np.arange(max_iter) * eval_dt + eval_dt
    Table_theta, Table_phi, Table_psi = helix_obj.theta_phi_psi
    Table_eta = np.arccos(np.sin(Table_theta) * np.sin(Table_phi))
    omega = Table_U[:, 3:]
    dP = np.vstack([np.cross(t1, t2) for t1, t2 in zip(omega, Table_P)])
    Table_dtheta = -dP[:, 2] / np.sin(np.abs(Table_theta))
    Table_dphi = (dP[:, 1] * np.cos(Table_phi) - dP[:, 0] * np.sin(Table_phi)) / np.sin(Table_theta)
    return Table_t, Table_X, Table_P, Table_P2, Table_theta, Table_phi, Table_psi, Table_eta


def do_calculate_ecoli_passive_Petsc(norm, ini_psi, max_t, update_fun='3bs', rtol=1e-6, atol=1e-9, eval_dt=0.001):
    importlib.reload(jm)
    norm = norm / np.linalg.norm(norm)
    planeShearRate = np.array((1, 0, 0))
    tcenter = np.zeros(3)
    tlateral_norm = np.random.sample(3)
    tlateral_norm = tlateral_norm / np.linalg.norm(tlateral_norm)
    tlateral_norm = tlateral_norm - norm * np.dot(norm, tlateral_norm)
    tlateral_norm = tlateral_norm / np.linalg.norm(tlateral_norm)
    P0 = norm / np.linalg.norm(norm)
    P20 = tlateral_norm / np.linalg.norm(tlateral_norm)
    ecoli_kwargs = {'name':         'ecoli_passive',
                    'center':       tcenter,
                    'norm':         P0,
                    'lateral_norm': P20,
                    'speed':        0,
                    'lbd':          np.nan,
                    'ini_psi':      ini_psi,
                    'table_name':   'planeShearRatex_1d_passive', }
    fileHandle = 'ShearTableProblem'
    helix_obj = jm.TablePetscObj(**ecoli_kwargs)
    helix_obj.set_update_para(fix_x=False, fix_y=False, fix_z=False,
                              update_fun=update_fun, rtol=rtol, atol=atol)
    problem = jm.ShearTableProblem(name=fileHandle, planeShearRate=planeShearRate)
    problem.add_obj(helix_obj)
    Table_t, Table_X, Table_P, Table_P2 = helix_obj.update_self(t1=max_t, eval_dt=eval_dt)
    Table_theta, Table_phi, Table_psi = helix_obj.theta_phi_psi
    Table_eta = np.arccos(np.sin(Table_theta) * np.sin(Table_phi))
    return Table_t, Table_X, Table_P, Table_P2, Table_theta, Table_phi, Table_psi, Table_eta


def do_calculate_ecoli_passive_RK(norm, ini_psi, max_t, update_fun=integrate.RK45, rtol=1e-6, atol=1e-9):
    importlib.reload(jm)
    norm = norm / np.linalg.norm(norm)
    planeShearRate = np.array((1, 0, 0))
    tcenter = np.zeros(3)
    tlateral_norm = np.random.sample(3)
    tlateral_norm = tlateral_norm / np.linalg.norm(tlateral_norm)
    tlateral_norm = tlateral_norm - norm * np.dot(norm, tlateral_norm)
    tlateral_norm = tlateral_norm / np.linalg.norm(tlateral_norm)
    P0 = norm / np.linalg.norm(norm)
    P20 = tlateral_norm / np.linalg.norm(tlateral_norm)
    ecoli_kwargs = {'name':         'ecoli_passive',
                    'center':       tcenter,
                    'norm':         P0,
                    'lateral_norm': P20,
                    'speed':        0,
                    'lbd':          np.nan,
                    'ini_psi':      ini_psi,
                    'table_name':   'planeShearRatex_1d_passive', }
    fileHandle = 'ShearTableProblem'
    helix_obj = jm.TableRtObj(**ecoli_kwargs)
    helix_obj.set_update_para(fix_x=False, fix_y=False, fix_z=False,
                              update_fun=update_fun, rtol=rtol, atol=atol)
    problem = jm.ShearTableProblem(name=fileHandle, planeShearRate=planeShearRate)
    problem.add_obj(helix_obj)
    Table_t, Table_X, Table_P, Table_P2 = helix_obj.update_self(t1=max_t)
    Table_theta, Table_phi, Table_psi = helix_obj.theta_phi_psi
    Table_eta = np.arccos(np.sin(Table_theta) * np.sin(Table_phi))
    return Table_t, Table_X, Table_P, Table_P2, Table_theta, Table_phi, Table_psi, Table_eta


def do_calculate_ecoli_passive(norm, ini_psi, eval_dt=0.1, max_iter=1000):
    importlib.reload(jm)
    norm = norm / np.linalg.norm(norm)
    planeShearRate = np.array((1, 0, 0))
    tcenter = np.zeros(3)
    tlateral_norm = np.random.sample(3)
    tlateral_norm = tlateral_norm / np.linalg.norm(tlateral_norm)
    tlateral_norm = tlateral_norm - norm * np.dot(norm, tlateral_norm)
    tlateral_norm = tlateral_norm / np.linalg.norm(tlateral_norm)
    P0 = norm / np.linalg.norm(norm)
    P20 = tlateral_norm / np.linalg.norm(tlateral_norm)
    ecoli_kwargs = {'name':         'ecoli_passive',
                    'center':       tcenter,
                    'norm':         P0,
                    'lateral_norm': P20,
                    'speed':        0,
                    'lbd':          np.nan,
                    'ini_psi':      ini_psi,
                    'omega_tail':   193.66659814,
                    'table_name':   'planeShearRatex_1d_passive', }
    fileHandle = 'ShearTableProblem'
    helix_obj = jm.TableEcoli(**ecoli_kwargs)
    helix_obj.set_update_para(fix_x=False, fix_y=False, fix_z=False, update_order=1)
    problem = jm.ShearTableProblem(name=fileHandle, planeShearRate=planeShearRate)
    problem.add_obj(helix_obj)
    for idx in tqdm_notebook(range(1, max_iter + 1)):
        problem.update_location(eval_dt, print_handle='%d / %d' % (idx, max_iter))
    Table_X = np.vstack(helix_obj.center_hist)
    Table_U = np.vstack(helix_obj.U_hist)
    Table_P = np.vstack(helix_obj.norm_hist)
    Table_P2 = np.vstack(helix_obj.lateral_norm_hist)
    Table_t = np.arange(max_iter) * eval_dt + eval_dt
    Table_theta, Table_phi, Table_psi = helix_obj.theta_phi_psi
    Table_eta = np.arccos(np.sin(Table_theta) * np.sin(Table_phi))
    omega = Table_U[:, 3:]
    dP = np.vstack([np.cross(t1, t2) for t1, t2 in zip(omega, Table_P)])
    Table_dtheta = -dP[:, 2] / np.sin(np.abs(Table_theta))
    Table_dphi = (dP[:, 1] * np.cos(Table_phi) - dP[:, 0] * np.sin(Table_phi)) / np.sin(Table_theta)
    return Table_t, Table_X, Table_P, Table_P2, Table_theta, Table_phi, Table_psi, Table_eta


def core_show_table_result(Table_t, Table_theta, Table_phi, Table_psi, Table_eta, Table_X, save_every=1):
    # show table results.
    fig = plt.figure(figsize=(20, 15))
    fig.patch.set_facecolor('white')
    ax0 = plt.subplot2grid((7, 6), (0, 0), rowspan=3, colspan=3, polar=True)
    ax1 = plt.subplot2grid((7, 6), (0, 3), colspan=3)
    ax2 = plt.subplot2grid((7, 6), (1, 3), colspan=3)
    ax3 = plt.subplot2grid((7, 6), (2, 3), colspan=3)
    ax4 = plt.subplot2grid((7, 6), (3, 3), colspan=3)
    axt = plt.subplot2grid((7, 6), (3, 0), colspan=3)
    ax5 = plt.subplot2grid((7, 6), (4, 0), rowspan=3, colspan=2)
    ax6 = plt.subplot2grid((7, 6), (4, 2), rowspan=3, colspan=2)
    ax7 = plt.subplot2grid((7, 6), (4, 4), rowspan=3, colspan=2)
    # polar version
    norm = plt.Normalize(Table_t.min(), Table_t.max())
    cmap = plt.get_cmap('jet')
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
    tfct = 10 ** np.floor(np.log10(Table_t.max())) / 100
    xticks = np.around(np.linspace(Table_t.min() / tfct, Table_t.max() / tfct, 5)) * tfct
    for axi, ty, axyi in zip((ax1, ax2, ax3, ax4, ax5, ax6, ax7, axt),
                             (Table_theta / np.pi, Table_phi / np.pi, Table_psi / np.pi, Table_eta / np.pi,
                              Table_X[:, 0], Table_X[:, 1], Table_X[:, 2],
                              np.hstack((np.diff(Table_t), 0)) / save_every),
                             ('$\\theta / \pi$', '$\\phi / \pi$', '$\\psi / \pi$', '$\\eta / \pi$',
                              '$center_x$', '$center_y$', '$center_z$', 'dt')):
        plt.sca(axi)
        axi.plot(Table_t, ty, '-*', label='Table')
        #     axi.set_xlabel('t', size=fontsize)
        #     axi.legend()
        axi.set_ylabel('%s' % axyi, size=fontsize * 0.7)
        axi.set_xticks(xticks)
        axi.set_xticklabels(xticks)
        plt.xticks(fontsize=fontsize * 0.5)
        plt.yticks(fontsize=fontsize * 0.5)
    for axi in (ax4, ax5, ax6, ax7):
        axi.set_xlabel('t', size=fontsize * 0.7)
    plt.tight_layout()
    return fig


def show_table_result(Table_t, Table_theta, Table_phi, Table_psi, Table_eta, Table_X, save_every=1):
    core_show_table_result(Table_t, Table_theta, Table_phi, Table_psi, Table_eta, Table_X, save_every)
    return True


def save_table_result(finename, Table_t, Table_theta, Table_phi, Table_psi, Table_eta, Table_X, save_every=1):
    fig = core_show_table_result(Table_t, Table_theta, Table_phi, Table_psi, Table_eta, Table_X, save_every)
    fig.savefig(finename, dpi=100)
    plt.close()
    return True


def make_table_video(Table_t, Table_X, Table_P, Table_P2,
                     Table_theta, Table_phi, Table_psi, Table_eta,
                     zm_fct=1, stp=1, interval=50):
    tl = Table_t.size
    video_length = tl // stp

    fig = plt.figure(figsize=(20, 15))
    fig.patch.set_facecolor('white')
    ax0 = plt.subplot2grid((6, 8), (0, 0), rowspan=6, colspan=6, projection='3d')
    ax6 = plt.subplot2grid((6, 8), (5, 6), colspan=2)
    ax1 = plt.subplot2grid((6, 8), (0, 6), colspan=2, sharex=ax6)
    ax2 = plt.subplot2grid((6, 8), (1, 6), colspan=2, sharex=ax6)
    ax3 = plt.subplot2grid((6, 8), (2, 6), colspan=2, sharex=ax6)
    ax4 = plt.subplot2grid((6, 8), (3, 6), colspan=2, sharex=ax6)
    ax5 = plt.subplot2grid((6, 8), (4, 6), colspan=2, sharex=ax6)
    for spine in ax0.spines.values():
        spine.set_visible(False)

    # stantart part
    tfct = 10 ** np.floor(np.log10(Table_t.max())) / 100
    xticks = np.around(np.linspace(Table_t.min() / tfct, Table_t.max() / tfct, 5)) * tfct
    for axi, ty, axyi in zip((ax1, ax2, ax3, ax4, ax5, ax6),
                             (Table_theta / np.pi, Table_phi / np.pi, Table_eta / np.pi,
                              Table_P[:, 0], Table_P[:, 1], Table_P[:, 2]),
                             ('$\\theta / \pi$', '$\\phi / \pi$', '$\\eta / \pi$',
                              '$Px$', '$Py$', '$Pz$')):
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

    # left part, animate of axis (which represent the object, i.e. helix, ecoli...)
    tX = Table_X[0]
    tP1 = Table_P[0]
    tP2 = Table_P2[0]
    tP1 = tP1 / np.linalg.norm(tP1) * zm_fct
    tP2 = tP2 / np.linalg.norm(tP2) * zm_fct
    tP3 = np.cross(tP1, tP2) / zm_fct
    tmp_line1 = ax0.plot([tX[0], tX[0] + tP1[0]],
                         [tX[1], tX[1] + tP1[1]],
                         [tX[2], tX[2] + tP1[2]], color='k')[0]
    tmp_line2 = ax0.plot([tX[0], tX[0] + tP2[0]],
                         [tX[1], tX[1] + tP2[1]],
                         [tX[2], tX[2] + tP2[2]], color='r')[0]
    tmp_line3 = ax0.plot([tX[0], tX[0] + tP3[0]],
                         [tX[1], tX[1] + tP3[1]],
                         [tX[2], tX[2] + tP3[2]], color='b')[0]
    trange = np.max(Table_X.max(axis=0) - Table_X.min(axis=0))
    tmid = (Table_X.max(axis=0) + Table_X.min(axis=0)) / 2
    ax0.set_xlim3d([tmid[0] - trange, tmid[0] + trange])
    ax0.set_xlabel('X')
    ax0.set_ylim3d([tmid[1] - trange, tmid[1] + trange])
    ax0.set_ylabel('Y')
    ax0.set_zlim3d([tmid[2] - trange, tmid[2] + trange])
    ax0.set_zlabel('Z')

    # right part, point indicates the time.
    scs = []
    for axi, ty, in zip((ax1, ax2, ax3, ax4, ax5, ax6),
                        (Table_theta / np.pi, Table_phi / np.pi, Table_eta / np.pi,
                         Table_P[:, 0], Table_P[:, 1], Table_P[:, 2])):
        plt.sca(axi)
        scs.append(axi.plot(Table_t[0], ty[0], 'or', markersize=fontsize * 0.3)[0])

    def update_fun(num, tl1, tl2, tl3, scs,
                   Table_t, Table_X, Table_P, Table_P2,
                   Table_theta, Table_phi, Table_psi, Table_eta,
                   zm_fct):
        num = num * stp

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

        # right, 2d plots
        for axi, ty, sci, in zip((ax1, ax2, ax3, ax4, ax5, ax6),
                                 (Table_theta / np.pi, Table_phi / np.pi, Table_eta / np.pi,
                                  Table_P[:, 0], Table_P[:, 1], Table_P[:, 2]),
                                 scs):
            sci.set_data(Table_t[num], ty[num])
        return tl1, tl2, tl3, scs

    anim = animation.FuncAnimation(fig, update_fun, video_length, interval=interval, blit=False,
                                   fargs=(tmp_line1, tmp_line2, tmp_line3, scs,
                                          Table_t, Table_X, Table_P, Table_P2,
                                          Table_theta, Table_phi, Table_psi, Table_eta,
                                          zm_fct), )
    return anim

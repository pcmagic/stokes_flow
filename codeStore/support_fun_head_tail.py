# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 11:05:23 2017

@author: zhangji
"""

# from datetime import datetime
# from time import time
# import dill
# import glob
# import importlib
# import numpy as np
# import scipy as sp
# import scipy.misc
# import pandas as pd
# import re
# import itertools
# import matplotlib.ticker as mtick
# from matplotlib.colors import ListedColormap, BoundaryNorm, PowerNorm, Normalize
# from mpl_toolkits.mplot3d import axes3d, Axes3D
# from sympy import symbols, simplify, series, exp
# from sympy.matrices import Matrix
# from sympy.solvers import solve
# from scipy import interpolate
# from ecoli_in_pipe import do_slenderbodytheory as do_SLB
# from tqdm.notebook import tqdm as tqdm_notebook
import pickle
from scanf import scanf
from scipy.optimize import leastsq, curve_fit
from scipy import interpolate, integrate, optimize, sparse
from scipy.interpolate import interp1d, interp2d
from IPython.display import display, HTML, Math
from src import slenderBodyTheory as slb
import os
import glob
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from codeStore import support_fun as spf

PWD = os.getcwd()
# font = {'size': 20}
# matplotlib.rc('font', **font)
np.set_printoptions(linewidth=110, precision=5)


def ax_base_fun(ax, ty, plotHeadle, xlabel='$r_t/R$', linestyle='*-'):
    for chi in ch:
        idxi = Idx.loc[chi]
        ti = ty.loc[chi]
        plotHeadle(zoom_factor[idxi], ti[idxi], linestyle, ms=10)
    ax.set_xlabel(xlabel, size='x-large')
    return True


def ax_plot(ax, ty, xlabel='$r_t/R$', linestyle='*-'):
    ax_base_fun(ax, ty, ax.plot, xlabel, linestyle)
    return True


def ax_semilogx(ax, ty, xlabel='$r_t/R$', linestyle='*-'):
    ax_base_fun(ax, ty, ax.semilogx, xlabel, linestyle)
    return True


def ax_semilogy(ax, ty, xlabel='$r_t/R$', linestyle='*-'):
    ax_base_fun(ax, ty, ax.semilogy, xlabel, linestyle)
    return True


# new theory: see paper 2019
def get_data_newTheory(subdir):
    t_dir = os.path.join(PWD, 'checkNewTheory', subdir)
    os.chdir(t_dir)
    txt_names = glob.glob('*.txt')
    uz = []
    wz = []
    zoom_factor = []
    pitch = []
    cycle = []
    lhInvrh = []
    rhInvrt = []

    for txt_name in txt_names:
        filename = os.path.join(t_dir, txt_name)
        with open(filename, 'r') as myinput:
            FILE_DATA = myinput.read()

        text_headle = ' non_dim_U \['
        temp1 = spf.read_array(text_headle, FILE_DATA, array_length=6)
        uz.append(temp1[2])
        wz.append(temp1[5])

        text_headle = ' geometry zoom factor is'
        temp1 = spf.read_array(text_headle, FILE_DATA, array_length=1)
        t_zf = temp1
        if np.isclose(t_zf, 1):
            t_zf = 0
        zoom_factor.append(t_zf)

        text_headle = ', helix pitch\:'
        temp1 = spf.read_array(text_headle, FILE_DATA, array_length=1)
        pitch.append(temp1)

        text_headle = ', helix cycle\:'
        temp1 = spf.read_array(text_headle, FILE_DATA, array_length=1)
        cycle.append(temp1)

        text_headle = '  head radius\:'
        temp1 = spf.read_array(text_headle, FILE_DATA, array_length=1)
        text_headle = '  helix radius\:'
        temp2 = spf.read_array(text_headle, FILE_DATA, array_length=1)
        text_headle = ', length\:'
        temp3 = spf.read_array(text_headle, FILE_DATA, array_length=1)
        lhInvrh.append(temp3 / temp1)
        rhInvrt.append(temp1 / temp2)

    os.chdir(PWD)

    data = pd.DataFrame({'uz':          np.hstack(uz),
                         'wz':          np.hstack(wz),
                         'zoom_factor': np.hstack(zoom_factor),
                         'pitch':       np.hstack(pitch),
                         'cycle':       np.hstack(cycle),
                         'lhInvrh':     np.hstack(lhInvrh),
                         'rhInvrt':     np.hstack(rhInvrt), })
    return data.pivot_table(index=['rhInvrt', 'cycle'], columns=['zoom_factor'])


def get_data_tail(subdir, myPWD=None):
    if myPWD is None:
        t_dir = os.path.join(PWD, 'tail', subdir)
    else:
        t_dir = os.path.join(myPWD, 'tail', subdir)
    os.chdir(t_dir)
    txt_names = glob.glob('*.txt')
    n_node = []
    matrix_time = []
    psi2 = []
    psi3 = []
    psi6 = []
    zoom_factor = []
    pitch = []
    cycle = []

    for txt_name in txt_names:
        filename = os.path.join(t_dir, txt_name)
        with open(filename, 'r') as myinput:
            FILE_DATA = myinput.read()

        text_headle = ', velocity nodes:'
        temp1 = spf.read_array(text_headle, FILE_DATA, array_length=1)
        n_node.append(temp1)

        text_headle = ': create matrix use:'
        temp1 = spf.read_array(text_headle, FILE_DATA, array_length=1)
        matrix_time.append(temp1)

        text_headle = ' geometry zoom factor is'
        temp1 = spf.read_array(text_headle, FILE_DATA, array_length=1)
        t_zf = temp1

        text_headle = 'tran tail resultant is \['
        temp1 = spf.read_array(text_headle, FILE_DATA, array_length=6)
        psi2.append(temp1[2] / t_zf ** 1)
        psi61 = temp1[5]
        text_headle = 'rota tail resultant is \['
        temp1 = spf.read_array(text_headle, FILE_DATA, array_length=6)
        psi62 = temp1[2]
        psi3.append(temp1[5] / t_zf ** 3)
        # psi6.append((psi61 + psi62) / 2 / t_zf ** 2)
        psi6.append(psi62 / t_zf ** 2)

        if np.abs(t_zf - 1) < 1e-3:
            t_zf = 0
        zoom_factor.append(t_zf)

        text_headle = ', helix pitch\:'
        temp1 = spf.read_array(text_headle, FILE_DATA, array_length=1)
        pitch.append(temp1)

        text_headle = ', helix cycle\:'
        temp1 = spf.read_array(text_headle, FILE_DATA, array_length=1)
        cycle.append(temp1)

    #         print(txt_name, pitch[-1], pitch[-1])
    os.chdir(PWD)

    data = pd.DataFrame({'n_node':      np.hstack(n_node),
                         'matrix_time': np.hstack(matrix_time),
                         'psi2':        np.hstack(psi2),
                         'psi3':        np.hstack(psi3),
                         'psi6':        np.hstack(psi6),
                         'zoom_factor': np.hstack(zoom_factor),
                         'pitch':       np.hstack(pitch),
                         'cycle':       np.hstack(cycle)})
    return data.pivot_table(index=['pitch', 'cycle'], columns=['zoom_factor'])


def get_data_repeat_tail(subdir, myPWD=None):
    if myPWD is None:
        t_dir = os.path.join(PWD, 'tail', subdir)
    else:
        t_dir = os.path.join(myPWD, 'tail', subdir)
    os.chdir(t_dir)
    txt_names = glob.glob('*.txt')
    n_node = []
    matrix_time = []
    psi2 = []
    psi3 = []
    psi6 = []
    zoom_factor = []
    pitch = []
    cycle = []

    for txt_name in txt_names:
        filename = os.path.join(t_dir, txt_name)
        with open(filename, 'r') as myinput:
            FILE_DATA = myinput.read()

        text_headle = ', velocity nodes:'
        temp1 = spf.read_array(text_headle, FILE_DATA, array_length=1)
        n_node.append(temp1)

        text_headle = ': create matrix use:'
        temp1 = spf.read_array(text_headle, FILE_DATA, array_length=1)
        matrix_time.append(temp1)

        text_headle = ' geometry zoom factor is'
        temp1 = spf.read_array(text_headle, FILE_DATA, array_length=1)
        t_zf = temp1

        text_headle = 'tran tail resultant is \['
        temp1 = spf.read_array(text_headle, FILE_DATA, array_length=6)
        psi2.append(temp1[2] / t_zf ** 1)
        psi61 = temp1[5]
        text_headle = 'rota tail resultant is \['
        temp1 = spf.read_array(text_headle, FILE_DATA, array_length=6)
        psi62 = temp1[2]
        psi3.append(temp1[5] / t_zf ** 3)
        # psi6.append((psi61 + psi62) / 2 / t_zf ** 2)
        psi6.append(psi62 / t_zf ** 2)

        if np.abs(t_zf - 1) < 1e-3:
            t_zf = 0
        zoom_factor.append(t_zf)

        text_headle = ', helix pitch\:'
        temp1 = spf.read_array(text_headle, FILE_DATA, array_length=1)
        pitch.append(temp1)

        text_headle = ', helix cycle\:'
        temp1 = spf.read_array(text_headle, FILE_DATA, array_length=1)
        text_headle = 'self repeat helix, repeat '
        temp2 = spf.read_array(text_headle, FILE_DATA, array_length=1)
        cycle.append(temp1 + temp2 - 1)

    #         print(txt_name, pitch[-1], pitch[-1])
    os.chdir(PWD)

    data = pd.DataFrame({'n_node':      np.hstack(n_node),
                         'matrix_time': np.hstack(matrix_time),
                         'psi2':        np.hstack(psi2),
                         'psi3':        np.hstack(psi3),
                         'psi6':        np.hstack(psi6),
                         'zoom_factor': np.hstack(zoom_factor),
                         'pitch':       np.hstack(pitch),
                         'cycle':       np.hstack(cycle)})
    return data.pivot_table(index=['pitch', 'cycle'], columns=['zoom_factor'])


def get_data_rotate_tail(foldername):
    tname = os.path.join(foldername, '*.txt')
    txt_name_list = glob.glob(tname)
    psi2 = []
    psi3 = []
    psi6 = []
    pitch = []
    cycle = []
    for txt_name in txt_name_list:
        with open(txt_name, 'r') as myinput:
            FILE_DATA = myinput.read()
        text_headle = ', helix pitch\:'
        temp1 = spf.read_array(text_headle, FILE_DATA, array_length=1)
        pitch.append(temp1)
        text_headle = ', helix cycle\:'
        temp1 = spf.read_array(text_headle, FILE_DATA, array_length=1)
        cycle.append(temp1)

        text_headle = ' geometry zoom factor is'
        temp1 = spf.read_array(text_headle, FILE_DATA, array_length=1)
        t_zf = temp1

        text_headle = 'tran tail resultant is \['
        temp1 = spf.read_array(text_headle, FILE_DATA, array_length=6)
        psi2.append(temp1[2] / t_zf ** 1)
        psi61 = temp1[5]
        text_headle = 'rota tail resultant is \['
        temp1 = spf.read_array(text_headle, FILE_DATA, array_length=6)
        psi62 = temp1[2]
        psi3.append(temp1[5] / t_zf ** 3)
        # psi6.append((psi61 + psi62) / 2 / t_zf ** 2)
        psi6.append(psi62 / t_zf ** 2)

    data = pd.DataFrame({'psi2':  np.hstack(psi2),
                         'psi3':  np.hstack(psi3),
                         'psi6':  np.hstack(psi6),
                         'pitch': np.hstack(pitch),
                         'cycle': np.hstack(cycle)})
    return data.pivot_table(index=['pitch'], columns=['cycle'])


def get_data_slb_tail(foldername):
    tname = os.path.join(foldername, '*.pickle')
    pickle_name_list = glob.glob(tname)
    psi2 = []
    psi3 = []
    psi6 = []
    pitch = []
    cycle = []
    for pickle_name in pickle_name_list:
        fileHandle = scanf('%s.pickle', pickle_name)[0]
        txt_name = '%s.txt' % fileHandle
        with open(txt_name, 'r') as myinput:
            FILE_DATA = myinput.read()
        text_headle = ', helix pitch\:'
        temp1 = spf.read_array(text_headle, FILE_DATA, array_length=1)
        pitch.append(temp1)
        text_headle = ', helix cycle\:'
        temp1 = spf.read_array(text_headle, FILE_DATA, array_length=1)
        cycle.append(temp1)

        with open(pickle_name, 'rb') as handle:
            tAt, _, tCt, ftr_info, frt_info = pickle.load(handle)
        x_slb0, f_slb0 = frt_info[0]
        x_slb1, f_slb1 = frt_info[1]
        f_slb0 = f_slb0.reshape((-1, 3))
        f_slb1 = f_slb1.reshape((-1, 3))
        ds = np.mean(np.linalg.norm(x_slb0[:-1] - x_slb0[1:], axis=-1))
        tBt = (np.sum(f_slb0[:, 2]) + np.sum(f_slb1[:, 2])) * ds
        psi2.append(tAt)
        psi3.append(tCt)
        psi6.append(tBt)

    data = pd.DataFrame({'psi2':  np.hstack(psi2),
                         'psi3':  np.hstack(psi3),
                         'psi6':  np.hstack(psi6),
                         'pitch': np.hstack(pitch),
                         'cycle': np.hstack(cycle)})
    return data.pivot_table(index=['pitch'], columns=['cycle'])


def get_data_head(type, psi_idx, myPWD=None):
    if myPWD is None:
        t_dir = os.path.join(PWD, 'head', type)
    else:
        t_dir = os.path.join(myPWD, 'head', type)
    os.chdir(t_dir)
    txt_names = glob.glob('*.txt')
    n_node = []
    matrix_time = []
    psi = []
    zoom_factor = []
    radius = []
    length = []
    err = []

    for txt_name in txt_names:
        filename = os.path.join(t_dir, txt_name)
        with open(filename, 'r') as myinput:
            FILE_DATA = myinput.read()

        text_headle = ', velocity nodes:'
        temp1 = spf.read_array(text_headle, FILE_DATA, array_length=1)
        n_node.append(temp1)

        text_headle = ': create matrix use:'
        temp1 = spf.read_array(text_headle, FILE_DATA, array_length=1)
        matrix_time.append(temp1)

        text_headle = ' head resultant is \['
        temp1 = spf.read_array(text_headle, FILE_DATA, array_length=6)
        psi.append(temp1[psi_idx])

        text_headle = ' geometry zoom factor is'
        temp1 = spf.read_array(text_headle, FILE_DATA, array_length=1)
        if np.abs(temp1 - 1) < 1e-3:
            temp1 = 0
        zoom_factor.append(temp1)

        text_headle = '  head radius\:'
        temp1 = spf.read_array(text_headle, FILE_DATA, array_length=1)
        radius.append(temp1)

        text_headle = ', length\:'
        temp1 = spf.read_array(text_headle, FILE_DATA, array_length=1)
        length.append(temp1)
        #         if not np.isfinite(temp1):
        #             print(txt_name)

        text_headle = 'velocity error of sphere \(total, x, y, z\)\:  \['
        temp1 = spf.read_array(text_headle, FILE_DATA, array_length=1)
        err.append(temp1)
    os.chdir(PWD)

    data = pd.DataFrame({'n_node':      np.hstack(n_node),
                         'matrix_time': np.hstack(matrix_time),
                         'psi':         np.hstack(psi),
                         'zoom_factor': np.hstack(zoom_factor),
                         'radius':      np.hstack(radius),
                         'err':         np.hstack(err),
                         'length':      np.hstack(length)})
    return data.dropna(how='all').pivot('length', 'zoom_factor')


# read rotation data
def get_rota_data_head(myPWD=None):
    data = get_data_head('rota', psi_idx=5, myPWD=myPWD)
    t1 = data.columns.levels[1].values.copy()
    t1[np.abs(t1) < 1e-3] = 1
    data.psi = data.psi.values / t1 ** 3
    return data


# read translation data
def get_tran_data_head(myPWD=None):
    data = get_data_head('tran', psi_idx=2, myPWD=myPWD)
    t1 = data.columns.levels[1].values.copy()
    t1[np.abs(t1) < 1e-3] = 1
    data.psi = data.psi.values / t1 ** 1
    return data


def merge_data(d1, d2):
    d0 = (d1 + d2) / 2
    t_index = ~(np.isnan(d0) & np.isfinite(d1))
    d0 = d0.where(t_index, d1)
    t_index = ~(np.isnan(d0) & np.isfinite(d2))
    d0 = d0.where(t_index, d2)
    return d0


def plot_head(dat_y, name_y='', ncol=1,
              xlabel='$r_t/R$', xlim=None, legend_title="aspect ratio"):
    ax_list = []
    for rhi in dat_y.index.levels[0]:
        title = '$rh$=%.3f' % rhi
        dat_yi = dat_y.loc[rhi]
        dat_xi = dat_yi.columns.values
        lh = dat_yi.index

        fig, ax = plt.subplots(nrows=1, ncols=1)
        fig.patch.set_facecolor('white')
        ax_list.append(ax)
        ax.set_title(title)
        for lhi in lh:
            dat_yii = dat_yi.loc[lhi]
            idxi = np.isfinite(dat_yii)
            ax.plot(dat_xi[idxi], dat_yii[idxi], '*-', ms=10)
        ax.set_xlabel(xlabel, fontsize=25)
        ax.set_ylabel(name_y, fontsize=25)
        ax.legend(lh, ncol=ncol, title=legend_title)
        if xlim is not None:
            ax.set_xlim(xlim)
    return ax_list


def plot_tail(dat_y, name_y='', ncol=1,
              xlabel='$r_t/R$', xlim=(-0.03, 0.83)):
    ax_list = []
    for phi in dat_y.index.levels[0]:
        theati = 180 / np.pi * np.arctan(2 * np.pi / phi)
        title = '$\\theta$=%.3f, ph=%.3f' % (theati, phi)
        dat_yi = dat_y.loc[phi]
        dat_xi = dat_yi.columns.values
        ch = dat_yi.index

        fig, ax = plt.subplots(nrows=1, ncols=1)
        fig.patch.set_facecolor('white')
        ax_list.append(ax)
        ax.set_title(title)
        for chi in ch:
            dat_yii = dat_yi.loc[chi]
            idxi = np.isfinite(dat_yii)
            ax.plot(dat_xi[idxi], dat_yii[idxi], '*-', ms=10)
        ax.set_xlabel(xlabel, fontsize=25)
        ax.set_ylabel(name_y, fontsize=25)
        ax.legend(ch, ncol=ncol, title="n")
        if xlim is not None:
            ax.set_xlim(xlim)
    return ax_list


def func_wt_uz(psi0, psi1, psi2, psi3, psi6):
    # given a constant torque:
    # ((psi2, psi6)    (v    (F_t
    #  (psi6, psi3)) *  w) =  T_t)

    wt = 1 / np.sqrt(psi3)  # T_t==1
    #     wt = psi1 / (psi1 + psi3)  # w_m==1
    uz = -psi6 / (psi0 + psi2) * wt
    return wt, uz


def get_data_head_symz(tdir, myPWD=None):
    if myPWD is None:
        t_dir = os.path.join(PWD, 'head', tdir)
    else:
        t_dir = os.path.join(myPWD, 'head', tdir)
    os.chdir(t_dir)
    txt_names = glob.glob('*.txt')
    n_node = []
    matrix_time = []
    psi0 = []
    psi1 = []
    zoom_factor = []
    length = []

    for txt_name in txt_names:
        filename = os.path.join(t_dir, txt_name)
        with open(filename, 'r') as myinput:
            FILE_DATA = myinput.read()

        text_headle = '  head radius\:'
        zfh = spf.read_array(text_headle, FILE_DATA, array_length=1)
        rs = zfh

        text_headle = ', velocity nodes:'
        temp1 = spf.read_array(text_headle, FILE_DATA, array_length=1)
        n_node.append(temp1)

        text_headle = ': create matrix use:'
        temp1 = spf.read_array(text_headle, FILE_DATA, array_length=1)
        matrix_time.append(temp1)

        text_headle = 'translational resistance is'
        temp1 = spf.read_array(text_headle, FILE_DATA, array_length=1)
        psi0.append(temp1 / zfh ** 1)

        text_headle = 'rotational resistance is'
        temp1 = spf.read_array(text_headle, FILE_DATA, array_length=1)
        psi1.append(temp1 / zfh ** 3)

        text_headle = ', length\:'
        temp2 = spf.read_array(text_headle, FILE_DATA, array_length=1)
        length.append(np.around(temp2 / rs, 3))
        if np.abs(zfh - 1) < 1e-3:
            zfh = 0
        zoom_factor.append(zfh)
    os.chdir(PWD)

    data = pd.DataFrame({'n_node':      np.hstack(n_node),
                         'matrix_time': np.hstack(matrix_time),
                         'psi0':        np.hstack(psi0),
                         'psi1':        np.hstack(psi1),
                         'zoom_factor': np.hstack(zoom_factor),
                         'length':      np.hstack(length)})
    return data.dropna(how='all').pivot('length', 'zoom_factor')


def fit_psi_tail(psi, phi, chi, zfi, chmin, chmax):
    x = psi.loc[phi][zfi].index
    y = psi.loc[phi][zfi].values
    idx = np.array(chmin <= x) & np.array(x <= chmax) & np.isfinite(x) & np.isfinite(y)
    popt_line, pcov = curve_fit(spf.func_line, x[idx], y[idx], maxfev=10000)
    return spf.func_line(chi, *popt_line)


def intp_psi_tail(psi, phi, chi, zfi):
    tpsi = psi.loc[phi][zfi]
    tx = tpsi.index
    ty = tpsi.values
    idxi = np.isfinite(tx) & np.isfinite(ty)
    tint = interp1d(tx[idxi], ty[idxi], kind='quadratic', fill_value='extrapolate')
    return tint(chi)


def intp_psi_head(psi, zf, kind='quadratic'):
    tx = psi.index
    ty = psi.values
    idxi = np.isfinite(tx) & np.isfinite(ty)
    int_psi = interp1d(tx[idxi], ty[idxi], kind=kind, fill_value='extrapolate')
    psi_zf = int_psi(zf)
    return psi_zf


# def fit_psi20_psi60(psi, phi, chi, chmin=3, chmax=np.inf, disp_var=None):
#     # try to fit ln(x) = a0*x/y + a1, so we have y = a0*x / (np.log(x) - a1)
#     tpsi = psi.loc[phi]
#     tx = tpsi.index * np.pi
#     theta = chi * np.pi
#     ty = tpsi.values
#     idxi = np.isfinite(tx) & np.isfinite(ty) & (chmin <= tx) & (tx <= chmax)
#     fit_x = tx[idxi] / ty[idxi]
#     fit_y = np.log(tx[idxi])
#     a0, a1 = spf.fit_line(None, fit_x, fit_y, 0, np.inf, ifprint=False)
#     if not disp_var is None:
#         display(Math('%s = \\dfrac{%f n_1 \\pi}{\\ln(n_1 \\pi ) %+f}' % (disp_var, a0, -1 * a1)))
#     return a0 * theta / (np.log(theta) - a1)
#
#
# def fit_psi30(psi, phi, chi, chmin=3, chmax=np.inf):
#     def func_psi30(x, a1):
#         y = a1 * x
#         return y
#
#     tpsi = psi.loc[phi]
#     tx = tpsi.index
#     ty = tpsi.values
#     idxi = np.isfinite(tx) & np.isfinite(ty) & (chmin <= tx) & (tx <= chmax)
#     fit_psi30, pcov = curve_fit(func_psi30, tx[idxi], ty[idxi], maxfev=1000)
#     return func_psi30(chi, *fit_psi30)


def psi20_psi30_psi60(psi2, psi3, psi6, phi, chi, chmin=4):
    if psi2.index.levels[1].values.min() <= chi <= psi2.index.levels[1].values.max():
        # in the range, intepolate
        psi2i_0 = intp_psi_tail(psi2, phi, chi, 0)
        psi3i_0 = intp_psi_tail(psi3, phi, chi, 0)
        psi6i_0 = -intp_psi_tail(-psi6, phi, chi, 0)
    else:
        # out of the range, fit
        psi20 = psi2[0].unstack()
        psi30 = psi3[0].unstack()
        psi60 = psi6[0].unstack()
        rtol = 1e-10
        atol = 1e-100
        max_iterate = 100
        rt1 = 1
        theta_fit = chi * np.pi
        cij = slb.iterate_fit_c22c33c23(phi, rt1, psi20.loc[phi], psi30.loc[phi], psi60.loc[phi],
                                        chmin=chmin, chmax=np.inf, rtol=rtol, atol=atol,
                                        max_iterate=max_iterate)
        fit_At, fit_Bt, fit_Ct, _ = slb.fit_AtBtCt_c22c23c33(theta_fit, ph, rt1, *cij)
        psi2i_0, psi3i_0, psi6i_0 = fit_At, fit_Ct, -fit_Bt
    return psi2i_0, psi3i_0, psi6i_0


def psi2_psi3_psi6(psi2, psi3, psi6, phi, chi, zft, chmin=4):
    # intepolate or fit psi2, psi3, psi6
    if psi2.index.levels[1].values.min() <= chi <= psi2.index.levels[1].values.max():
        # in the range, intepolate
        psi2i_use = intp_psi_tail(psi2, phi, chi, zft)
        psi3i_use = intp_psi_tail(psi3, phi, chi, zft)
        psi6i_use = intp_psi_tail(psi6, phi, chi, zft)
    else:
        # out of the range, fit
        psi2i_use = fit_psi_tail(psi2, phi, chi, zft, chmin, np.inf)
        psi3i_use = fit_psi_tail(psi3, phi, chi, zft, chmin, np.inf)
        psi6i_use = fit_psi_tail(psi6, phi, chi, zft, chmin, np.inf)
    return psi2i_use, psi3i_use, psi6i_use


def fit_psi00(psi00, lh, lh_min, lh_max):
    def fun_t2(tx, c):
        return 2 * np.pi * tx / (np.log(tx) - 1) + c

    fit_range = (lh_min, lh_max)
    tidx = np.isfinite(psi00.values)
    tx = psi00.index.values[tidx]
    ty = psi00.values[tidx]
    tidx = np.logical_and(tx >= fit_range[0], tx <= fit_range[1])
    pt2, _ = optimize.curve_fit(fun_t2, tx[tidx], ty[tidx], p0=(0,))
    return fun_t2(lh, *pt2)


def fit_psi10(psi10, lh, lh_min, lh_max):
    def fun_t2(tx, c):
        return tx * c

    fit_range = (lh_min, lh_max)
    tidx = np.isfinite(psi10.values)
    tx = psi10.index.values[tidx]
    ty = psi10.values[tidx]
    tidx = np.logical_and(tx >= fit_range[0], tx <= fit_range[1])
    pt2, _ = optimize.curve_fit(fun_t2, tx[tidx], ty[tidx], p0=(0,))
    return fun_t2(lh, *pt2)


def psi0_psi1(psi0, psi1, lhi, rh_rt, zft):
    zfh = rh_rt * zft
    if lhi in psi0.index.values:  # use calculated data.
        # psi0
        psi0i = psi0.loc[lhi] * rh_rt
        psi0i_use = intp_psi_head(psi0i, zfh, kind='quadratic')
        # psi1
        psi1i = psi1.loc[lhi] * rh_rt ** 3
        psi1i_use = intp_psi_head(psi1i, zfh, kind='quadratic')
    else:  # fit the trend of parameters.
        # psi0
        int_a0 = interp1d(psi0.columns.values, fit_psi0[:, 0], kind='quadratic',
                          fill_value='extrapolate')
        int_a1 = interp1d(psi0.columns.values, fit_psi0[:, 1], kind='quadratic',
                          fill_value='extrapolate')
        psi0i_use = func_psi0(lhi, int_a0(zfh), int_a1(zfh)) * rh_rt
        # psi1
        int_a0 = interp1d(psi1.columns.values, fit_psi1[:, 0], kind='quadratic',
                          fill_value='extrapolate')
        int_a1 = interp1d(psi1.columns.values, fit_psi1[:, 1], kind='quadratic',
                          fill_value='extrapolate')
        psi1i_use = func_psi1(lhi, int_a0(zfh), int_a1(zfh)) * rh_rt ** 3
    #         if zft == 0:
    #             psi1i_use =  func_psi1(lhi, *fit_psi1[0, :]) * rh_rt ** 3
    #         else:
    #             t1 = 1 / zfh
    #             psi1i_use = 4 * np.pi * (t1**2) / (t1**2 - 1) * lhi * rh_rt ** 3
    return psi0i_use, psi1i_use


def psi00_psi10(lhi, rh_rt, lhmin=40, lhmax=np.inf):
    # psi00: try to fit x/ln(x) = b0*y + b1, so we have {a0 = 1/b0, a1 = -b1/b0}
    #   where y = a0 * x / np.log(x) + a1.
    tpsi = psi0[0]


class AhChAtBtCt_fit_intp():
    def __init__(self, ph, rt1, psi0, psi1, psi2, psi3, psi6, psi20=None, psi30=None, psi60=None):
        self._ph = ph
        self._rt1 = rt1
        self._psi0 = psi0
        self._psi1 = psi1
        self._psi2 = psi2
        self._psi3 = psi3
        self._psi6 = psi6
        self._psi20 = psi2[0].unstack() if psi20 is None else psi20
        self._psi30 = psi3[0].unstack() if psi30 is None else psi30
        self._psi60 = psi6[0].unstack() if psi60 is None else psi60
        self._popt_psi00 = None
        self._popt_psi10 = None
        self._tail0_fit_info = None
        self._fit_psi0_info = None
        self._fit_psi1_info = None
        self._fit_psi2_info = None
        self._fit_psi3_info = None
        self._fit_psi6_info = None
        self._int_psi0_info = None
        self._int_psi1_info = None
        self._int_psi2_info = None
        self._int_psi3_info = None
        self._int_psi6_info = None

    def fun_psi10(self, tx, c):
        return tx * c

    def fun_psi00(self, tx, c):
        return 2 * np.pi * tx / (np.log(tx) + np.log(2) - 3 / 2) + c

    def func_psi0(self, x, a0, a1):
        y = a0 + a1 * x
        return y

    def func_psi1(self, x, a0, a1):
        # y = a0 * 0 + a1 * x
        y = a0 + a1 * x
        return y

    def func_psi2(self, x, a0, a1):
        y = a0 + a1 * x
        return y

    def func_psi3(self, x, a0, a1):
        y = a0 + a1 * x
        return y

    def func_psi6(self, x, a0, a1):
        y = a0 + a1 * x
        return y

    def cal_fit_psi_pipe_list(self, data, fit_fun, x_min=50, x_max=np.inf):
        tx = data.index
        fit_psi_list = []
        for zfi in data.columns:
            if np.isclose(zfi, 0):
                continue
            ty = data[zfi]
            idxi = np.isfinite(tx) & np.isfinite(ty) & \
                   np.array(tx >= x_min) & np.array(tx <= x_max)
            fit_psii, _ = curve_fit(fit_fun, tx[idxi], ty[idxi], maxfev=10000)
            # print(zfi)
            # print(tx[idxi], ty[idxi])
            # print(_)
            fit_psi_list.append(fit_psii)
        fit_psi_list = np.vstack(fit_psi_list)
        return fit_psi_list

    def fit_prepare(self, lh_inf_min=50, lh_inf_max=100, lh_min=50,
                    ch_inf_min=3, ch_inf_max=np.inf, ch_min=4,
                    c22c33c23_fun=slb.iterate_fit_c22c33c23_v2,
                    rtol=1e-10, atol=1e-100, max_iterate=100, ):
        ph = self._ph
        rt1 = self._rt1

        # A_h^\infty and C_h^\infty
        psi0 = self._psi0
        psi1 = self._psi1
        psi00 = psi0[0]
        psi10 = psi1[0]
        fit_range = (lh_inf_min, lh_inf_max)
        # A_h^\infty
        tidx = np.isfinite(psi00.values)
        tx = psi00.index.values[tidx]
        ty = psi00.values[tidx]
        tidx = np.logical_and(tx >= fit_range[0], tx <= fit_range[1])
        # noinspection PyTypeChecker
        self._popt_psi00, _ = optimize.curve_fit(self.fun_psi00, tx[tidx], ty[tidx], p0=(0,))
        # C_h^\infty
        tidx = np.isfinite(psi10.values)
        tx = psi10.index.values[tidx]
        ty = psi10.values[tidx]
        tidx = np.logical_and(tx >= fit_range[0], tx <= fit_range[1])
        # noinspection PyTypeChecker
        self._popt_psi10, _ = optimize.curve_fit(self.fun_psi10, tx[tidx], ty[tidx], p0=(0,))

        # A_t^\infty, B_t^\infty, C_t^\infty
        psi20 = self._psi20
        psi30 = self._psi30
        psi60 = self._psi60
        tfit_info = c22c33c23_fun(ph, rt1, psi20.loc[ph], psi30.loc[ph], psi60.loc[ph],
                                  chmin=ch_inf_min, chmax=ch_inf_max, rtol=rtol, atol=atol,
                                  max_iterate=max_iterate)
        self._tail0_fit_info = tfit_info

        # A_h, C_h, A_t, B_t, C_t
        # print('fit_psi0_info')
        self._fit_psi0_info = self.cal_fit_psi_pipe_list(self._psi0, self.func_psi0, x_min=lh_min)
        # print(self._fit_psi0_info)
        # print()
        # print('fit_psi1_info')
        self._fit_psi1_info = self.cal_fit_psi_pipe_list(self._psi1, self.func_psi1, x_min=lh_min)
        # print(self._fit_psi1_info)
        # print()
        # print('fit_psi2_info')
        self._fit_psi2_info = self.cal_fit_psi_pipe_list(self._psi2.loc[ph], self.func_psi2,
                                                         x_min=ch_min)
        # print('fit_psi3_info')
        self._fit_psi3_info = self.cal_fit_psi_pipe_list(self._psi3.loc[ph], self.func_psi3,
                                                         x_min=ch_min)
        # print('fit_psi6_info')
        self._fit_psi6_info = self.cal_fit_psi_pipe_list(self._psi6.loc[ph], self.func_psi6,
                                                         x_min=ch_min)
        return True

    def cal_int_psi_list(self, data0, data, kind='quadratic'):
        def _do(psi):
            tx = psi.index.values
            ty = psi.values
            idxi = np.isfinite(tx) & np.isfinite(ty)
            int_psi = interp1d(tx[idxi], ty[idxi], kind=kind, fill_value='extrapolate')
            int_psi_info.append(int_psi)

        int_psi_info = []
        _do(data0)
        for zfi in data.columns[1:]:
            _do(data[zfi])
        return int_psi_info

    def intp_psi_prepare(self, kind='quadratic'):
        ph = self._ph
        self._int_psi0_info = self.cal_int_psi_list(self._psi0[0], self._psi0, kind)
        self._int_psi1_info = self.cal_int_psi_list(self._psi1[0], self._psi1, kind)
        self._int_psi2_info = self.cal_int_psi_list(self._psi20.loc[ph], self._psi2.loc[ph], kind)
        self._int_psi3_info = self.cal_int_psi_list(self._psi30.loc[ph], self._psi3.loc[ph], kind)
        self._int_psi6_info = self.cal_int_psi_list(self._psi60.loc[ph], self._psi6.loc[ph], kind)

    def get_head(self, lh, zf, kind='quadratic'):
        lh = np.array((lh))
        psi0_list = []
        zfi = 0
        t1 = self._psi0.index.values[np.isfinite(self._psi0[zfi].values)].max()
        tidxa = lh >= t1
        tidxb = np.logical_not(tidxa)
        t2 = np.zeros_like(lh)
        t2[tidxa] = self.fun_psi00(lh[tidxa], *self._popt_psi00)
        t2[tidxb] = self._int_psi0_info[0](lh[tidxb])
        psi0_list.append(t2)
        for fit_psi0_info, int_psi0_info, zfi in zip(self._fit_psi0_info,
                                                     self._int_psi0_info[1:],
                                                     self._psi0.columns.values[1:]):
            t1 = self._psi0.index.values[np.isfinite(self._psi0[zfi].values)].max()
            tidxa = lh >= t1
            tidxb = np.logical_not(tidxa)
            t2 = np.zeros_like(lh)
            t2[tidxa] = self.func_psi0(lh[tidxa], *fit_psi0_info)
            t2[tidxb] = int_psi0_info(lh[tidxb])
            psi0_list.append(t2)
        tx = self._psi0.columns.values
        t_psi0 = interpolate.interp1d(tx, np.vstack(psi0_list).T, kind=kind)(zf)

        psi1_list = []
        zfi = 0
        t1 = self._psi1.index.values[np.isfinite(self._psi1[zfi].values)].max()
        tidxa = lh >= t1
        tidxb = np.logical_not(tidxa)
        t2 = np.zeros_like(lh)
        t2[tidxa] = self.fun_psi10(lh[tidxa], *self._popt_psi10)
        t2[tidxb] = self._int_psi1_info[0](lh[tidxb])
        psi1_list.append(t2)
        for fit_psi1_info, int_psi1_info, zfi in zip(self._fit_psi1_info,
                                                     self._int_psi1_info[1:],
                                                     self._psi1.columns.values[1:]):
            t1 = self._psi1.index.values[np.isfinite(self._psi1[zfi].values)].max()
            tidxa = lh >= t1
            tidxb = np.logical_not(tidxa)
            t2 = np.zeros_like(lh)
            t2[tidxa] = self.func_psi1(lh[tidxa], *fit_psi1_info)
            t2[tidxb] = int_psi1_info(lh[tidxb])
            psi1_list.append(t2)
        tx = self._psi1.columns.values
        t_psi1 = interpolate.interp1d(tx, np.vstack(psi1_list).T, kind=kind)(zf)
        return t_psi0, t_psi1

    def get_head_v2(self, kappa, zf):
        # here we assume rh = 1, R = rh / zf (or zf = rh / R).
        kappa = np.hstack((kappa,))
        # assert np.all(kappa > 99)

        fun_ah = lambda zf: 2 * np.pi * (4 * zf ** 4 * np.log(zf)
                                         - 3 * zf ** 4 + 4 * zf ** 2 - 1) \
                            / (-zf ** 4 * np.log(zf) + zf ** 4 - 2 * zf ** 2
                               + np.log(zf) + 1)
        fun_ahinf = lambda kappa: 2 * np.pi / (np.log(kappa) + (np.log(2) - 3 / 2))
        if np.any(zf == 0):
            t_psi0 = fun_ahinf(kappa) * kappa
        elif np.all(zf > 0):
            t_psi0 = fun_ah(zf) * kappa
        else:
            raise ValueError('wrong zf')

        fun_ch = lambda zf: 4 * np.pi / (1 - (zf) ** 2)
        if np.any(zf == 0):
            t_psi1 = 4 * np.pi * kappa
        elif np.all(zf > 0):
            t_psi1 = fun_ch(zf) * kappa
        else:
            raise ValueError('wrong zf')
        return t_psi0, t_psi1

    def get_tail(self, ch, zf, AtBtCt_fun=slb.fit_AtBtCt_c22c23c33_v2, kind='quadratic'):
        ch = np.hstack(ch)
        ph = self._ph
        rt1 = self._rt1
        tail0_fit_info = self._tail0_fit_info

        psi2_list = []
        psi3_list = []
        psi6_list = []

        t1 = self._psi20.loc[ph].index.values[np.isfinite(self._psi20.loc[ph].values)].max()
        tidxa = ch >= t1
        tidxb = np.logical_not(tidxa)
        theta_fit = ch[tidxa] * np.pi
        fit_At, fit_Bt, fit_Ct, _ = AtBtCt_fun(theta_fit, ph, rt1, *tail0_fit_info)
        t2 = np.zeros_like(ch)
        t2[tidxa] = fit_At
        t2[tidxb] = self._int_psi2_info[0](ch[tidxb])
        psi2_list.append(t2)
        t2 = np.zeros_like(ch)
        t2[tidxa] = fit_Ct
        t2[tidxb] = self._int_psi3_info[0](ch[tidxb])
        psi3_list.append(t2)
        t2 = np.zeros_like(ch)
        t2[tidxa] = -1 * fit_Bt
        t2[tidxb] = self._int_psi6_info[0](ch[tidxb])
        psi6_list.append(t2)

        # # dbg
        # fig, axs = plt.subplots(nrows=2, ncols=1, dpi=300)
        # fig.patch.set_facecolor('white')
        # axi = axs[0]
        # axi.plot(ch[tidxa], -t2[tidxa], '.')
        # axi = axs[1]
        # psi60_merge = self._psi60
        # tx, ty = psi60_merge.loc[2.5].index.values, psi60_merge.loc[2.5].values
        # tidx = np.isfinite(ty)
        # axi.plot(ch[tidxb], -t2[tidxb], '.')
        # axi.plot(tx[tidx], -ty[tidx], '.')

        for fit_psi2_info, int_psi2_info, zfi in zip(self._fit_psi2_info,
                                                     self._int_psi2_info[1:],
                                                     self._psi2.columns.values[1:]):
            t1 = self._psi2[zfi].loc[ph].index.values[
                np.isfinite(self._psi2[zfi].loc[ph].values)].max()
            tidxa = ch >= t1
            tidxb = np.logical_not(tidxa)
            t2 = np.zeros_like(ch)
            t2[tidxa] = self.func_psi2(ch[tidxa], *fit_psi2_info)
            t2[tidxb] = int_psi2_info(ch[tidxb])
            psi2_list.append(t2)
        tx = self._psi2.columns.values
        t_psi2 = interpolate.interp1d(tx, np.vstack(psi2_list).T, kind=kind)(zf)

        for fit_psi3_info, int_psi3_info, zfi in zip(self._fit_psi3_info,
                                                     self._int_psi3_info[1:],
                                                     self._psi3.columns.values[1:]):
            t1 = self._psi3[zfi].loc[ph].index.values[
                np.isfinite(self._psi3[zfi].loc[ph].values)].max()
            tidxa = ch >= t1
            tidxb = np.logical_not(tidxa)
            t2 = np.zeros_like(ch)
            t2[tidxa] = self.func_psi3(ch[tidxa], *fit_psi3_info)
            t2[tidxb] = int_psi3_info(ch[tidxb])
            psi3_list.append(t2)
        tx = self._psi3.columns.values
        t_psi3 = interpolate.interp1d(tx, np.vstack(psi3_list).T, kind=kind)(zf)

        for fit_psi6_info, int_psi6_info, zfi in zip(self._fit_psi6_info,
                                                     self._int_psi6_info[1:],
                                                     self._psi6.columns.values[1:]):
            t1 = self._psi6[zfi].loc[ph].index.values[
                np.isfinite(self._psi6[zfi].loc[ph].values)].max()
            tidxa = ch >= t1
            tidxb = np.logical_not(tidxa)
            t2 = np.zeros_like(ch)
            t2[tidxa] = self.func_psi6(ch[tidxa], *fit_psi6_info)
            t2[tidxb] = int_psi6_info(ch[tidxb])
            psi6_list.append(t2)
        tx = self._psi6.columns.values
        t_psi6 = interpolate.interp1d(tx, np.vstack(psi6_list).T, kind=kind)(zf)
        return t_psi2, t_psi3, t_psi6

    def at(self, zf):
        assert zf in self._psi2.columns.values
        assert zf > 0
        # print(self._psi2.columns.values)
        # print(np.where(self._psi2.columns.values == zf)[0][0])
        # print(self._fit_psi2_info)
        t1 = self._fit_psi2_info[np.where(self._psi2.columns.values == zf)[0][0] - 1]
        return t1[1]

    def bt(self, zf):
        assert zf in self._psi6.columns.values
        assert zf > 0
        t1 = self._fit_psi6_info[np.where(self._psi6.columns.values == zf)[0][0] - 1]
        return t1[1]

    def ct(self, zf):
        assert zf in self._psi3.columns.values
        assert zf > 0
        t1 = self._fit_psi3_info[np.where(self._psi3.columns.values == zf)[0][0] - 1]
        return t1[1]

    def tail0_fit_info(self):
        return self._tail0_fit_info

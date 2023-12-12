# coding: utf-8
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 2020

@author: zhangji
"""

import matplotlib
import subprocess
import os

# matplotlib.use('agg')

from petsc4py import PETSc
import numpy as np
import pickle
import re
from tqdm.notebook import tqdm as tqdm_notebook
from scipy import interpolate
from src import jeffery_model as jm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, zoomed_inset_axes
from matplotlib import colors as mcolors
from tqdm import tqdm
from codeStore import support_fun_bck as spf
from codeStore import support_fun_table as spf_tb


def do_GivenFlowObj_kwargs(ini_center, ini_theta, ini_phi, ini_psi,
                           omega_tail, table_name, name='obj'):
    obj_kwargs = {'name':       name,
                  'center':     ini_center,
                  'ini_theta':  ini_theta,
                  'ini_phi':    ini_phi,
                  'ini_psi':    ini_psi,
                  'speed':      np.nan,
                  'lbd':        np.nan,
                  'omega_tail': omega_tail,
                  'table_name': table_name, }
    return obj_kwargs


def do_GivenFlowObj(ini_theta, ini_phi, ini_psi, max_t, table_name,
                    update_fun='3bs', rtol=1e-6, atol=1e-9, eval_dt=0.001, ini_t=0,
                    save_every=1, tqdm_fun=tqdm_notebook,
                    omega_tail=0, ini_center=np.zeros(3),
                    problemHandle=jm._JefferyProblem, **problem_kwargs):
    problem = problemHandle(**problem_kwargs)
    obj_kwargs = do_GivenFlowObj_kwargs(ini_center, ini_theta, ini_phi, ini_psi,
                                        omega_tail=omega_tail, table_name=table_name,
                                        name='GivenFlowObj')
    obj = jm.GivenFlowObj(**obj_kwargs)
    obj.set_update_para(fix_x=False, fix_y=False, fix_z=False, update_fun=update_fun,
                        rtol=rtol, atol=atol, save_every=save_every, tqdm_fun=tqdm_fun)
    problem.add_obj(obj)
    base_t, base_dt, base_X, base_thphps, base_U, base_W, base_psi_t \
        = obj.update_self(t0=ini_t, t1=max_t, eval_dt=eval_dt)
    return base_t, base_dt, base_X, base_thphps, base_U, base_W, base_psi_t


def show_fun(show_handle, *args, **kwargs):
    show_handle(*args, **kwargs)
    return True


def core_show_theta_phi(base_t, base_thphps, base_psi_t, fig=None, figsize=np.array((7, 7)),
                        dpi=200, markersize=3):
    base_theta, base_phi, base_psi = base_thphps[:, 0], base_thphps[:, 1], base_psi_t

    # polar version of theta-phi
    if fig is None:
        fig = plt.figure(figsize=figsize, dpi=dpi)
    else:
        fig.clf()
    fig.patch.set_facecolor('white')
    ax1 = fig.add_subplot(111, projection='polar')
    ax1.set_ylim(0, np.pi)
    #     ax1.xaxis.set_ticklabels(['$\dfrac{%d}{8}2\pi$' % i0 for i0 in np.arange(8)])
    ax1.yaxis.set_ticklabels([])
    norm = plt.Normalize(base_t.min(), base_t.max())
    cmap = plt.get_cmap('jet')
    ax1.plot(base_phi, base_theta, '-', alpha=0.2)
    ax1.scatter(base_phi[0], base_theta[0], c='k', s=markersize * 20, marker='*')
    lc = ax1.scatter(base_phi, base_theta, c=base_t, cmap=cmap, norm=norm, s=markersize)
    ax1.set_title('$\\theta$ (radial coordinate) $-$ $\\phi$ (angular coordinate)', y=1.1)
    plt.tight_layout()
    clb = fig.colorbar(lc, ax=ax1, orientation="vertical")
    clb.ax.set_title('$t$')
    return fig


def core_show_thphps_t(base_t, base_thphps, base_psi_t, fig=None, line_fmt='-',
                       figsize=np.array((7, 7)),
                       dpi=200):
    base_theta, base_phi, base_psi = base_thphps[:, 0], base_thphps[:, 1], base_psi_t

    # polar version of theta-phi
    if fig is None:
        fig = plt.figure(figsize=figsize, dpi=dpi)
    else:
        fig.clf()
    fig.patch.set_facecolor('white')
    axs = [fig.add_subplot(3, 1, i0 + 1) for i0 in range(3)]

    for axi, ty, ylabel in zip(axs, (base_theta, base_phi, base_psi),
                               ('$\\theta$', '$\\phi$', '$\\psi$',)):
        axi.plot(base_t, ty, line_fmt)
        axi.set_ylabel(ylabel)
    axs[-1].set_xlabel('$t$')
    plt.tight_layout()
    return fig


def core_show_thphps_X_t(base_t, base_thphps, base_psi_t, base_X, fig=None, line_fmt='-',
                         figsize=np.array((7, 7)), dpi=200):
    base_theta, base_phi, base_psi = base_thphps[:, 0], base_thphps[:, 1], base_psi_t

    # polar version of theta-phi
    if fig is None:
        fig = plt.figure(figsize=figsize, dpi=dpi)
    else:
        fig.clf()
    fig.patch.set_facecolor('white')
    axs = [fig.add_subplot(3, 2, i0 + 1) for i0 in range(6)]
    ty_list = (base_theta, base_X[:, 0], base_phi, base_X[:, 1], base_psi, base_X[:, 2])
    if spf_tb.latex_installed:
        tylabel_list = ('$\\theta$', '$\\textbf{X}_1$',
                        '$\\phi$', '$\\textbf{X}_2$',
                        '$\\psi$', '$\\textbf{X}_3$')
    else:
        tylabel_list = ('$\\theta$', '$X_1$', '$\\phi$', '$X_2$', '$\\psi$', '$X_3$')
    axs[4].set_xlabel('$t$')
    axs[5].set_xlabel('$t$')

    for axi, ty, ylabel in zip(axs[0::2], ty_list[0::2], tylabel_list[0::2]):
        for i0, i1 in spf_tb.separate_angle_idx(ty):
            axi.plot(base_t[i0:i1], ty[i0:i1], line_fmt, color='#1f77b4')
        axi.set_ylabel(ylabel)

    for axi, ty, ylabel in zip(axs[1::2], ty_list[1::2], tylabel_list[1::2]):
        axi.plot(base_t, ty, line_fmt)
        axi.set_ylabel(ylabel)

    plt.tight_layout()
    return fig


def core_show_P1P2_t(base_t, base_thphps, base_psi_t, fig=None, line_fmt='-',
                     figsize=np.array((7, 7)), dpi=200):
    base_theta, base_phi, base_psi = base_thphps[:, 0], base_thphps[:, 1], base_psi_t
    P1, P2 = get_P1_P2(base_theta, base_phi, base_psi)

    # polar version of theta-phi
    if fig is None:
        fig = plt.figure(figsize=figsize, dpi=dpi)
    else:
        fig.clf()
    fig.patch.set_facecolor('white')
    axs = [fig.add_subplot(3, 2, i0 + 1) for i0 in range(6)]
    ty_list = (P1[:, 0], P2[:, 0],
               P1[:, 1], P2[:, 1],
               P1[:, 2], P2[:, 2], )
    tylabel_list = ('$p_{11}$', '$p_{11}$',
                    '$p_{12}$', '$p_{12}$',
                    '$p_{13}$', '$p_{13}$')
    axs[4].set_xlabel('$t$')
    axs[5].set_xlabel('$t$')

    for axi, ty, ylabel in zip(axs, ty_list, tylabel_list):
        axi.plot(base_t, ty, line_fmt)
        axi.set_ylabel(ylabel)

    plt.tight_layout()
    return fig


def core_show_P(base_t, base_thphps, base_psi_t, fig=None, figsize=np.array((7, 7)), dpi=200):
    base_theta, base_phi, base_psi = base_thphps[:, 0], base_thphps[:, 1], base_psi_t
    base_P = np.vstack((np.sin(base_theta) * np.cos(base_phi),
                        np.sin(base_theta) * np.sin(base_phi),
                        np.cos(base_theta))).T

    if fig is None:
        fig = plt.figure(figsize=figsize, dpi=dpi)
    else:
        fig.clf()
    fig.patch.set_facecolor('white')
    ax0 = fig.add_subplot(1, 1, 1, projection='3d')
    ax0.set_xlim(-1.1, 1.1)
    ax0.set_ylim(-1.1, 1.1)
    ax0.set_zlim(-1.1, 1.1)
    if spf_tb.latex_installed:
        ax0.set_title('$\\bm p$')
        ax0.set_xlabel('$\\textbf{X}_1$')
        ax0.set_ylabel('$\\textbf{X}_2$')
        ax0.set_zlabel('$\\textbf{X}_3$')
    else:
        ax0.set_title('$p$')
        ax0.set_xlabel('$X_1$')
        ax0.set_ylabel('$X_2$')
        ax0.set_zlabel('$X_3$')

    # Create the 3D-line collection object
    norm = plt.Normalize(base_t.min(), base_t.max())
    cmap = plt.get_cmap('jet')
    points = base_P.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = Line3DCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(base_t)
    ax0.add_collection3d(lc, zs=points[:, :, 2].flatten(), zdir='z')

    # plot jeffery sphere
    u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 100), np.linspace(0, np.pi, 100))
    tr = 1
    tx = np.cos(u) * np.sin(v) * tr
    ty = np.sin(u) * np.sin(v) * tr
    tz = np.cos(v) * tr
    color1 = plt.get_cmap('gray')(np.linspace(0.2, 0.8, 256))
    cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', color1)
    ax0.plot_surface(tx, ty, tz, rstride=1, cstride=1, cmap=cmap, edgecolor='none', alpha=0.2)

    # plot projections
    ax0.plot(np.ones_like(points[:, :, 0].flatten()) * ax0.get_xlim()[0],
             points[:, :, 1].flatten(),
             points[:, :, 2].flatten(), alpha=0.2)
    ax0.plot(points[:, :, 0].flatten(),
             np.ones_like(points[:, :, 1].flatten()) * ax0.get_ylim()[1],
             points[:, :, 2].flatten(), alpha=0.2)
    ax0.plot(points[:, :, 0].flatten(), points[:, :, 1].flatten(),
             np.ones_like(points[:, :, 2].flatten()) * ax0.get_zlim()[0], alpha=0.2)
    plt.tight_layout()
    cax0 = inset_axes(ax0, width="100%", height="5%", bbox_to_anchor=(0.0, 0.05, 1, 1),
                      loc=1, bbox_transform=ax0.transAxes, borderpad=0, )
    clb = fig.colorbar(lc, cax=cax0, orientation="horizontal")
    clb.ax.set_title('$t$')
    return fig


def core_show_X(base_t, base_X, base_U, fig=None, figsize=np.array((7, 7)), dpi=200):
    if fig is None:
        fig = plt.figure(figsize=figsize, dpi=dpi)
    else:
        fig.clf()
    fig.patch.set_facecolor('white')
    ax0 = fig.add_subplot(1, 1, 1, projection='3d')
    if spf_tb.latex_installed:
        ax0.set_title('$\\bm{X}_c$')
        ax0.set_xlabel('$\\textbf{X}_1$')
        ax0.set_ylabel('$\\textbf{X}_2$')
        ax0.set_zlabel('$\\textbf{X}_3$')
    else:
        ax0.set_title('$X_c$')
        ax0.set_xlabel('$X_1$')
        ax0.set_ylabel('$X_2$')
        ax0.set_zlabel('$X_3$')

    spf.colorline3d(base_X, base_t, quiver_length_fct=0.011, clb_title='$t$',
                    show_project=True, tu=None, nu_show=50, return_fig=True,
                    ax0=ax0, tcl_lim=None, tcl_fontsize=10)
    return fig


def pick_problem(data, **problem_kwargs):
    my_logger = problem_kwargs['my_logger']
    rank = problem_kwargs['rank']
    fileHandle = problem_kwargs['fileHandle']

    base_t, base_dt, base_X, base_thphps, base_U, base_W, base_psi_t = data
    # base_theta, base_phi, base_psi = base_thphps[:, 0], base_thphps[:, 1], base_psi_t
    save_list = ('base_t', 'base_dt', 'base_X', 'base_thphps', 'base_U', 'base_W', 'base_psi_t',)
    t_pick = {}
    for var_name in save_list:
        t_pick[var_name] = locals()[var_name]
    problem_kwargs['my_logger'] = ''
    t_pick['problem_kwargs'] = problem_kwargs

    pickle_name = '%s.pickle' % fileHandle
    with open(pickle_name, 'wb') as handle:
        pickle.dump(t_pick, handle, protocol=4)
    if rank == 0:
        my_logger.info('-->save to %s. ' % pickle_name)
    return True


def get_P1_P2(base_theta, base_phi, base_psi):
    P1, P2 = [], []
    for theta, phi, psi in zip(base_theta, base_phi, base_psi):
        rotM = spf_tb.Rloc2glb(theta, phi, psi)
        P1.append(rotM[:, 2])
        P2.append(rotM[:, 0])
    return np.vstack(P1), np.vstack(P2)


def save_fig_result_v2(data, figsize=np.array((16, 9)) * 1.5, dpi=100, **problem_kwargs):
    my_logger = problem_kwargs['my_logger']
    rank = problem_kwargs['rank']
    fileHandle = problem_kwargs['fileHandle']

    base_t, base_dt, base_X, base_thphps, base_U, base_W, base_psi_t = data
    base_theta, base_phi, base_psi = base_thphps[:, 0], base_thphps[:, 1], base_psi_t
    base_eta = np.arccos(np.sin(base_theta) * np.sin(base_phi))
    base_P1, base_P2 = get_P1_P2(base_theta, base_phi, base_psi)

    if rank == 0:
        fig_name = '%s.jpg' % fileHandle
        spf_tb.save_table_result_v2(fig_name, base_t, base_dt, base_X, base_P1, base_P2,
                                    base_theta, base_phi, base_psi, base_eta,
                                    figsize=figsize, dpi=dpi, resampling=False, move_z=False)
        my_logger.info('-->save to %s' % fig_name)

    return True


def show_fig_result_v2(data, figsize=np.array((16, 9)) * 1.5, dpi=100):
    base_t, base_dt, base_X, base_thphps, base_U, base_W, base_psi_t = data
    base_theta, base_phi, base_psi = base_thphps[:, 0], base_thphps[:, 1], base_psi_t
    base_eta = np.arccos(np.sin(base_theta) * np.sin(base_phi))
    base_P1, base_P2 = get_P1_P2(base_theta, base_phi, base_psi)

    spf_tb.show_table_result_v2(base_t, base_dt, base_X, base_P1, base_P2,
                                base_theta, base_phi, base_psi, base_eta,
                                figsize=figsize, dpi=dpi, resampling=False, move_z=False)
    return True


def load_rand_data_pickle_dir_instant(t_dir, t_headle='(.*?).pickle', n_load=None, rand_mode=False,
                                      t_start=0, t_stop=None, t_step=1):
    t_path = os.listdir(t_dir)
    filename_list = [filename for filename in t_path if re.match(t_headle, filename) is not None]
    n_load = len(filename_list) if n_load is None else n_load
    assert n_load <= len(filename_list)
    if rand_mode:
        tidx = np.random.choice(len(filename_list), n_load, replace=False)
    else:
        tidx = np.arange(n_load)
    use_filename_list = np.array(filename_list)[tidx]
    if t_stop is None:
        tname = use_filename_list[0]
        tpath = os.path.join(t_dir, tname)
        with open(tpath, 'rb') as handle:
            tpick = pickle.load(handle)
        base_t = tpick['base_t'][1:]
        t_stop = base_t.max()

    pickle_path_list = []
    idx_list = []
    intp_X_list = []
    intp_t = np.arange(t_start, t_stop, t_step)
    for i0, tname in enumerate(tqdm_notebook(use_filename_list)):
        tpath = os.path.join(t_dir, tname)
        with open(tpath, 'rb') as handle:
            tpick = pickle.load(handle)
        pickle_path_list.append(tpath)
        idx_list.append(i0)

        base_t = tpick['base_t'][1:]
        base_X = tpick['base_X'][1:]
        int_fun_X = interpolate.interp1d(base_t, base_X, kind='quadratic', axis=0)
        intp_X = int_fun_X(intp_t)
        intp_X_list.append(intp_X)
    pickle_path_list = np.array(pickle_path_list)
    idx_list = np.hstack(idx_list)
    intp_X_list = np.dstack(intp_X_list)  # (time, coord, caseid)
    return pickle_path_list, idx_list, intp_t, intp_X_list


# def resampling_pickle_dir(t_dir, rs_dt, t_headle='(.*?).pickle', n_load=None, rand_mode=False):
#     t_path = os.listdir(t_dir)
#     filename_list = [filename for filename in t_path if re.match(t_headle, filename) is not None]
#     n_load = len(filename_list) if n_load is None else n_load
#     assert len(filename_list) > 0
#     assert n_load <= len(filename_list)
#     tidx = np.arange(n_load) if rand_mode else np.random.choice(len(filename_list),
#                                                                 n_load, replace=False)
#     use_filename_list = np.array(filename_list)[tidx]


def every_pickle_dir(t_dir, save_every, new_dir, t_headle='(.*?).pickle',
                     n_load=None, rand_mode=False, tqdm_fun=tqdm_notebook, **kwargs):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
        print('make folder %s' % new_dir)
    else:
        print('exist folder %s' % new_dir)
    comm = PETSc.COMM_WORLD.tompi4py()
    rank = comm.Get_rank()

    t_path = os.listdir(t_dir)
    filename_list = [filename for filename in t_path if re.match(t_headle, filename) is not None]
    n_load = len(filename_list) if n_load is None else n_load
    assert len(filename_list) > 0
    assert n_load <= len(filename_list)
    tidx = np.arange(n_load) if rand_mode else np.random.choice(len(filename_list),
                                                                n_load, replace=False)
    use_filename_list = np.array(filename_list)[tidx]

    for i0, tname in enumerate(tqdm_fun(use_filename_list)):
        tpath = os.path.join(t_dir, tname)
        with open(tpath, 'rb') as handle:
            tpick = pickle.load(handle)
        base_t = tpick['base_t'][::save_every]
        base_dt = tpick['base_dt'][::save_every]
        base_X = tpick['base_X'][::save_every]
        base_thphps = tpick['base_thphps'][::save_every]
        base_U = tpick['base_U'][::save_every]
        base_W = tpick['base_W'][::save_every]
        base_psi_t = tpick['base_psi_t'][::save_every]
        problem_kwargs = tpick['problem_kwargs']

        save_list = ('base_t', 'base_dt', 'base_X', 'base_thphps',
                     'base_U', 'base_W', 'base_psi_t',)
        t_pick = {}
        for var_name in save_list:
            t_pick[var_name] = locals()[var_name]
        t_pick['problem_kwargs'] = problem_kwargs

        pickle_name = os.path.join(new_dir, tname)
        with open(pickle_name, 'wb') as handle:
            pickle.dump(t_pick, handle, protocol=4)
        # if rank == 0:
        #     print('-->save to %s ' % pickle_name)
    return True

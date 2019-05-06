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

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
import re
from scanf import scanf

# from scipy.optimize import curve_fit

PWD = os.getcwd()
font = {'size': 20}
matplotlib.rc('font', **font)
np.set_printoptions(linewidth=90, precision=5)

markerstyle_list = ['^', 'v', 'o', 's', 'p', 'd', 'H',
                    '1', '2', '3', '4', '8', 'P', '*',
                    'h', '+', 'x', 'X', 'D', '|', '_', ]


def read_array(text_headle, FILE_DATA, array_length=6):
    t_match = re.search(text_headle, FILE_DATA)
    if t_match is not None:
        t1 = t_match.end()
        myformat = ('%f ' * array_length)[:-1]
        temp1 = np.array(scanf(myformat, FILE_DATA[t1:]))
    else:
        temp1 = np.ones(array_length)
        temp1[:] = np.nan
    return temp1


def fit_line(ax, x, y, x0, x1, ifprint=1, linestyle='-.', linewidth=1, extendline=False,
             color='k', alpha=0.7):
    idx = np.array(x >= x0) & np.array(x <= x1) & np.isfinite(x) & np.isfinite(y)
    tx = x[idx]
    ty = y[idx]
    fit_para = np.polyfit(tx, ty, 1)
    pol_y = np.poly1d(fit_para)
    if extendline:
        fit_x = np.linspace(x.min(), x.max(), 30)
    else:
        fit_x = np.linspace(max(x.min(), x0), min(x.max(), x1), 30)
    if ax is not None:
        ax.plot(fit_x, pol_y(fit_x), linestyle, linewidth=linewidth,
                color=color, alpha=alpha)
    if ifprint:
        print('y = %f + %f * x' % (fit_para[1], fit_para[0]), 'in range', (x[idx].min(), x[idx].max()))
    return fit_para


def fit_power_law(ax, x, y, x0, x1, ifprint=1, linestyle='-.', linewidth=1, extendline=False,
                  color='k', alpha=0.7):
    idx = np.array(x >= x0) & np.array(x <= x1) & np.isfinite((np.log10(x))) & np.isfinite((np.log10(y)))
    tx = np.log10(x[idx])
    ty = np.log10(y[idx])
    fit_para = np.polyfit(tx, ty, 1)
    pol_y = np.poly1d(fit_para)

    if extendline:
        fit_x = np.log10(np.linspace(x.min(), x.max(), 30))
    else:
        fit_x = np.log10(np.linspace(max(x.min(), x0), min(x.max(), x1), 30))
    if ax is not None:
        ax.loglog(10 ** fit_x, 10 ** pol_y(fit_x), linestyle, linewidth=linewidth,
                  color=color, alpha=alpha)
    if ifprint:
        print('log(y) = %f + %f * log(x)' % (fit_para[1], fit_para[0]), 'in range', (10 ** tx.min(), 10 ** tx.max()))
        print('ln(y) = %f + %f * ln(x)' % (fit_para[1] * np.log(10), fit_para[0]), 'in range',
              (10 ** tx.min(), 10 ** tx.max()))
    return fit_para


def fit_semilogy(ax, x, y, x0, x1, ifprint=1, linestyle='-.', linewidth=1, extendline=False,
                 color='k', alpha=0.7):
    idx = np.array(x >= x0) & np.array(x <= x1) & np.isfinite(x) & np.isfinite(np.log10(y))
    tx = x[idx]
    ty = np.log10(y[idx])
    fit_para = np.polyfit(tx, ty, 1)
    pol_y = np.poly1d(fit_para)
    if extendline:
        fit_x = np.linspace(x.min(), x.max(), 30)
    else:
        fit_x = np.linspace(max(x.min(), x0), min(x.max(), x1), 30)
    if ax is not None:
        ax.plot(fit_x, 10 ** pol_y(fit_x), linestyle, linewidth=linewidth,
                color=color, alpha=alpha)
    if ifprint:
        print('log(y) = %f + %f * x' % (fit_para[1], fit_para[0]), 'in range', (tx.min(), tx.max()))
        fit_para = fit_para * np.log(10)
        print('ln(y) = %f + %f * x' % (fit_para[1], fit_para[0]), 'in range', (tx.min(), tx.max()))
    return fit_para


def get_simulate_data(eq_dir):
    txt_names = glob.glob(eq_dir + '/*.txt')

    absU = []  # abosultely velocity
    absF = []  # force of head
    zf = []  # zoom factor
    wm = []  # motor spin
    for txt_name in txt_names:
        with open(txt_name, 'r') as myinput:
            FILE_DATA = myinput.read()

        text_headle = 'absolute ref U \['
        absU.append(read_array(text_headle, FILE_DATA, array_length=6))
        text_headle = '\] and \['
        wm.append(read_array(text_headle, FILE_DATA, array_length=6))
        text_headle = 'head resultant is \['
        absF.append(read_array(text_headle, FILE_DATA, array_length=6))
        text_headle = ' geometry zoom factor is'
        temp1 = read_array(text_headle, FILE_DATA, array_length=1)
        zf.append(0 if np.isclose(temp1, 1) else temp1)
    absU = np.vstack(absU)
    wm = np.vstack(wm)
    absF = np.vstack(absF)
    zf = np.hstack(zf)
    tzf = zf.copy()
    tzf[zf == 0] = 1

    data = pd.DataFrame({'uz': absU[:, 2] / tzf,
                         'wh': absU[:, 5],
                         'wm': wm[:, 5],
                         'fh': absF[:, 2] / tzf,
                         'Th': absF[:, 5] / (tzf ** 3) * (1 - 0.1 * zf),
                         'zf': zf}).dropna(how='all').pivot_table(index='zf')
    Th = data.Th
    uz = data.uz
    uz[uz < 0] = 0
    wm = data.wm
    wh = data.wh
    return uz, wm, wh, Th


def write_pbs_head(fpbs, job_name):
    fpbs.write('#! /bin/bash\n')
    fpbs.write('#PBS -M zhangji@csrc.ac.cn\n')
    fpbs.write('#PBS -l nodes=1:ppn=24\n')
    fpbs.write('#PBS -l walltime=72:00:00\n')
    fpbs.write('#PBS -q common\n')
    fpbs.write('#PBS -N %s\n' % job_name)
    fpbs.write('\n')
    fpbs.write('cd $PBS_O_WORKDIR\n')
    fpbs.write('\n')


def write_pbs_head_q03(fpbs, job_name):
    fpbs.write('#! /bin/bash\n')
    fpbs.write('#PBS -M zhangji@csrc.ac.cn\n')
    fpbs.write('#PBS -l nodes=1:ppn=24\n')
    fpbs.write('#PBS -l walltime=72:00:00\n')
    fpbs.write('#PBS -q q03\n')
    fpbs.write('#PBS -N %s\n' % job_name)
    fpbs.write('\n')
    fpbs.write('cd $PBS_O_WORKDIR\n')
    fpbs.write('\n')


def write_pbs_head_newturb(fpbs, job_name):
    fpbs.write('#!/bin/sh\n')
    fpbs.write('#PBS -M zhangji@csrc.ac.cn\n')
    fpbs.write('#PBS -l nodes=1:ppn=24,walltime=24:00:00\n')
    fpbs.write('#PBS -N %s\n' % job_name)
    fpbs.write('\n')
    fpbs.write('cd $PBS_O_WORKDIR\n')
    fpbs.write('source /storage/zhang/.bashrc\n')
    fpbs.write('\n')


def set_axes_equal(ax):
    if ax.name == "3d":
        '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
        cubes as cubes, etc..  This is one possible solution to Matplotlib's
        ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

        Input
          ax: a matplotlib axis, e.g., as output from plt.gca().
        '''

        limits = np.array([
            ax.get_xlim3d(),
            ax.get_ylim3d(),
            ax.get_zlim3d(),
        ])

        origin = np.mean(limits, axis=1)
        radius = 0.6 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
        ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
        ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
        ax.set_zlim3d([origin[2] - radius, origin[2] + radius])
    else:
        limits = np.array([
            ax.get_xlim(),
            ax.get_ylim(),
        ])

        origin = np.mean(limits, axis=1)
        radius = 0.6 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
        ax.set_xlim([origin[0] - radius, origin[0] + radius])
        ax.set_ylim([origin[1] - radius, origin[1] + radius])
    return ax


# Topics: line, color, LineCollection, cmap, colorline, codex
'''
Defines a function colorline that draws a (multi-)colored 2D line with coordinates x and y.
The color is taken from optional data in z, and creates a LineCollection.

z can be:
- empty, in which case a default coloring will be used based on the position along the input arrays
- a single number, for a uniform color [this can also be accomplished with the usual plt.plot]
- an array of the length of at least the same length as x, to color according to this data
- an array of a smaller length, in which case the colors are repeated along the curve

The function colorline returns the LineCollection created, which can be modified afterwards.

See also: plt.streamplot
'''


# Data manipulation:
def make_segments(x, y):
    '''
    Create list of line segments from x and y coordinates, in the correct format for LineCollection:
    an array of the form   numlines x (points per line) x 2 (x and y) array
    '''

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    return segments


# Interface to LineCollection:
def colorline(x, y, z=None, cmap=plt.get_cmap('copper'), ax=None, norm=plt.Normalize(0.0, 1.0),
              label='', linewidth=3, alpha=1.0):
    '''
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    '''
    from matplotlib.collections import LineCollection

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))
    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])
    z = np.asarray(z)

    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1)
        fig.patch.set_facecolor('white')
    else:
        plt.sca(ax)
        fig = plt.gcf()

    segments = make_segments(x, y)
    lc = LineCollection(segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha)
    ax.add_collection(lc)
    return lc


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

# coding: utf-8
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 11:05:23 2017

@author: zhangji
"""

from matplotlib import pyplot as plt

# plt.rcParams['figure.figsize'] = (18.5, 10.5)
# fontsize = 40

import os
# import glob
import numpy as np
from datetime import datetime
# import matplotlib
import re
from scanf import scanf
from scipy import interpolate, integrate
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.colors import Normalize
from matplotlib.ticker import Locator
from matplotlib.collections import LineCollection
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker

# from scipy.optimize import curve_fit

# font = {'size': 20}
# matplotlib.rc('font', **font)
# np.set_printoptions(linewidth=90, precision=5)

markerstyle_list = ['^', 'v', 'o', 's', 'p', 'd', 'H',
                    '1', '2', '3', '4', '8', 'P', '*',
                    'h', '+', 'x', 'X', 'D', '|', '_', ]

color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2',
              '#7f7f7f', '#bcbd22', '#17becf']


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


class fullprint:
    'context manager for printing full numpy arrays'

    def __init__(self, **kwargs):
        kwargs.setdefault('threshold', np.inf)
        self.opt = kwargs

    def __enter__(self):
        self._opt = np.get_printoptions()
        np.set_printoptions(**self.opt)

    def __exit__(self, type, value, traceback):
        np.set_printoptions(**self._opt)


def func_line(x, a0, a1):
    y = a0 + a1 * x
    return y


def fit_line(ax, x, y, x0, x1, ifprint=1, linestyle='-.', linewidth=1, extendline=False,
             color='k', alpha=0.7):
    idx = np.array(x >= x0) & np.array(x <= x1) & np.isfinite(x) & np.isfinite(y)
    tx = x[idx]
    ty = y[idx]
    fit_para = np.polyfit(tx, ty, 1)
    pol_y = np.poly1d(fit_para)
    if extendline:
        fit_x = np.linspace(x.min(), x.max(), 100)
    else:
        fit_x = np.linspace(max(x.min(), x0), min(x.max(), x1), 100)
    if ax is not None:
        ax.plot(fit_x, pol_y(fit_x), linestyle, linewidth=linewidth,
                color=color, alpha=alpha)
    if ifprint:
        print('y = %f + %f * x' % (fit_para[1], fit_para[0]), 'in range',
              (x[idx].min(), x[idx].max()))
    return fit_para


def fit_power_law(ax, x, y, x0, x1, ifprint=1, linestyle='-.', linewidth=1, extendline=False,
                  color='k', alpha=0.7):
    idx = np.array(x >= x0) & np.array(x <= x1) & np.isfinite((np.log10(x))) & np.isfinite(
        (np.log10(y)))
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
        print('log(y) = %f + %f * log(x)' % (fit_para[1], fit_para[0]), 'in range',
              (10 ** tx.min(), 10 ** tx.max()))
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


def norm_self(v):
    return v / np.linalg.norm(v)


def angle_2vectors(v1, v2, vct_direct=None):
    v1 = norm_self(np.array(v1).ravel())
    v2 = norm_self(np.array(v2).ravel())
    err_msg = 'inputs are not 3 dimensional vectors. '
    assert v1.size == 3, err_msg
    assert v2.size == 3, err_msg
    t1 = np.dot(v1, v2)
    if vct_direct is None:
        sign = 1
    else:
        vct_direct = norm_self(np.array(vct_direct).ravel())
        assert vct_direct.size == 3, err_msg
        sign = np.sign(np.dot(vct_direct, np.cross(v1, v2)))
    theta = sign * np.arccos(t1)
    return theta


def get_rot_matrix(*args, **kwargs):
    from src import support_class as spc
    return spc.get_rot_matrix(*args, **kwargs)


def mycot(x):
    return 1 / np.tan(x)


def mycsc(x):
    return 1 / np.sin(x)


def mysec(x):
    return 1 / np.cos(x)


def write_pbs_head(fpbs, job_name, nodes=1):
    fpbs.write('#! /bin/bash\n')
    fpbs.write('#PBS -M zhangji@csrc.ac.cn\n')
    fpbs.write('#PBS -l nodes=%d:ppn=24\n' % nodes)
    fpbs.write('#PBS -l walltime=72:00:00\n')
    fpbs.write('#PBS -q common\n')
    fpbs.write('#PBS -N %s\n' % job_name)
    fpbs.write('\n')
    fpbs.write('cd $PBS_O_WORKDIR\n')
    fpbs.write('\n')
    return True


def write_pbs_head_dbg(fpbs, job_name, nodes=1):
    assert np.isclose(nodes, 1)
    fpbs.write('#! /bin/bash\n')
    fpbs.write('#PBS -M zhangji@csrc.ac.cn\n')
    fpbs.write('#PBS -l nodes=%d:ppn=24\n' % nodes)
    fpbs.write('#PBS -l walltime=24:00:00\n')
    fpbs.write('#PBS -q debug\n')
    fpbs.write('#PBS -N %s\n' % job_name)
    fpbs.write('\n')
    fpbs.write('cd $PBS_O_WORKDIR\n')
    fpbs.write('\n')
    return True


def write_pbs_head_serial(fpbs, job_name, nodes=1):
    assert np.isclose(nodes, 1)
    fpbs.write('#! /bin/bash\n')
    fpbs.write('#PBS -M zhangji@csrc.ac.cn\n')
    fpbs.write('#PBS -l nodes=%d:ppn=1\n' % nodes)
    fpbs.write('#PBS -l walltime=1000:00:00\n')
    fpbs.write('#PBS -q serial\n')
    fpbs.write('#PBS -N %s\n' % job_name)
    fpbs.write('\n')
    fpbs.write('cd $PBS_O_WORKDIR\n')
    fpbs.write('\n')
    return True


def write_pbs_head_q03(fpbs, job_name, nodes=1):
    fpbs.write('#! /bin/bash\n')
    fpbs.write('#PBS -M zhangji@csrc.ac.cn\n')
    fpbs.write('#PBS -l nodes=%d:ppn=24\n' % nodes)
    fpbs.write('#PBS -l walltime=72:00:00\n')
    fpbs.write('#PBS -q q03\n')
    fpbs.write('#PBS -N %s\n' % job_name)
    fpbs.write('\n')
    fpbs.write('cd $PBS_O_WORKDIR\n')
    fpbs.write('\n')
    return True


def write_pbs_head_newturb(fpbs, job_name, nodes=1):
    fpbs.write('#!/bin/sh\n')
    fpbs.write('#PBS -M zhangji@csrc.ac.cn\n')
    fpbs.write('#PBS -l nodes=%d:ppn=24\n' % nodes)
    fpbs.write('#PBS -l walltime=24:00:00\n')
    fpbs.write('#PBS -N %s\n' % job_name)
    fpbs.write('\n')
    fpbs.write('cd $PBS_O_WORKDIR\n')
    fpbs.write('source /storage/zhang/.bashrc\n')
    fpbs.write('\n')
    return True


def write_pbs_head_haiguang(fpbs, job_name, nodes=1):
    fpbs.write('#!/bin/sh\n')
    fpbs.write('# run the job in the main node directly. ')
    fpbs.write('\n')
    return True


def _write_main_run_top(frun, main_hostname='ln0'):
    frun.write('t_dir=$PWD \n')
    frun.write('bash_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )" \n\n')

    # check if the script run on the main node.
    frun.write('if [ $(hostname) == \'%s\' ]; then\n' % main_hostname)
    frun.write('    echo \'this node is %s. \' \n' % main_hostname)
    frun.write('else \n')
    frun.write('    echo \'please run in the node %s. \' \n' % main_hostname)
    frun.write('    exit \n')
    frun.write('fi \n\n')
    return True


def write_main_run(write_pbs_head, job_dir, ncase):
    tname = os.path.join(job_dir, 'main_run.pbs')
    print('ncase =', ncase)
    print('write parallel pbs file to %s' % tname)
    with open(tname, 'w') as fpbs:
        write_pbs_head(fpbs, job_dir, nodes=ncase)
        fpbs.write('seq 0 %d | parallel -j 1 -u --sshloginfile $PBS_NODEFILE \\\n' % (ncase - 1))
        fpbs.write('\"cd $PWD;echo $PWD;bash myscript.csh {}\"')
    return True


def write_main_run_comm_list(comm_list, txt_list, use_node, njob_node, job_dir,
                             write_pbs_head000=write_pbs_head, n_job_pbs=None,
                             random_order=False, ):
    def _parallel_pbs_ln0(n_use_comm, njob_node, csh_name):
        t2 = 'seq 0 %d | parallel -j %d -u ' % (n_use_comm - 1, njob_node)
        t2 = t2 + ' --sshloginfile $PBS_NODEFILE --sshdelay 0.1 '
        t2 = t2 + ' "cd $PWD; echo $PWD; echo; bash %s {} true " \n\n' % csh_name
        return t2

    def _parallel_pbs_newturb(n_use_comm, njob_node, csh_name):
        t2 = 'seq 0 %d | parallel -j %d -u ' % (n_use_comm - 1, njob_node)
        t2 = t2 + ' --sshdelay 0.1 '
        t2 = t2 + ' "cd $PWD; echo $PWD; echo; bash %s {} true " \n\n' % csh_name
        return t2

    PWD = os.getcwd()
    comm_list = np.array(comm_list)
    txt_list = np.array(txt_list)
    t_path = os.path.join(PWD, job_dir)
    if not os.path.exists(t_path):
        os.makedirs(t_path)
        print('make folder %s' % t_path)
    else:
        print('exist folder %s' % t_path)
    n_case = len(comm_list)
    if n_job_pbs is None:
        n_job_pbs = use_node * njob_node
    n_pbs = (n_case // n_job_pbs) + np.sign(n_case % n_job_pbs)
    if random_order:
        tidx = np.arange(n_case)
        np.random.shuffle(tidx)
        comm_list = comm_list[tidx]
        txt_list = txt_list[tidx]

    # generate comm_list.sh
    t_name0 = os.path.join(t_path, 'comm_list.sh')
    with open(t_name0, 'w') as fcomm:
        for i0, ts, f in zip(range(n_case), comm_list, txt_list):
            fcomm.write('%s > %s.txt 2> %s.err \n' % (ts, f, f))
            fcomm.write('echo \'%d / %d, %s start.\'  \n\n' % (i0 + 1, n_case, f))

    assert callable(write_pbs_head000)
    if write_pbs_head000 is write_pbs_head:
        main_hostname = 'ln0'
        _parallel_pbs_use = _parallel_pbs_ln0
        run_fun = 'qsub %s\n\n'
    # elif write_pbs_head000 is write_pbs_head_q03:
    #     main_hostname = 'ln0'
    #     _parallel_pbs_use = _parallel_pbs_ln0
    #     run_fun = 'qsub %s\n\n'
    elif write_pbs_head000 is write_pbs_head_dbg:
        main_hostname = 'ln0'
        _parallel_pbs_use = _parallel_pbs_ln0
        run_fun = 'qsub %s\n\n'
    elif write_pbs_head000 is write_pbs_head_q03:
        main_hostname = 'ln0'
        _parallel_pbs_use = _parallel_pbs_ln0
        run_fun = 'qsub %s\n\n'
    elif write_pbs_head000 is write_pbs_head_serial:
        main_hostname = 'ln0'
        _parallel_pbs_use = _parallel_pbs_ln0
        run_fun = 'qsub %s\n\n'
    elif write_pbs_head000 is write_pbs_head_newturb:
        main_hostname = 'newturb'
        _parallel_pbs_use = _parallel_pbs_newturb
        run_fun = 'qsub %s\n\n'
        assert np.isclose(use_node, 1)
    elif write_pbs_head000 is write_pbs_head_haiguang:
        main_hostname = 'bogon'
        _parallel_pbs_use = _parallel_pbs_newturb
        run_fun = 'cd $bash_dir \nnohup bash %s &\ncd $t_dir\n\n'
        assert np.isclose(use_node, 1)
    else:
        raise ValueError('wrong write_pbs_head000')
    # generate .pbs file and .csh file
    t_name0 = os.path.join(t_path, 'main_run.sh')
    with open(t_name0, 'w') as frun:
        _write_main_run_top(frun, main_hostname=main_hostname)
        # noinspection PyTypeChecker
        for t1 in np.arange(n_pbs, dtype='int'):
            use_comm = comm_list[t1 * n_job_pbs: np.min(((t1 + 1) * n_job_pbs, n_case))]
            use_txt = txt_list[t1 * n_job_pbs: np.min(((t1 + 1) * n_job_pbs, n_case))]
            n_use_comm = len(use_comm)
            tnode = np.min((use_node, np.ceil(n_use_comm / njob_node)))
            pbs_name = 'run%03d.pbs' % t1
            csh_name = 'run%03d.csh' % t1
            # generate .pbs file
            t_name = os.path.join(t_path, pbs_name)
            with open(t_name, 'w') as fpbs:
                # pbs_head = '%s_%s' % (job_dir, pbs_name)
                pbs_head = '%s_%d' % (job_dir, t1)
                write_pbs_head000(fpbs, pbs_head, nodes=tnode)
                fpbs.write(_parallel_pbs_use(n_use_comm, njob_node, csh_name))
            # generate .csh file for submit
            t_name = os.path.join(t_path, csh_name)
            with open(t_name, 'w') as fcsh:
                fcsh.write('#!/bin/csh -fe \n\n')
                t2 = 'comm_list=('
                for t3 in use_comm:
                    t2 = t2 + '"%s" ' % t3
                t2 = t2 + ') \n\n'
                fcsh.write(t2)
                t2 = 'txt_list=('
                for t3 in use_txt:
                    t2 = t2 + '"%s" ' % t3
                t2 = t2 + ') \n\n'
                fcsh.write(t2)
                fcsh.write('echo ${comm_list[$1]} \'>\' ${txt_list[$1]}.txt'
                           ' \'2>\' ${txt_list[$1]}.err \n')
                fcsh.write('echo $(expr $1 + 1) / %d, ${txt_list[$1]} start.  \n' % n_case)
                fcsh.write('echo \n')
                fcsh.write('if [ ${2:-false} = true ]; then \n')
                fcsh.write('    ${comm_list[$1]} > ${txt_list[$1]}.txt 2> ${txt_list[$1]}.err \n')
                fcsh.write('fi \n\n')
            frun.write(run_fun % pbs_name)
        frun.write('\n')
    print('input %d cases.' % n_case)
    print('generate %d pbs files in total.' % n_pbs)
    if random_order:
        print(' --->>random order mode is ON. ')
    print('Command of first case is:')
    print(comm_list[0])
    return True


def write_main_run_local(comm_list, njob_node, job_dir, random_order=False,
                         local_hostname='JiUbuntu'):
    PWD = os.getcwd()
    comm_list = np.array(comm_list)
    n_comm = comm_list.size
    sh_name = 'main_run.sh'
    pbs_name = 'pbs.main_run'
    csh_name = 'csh.main_run'

    t_path = os.path.join(PWD, job_dir)
    if not os.path.exists(t_path):
        os.makedirs(t_path)
        print('make folder %s' % t_path)
    else:
        print('exist folder %s' % t_path)

    if random_order:
        tidx = np.arange(n_comm)
        np.random.shuffle(tidx)
        comm_list = comm_list[tidx]

    # generate comm_list.sh
    t_name0 = os.path.join(t_path, 'comm_list.sh')
    with open(t_name0, 'w') as fcomm:
        for i0, ts in enumerate(comm_list):
            fcomm.write('%s \n' % ts)
            fcomm.write('echo \'%d / %d start.\'  \n\n' % (i0 + 1, n_comm))

    # generate .pbs file
    t_name = os.path.join(t_path, pbs_name)
    with open(t_name, 'w') as fpbs:
        fpbs.write('#!/bin/sh\n')
        fpbs.write('# run the job locally. \n')
        fpbs.write('echo start job at $(date) \n')
        t2 = 'seq 0 %d | parallel -j %d -u ' % (n_comm - 1, njob_node)
        t2 = t2 + ' --sshdelay 0.1 '
        t2 = t2 + ' "cd $PWD; echo $PWD; echo; bash %s {} true " \n' % csh_name
        fpbs.write(t2)
        fpbs.write('echo finish job at $(date) \n')
        fpbs.write('\n')

    # generate .csh file
    t_name = os.path.join(t_path, csh_name)
    with open(t_name, 'w') as fcsh:
        fcsh.write('#!/bin/csh -fe \n\n')
        t2 = 'comm_list=('
        for t3 in comm_list:
            t2 = t2 + '"%s" ' % t3
        t2 = t2 + ') \n\n'
        fcsh.write(t2)
        fcsh.write('echo ${comm_list[$1]} \n')
        fcsh.write('echo $(expr $1 + 1) / %d start.  \n' % n_comm)
        fcsh.write('echo \n')
        fcsh.write('if [ ${2:-false} = true ]; then \n')
        fcsh.write('    ${comm_list[$1]} \n')
        fcsh.write('fi \n\n')

    # generate .sh file
    t_name = os.path.join(t_path, sh_name)
    with open(t_name, 'w') as fsh:
        # check if the script run on the main node.
        fsh.write('if [ $(hostname) == \'%s\' ]; then\n' % local_hostname)
        fsh.write('    echo DO NOT run in the node $HOSTNAME. \n')
        fsh.write('    exit \n')
        fsh.write('else \n')
        fsh.write('    echo This node is $HOSTNAME. \n')
        fsh.write('fi \n\n')

        fsh.write('t_dir=$PWD \n')
        fsh.write('bash_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" '
                  '>/dev/null 2>&1 && pwd )" \n')
        fsh.write('echo Current path: \n')
        fsh.write('echo $bash_dir \n')
        fsh.write('cd $bash_dir \n')
        nohup_name = 'nohup_%s.out' % '$(date +"%Y%m%d_%H%M%S")'
        fsh.write('nohup bash %s > %s 2>&1 & \n' % (pbs_name, nohup_name))
        fsh.write('echo Try the command to see the output information. \n')
        fsh.write('echo tail -f %s \n' % nohup_name)
        fsh.write('cd $t_dir \n')
        fsh.write('\n')

    print('Input %d cases. ' % n_comm)
    print('Random order mode is %s. ' % random_order)
    print('Command of first case is:')
    print(comm_list[0])
    return True


def write_myscript(job_name_list, job_dir):
    t1 = ' '.join(['\"%s\"' % job_name for job_name in job_name_list])
    tname = os.path.join(job_dir, 'myscript.csh')
    print('write myscript csh file to %s' % tname)
    with open(tname, 'w') as fcsh:
        fcsh.write('#!/bin/sh -fe\n')
        fcsh.write('job_name_list=(%s)\n' % t1)
        fcsh.write('\n')
        fcsh.write('echo ${job_name_list[$1]}\n')
        fcsh.write('cd ${job_name_list[$1]}\n')
        fcsh.write('bash ${job_name_list[$1]}.sh\n')
    return True


def set_axes_equal(ax, rad_fct=0.5):
    figsize = ax.figure.get_size_inches()
    l1, l2 = ax.get_position().bounds[2:] * figsize
    lmax = np.max((l1, l2))

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
        radius = rad_fct * np.max(np.abs(limits[:, 1] - limits[:, 0]))
        radius_x = l1 / lmax * radius
        radius_y = l1 / lmax * radius
        radius_z = l2 / lmax * radius
        ax.set_xlim3d([origin[0] - radius_x, origin[0] + radius_x])
        ax.set_ylim3d([origin[1] - radius_y, origin[1] + radius_y])
        ax.set_zlim3d([origin[2] - radius_z, origin[2] + radius_z])
    else:
        limits = np.array([
            ax.get_xlim(),
            ax.get_ylim(),
        ])

        origin = np.mean(limits, axis=1)
        radius = rad_fct * np.max(np.abs(limits[:, 1] - limits[:, 0]))
        radius_x = l1 / lmax * radius
        radius_y = l2 / lmax * radius
        ax.set_xlim([origin[0] - radius_x, origin[0] + radius_x])
        ax.set_ylim([origin[1] - radius_y, origin[1] + radius_y])
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

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, x.size)
    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])
    z = np.asarray(z)

    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1)
        fig.patch.set_facecolor('white')
    else:
        plt.sca(ax)
        # fig = plt.gcf()

    segments = make_segments(x, y)
    lc = LineCollection(segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha)
    ax.add_collection(lc)
    return lc


def colorline3d(tnodes, tcl, quiver_length_fct=None, clb_title='', show_project=False, tu=None,
                nu_show=50, return_fig=False, ax0=None, tcl_lim=None, tcl_fontsize=10,
                cmap=plt.get_cmap('jet')):
    if ax0 is None:
        fig = plt.figure(figsize=(8, 8), dpi=100)
        fig.patch.set_facecolor('white')
        ax0 = fig.add_subplot(1, 1, 1, projection='3d')
    else:
        assert hasattr(ax0, 'get_zlim')
        plt.sca(ax0)
        fig = plt.gcf()
    if tcl_lim is None:
        tcl_lim = (tcl.min(), tcl.max())
    ax0.plot(tnodes[:, 0], tnodes[:, 1], tnodes[:, 2]).pop(0).remove()
    cax1 = inset_axes(ax0, width="80%", height="5%", bbox_to_anchor=(0.1, 0.1, 0.8, 1),
                      loc=9, bbox_transform=ax0.transAxes, borderpad=0, )
    norm = plt.Normalize(*tcl_lim)
    cmap = cmap
    # Create the 3D-line collection object
    points = tnodes.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = Line3DCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(tcl)
    ax0.add_collection3d(lc, zs=points[:, :, 2].flatten(), zdir='z')
    clb = fig.colorbar(lc, cax=cax1, orientation="horizontal")
    clb.ax.tick_params(labelsize=tcl_fontsize)
    clb.ax.set_title(clb_title)
    clb_ticks = np.linspace(*tcl_lim, 5)
    clb.set_ticks(clb_ticks)
    clb.ax.set_yticklabels(clb_ticks)
    set_axes_equal(ax0)
    if show_project:
        ax0.plot(np.ones_like(tnodes[:, 0]) * ax0.get_xlim()[0], tnodes[:, 1], tnodes[:, 2], '--k',
                 alpha=0.2)
        ax0.plot(tnodes[:, 0], np.ones_like(tnodes[:, 1]) * ax0.get_ylim()[1], tnodes[:, 2], '--k',
                 alpha=0.2)
        ax0.plot(tnodes[:, 0], tnodes[:, 1], np.ones_like(tnodes[:, 0]) * ax0.get_zlim()[0], '--k',
                 alpha=0.2)
    if not tu is None:
        assert not quiver_length_fct is None
        t_stp = np.max((1, tu.shape[0] // nu_show))
        color_len = tnodes[::t_stp, 0].size
        quiver_length = np.max(tnodes.max(axis=0) - tnodes.min(axis=0)) * quiver_length_fct
        # colors = [cmap(1.0 * i / color_len) for i in range(color_len)]
        # ax0.quiver(tnodes[::t_stp, 0], tnodes[::t_stp, 1], tnodes[::t_stp, 2],
        #            tu[::t_stp, 0], tu[::t_stp, 1], tu[::t_stp, 2],
        #            length=quiver_length, arrow_length_ratio=0.2, pivot='tail', normalize=False,
        #            colors=colors)
        ax0.quiver(tnodes[::t_stp, 0], tnodes[::t_stp, 1], tnodes[::t_stp, 2],
                   tu[::t_stp, 0], tu[::t_stp, 1], tu[::t_stp, 2],
                   length=quiver_length, arrow_length_ratio=0.2, pivot='tail', normalize=False,
                   colors='k')
    plt.sca(ax0)
    ax0.set_xlabel('$X_1$')
    ax0.set_ylabel('$X_2$')
    ax0.set_zlabel('$X_3$')
    # for spine in ax0.spines.values():
    #     spine.set_visible(False)
    # plt.tight_layout()

    t1 = fig if return_fig else True
    return t1


def add_inset(ax0, rect, *args, **kwargs):
    box = ax0.get_position()
    xlim = ax0.get_xlim()
    ylim = ax0.get_ylim()
    inptx = interpolate.interp1d(xlim, (0, box.x1 - box.x0))
    inpty = interpolate.interp1d(ylim, (0, box.y1 - box.y0))
    left = inptx(rect[0]) + box.x0
    bottom = inpty(rect[1]) + box.y0
    width = inptx(rect[2] + rect[0]) - inptx(rect[0])
    height = inpty(rect[3] + rect[1]) - inpty(rect[1])
    new_rect = np.hstack((left, bottom, width, height))
    return ax0.figure.add_axes(new_rect, *args, **kwargs)


def multicolor_ylabel(ax, list_of_strings, list_of_colors, axis='x', anchorpad=0, **kw):
    """this function creates axes labels with multiple colors
    ax specifies the axes object where the labels should be drawn
    list_of_strings is a list of all of the text items
    list_if_colors is a corresponding list of colors for the strings
    axis='x', 'y', or 'both' and specifies which label(s) should be drawn"""

    # x-axis label
    if axis == 'x' or axis == 'both':
        boxes = [TextArea(text, textprops=dict(color=color, ha='left', va='bottom', **kw))
                 for text, color in zip(list_of_strings, list_of_colors)]
        xbox = HPacker(children=boxes, align="center", pad=0, sep=5)
        anchored_xbox = AnchoredOffsetbox(loc='lower left', child=xbox, pad=anchorpad,
                                          frameon=False, bbox_to_anchor=(0.2, -0.09),
                                          bbox_transform=ax.transAxes, borderpad=0.)
        ax.add_artist(anchored_xbox)

    # y-axis label
    if axis == 'y' or axis == 'both':
        boxes = [TextArea(text, textprops=dict(color=color, ha='left', va='bottom',
                                               rotation=90, **kw))
                 for text, color in zip(list_of_strings[::-1], list_of_colors)]
        ybox = VPacker(children=boxes, align="center", pad=0, sep=5)
        anchored_ybox = AnchoredOffsetbox(loc='lower left', child=ybox, pad=anchorpad,
                                          frameon=False, bbox_to_anchor=(-0.105, 0.25),
                                          bbox_transform=ax.transAxes, borderpad=0.)
        ax.add_artist(anchored_ybox)


class MinorSymLogLocator(Locator):
    """
    Dynamically find minor tick positions based on the positions of
    major ticks for a symlog scaling.
    """

    def __init__(self, linthresh):
        """
        Ticks will be placed between the major ticks.
        The placement is linear for x between -linthresh and linthresh,
        otherwise its logarithmically
        """
        self.linthresh = linthresh

    def __call__(self):
        'Return the locations of the ticks'
        majorlocs = self.axis.get_majorticklocs()
        view_interval = self.axis.get_view_interval()
        if view_interval[-1] > majorlocs[-1]:
            majorlocs = np.hstack((majorlocs, view_interval[-1]))
        assert np.all(majorlocs >= 0)
        if np.isclose(majorlocs[0], 0):
            majorlocs = majorlocs[1:]

        # # iterate through minor locs, handle the lowest part, old version
        # minorlocs = []
        # for i in range(1, len(majorlocs)):
        #     majorstep = majorlocs[i] - majorlocs[i - 1]
        #     if abs(majorlocs[i - 1] + majorstep / 2) < self.linthresh:
        #         ndivs = 10
        #     else:
        #         ndivs = 9
        #     minorstep = majorstep / ndivs
        #     locs = np.arange(majorlocs[i - 1], majorlocs[i], minorstep)[1:]
        #     minorlocs.extend(locs)

        # iterate through minor locs, handle the lowest part, my version
        minorlocs = []
        for i in range(1, len(majorlocs)):
            tloc = majorlocs[i - 1]
            tgap = majorlocs[i] - majorlocs[i - 1]
            tstp = majorlocs[i - 1] * self.linthresh * 10
            while tloc < tgap and not np.isclose(tloc, tgap):
                tloc = tloc + tstp
                minorlocs.append(tloc)
        return self.raise_if_exceeds(np.array(minorlocs))

    def tick_values(self, vmin, vmax):
        raise NotImplementedError('Cannot get tick locations for a '
                                  '%s type.' % type(self))


# user define color norm
class midPowerNorm(Normalize):
    def __init__(self, gamma=10, midpoint=1, vmin=None, vmax=None, clip=False):
        Normalize.__init__(self, vmin, vmax, clip)
        assert gamma > 1
        self.gamma = gamma
        self.midpoint = midpoint

    def __call__(self, value, clip=None):
        if clip is None:
            clip = self.clip

        result, is_scalar = self.process_value(value)

        self.autoscale_None(result)
        gamma = self.gamma
        midpoint = self.midpoint
        logmid = np.log(midpoint) / np.log(gamma)
        vmin, vmax = self.vmin, self.vmax
        if vmin > vmax:
            raise ValueError("minvalue must be less than or equal to maxvalue")
        elif vmin == vmax:
            result.fill(0)
        else:
            if clip:
                mask = np.ma.getmask(result)
                result = np.ma.array(np.clip(result.filled(vmax), vmin, vmax),
                                     mask=mask)
            resdat = result.data
            tidx1 = resdat < midpoint
            tidx2 = np.logical_not(tidx1)
            resdat1 = np.log(resdat[tidx1]) / np.log(gamma)
            v1 = np.log(vmin) / np.log(gamma)
            tx, ty = [v1, logmid], [0, 0.5]
            #             print(resdat1, tx, ty)
            tuse1 = np.interp(resdat1, tx, ty)
            resdat2 = np.log(resdat[tidx2]) / np.log(gamma)
            v2 = np.log(vmax) / np.log(gamma)
            tx, ty = [logmid, v2], [0.5, 1]
            tuse2 = np.interp(resdat2, tx, ty)
            resdat[tidx1] = tuse1
            resdat[tidx2] = tuse2
            result = np.ma.array(resdat, mask=result.mask, copy=False)
        return result


# class zeroPowerNorm(Normalize):
#     def __init__(self, gamma=10, linthresh=1, linscale=1, vmin=None, vmax=None, clip=False):
#         Normalize.__init__(self, vmin, vmax, clip)
#         assert gamma > 1
#         self.gamma = gamma
#         self.midpoint = 0
#         assert vmin < 0
#         assert vmax > 0
#         self.linthresh = linthresh
#         self.linscale = linscale
#
#     def __call__(self, value, clip=None):
#         if clip is None:
#             clip = self.clip
#         result, is_scalar = self.process_value(value)
#
#         self.autoscale_None(result)
#         gamma = self.gamma
#         midpoint = self.midpoint
#         linthresh = self.linthresh
#         linscale = self.linscale
#         vmin, vmax = self.vmin, self.vmax
#
#         if clip:
#             mask = np.ma.getmask(result)
#             result = np.ma.array(np.clip(result.filled(vmax), vmin, vmax),
#                                  mask=mask)
#         assert result.max() > 0
#         assert result.min() < 0
#
#         mag0 = np.log(result.max()) / np.log(linthresh)
#         mag2 = np.log(-result.min()) / np.log(linthresh)
#         mag1 = linscale / (linscale + mag0 + mag2)
#         b0 = mag0 / (mag0 + mag1 + mag2)
#         b1 = (mag0 + mag1) / (mag0 + mag1 + mag2)
#
#         resdat = result.data
#         tidx0 = (resdat > -np.inf) * (resdat <= -linthresh)
#         tidx1 = (resdat > -linthresh) * (resdat <= linthresh)
#         tidx2 = (resdat > linthresh) * (resdat <= np.inf)
#         resdat0 = np.log(-resdat[tidx0]) / np.log(gamma)
#         resdat1 = resdat[tidx1]
#         resdat2 = np.log(resdat[tidx2]) / np.log(gamma)
#         #
#         tx, ty = [np.log(-vmin) / np.log(gamma), np.log(linthresh) / np.log(gamma)], [0, b0]
#         tuse0 = np.interp(resdat0, tx, ty)
#         #
#         tx, ty = [-linthresh, linthresh], [b0, b1]
#         tuse1 = np.interp(resdat1, tx, ty)
#
#         tx, ty = [v1, logmid], [0, 0.5]
#         #             print(resdat1, tx, ty)
#         tuse1 = np.interp(resdat1, tx, ty)
#         resdat2 = np.log(resdat[tidx2]) / np.log(gamma)
#         v2 = np.log(vmax) / np.log(gamma)
#         tx, ty = [logmid, v2], [0.5, 1]
#         tuse2 = np.interp(resdat2, tx, ty)
#         resdat[tidx1] = tuse1
#         resdat[tidx2] = tuse2
#         result = np.ma.array(resdat, mask=result.mask, copy=False)
#         return result


# user define color norm
class midLinearNorm(Normalize):
    def __init__(self, midpoint=1, vmin=None, vmax=None, clip=False):
        # clip: see np.clip, Clip (limit) the values in an array.
        # assert 1 == 2
        Normalize.__init__(self, vmin, vmax, clip)
        self.midpoint = midpoint

    def __call__(self, value, clip=None):
        if clip is None:
            clip = self.clip
        result, is_scalar = self.process_value(value)
        # print(type(result))

        self.autoscale_None(result)
        midpoint = self.midpoint
        vmin, vmax = self.vmin, self.vmax
        if vmin > vmax:
            raise ValueError("minvalue must be less than or equal to maxvalue")
        elif vmin == vmax:
            result.fill(0)
        else:
            if clip:
                mask = np.ma.getmask(result)
                result = np.ma.array(np.clip(result.filled(vmax), vmin, vmax), mask=mask)
            resdat = result.data
            tidx1 = resdat < midpoint
            tidx2 = np.logical_not(tidx1)
            resdat1 = resdat[tidx1]
            if vmin < midpoint:
                tx, ty = [vmin, midpoint], [0, 0.5]
                tuse1 = np.interp(resdat1, tx, ty)
            else:
                tuse1 = np.zeros_like(resdat1)
            resdat2 = resdat[tidx2]
            if vmax > midpoint:
                tx, ty = [midpoint, vmax], [0.5, 1]
                tuse2 = np.interp(resdat2, tx, ty)
            else:
                tuse2 = np.zeros_like(resdat2)
            resdat[tidx1] = tuse1
            resdat[tidx2] = tuse2
            result = np.ma.array(resdat, mask=result.mask, copy=False)
        return result


class TwoSlopeNorm(Normalize):
    # noinspection PyMissingConstructor
    def __init__(self, vcenter, vmin=None, vmax=None):
        """
        Normalize data with a set center.

        Useful when mapping data with an unequal rates of change around a
        conceptual center, e.g., data that range from -2 to 4, with 0 as
        the midpoint.

        Parameters
        ----------
        vcenter : float
            The data value that defines ``0.5`` in the normalization.
        vmin : float, optional
            The data value that defines ``0.0`` in the normalization.
            Defaults to the min value of the dataset.
        vmax : float, optional
            The data value that defines ``1.0`` in the normalization.
            Defaults to the the max value of the dataset.

        Examples
        --------
        This maps data value -4000 to 0., 0 to 0.5, and +10000 to 1.0; data
        between is linearly interpolated::

            >>> import matplotlib.colors as mcolors
            >>> offset = mcolors.TwoSlopeNorm(vmin=-4000.,
                                              vcenter=0., vmax=10000)
            >>> data = [-4000., -2000., 0., 2500., 5000., 7500., 10000.]
            >>> offset(data)
            array([0., 0.25, 0.5, 0.625, 0.75, 0.875, 1.0])
        """

        self.vcenter = vcenter
        self.vmin = vmin
        self.vmax = vmax
        if vcenter is not None and vmax is not None and vcenter >= vmax:
            raise ValueError('vmin, vcenter, and vmax must be in '
                             'ascending order')
        if vcenter is not None and vmin is not None and vcenter <= vmin:
            raise ValueError('vmin, vcenter, and vmax must be in '
                             'ascending order')

    def autoscale_None(self, A):
        """
        Get vmin and vmax, and then clip at vcenter
        """
        super().autoscale_None(A)
        if self.vmin > self.vcenter:
            self.vmin = self.vcenter
        if self.vmax < self.vcenter:
            self.vmax = self.vcenter

    def __call__(self, value, clip=None):
        """
        Map value to the interval [0, 1]. The clip argument is unused.
        """
        result, is_scalar = self.process_value(value)
        self.autoscale_None(result)  # sets self.vmin, self.vmax if None

        if not self.vmin <= self.vcenter <= self.vmax:
            raise ValueError("vmin, vcenter, vmax must increase monotonically")
        result = np.ma.masked_array(
            np.interp(result, [self.vmin, self.vcenter, self.vmax],
                      [0, 0.5, 1.]), mask=np.ma.getmask(result))
        if is_scalar:
            result = np.atleast_1d(result)[0]
        return result


class Arrow3D(FancyArrowPatch):
    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)
        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)


def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    '''Add an 3d arrow to an `Axes3D` instance.'''
    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)


setattr(Axes3D, 'arrow3D', _arrow3D)

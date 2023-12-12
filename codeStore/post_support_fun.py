# coding: utf-8
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 11:05:23 2017

@author: zhangji
"""

# from matplotlib import pyplot as plt
# plt.rcParams['figure.figsize'] = (18.5, 10.5)
# fontsize = 40
import codeStore.support_fun_bck as spf
# import os
import glob
import numpy as np
# import matplotlib
# import re
# from scanf import scanf
# from scipy import interpolate, integrate
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# from mpl_toolkits.mplot3d.art3d import Line3DCollection

# PWD = os.getcwd()
# font = {'size': 20}
# matplotlib.rc('font', **font)
# np.set_printoptions(linewidth=90, precision=5)


def read_array(text_headle, FILE_DATA, array_length=6):
    return spf.read_array(text_headle, FILE_DATA, array_length)


def func_line(x, a0, a1):
    return spf.func_line(x, a0, a1)


def fit_line(ax, x, y, x0, x1, ifprint=1):
    return spf.fit_line(ax, x, y, x0, x1, ifprint)


def get_simulate_data(eq_dir):
    import pandas as pd

    absU = []  # abosultely velocity
    absF = []  # force of head
    zf = []  # zoom factor
    wm = []  # motor spin
    txt_names = glob.glob(eq_dir + '/*.txt')
    for txt_name in txt_names:
        with open(txt_name, 'r') as myinput:
            FILE_DATA = myinput.read()

        text_headle = 'absolute ref U \['
        absU.append(read_array(text_headle, FILE_DATA, array_length=6))

        text_headle = '\] and \['
        t1 = read_array(text_headle, FILE_DATA, array_length=6)
        if np.all(np.isfinite(t1)):
            wm.append(read_array(text_headle, FILE_DATA, array_length=6))
        else:
            text_headle = 'sphere_0: relative velocity \['
            t1 = read_array(text_headle, FILE_DATA, array_length=6)
            text_headle = 'helix_0: relative velocity \['
            t2 = read_array(text_headle, FILE_DATA, array_length=6)
            t3 = t2 - t1
            wm.append(t3)
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
    tzf[np.isclose(zf, 0)] = 1

    data = pd.DataFrame({'uz': absU[:, 2] / tzf,
                         'wh': absU[:, 5],
                         'wm': wm[:, 5],
                         'fh': absF[:, 2] / tzf,
                         'Th': absF[:, 5] / (tzf ** 3) * (1 - 0.1 * zf),
                         'zf': zf}).dropna(how='all').pivot_table(index='zf')
    Th = data.Th
    uz = data.uz
    # uz[uz < 0] = 0
    uz[uz < 0] = np.abs(uz[uz < 0]) * 0.1
    wm = data.wm
    wh = data.wh
    return uz, wm, wh, Th

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
from scipy.optimize import curve_fit

PWD = os.getcwd()
font = {'size': 20}
matplotlib.rc('font', **font)
np.set_printoptions(linewidth=90, precision=5)


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

def func_line(x, a0, a1):
    y = a0 + a1 * x
    return y

def fit_line(ax, x, y, x0, x1, ifprint=1):
    idx = np.array(x > x0) & np.array(x < x1) & np.isfinite(x) & np.isfinite(y)
    fit_line, pcov = curve_fit(func_line, x[idx], y[idx], maxfev=10000)
    fit_x = np.linspace(x.min(), x.max(), 30)
    if ax is not None:
        ax.plot(fit_x, func_line(fit_x, *fit_line), '-.', color='k')
    if ifprint:
        print('y = %f + %f * x' % (fit_line[0], fit_line[1]), 'in range', (x0, x1))
    return fit_line

def get_simulate_data(eq_dir):
    txt_names = glob.glob(eq_dir + '/*.txt')

    absU = []  # abosultely velocity
    absF = []  # force of head
    zf = []    # zoom factor
    wm = []    # motor spin
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
    tzf[zf==0] = 1

    data = pd.DataFrame({'uz': absU[:, 2] / tzf, 
                        'wh': absU[:, 5], 
                        'wm': wm[:, 5], 
                        'fh': absF[:, 2] / tzf, 
                        'Th': absF[:, 5] / (tzf ** 3) * (1 - 0.1*zf), 
                        'zf': zf}).dropna(how='all').pivot_table(index='zf')
    Th = data.Th
    uz = data.uz
    uz[uz<0] = 0
    wm = data.wm
    wh = data.wh
    return uz, wm, wh, Th
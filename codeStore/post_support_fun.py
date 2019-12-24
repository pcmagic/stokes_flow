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
import codeStore.support_fun as spf

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
    return spf.get_simulate_data(eq_dir)
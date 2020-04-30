# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on

@author: zhangji
"""

from time import time
from typing import Any, Tuple

import numpy as np
from scipy import interpolate, integrate, optimize, sparse
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook

# import os
# import scipy as sp
# import pandas as pd
# import re
# from scanf import scanf
# from matplotlib import pyplot as plt
# from mpl_toolkits.mplot3d import axes3d, Axes3D
# from scipy.optimize import leastsq, curve_fit
# from IPython.display import display, HTML
# from codeStore import support_fun as spf

# support functions
# helix curve
x1_fun = lambda theta, ph, rt1, rt2: np.array(
        (rt1 * np.cos(theta),
         rt1 * np.sin(theta),
         ph * theta / (2 * np.pi))).T
x2_fun = lambda theta, ph, rt1, rt2: np.array(
        (rt1 * np.cos(theta + np.pi),
         rt1 * np.sin(theta + np.pi),
         ph * theta / (2 * np.pi))).T
_arc_length = lambda theta, ph, rt1, rt2: np.sqrt(ph ** 2 + 4 * np.pi ** 2 * rt1 ** 2)
T1_fun = lambda theta, ph, rt1, rt2: np.array(
        ((-2 * np.pi * rt1 * np.sin(theta)),
         (2 * np.pi * rt1 * np.cos(theta)),
         np.ones_like(theta) * ph)).T / _arc_length(theta, ph, rt1, rt2)
T2_fun = lambda theta, ph, rt1, rt2: np.array(
        ((2 * np.pi * rt1 * np.sin(theta)),
         (-2 * np.pi * rt1 * np.cos(theta)),
         np.ones_like(theta) * ph)).T / _arc_length(theta, ph, rt1, rt2)
N1_fun = lambda theta, ph, rt1, rt2: np.array(
        ((ph * np.sin(theta)),
         -((ph * np.cos(theta))),
         np.ones_like(theta) * (2 * np.pi * rt1))).T / _arc_length(theta, ph, rt1, rt2)
B1_fun = lambda theta, ph, rt1, rt2: np.array(
        (-np.cos(theta), -np.sin(theta), np.zeros_like(theta))).T
# RotF1_fun = lambda theta, ph, rt1, rt2: np.array(
#         ((np.cos(theta), -np.sin(theta), 0),
#          (np.sin(theta), np.cos(theta), 0),
#          (0, 0, 1)))
# RotF2_fun = lambda theta, ph, rt1, rt2: np.array(
#         ((-np.cos(theta), np.sin(theta), 0),
#          (-np.sin(theta), -np.cos(theta), 0),
#          (0, 0, 1)))
Fn1Mat_fun = lambda theta, ph, rt1, rt2: \
    np.eye(3) - np.outer(T1_fun(theta, ph, rt1, rt2), T1_fun(theta, ph, rt1, rt2))
Fn2Mat_fun = lambda theta, ph, rt1, rt2: \
    np.eye(3) - np.outer(T2_fun(theta, ph, rt1, rt2), T2_fun(theta, ph, rt1, rt2))
dist_th1_fun = lambda theta, ph, rt1, rt2: np.linalg.norm(
        x1_fun(theta, ph, rt1, rt2) - x1_fun(0, ph, rt1, rt2))
dist_th2_fun = lambda theta, ph, rt1, rt2: np.linalg.norm(
        x2_fun(theta, ph, rt1, rt2) - x2_fun(0, ph, rt1, rt2))
natu_cut_th1_fun = lambda theta, ph, rt1, rt2: \
    dist_th1_fun(theta, ph, rt1, rt2) - rt2 * np.sqrt(np.e) / 2


def hlx1_info_fun(ph, rt1, rt2, ch, nth=None, hlx_node_fun=x1_fun,
                  check_dth=True):
    S = _arc_length(0, ph, rt1, rt2)
    natu_cut = rt2 * np.sqrt(np.e) / 2
    # natu_cut_th = optimize.brentq(natu_cut_th1_fun, 0, np.pi, args=(ph, rt1, rt2))
    natu_cut_th = natu_cut / S * (2 * np.pi)
    # noinspection PyTypeChecker
    max_nth = np.floor(2 * np.pi * ch / (natu_cut_th * 2)).astype(int)
    hlx_th = np.arctan(2 * np.pi * rt1 / ph)
    if nth is None:
        nth = max_nth
    dth = 2 * np.pi * ch / nth
    # th_list = np.linspace(0, 2 * np.pi * ch, nth, endpoint=False) + dth / 2
    th_list = np.linspace(-np.pi * ch, np.pi * ch, nth, endpoint=False) + dth / 2
    nodes = hlx_node_fun(th_list, ph, rt1, rt2)
    dS = S * dth / (2 * np.pi)
    tdic = {'natu_cut':    natu_cut,
            'natu_cut_th': natu_cut_th,
            'hlx_th':      hlx_th,
            'dth':         dth,
            'th_list':     th_list,
            'nth':         nth,
            'nodes':       nodes,
            'S':           S,
            'dS':          dS, }

    if check_dth:
        err_msg = 'nth is too large. nth <= %d' % max_nth
        assert dth > (2 * natu_cut_th), err_msg
    return tdic


def hlxn_info_fun(ph, rt1, rt2, ch, nth=None, hlx_node_fun_list=(x1_fun,),
                  check_dth=True):
    S = np.sqrt(4 * np.pi ** 2 * rt1 ** 2 + ph ** 2)
    natu_cut = rt2 * np.sqrt(np.e) / 2
    # natu_cut_th = optimize.brentq(natu_cut_th1_fun, 0, np.pi, args=(ph, rt1, rt2))
    natu_cut_th = natu_cut / S * (2 * np.pi)
    # noinspection PyTypeChecker
    max_nth = np.floor(2 * np.pi * ch / (natu_cut_th * 2)).astype(int)
    hlx_th = np.arctan(2 * np.pi * rt1 / ph)
    if nth is None:
        nth = max_nth
    dth = 2 * np.pi * ch / nth
    # th_list = np.linspace(0, 2 * np.pi * ch, nth, endpoint=False) + dth / 2
    th_list = np.linspace(-np.pi * ch, np.pi * ch, nth, endpoint=False) + dth / 2
    nodes = np.vstack([tx_fun(th_list, ph, rt1, rt2) for tx_fun in hlx_node_fun_list])
    dS = S * dth / (2 * np.pi)
    tdic = {'natu_cut':    natu_cut,
            'natu_cut_th': natu_cut_th,
            'hlx_th':      hlx_th,
            'dth':         dth,
            'th_list':     th_list,
            'nth':         nth,
            'nodes':       nodes,
            'S':           S,
            'dS':          dS, }

    if check_dth:
        err_msg = 'nth is too large. nth <= %d' % max_nth
        assert dth > (2 * natu_cut_th), err_msg
    return tdic


def stokeslets_matrix_mij(u_theta, f_theta, ph, rt1, rt2, u_node_fun=x1_fun,
                          f_node_fun=x1_fun) -> np.ndarray:
    # mij = S(i, :), along f.
    u_node = u_node_fun(u_theta, ph, rt1, rt2)
    f_nodes = f_node_fun(f_theta, ph, rt1, rt2)
    t_m = np.zeros((3, f_nodes.size))
    dxi = (u_node - f_nodes).T
    dx0 = dxi[0]
    dx1 = dxi[1]
    dx2 = dxi[2]
    dr2 = np.sum(dxi ** 2, axis=0)
    dr1 = np.sqrt(dr2)
    dr3 = dr1 * dr2
    temp1 = 1 / (dr1 * (8 * np.pi))  # 1/r^1
    temp2 = 1 / (dr3 * (8 * np.pi))  # 1/r^3
    t_m[0, 0::3] = temp2 * dx0 * dx0 + temp1
    t_m[0, 1::3] = temp2 * dx0 * dx1
    t_m[0, 2::3] = temp2 * dx0 * dx2
    t_m[1, 0::3] = temp2 * dx1 * dx0
    t_m[1, 1::3] = temp2 * dx1 * dx1 + temp1
    t_m[1, 2::3] = temp2 * dx1 * dx2
    t_m[2, 0::3] = temp2 * dx2 * dx0
    t_m[2, 1::3] = temp2 * dx2 * dx1
    t_m[2, 2::3] = temp2 * dx2 * dx2 + temp1
    return t_m


def stokeslets_matrix_mij2(u_theta, f_theta, fidx, ph, rt1, rt2, u_node_fun=x1_fun,
                           f_node_fun=x1_fun) -> np.ndarray:
    # mj = S(:, j), along u
    u_nodes = u_node_fun(u_theta, ph, rt1, rt2)
    f_node = f_node_fun(f_theta, ph, rt1, rt2)
    t_m = np.zeros((u_nodes.size, 3))
    dxi = (u_nodes - f_node).T
    dx0 = dxi[0]
    dx1 = dxi[1]
    dx2 = dxi[2]
    dr2 = np.sum(dxi ** 2, axis=0)
    if u_node_fun is f_node_fun:
        dr2[fidx] = np.inf
    dr1 = np.sqrt(dr2)
    dr3 = dr1 * dr2
    temp1 = 1 / (dr1 * (8 * np.pi))  # 1/r^1
    temp2 = 1 / (dr3 * (8 * np.pi))  # 1/r^3
    t_m[0::3, 0] = temp2 * dx0 * dx0 + temp1
    t_m[0::3, 1] = temp2 * dx0 * dx1
    t_m[0::3, 2] = temp2 * dx0 * dx2
    t_m[1::3, 0] = temp2 * dx1 * dx0
    t_m[1::3, 1] = temp2 * dx1 * dx1 + temp1
    t_m[1::3, 2] = temp2 * dx1 * dx2
    t_m[2::3, 0] = temp2 * dx2 * dx0
    t_m[2::3, 1] = temp2 * dx2 * dx1
    t_m[2::3, 2] = temp2 * dx2 * dx2 + temp1
    return t_m


def doublets_matrix_dij2(u_theta, f_theta, fidx, ph, rt1, rt2, u_node_fun=x1_fun,
                         f_node_fun=x1_fun) -> np.ndarray:
    # dj = D(:, j), along u
    u_nodes = u_node_fun(u_theta, ph, rt1, rt2)
    f_node = f_node_fun(f_theta, ph, rt1, rt2)
    t_d = np.zeros((u_nodes.size, 3))
    dxi = (u_nodes - f_node).T
    dx0 = dxi[0]
    dx1 = dxi[1]
    dx2 = dxi[2]
    dr2 = np.sum(dxi ** 2, axis=0)
    if u_node_fun is f_node_fun:
        dr2[fidx] = np.inf
    dr1 = np.sqrt(dr2)
    dr3 = dr1 * dr2
    dr5 = dr3 * dr2
    temp1 = rt2 ** 2 / (2 * dr3 * (8 * np.pi))  # 1/r^1
    temp2 = rt2 ** 2 / (2 * dr5 * (8 * np.pi))  # 1/r^3
    t_d[0::3, 0] = temp2 * dx0 * dx0 + temp1
    t_d[0::3, 1] = temp2 * dx0 * dx1
    t_d[0::3, 2] = temp2 * dx0 * dx2
    t_d[1::3, 0] = temp2 * dx1 * dx0
    t_d[1::3, 1] = temp2 * dx1 * dx1 + temp1
    t_d[1::3, 2] = temp2 * dx1 * dx2
    t_d[2::3, 0] = temp2 * dx2 * dx0
    t_d[2::3, 1] = temp2 * dx2 * dx1
    t_d[2::3, 2] = temp2 * dx2 * dx2 + temp1
    return t_d


# Lightill Slender Body Theory, this version assume mesh size == local part size.
def Lightill_slenderBody_matrix_1hlx(ph, rt1, rt2, ch,
                                     hlx_node_fun=x1_fun, FnMat_fun=Fn1Mat_fun,
                                     epsabs=1e-200, epsrel=1e-08, limit=10000,
                                     workers=1) -> np.ndarray:
    hlx_info = hlx1_info_fun(ph, rt1, rt2, ch, hlx_node_fun=hlx_node_fun)
    dth = hlx_info['dth']
    th_list = hlx_info['th_list']
    nth = hlx_info['nth']
    S = hlx_info['S']
    intFct = S / (2 * np.pi)

    m = np.zeros((nth * 3, nth * 3))

    # # part 1, nonlocal part, version 1, ignore the local variation of tsij
    # for i0, th0 in enumerate(tqdm_notebook(th_list)):  # u_node
    #     i2 = i0 * 3
    #     m[i2:i2 + 3, :] = stokeslets_matrix_mij(th0, th_list, ph, rt1, rt2,
    #                                             u_node_fun, f_node_fun) * dS

    # # part 1, nonlocal part, version 2, locally integration of tsij, loop along i and j.
    # warpper_mij = lambda theta: stokeslets_matrix_mij(th0, theta, ph, rt1, rt2,
    #                                                   u_node_fun, f_node_fun)
    # norm_diag = np.linalg.norm(Fn1Mat_fun(0, ph, rt1, rt2) / (4 * np.pi))
    # for i0, th0 in enumerate(tqdm_notebook(th_list)):  # u_node
    #     i2 = i0 * 3
    #     for i1, th1 in enumerate(th_list):  # f_node
    #         i3 = i1 * 3
    #         if i0 == i1:
    #             continue
    #         theta_a = th1 - dth / 2
    #         theta_b = th1 + dth / 2
    #         tsij = integrate.quad_vec(warpper_mij, theta_a, theta_b, workers=workers,
    #                                   epsabs=norm_diag * epsabs, epsrel=epsrel, limit=limit, )[0]
    #         m[i2:i2 + 3, i3:i3 + 3] = tsij * intFct

    # part 1, nonlocal part, version 3, locally integration of tsij, loop along j (force points).
    warpper_mij2 = lambda theta: stokeslets_matrix_mij2(th_list, theta, i0, ph, rt1, rt2,
                                                        hlx_node_fun, hlx_node_fun)
    # for i0, th0 in enumerate(tqdm_notebook(th_list)):  # f_node
    for i0, th0 in enumerate(th_list):  # f_node
        i2 = i0 * 3
        theta_a = th0 - dth / 2
        theta_b = th0 + dth / 2
        tsij = integrate.quad_vec(warpper_mij2, theta_a, theta_b, workers=workers,
                                  epsabs=epsabs, epsrel=epsrel, limit=limit, )[0]
        m[:, i2:i2 + 3] = tsij * intFct

    # part 2, local part.
    for i0, th in enumerate(th_list):
        i1 = i0 * 3
        m[i1:i1 + 3, i1:i1 + 3] = FnMat_fun(th, ph, rt1, rt2) / (4 * np.pi)
    return m


# Lightill Slender Body Theory, this version assert mesh size > local part size.
def Lightill_slenderBody_matrix_1hlx_v2(ph, rt1, rt2, ch, nth=None,
                                        hlx_node_fun=x1_fun, FnMat_fun=Fn1Mat_fun,
                                        epsabs=1e-200, epsrel=1e-08, limit=10000,
                                        workers=1) -> np.ndarray:
    hlx_info = hlx1_info_fun(ph, rt1, rt2, ch, nth=nth, hlx_node_fun=hlx_node_fun)
    dth = hlx_info['dth']
    th_list = hlx_info['th_list']
    nth = hlx_info['nth']
    S = hlx_info['S']
    natu_cut_th = hlx_info['natu_cut_th']
    intFct = S / (2 * np.pi)

    m = np.zeros((nth * 3, nth * 3))
    # part 1, nonlocal part, version 3, locally integration of tsij, loop along j (force points).
    warpper_mij2 = lambda theta: stokeslets_matrix_mij2(th_list, theta, i0, ph, rt1, rt2,
                                                        hlx_node_fun, hlx_node_fun)
    # for i0, th0 in enumerate(tqdm_notebook(th_list)):  # f_node
    for i0, th0 in enumerate(th_list):  # f_node
        i2 = i0 * 3
        theta_a = th0 - dth / 2
        theta_b = th0 + dth / 2
        tsij = integrate.quad_vec(warpper_mij2, theta_a, theta_b, workers=workers,
                                  epsabs=epsabs, epsrel=epsrel, limit=limit, )[0]
        m[:, i2:i2 + 3] = tsij * intFct

    # part 2, local part.
    warpper_mij = lambda theta: stokeslets_matrix_mij(th, theta, ph, rt1, rt2,
                                                      hlx_node_fun, hlx_node_fun)
    for i0, th in enumerate(th_list):
        i1 = i0 * 3
        theta_a = th - dth / 2
        theta_b = th + dth / 2
        t1 = FnMat_fun(th, ph, rt1, rt2) / (4 * np.pi)
        t2 = integrate.quad_vec(warpper_mij, th + natu_cut_th, theta_b, workers=workers,
                                epsabs=epsabs, epsrel=epsrel, limit=limit, )[0]
        t3 = integrate.quad_vec(warpper_mij, theta_a, th - natu_cut_th, workers=workers,
                                epsabs=epsabs, epsrel=epsrel, limit=limit, )[0]
        m[i1:i1 + 3, i1:i1 + 3] = t1 + (t2 + t3) * intFct
    return m


# def KRJ_stokeslets_mij2(u_theta, f_theta, fidx, ph, rt1, rt2,
#                         u_node_fun=x1_fun, f_node_fun=x1_fun, T_fun=T1_fun):
#     # inner_mj = S(:, j), along u
#     S = _arc_length(0, ph, rt1, rt2)
#     intFct = S / (2 * np.pi)
#     su = u_theta * intFct
#     sf = f_theta * intFct
#     t = T_fun(u_theta, ph, rt1, rt2)
#     ds = np.abs(sf - su)
#     if u_node_fun is f_node_fun:
#         ds[fidx] = np.inf
#     t_m = np.vstack([(np.eye(3) + np.outer(ti, ti)) / dsi
#                      for ti, dsi in zip(t.reshape(-1, 3), ds)])
#     return t_m
#
#
# def KRJ_local_part(u_theta, theta_a, theta_b, l, ph, rt1, rt2,
#                    u_node_fun=x1_fun, T_fun=T1_fun):
#     t = T_fun(u_theta, ph, rt1, rt2)
#     intFct = S / (2 * np.pi)
#     su = u_theta * intFct
#     Lsbt = np.log(4 * (l * su - su ** 2) / rt2 ** 2)
#     t1 = np.eye(3)
#     t2 = np.outer(t, t)
#     t3 = (t1 + t2)
#     t_m = Lsbt * t3 + t1 - 3 * t2
#     return t_m


# Keller-Rubinow-Johnson Slender Body Theory, this version assert mesh size > local part size.
def KRJ_slenderBody_matrix_1hlx(ph, rt1, rt2, ch, nth=None,
                                hlx_node_fun=x1_fun, T_fun=T1_fun,
                                epsabs=1e-200, epsrel=1e-08, limit=10000,
                                workers=1) -> np.ndarray:
    hlx_info = hlx1_info_fun(ph, rt1, rt2, ch, nth=nth, hlx_node_fun=hlx_node_fun, check_dth=False)
    dth = hlx_info['dth']
    th_list = hlx_info['th_list']
    nth = hlx_info['nth']
    S = hlx_info['S']
    intFct = S / (2 * np.pi)

    m = np.zeros((nth * 3, nth * 3))
    # part 1, nonlocal part, version 3, locally integration of tsij, loop along j (force points).
    warpper_mij2 = lambda theta: stokeslets_matrix_mij2(th_list, theta, i0, ph, rt1, rt2,
                                                        hlx_node_fun, hlx_node_fun)
    # for i0, th0 in enumerate(tqdm_notebook(th_list)):  # f_node
    for i0, th0 in enumerate(th_list):  # f_node
        i2 = i0 * 3
        theta_a = th0 - dth / 2
        theta_b = th0 + dth / 2
        tsij = integrate.quad_vec(warpper_mij2, theta_a, theta_b, workers=workers,
                                  epsabs=epsabs, epsrel=epsrel, limit=limit, )[0]
        m[:, i2:i2 + 3] = tsij * intFct

    # part 2, local part.
    min_th = th_list[0] - dth / 2
    max_th = th_list[-1] + dth / 2
    l_min = min_th * intFct
    l_max = max_th * intFct
    for i0, th in enumerate(th_list):
        i1 = i0 * 3
        s0 = th * intFct
        s_a = (th - dth / 2) * intFct
        s_b = (th + dth / 2) * intFct
        t = T_fun(th, ph, rt1, rt2)
        # rt2_use = rt2 * 4 / np.pi # This factor following Rodenborn2013
        rt2_use = rt2
        Lsbt = np.log((-l_max * l_min + (l_max + l_min) * s0 - s0 ** 2) / rt2_use ** 2)
        ta = np.eye(3)
        tb = np.outer(t, t)
        tc = ta + tb
        t1 = Lsbt * tc + ta - 3 * tb
        t2 = np.log((l_min - s0) / (s_a - s0))
        t3 = np.log((l_max - s0) / (s_b - s0))
        tint = (t2 + t3) * tc
        m[i1:i1 + 3, i1:i1 + 3] = (t1 - tint) / (8 * np.pi)

    # # part 2, local part, Rodenborn2013 version
    # max_th = th_list[-1] + dth / 2
    # for i0, thi in enumerate(th_list):
    #     i1 = i0 * 3
    #     t = T_fun(thi, ph, rt1, rt2)
    #     rt2_use = rt2 * 4 / np.pi # This factor following Rodenborn2013
    #     kappa = rt2_use / (max_th * intFct)
    #     ds = np.abs(np.arange(nth) - i0) * 1.
    #     ds[i0] = np.inf
    #     K = 1 / 2 * (np.sum(1 / ds) + np.log(kappa ** 2 * np.e))
    #     t1 = np.eye(3) + np.outer(t, t)
    #     t2 = np.eye(3) - np.outer(t, t)
    #     m[i1:i1 + 3, i1:i1 + 3] = (-K * t1 + t2) / (4 * np.pi)
    return m


# Lightill Slender Body Theory, this version assume mesh size == local part size.
def Lightill_slenderBody_matrix_nhlx(ph, rt1, rt2, ch,
                                     hlx_node_fun_list=(x1_fun,), FnMat_fun_list=(Fn1Mat_fun,),
                                     epsabs=1e-200, epsrel=1e-08, limit=10000,
                                     workers=1) -> np.ndarray:
    hlx_info = hlxn_info_fun(ph, rt1, rt2, ch, hlx_node_fun_list=hlx_node_fun_list)
    dth = hlx_info['dth']
    th_list = hlx_info['th_list']
    nth = hlx_info['nth']
    S = hlx_info['S']
    intFct = S / (2 * np.pi)
    err_msg = 'tube size of u_node_fun_list and f_node_fun_list is different. '
    assert len(hlx_node_fun_list) == len(hlx_node_fun_list), err_msg
    n_hlx = len(hlx_node_fun_list)

    m = np.zeros((nth * n_hlx * 3, nth * n_hlx * 3))

    # part 1, nonlocal part, version 3, locally integration of tsij, loop along j (force points).
    warpper_mij2 = lambda theta: stokeslets_matrix_mij2(th_list, theta, i0, ph, rt1, rt2,
                                                        u_node_fun, f_node_fun)
    # for i5, u_node_fun in enumerate(tqdm_notebook(u_node_fun_list)):
    for i5, u_node_fun in enumerate(hlx_node_fun_list):
        idx0 = nth * i5 * 3
        for i6, f_node_fun in enumerate(hlx_node_fun_list):
            idx1 = nth * i6 * 3
            for i0, th0 in enumerate(th_list):  # f_node
                i2 = idx1 + i0 * 3
                theta_a = th0 - dth / 2
                theta_b = th0 + dth / 2
                tsij = integrate.quad_vec(warpper_mij2, theta_a, theta_b, workers=workers,
                                          epsabs=epsabs, epsrel=epsrel, limit=limit, )[0]
                m[idx0:idx0 + (3 * nth), i2:i2 + 3] = tsij * intFct

    # part 2, local part.
    for i5, (u_node_fun, f_node_fun, FnMat_fun) in enumerate(
            zip(hlx_node_fun_list, hlx_node_fun_list, FnMat_fun_list)):
        idx0 = nth * i5 * 3
        for i0, th in enumerate(th_list):
            i1 = idx0 + i0 * 3
            m[i1:i1 + 3, i1:i1 + 3] = FnMat_fun(th, ph, rt1, rt2) / (4 * np.pi)
    return m


# Lightill Slender Body Theory, this version assert mesh size > local part size.
def Lightill_slenderBody_matrix_nhlx_v2(ph, rt1, rt2, ch, nth=None,
                                        hlx_node_fun_list=(x1_fun,), FnMat_fun_list=(Fn1Mat_fun,),
                                        epsabs=1e-200, epsrel=1e-08, limit=10000,
                                        workers=1) -> np.ndarray:
    hlx_info = hlxn_info_fun(ph, rt1, rt2, ch, nth=nth, hlx_node_fun_list=hlx_node_fun_list)
    dth = hlx_info['dth']
    th_list = hlx_info['th_list']
    nth = hlx_info['nth']
    S = hlx_info['S']
    natu_cut_th = hlx_info['natu_cut_th']
    intFct = S / (2 * np.pi)
    err_msg = 'tube size of u_node_fun_list and f_node_fun_list is different. '
    assert len(hlx_node_fun_list) == len(hlx_node_fun_list), err_msg
    n_hlx = len(hlx_node_fun_list)

    m = np.zeros((nth * n_hlx * 3, nth * n_hlx * 3))

    # part 1, nonlocal part, version 3, locally integration of tsij, loop along j (force points).
    warpper_mij2 = lambda theta: stokeslets_matrix_mij2(th_list, theta, i0, ph, rt1, rt2,
                                                        u_node_fun, f_node_fun)
    # for i5, u_node_fun in enumerate(tqdm_notebook(u_node_fun_list)):
    for i5, u_node_fun in enumerate(hlx_node_fun_list):
        idx0 = nth * i5 * 3
        for i6, f_node_fun in enumerate(hlx_node_fun_list):
            idx1 = nth * i6 * 3
            for i0, th0 in enumerate(th_list):  # f_node
                i2 = idx1 + i0 * 3
                theta_a = th0 - dth / 2
                theta_b = th0 + dth / 2
                tsij = integrate.quad_vec(warpper_mij2, theta_a, theta_b, workers=workers,
                                          epsabs=epsabs, epsrel=epsrel, limit=limit, )[0]
                m[idx0:idx0 + (3 * nth), i2:i2 + 3] = tsij * intFct

    # part 2, local part.
    warpper_mij = lambda theta: stokeslets_matrix_mij(th, theta, ph, rt1, rt2, u_node_fun,
                                                      f_node_fun)
    for i5, (u_node_fun, f_node_fun, FnMat_fun) in enumerate(
            zip(hlx_node_fun_list, hlx_node_fun_list, FnMat_fun_list)):
        idx0 = nth * i5 * 3
        for i0, th in enumerate(th_list):
            i1 = idx0 + i0 * 3
            theta_a = th - dth / 2
            theta_b = th + dth / 2
            t1 = FnMat_fun(th, ph, rt1, rt2) / (4 * np.pi)
            t2 = integrate.quad_vec(warpper_mij, th + natu_cut_th, theta_b, workers=workers,
                                    epsabs=epsabs, epsrel=epsrel, limit=limit, )[0]
            t3 = integrate.quad_vec(warpper_mij, theta_a, th - natu_cut_th, workers=workers,
                                    epsabs=epsabs, epsrel=epsrel, limit=limit, )[0]
            m[i1:i1 + 3, i1:i1 + 3] = t1 + (t2 + t3) * intFct
    return m


# Lightill Slender Body Theory, this version assert mesh size > local part size,
#   This version contain the influnce of doublet from the other helixs.
def Lightill_slenderBody_matrix_nhlx_v3(ph, rt1, rt2, ch, nth=None,
                                        hlx_node_fun_list=(x1_fun,), FnMat_fun_list=(Fn1Mat_fun,),
                                        epsabs=1e-200, epsrel=1e-08, limit=10000,
                                        workers=1) -> np.ndarray:
    hlx_info = hlxn_info_fun(ph, rt1, rt2, ch, nth=nth, hlx_node_fun_list=hlx_node_fun_list)
    dth = hlx_info['dth']
    th_list = hlx_info['th_list']
    nth = hlx_info['nth']
    S = hlx_info['S']
    natu_cut_th = hlx_info['natu_cut_th']
    intFct = S / (2 * np.pi)
    err_msg = 'tube size of u_node_fun_list and f_node_fun_list is different. '
    assert len(hlx_node_fun_list) == len(hlx_node_fun_list), err_msg
    n_hlx = len(hlx_node_fun_list)

    m = np.zeros((nth * n_hlx * 3, nth * n_hlx * 3))

    # part 1, nonlocal part, version 3, locally integration of tsij, loop along j (force points).
    warpper_mij2 = lambda theta: \
        stokeslets_matrix_mij2(th_list, theta, i0, ph, rt1, rt2, u_node_fun, f_node_fun)

    def warpper_dij2(theta):
        # ignore self interaction.
        if u_node_fun is f_node_fun:
            tdij = np.zeros((th_list.size * 3, 3))
        else:
            tdij = doublets_matrix_dij2(th_list, theta, i0, ph, rt1, rt2,
                                        u_node_fun, f_node_fun)
        return tdij

    # for i5, u_node_fun in enumerate(tqdm_notebook(u_node_fun_list)):
    for i5, u_node_fun in enumerate(hlx_node_fun_list):
        idx0 = nth * i5 * 3
        for i6, f_node_fun in enumerate(hlx_node_fun_list):
            idx1 = nth * i6 * 3
            for i0, th0 in enumerate(th_list):  # f_node
                i2 = idx1 + i0 * 3
                theta_a = th0 - dth / 2
                theta_b = th0 + dth / 2
                tsij = integrate.quad_vec(warpper_mij2, theta_a, theta_b, workers=workers,
                                          epsabs=epsabs, epsrel=epsrel, limit=limit, )[0]
                tdij = integrate.quad_vec(warpper_dij2, theta_a, theta_b, workers=workers,
                                          epsabs=epsabs, epsrel=epsrel, limit=limit, )[0]
                m[idx0:idx0 + (3 * nth), i2:i2 + 3] = (tsij + tdij) * intFct
                # # dbg
                # if not u_node_fun is f_node_fun:
                #     print(np.nanmax(np.abs(tdij / tsij)))

    # part 2, local part.
    warpper_mij = lambda theta: \
        stokeslets_matrix_mij(th, theta, ph, rt1, rt2, u_node_fun, f_node_fun)
    for i5, (u_node_fun, f_node_fun, FnMat_fun) in enumerate(
            zip(hlx_node_fun_list, hlx_node_fun_list, FnMat_fun_list)):
        idx0 = nth * i5 * 3
        for i0, th in enumerate(th_list):
            i1 = idx0 + i0 * 3
            theta_a = th - dth / 2
            theta_b = th + dth / 2
            t1 = FnMat_fun(th, ph, rt1, rt2) / (4 * np.pi)
            t2 = integrate.quad_vec(warpper_mij, th + natu_cut_th, theta_b, workers=workers,
                                    epsabs=epsabs, epsrel=epsrel, limit=limit, )[0]
            t3 = integrate.quad_vec(warpper_mij, theta_a, th - natu_cut_th, workers=workers,
                                    epsabs=epsabs, epsrel=epsrel, limit=limit, )[0]
            m[i1:i1 + 3, i1:i1 + 3] = t1 + (t2 + t3) * intFct
    return m


# Keller-Rubinow-Johnson Slender Body Theory, this version assert mesh size > local part size.
def KRJ_slenderBody_matrix_nhlx(ph, rt1, rt2, ch, nth=None,
                                hlx_node_fun_list=(x1_fun,), T_fun_list=(T1_fun,),
                                epsabs=1e-200, epsrel=1e-08, limit=10000,
                                workers=1, use_tqdm_notebook=False) -> np.ndarray:
    def _do_m(i0, th0):
        i2 = idx1 + i0 * 3
        theta_a = th0 - dth / 2
        theta_b = th0 + dth / 2
        tsij = integrate.quad_vec(warpper_mij2, theta_a, theta_b, workers=workers,
                                  epsabs=epsabs, epsrel=epsrel, limit=limit, )[0]
        m[idx0:idx0 + (3 * nth), i2:i2 + 3] = tsij * intFct

    hlx_info = hlxn_info_fun(ph, rt1, rt2, ch, nth=nth, hlx_node_fun_list=hlx_node_fun_list,
                             check_dth=False)
    dth = hlx_info['dth']
    th_list = hlx_info['th_list']
    nth = hlx_info['nth']
    S = hlx_info['S']
    intFct = S / (2 * np.pi)
    err_msg = 'tube size of u_node_fun_list and f_node_fun_list is different. '
    assert len(hlx_node_fun_list) == len(hlx_node_fun_list), err_msg
    n_hlx = len(hlx_node_fun_list)

    m = np.zeros((nth * n_hlx * 3, nth * n_hlx * 3))

    # part 1, nonlocal part, version 3, locally integration of tsij, loop along j (force points).
    warpper_mij2 = lambda theta: \
        stokeslets_matrix_mij2(th_list, theta, i0, ph, rt1, rt2, u_node_fun, f_node_fun)
    # for i5, u_node_fun in enumerate(tqdm_notebook(u_node_fun_list)):
    for i5, u_node_fun in enumerate(hlx_node_fun_list):
        idx0 = nth * i5 * 3
        for i6, f_node_fun in enumerate(hlx_node_fun_list):
            idx1 = nth * i6 * 3
            if use_tqdm_notebook:
                desc = '%d / %d, %d / %d' % (i5 + 1, n_hlx, i6 + 1, n_hlx)
                for i0, th0 in enumerate(tqdm_notebook(th_list, desc=desc)):  # f_node
                    _do_m(i0, th0)
            else:
                for i0, th0 in enumerate(th_list):  # f_node
                    _do_m(i0, th0)

    # part 2, local part.
    min_th = th_list[0] - dth / 2
    max_th = th_list[-1] + dth / 2
    s_min = min_th * intFct
    s_max = max_th * intFct
    warpper_mij_self = lambda theta: \
        stokeslets_matrix_mij(th, theta, ph, rt1, rt2, u_node_fun, f_node_fun) * 8 * np.pi \
        - tc / (np.abs(th - theta) * intFct)
    for i5, (u_node_fun, f_node_fun, T_fun) in enumerate(
            zip(hlx_node_fun_list, hlx_node_fun_list, T_fun_list)):
        idx0 = nth * i5 * 3
        for i0, th in enumerate(th_list):
            i1 = idx0 + i0 * 3
            theta_a = th - dth / 2
            theta_b = th + dth / 2
            s_a = theta_a * intFct
            s_b = theta_b * intFct
            t = T_fun(th, ph, rt1, rt2)
            # rt2_use = rt2 * 4 / np.pi # This factor following Rodenborn2013
            rt2_use = rt2
            Lsbt = np.log((-s_max * s_min + (s_max + s_min) * th - th ** 2) / rt2_use ** 2)
            ta = np.eye(3)
            tb = np.outer(t, t)
            tc = ta + tb
            t1 = Lsbt * tc + ta - 3 * tb
            s0 = th * intFct
            t2 = np.log((s_min - s0) / (s_a - s0))
            t3 = np.log((s_max - s0) / (s_b - s0))
            tint_other = (t2 + t3) * tc
            # tfct = 100
            # tint_self = integrate.quad_vec(warpper_mij_self, theta_a, th - dth / tfct,
            #                                workers=workers, epsabs=epsabs, epsrel=epsrel,
            #                                limit=limit, )[0] + \
            #             integrate.quad_vec(warpper_mij_self, th + dth / tfct, theta_b,
            #                                workers=workers, epsabs=epsabs, epsrel=epsrel,
            #                                limit=limit, )[0]
            tint_self = 0
            m[i1:i1 + 3, i1:i1 + 3] = (t1 - tint_other - tint_self) / (8 * np.pi)
    return m


def AtBtCt(nodes, dS, M, gmres_maxiter=300, ifprint=False):
    # translation
    Uz = 1
    Wz = 0
    u = np.zeros(nodes.size)
    u[0::3] = -Wz * nodes[:, 1]
    u[1::3] = Wz * nodes[:, 0]
    u[2::3] = Uz
    t0 = time()
    ftr = sparse.linalg.gmres(M, u, restart=gmres_maxiter, maxiter=gmres_maxiter)[0]
    At = np.sum(ftr[2::3]) * dS
    Bt1 = -np.sum(nodes[:, 0] * ftr[1::3] - nodes[:, 1] * ftr[0::3]) * dS
    t1 = time()
    if ifprint:
        print('solve matrix equation use %fs' % (t1 - t0))

    # rotation
    Uz = 0
    Wz = 1
    u = np.zeros(nodes.size)
    u[0::3] = -Wz * nodes[:, 1]
    u[1::3] = Wz * nodes[:, 0]
    u[2::3] = Uz
    t0 = time()
    frt = sparse.linalg.gmres(M, u, restart=gmres_maxiter, maxiter=gmres_maxiter)[0]
    Bt2 = -np.sum(frt[2::3]) * dS
    Ct = np.sum(nodes[:, 0] * frt[1::3] - nodes[:, 1] * frt[0::3]) * dS
    t1 = time()
    Bt = (Bt1 + Bt2) / 2
    if ifprint:
        print('solve matrix equation use %fs' % (t1 - t0))
        print('At=%f, Bt=%f, Ct=%f, rel_err of Bt is %e' % (At, Bt, Ct, (Bt1 - Bt2) / Bt))
    return At, Bt, Ct, ftr, frt


def Lightill_AtBtCt_1hlx(ph, rt1, rt2, ch, nth=None, gmres_maxiter=300,
                         hlx_node_fun=x1_fun, FnMat_fun=Fn1Mat_fun,
                         intsij_epsabs=1e-200, intsij_epsrel=1e-08, intsij_limit=10000,
                         intsij_workers=1, ifprint=False):
    hlx1_info = hlx1_info_fun(ph, rt1, rt2, ch, nth=nth, hlx_node_fun=hlx_node_fun)
    natu_cut_th = hlx1_info['natu_cut_th']
    dth = hlx1_info['dth']
    nth = hlx1_info['nth']
    nodes = hlx1_info['nodes']
    dS = hlx1_info['dS']
    if ifprint:
        print('--->> ch=%f, 2*natu_cut_th=%f, dth=%f, rel_err=%e' % (
            ch, 2 * natu_cut_th, dth, (dth - 2 * natu_cut_th) / natu_cut_th))
        print('nth =', nth)
    t0 = time()
    M = Lightill_slenderBody_matrix_1hlx_v2(ph, rt1, rt2, ch, nth=nth,
                                            hlx_node_fun=hlx_node_fun, FnMat_fun=FnMat_fun,
                                            epsabs=intsij_epsabs, epsrel=intsij_epsrel,
                                            limit=intsij_limit, workers=intsij_workers)
    # M = Lightill_slenderBody_matrix_1hlx(ph, rt1, rt2, ch,
    #                                         hlx_node_fun=hlx_node_fun, FnMat_fun=FnMat_fun,
    #                                         epsabs=intsij_epsabs, epsrel=intsij_epsrel,
    #                                         limit=intsij_limit, workers=intsij_workers)
    t1 = time()
    if ifprint:
        print('create M matrix use %fs' % (t1 - t0))

    At, Bt, Ct, ftr, frt = AtBtCt(nodes, dS, M, gmres_maxiter=gmres_maxiter, ifprint=ifprint)
    return At, Bt, Ct, ftr, frt


def KRJ_AtBtCt_1hlx(ph, rt1, rt2, ch, nth=None, gmres_maxiter=300,
                    hlx_node_fun=x1_fun, T_fun=T1_fun,
                    intsij_epsabs=1e-200, intsij_epsrel=1e-08, intsij_limit=10000,
                    intsij_workers=1, ifprint=False):
    hlx1_info = hlx1_info_fun(ph, rt1, rt2, ch, nth=nth, hlx_node_fun=hlx_node_fun, check_dth=False)
    natu_cut_th = hlx1_info['natu_cut_th']
    dth = hlx1_info['dth']
    nth = hlx1_info['nth']
    nodes = hlx1_info['nodes']
    dS = hlx1_info['dS']
    if ifprint:
        print('--->> ch=%f, 2*natu_cut_th=%f, dth=%f, rel_err=%e' % (
            ch, 2 * natu_cut_th, dth, (dth - 2 * natu_cut_th) / natu_cut_th))
        print('nth =', nth)
    t0 = time()
    M = KRJ_slenderBody_matrix_1hlx(ph, rt1, rt2, ch, nth=nth,
                                    hlx_node_fun=hlx_node_fun, T_fun=T_fun,
                                    epsabs=intsij_epsabs, epsrel=intsij_epsrel,
                                    limit=intsij_limit, workers=intsij_workers)
    t1 = time()
    if ifprint:
        print('create M matrix use %fs' % (t1 - t0))

    At, Bt, Ct, ftr, frt = AtBtCt(nodes, dS, M, gmres_maxiter=gmres_maxiter, ifprint=ifprint)
    return At, Bt, Ct, ftr, frt


def Lightill_AtBtCt_nhlx(ph, rt1, rt2, ch, nth=None, gmres_maxiter=300,
                         hlx_node_fun_list=(x1_fun,), FnMat_fun_list=(Fn1Mat_fun,),
                         intsij_epsabs=1e-200, intsij_epsrel=1e-08, intsij_limit=10000,
                         intsij_workers=1, ifprint=False):
    hlxn_info = hlxn_info_fun(ph, rt1, rt2, ch, nth=nth, hlx_node_fun_list=hlx_node_fun_list)
    natu_cut_th = hlxn_info['natu_cut_th']
    dth = hlxn_info['dth']
    nth = hlxn_info['nth']
    nodes = hlxn_info['nodes']
    dS = hlxn_info['dS']
    if ifprint:
        print('--->> ch=%f, 2*natu_cut_th=%f, dth=%f, rel_err=%e' % (
            ch, 2 * natu_cut_th, dth, (dth - 2 * natu_cut_th) / natu_cut_th))
        print('nth = %d × %d ' % (nth, len(hlx_node_fun_list)))
    t0 = time()
    # M = Lightill_slenderBody_matrix_nhlx(ph, rt1, rt2, ch,
    #                                      hlx_node_fun_list=hlx_node_fun_list,
    #                                      FnMat_fun_list=FnMat_fun_list,
    #                                      epsabs=intsij_epsabs, epsrel=intsij_epsrel,
    #                                      limit=intsij_limit, workers=intsij_workers)
    M = Lightill_slenderBody_matrix_nhlx_v2(ph, rt1, rt2, ch, nth=nth,
                                            hlx_node_fun_list=hlx_node_fun_list,
                                            FnMat_fun_list=FnMat_fun_list,
                                            epsabs=intsij_epsabs, epsrel=intsij_epsrel,
                                            limit=intsij_limit, workers=intsij_workers)
    # M = Lightill_slenderBody_matrix_nhlx_v3(ph, rt1, rt2, ch, nth=nth,
    #                                         hlx_node_fun_list=hlx_node_fun_list,
    #                                         FnMat_fun_list=FnMat_fun_list,
    #                                         epsabs=intsij_epsabs, epsrel=intsij_epsrel,
    #                                         limit=intsij_limit, workers=intsij_workers)
    t1 = time()
    if ifprint:
        print('create M matrix use %fs' % (t1 - t0))

    At, Bt, Ct, ftr, frt = AtBtCt(nodes, dS, M, gmres_maxiter=gmres_maxiter, ifprint=ifprint)
    return At, Bt, Ct, ftr, frt


def KRJ_AtBtCt_nhlx(ph, rt1, rt2, ch, nth=None, gmres_maxiter=300,
                    hlx_node_fun_list=(x1_fun,), T_fun_list=(T1_fun,),
                    intsij_epsabs=1e-200, intsij_epsrel=1e-08, intsij_limit=10000,
                    intsij_workers=1, ifprint=False, use_tqdm_notebook=False):
    hlxn_info = hlxn_info_fun(ph, rt1, rt2, ch, nth=nth, hlx_node_fun_list=hlx_node_fun_list,
                              check_dth=False)
    natu_cut_th = hlxn_info['natu_cut_th']
    dth = hlxn_info['dth']
    nth = hlxn_info['nth']
    nodes = hlxn_info['nodes']
    dS = hlxn_info['dS']
    if ifprint:
        print('--->> ch=%f, 2*natu_cut_th=%f, dth=%f, rel_err=%e' % (
            ch, 2 * natu_cut_th, dth, (dth - 2 * natu_cut_th) / natu_cut_th))
        print('nth = %d × %d ' % (nth, len(hlx_node_fun_list)))
    t0 = time()
    M = KRJ_slenderBody_matrix_nhlx(ph, rt1, rt2, ch, nth=nth,
                                    hlx_node_fun_list=hlx_node_fun_list, T_fun_list=T_fun_list,
                                    epsabs=intsij_epsabs, epsrel=intsij_epsrel,
                                    limit=intsij_limit, workers=intsij_workers,
                                    use_tqdm_notebook=use_tqdm_notebook)
    t1 = time()
    if ifprint:
        print('create M matrix use %fs' % (t1 - t0))

    At, Bt, Ct, ftr, frt = AtBtCt(nodes, dS, M, gmres_maxiter=gmres_maxiter, ifprint=ifprint)
    return At, Bt, Ct, ftr, frt


# Lighthill Slender body theory based asymtotic Theory
# sij_1tail = lambda theta, ph, rt1, rt2: np.array(((((16*np.pi**2*rt1**2 + ph**2*theta**2)*np.cos(theta) - 2*np.pi**2*rt1**2*(5 + 3*np.cos(2*theta)))/(2.*(8*np.pi**2*rt1**2 + ph**2*theta**2 - 8*np.pi**2*rt1**2*np.cos(theta))**1.5),0,0),(0,((8*np.pi**2*rt1**2 + ph**2*theta**2)*np.cos(theta) - 2*np.pi**2*rt1**2*(1 + 3*np.cos(2*theta)))/(2.*(8*np.pi**2*rt1**2 + ph**2*theta**2 - 8*np.pi**2*rt1**2*np.cos(theta))**1.5),(ph*np.pi*rt1*theta*np.sin(theta))/(8*np.pi**2*rt1**2 + ph**2*theta**2 - 8*np.pi**2*rt1**2*np.cos(theta))**1.5),(0,(ph*np.pi*rt1*theta*np.sin(theta))/(8*np.pi**2*rt1**2 + ph**2*theta**2 - 8*np.pi**2*rt1**2*np.cos(theta))**1.5,(4*np.pi**2*rt1**2 + ph**2*theta**2 - 4*np.pi**2*rt1**2*np.cos(theta))/(8*np.pi**2*rt1**2 + ph**2*theta**2 - 8*np.pi**2*rt1**2*np.cos(theta))**1.5)))
# sij = lambda theta, ph, rt1, rt2: np.array(((((-((16*np.pi**2*rt1**2 + ph**2*theta**2)*np.cos(theta)) - 2*np.pi**2*rt1**2*(5 + 3*np.cos(2*theta)))/(8*np.pi**2*rt1**2 + ph**2*theta**2 + 8*np.pi**2*rt1**2*np.cos(theta))**1.5 + ((16*np.pi**2*rt1**2 + ph**2*theta**2)*np.cos(theta) - 2*np.pi**2*rt1**2*(5 + 3*np.cos(2*theta)))/(8*np.pi**2*rt1**2 + ph**2*theta**2 - 8*np.pi**2*rt1**2*np.cos(theta))**1.5)/2.,0,0),(0,((-((8*np.pi**2*rt1**2 + ph**2*theta**2)*np.cos(theta)) - 2*np.pi**2*rt1**2*(1 + 3*np.cos(2*theta)))/(8*np.pi**2*rt1**2 + ph**2*theta**2 + 8*np.pi**2*rt1**2*np.cos(theta))**1.5 + ((8*np.pi**2*rt1**2 + ph**2*theta**2)*np.cos(theta) - 2*np.pi**2*rt1**2*(1 + 3*np.cos(2*theta)))/(8*np.pi**2*rt1**2 + ph**2*theta**2 - 8*np.pi**2*rt1**2*np.cos(theta))**1.5)/2.,ph*np.pi*rt1*theta*((8*np.pi**2*rt1**2 + ph**2*theta**2 - 8*np.pi**2*rt1**2*np.cos(theta))**(-1.5) - (8*np.pi**2*rt1**2 + ph**2*theta**2 + 8*np.pi**2*rt1**2*np.cos(theta))**(-1.5))*np.sin(theta)),(0,ph*np.pi*rt1*theta*((8*np.pi**2*rt1**2 + ph**2*theta**2 - 8*np.pi**2*rt1**2*np.cos(theta))**(-1.5) - (8*np.pi**2*rt1**2 + ph**2*theta**2 + 8*np.pi**2*rt1**2*np.cos(theta))**(-1.5))*np.sin(theta),(4*np.pi**2*rt1**2 + ph**2*theta**2 - 4*np.pi**2*rt1**2*np.cos(theta))/(8*np.pi**2*rt1**2 + ph**2*theta**2 - 8*np.pi**2*rt1**2*np.cos(theta))**1.5 + (4*np.pi**2*rt1**2 + ph**2*theta**2 + 4*np.pi**2*rt1**2*np.cos(theta))/(8*np.pi**2*rt1**2 + ph**2*theta**2 + 8*np.pi**2*rt1**2*np.cos(theta))**1.5)))
# dij = lambda theta, ph, rt1, rt2: np.array((((rt2**2*np.cos(theta)*(8*np.pi**4*rt1**2 - 2*ph**2*np.pi**2*theta**2 + 8*np.pi**4*rt1**2*np.cos(theta)*(-4 + 3*np.cos(theta)))*(-1 + (4*np.pi**2*rt1**2*np.sin(theta)**2)/(ph**2 + 4*np.pi**2*rt1**2)))/(2.*(8*np.pi**2*rt1**2 + ph**2*theta**2 - 8*np.pi**2*rt1**2*np.cos(theta))**2.5),(-48*np.pi**6*rt1**4*rt2**2*(-1 + np.cos(theta))*np.cos(theta)**2*np.sin(theta)**2)/((ph**2 + 4*np.pi**2*rt1**2)*(8*np.pi**2*rt1**2 + ph**2*theta**2 - 8*np.pi**2*rt1**2*np.cos(theta))**2.5),(-12*ph**2*np.pi**4*rt1**2*rt2**2*theta*(-1 + np.cos(theta))*np.sin(theta))/((ph**2 + 4*np.pi**2*rt1**2)*(8*np.pi**2*rt1**2 + ph**2*theta**2 - 8*np.pi**2*rt1**2*np.cos(theta))**2.5)),((-48*np.pi**6*rt1**4*rt2**2*(-1 + np.cos(theta))*np.cos(theta)**2*np.sin(theta)**2)/((ph**2 + 4*np.pi**2*rt1**2)*(8*np.pi**2*rt1**2 + ph**2*theta**2 - 8*np.pi**2*rt1**2*np.cos(theta))**2.5),-((np.pi**2*rt2**2*np.cos(theta)*(-1 + (4*np.pi**2*rt1**2*np.cos(theta)**2)/(ph**2 + 4*np.pi**2*rt1**2))*(2*np.pi**2*rt1**2 + ph**2*theta**2 + 2*np.pi**2*rt1**2*(-4*np.cos(theta) + 3*np.cos(2*theta))))/(8*np.pi**2*rt1**2 + ph**2*theta**2 - 8*np.pi**2*rt1**2*np.cos(theta))**2.5),(6*ph**2*np.pi**4*rt1**2*rt2**2*theta*np.sin(2*theta))/((ph**2 + 4*np.pi**2*rt1**2)*(8*np.pi**2*rt1**2 + ph**2*theta**2 - 8*np.pi**2*rt1**2*np.cos(theta))**2.5)),((-12*ph**2*np.pi**4*rt1**2*rt2**2*theta*(-1 + np.cos(theta))*np.cos(theta)*np.sin(theta))/((ph**2 + 4*np.pi**2*rt1**2)*(8*np.pi**2*rt1**2 + ph**2*theta**2 - 8*np.pi**2*rt1**2*np.cos(theta))**2.5),(6*ph**2*np.pi**4*rt1**2*rt2**2*theta*np.cos(theta)*np.sin(2*theta))/((ph**2 + 4*np.pi**2*rt1**2)*(8*np.pi**2*rt1**2 + ph**2*theta**2 - 8*np.pi**2*rt1**2*np.cos(theta))**2.5),(-8*np.pi**4*rt1**2*rt2**2*(-4*np.pi**2*rt1**2 + ph**2*theta**2 + 4*np.pi**2*rt1**2*np.cos(theta)))/((ph**2 + 4*np.pi**2*rt1**2)*(8*np.pi**2*rt1**2 + ph**2*theta**2 - 8*np.pi**2*rt1**2*np.cos(theta))**2.5))))
_sij_1hlx_fun = lambda theta, ph, rt1: np.array(((((
                                                           16 * np.pi ** 2 * rt1 ** 2 + ph ** 2 * theta ** 2) * np.cos(
        theta) - 2 * np.pi ** 2 * rt1 ** 2 * (5 + 3 * np.cos(2 * theta))) / (2. * (
        8 * np.pi ** 2 * rt1 ** 2 + ph ** 2 * theta ** 2 - 8 * np.pi ** 2 * rt1 ** 2 * np.cos(
        theta)) ** 1.5), 0, 0), (0, ((8 * np.pi ** 2 * rt1 ** 2 + ph ** 2 * theta ** 2) * np.cos(
        theta) - 2 * np.pi ** 2 * rt1 ** 2 * (1 + 3 * np.cos(2 * theta))) / (2. * (
        8 * np.pi ** 2 * rt1 ** 2 + ph ** 2 * theta ** 2 - 8 * np.pi ** 2 * rt1 ** 2 * np.cos(
        theta)) ** 1.5), (ph * np.pi * rt1 * theta * np.sin(theta)) / (
                                         8 * np.pi ** 2 * rt1 ** 2 + ph ** 2 * theta ** 2 - 8 * np.pi ** 2 * rt1 ** 2 * np.cos(
                                         theta)) ** 1.5), (0, (
        ph * np.pi * rt1 * theta * np.sin(theta)) / (
                                                                   8 * np.pi ** 2 * rt1 ** 2 + ph ** 2 * theta ** 2 - 8 * np.pi ** 2 * rt1 ** 2 * np.cos(
                                                                   theta)) ** 1.5, (
                                                                   4 * np.pi ** 2 * rt1 ** 2 + ph ** 2 * theta ** 2 - 4 * np.pi ** 2 * rt1 ** 2 * np.cos(
                                                                   theta)) / (
                                                                   8 * np.pi ** 2 * rt1 ** 2 + ph ** 2 * theta ** 2 - 8 * np.pi ** 2 * rt1 ** 2 * np.cos(
                                                                   theta)) ** 1.5)))
_sij_2hlx_fun = lambda theta, ph, rt1: np.array(((((-(
        (16 * np.pi ** 2 * rt1 ** 2 + ph ** 2 * theta ** 2) * np.cos(
        theta)) - 2 * np.pi ** 2 * rt1 ** 2 * (5 + 3 * np.cos(2 * theta))) / (
                                                           8 * np.pi ** 2 * rt1 ** 2 + ph ** 2 * theta ** 2 + 8 * np.pi ** 2 * rt1 ** 2 * np.cos(
                                                           theta)) ** 1.5 + ((
                                                                                     16 * np.pi ** 2 * rt1 ** 2 + ph ** 2 * theta ** 2) * np.cos(
        theta) - 2 * np.pi ** 2 * rt1 ** 2 * (5 + 3 * np.cos(2 * theta))) / (
                                                           8 * np.pi ** 2 * rt1 ** 2 + ph ** 2 * theta ** 2 - 8 * np.pi ** 2 * rt1 ** 2 * np.cos(
                                                           theta)) ** 1.5) / 2., 0, 0), (0, ((-(
        (8 * np.pi ** 2 * rt1 ** 2 + ph ** 2 * theta ** 2) * np.cos(
        theta)) - 2 * np.pi ** 2 * rt1 ** 2 * (1 + 3 * np.cos(2 * theta))) / (
                                                                                                     8 * np.pi ** 2 * rt1 ** 2 + ph ** 2 * theta ** 2 + 8 * np.pi ** 2 * rt1 ** 2 * np.cos(
                                                                                                     theta)) ** 1.5 + (
                                                                                                     (
                                                                                                             8 * np.pi ** 2 * rt1 ** 2 + ph ** 2 * theta ** 2) * np.cos(
                                                                                                     theta) - 2 * np.pi ** 2 * rt1 ** 2 * (
                                                                                                             1 + 3 * np.cos(
                                                                                                             2 * theta))) / (
                                                                                                     8 * np.pi ** 2 * rt1 ** 2 + ph ** 2 * theta ** 2 - 8 * np.pi ** 2 * rt1 ** 2 * np.cos(
                                                                                                     theta)) ** 1.5) / 2.,
                                                                                         ph * np.pi * rt1 * theta * (
                                                                                                 (
                                                                                                         8 * np.pi ** 2 * rt1 ** 2 + ph ** 2 * theta ** 2 - 8 * np.pi ** 2 * rt1 ** 2 * np.cos(
                                                                                                         theta)) ** (
                                                                                                     -1.5) - (
                                                                                                         8 * np.pi ** 2 * rt1 ** 2 + ph ** 2 * theta ** 2 + 8 * np.pi ** 2 * rt1 ** 2 * np.cos(
                                                                                                         theta)) ** (
                                                                                                     -1.5)) * np.sin(
                                                                                                 theta)),
                                                 (0,
                                                  ph * np.pi * rt1 * theta * (
                                                          (
                                                                  8 * np.pi ** 2 * rt1 ** 2 + ph ** 2 * theta ** 2 - 8 * np.pi ** 2 * rt1 ** 2 * np.cos(
                                                                  theta)) ** (
                                                              -1.5) - (
                                                                  8 * np.pi ** 2 * rt1 ** 2 + ph ** 2 * theta ** 2 + 8 * np.pi ** 2 * rt1 ** 2 * np.cos(
                                                                  theta)) ** (
                                                              -1.5)) * np.sin(
                                                          theta),
                                                  (
                                                          4 * np.pi ** 2 * rt1 ** 2 + ph ** 2 * theta ** 2 - 4 * np.pi ** 2 * rt1 ** 2 * np.cos(
                                                          theta)) / (
                                                          8 * np.pi ** 2 * rt1 ** 2 + ph ** 2 * theta ** 2 - 8 * np.pi ** 2 * rt1 ** 2 * np.cos(
                                                          theta)) ** 1.5 + (
                                                          4 * np.pi ** 2 * rt1 ** 2 + ph ** 2 * theta ** 2 + 4 * np.pi ** 2 * rt1 ** 2 * np.cos(
                                                          theta)) / (
                                                          8 * np.pi ** 2 * rt1 ** 2 + ph ** 2 * theta ** 2 + 8 * np.pi ** 2 * rt1 ** 2 * np.cos(
                                                          theta)) ** 1.5)))
_dij_2hlx_fun = lambda theta, ph, rt1, rt2: np.array(((-(rt2 ** 2 * np.cos(theta) * (
        8 * np.pi ** 4 * rt1 ** 2 - 2 * ph ** 2 * np.pi ** 2 * theta ** 2 + 8 * np.pi ** 4 * rt1 ** 2 * np.cos(
        theta) * (4 + 3 * np.cos(theta))) * (-1 + (
        4 * np.pi ** 2 * rt1 ** 2 * np.sin(theta) ** 2) / (
                                                     ph ** 2 + 4 * np.pi ** 2 * rt1 ** 2))) / (
                                                               2. * (
                                                               8 * np.pi ** 2 * rt1 ** 2 + ph ** 2 * theta ** 2 + 8 * np.pi ** 2 * rt1 ** 2 * np.cos(
                                                               theta)) ** 2.5), (
                                                               48 * np.pi ** 6 * rt1 ** 4 * rt2 ** 2 * np.cos(
                                                               theta) ** 2 * (1 + np.cos(
                                                               theta)) * np.sin(theta) ** 2) / ((
                                                                                                        ph ** 2 + 4 * np.pi ** 2 * rt1 ** 2) * (
                                                                                                        8 * np.pi ** 2 * rt1 ** 2 + ph ** 2 * theta ** 2 + 8 * np.pi ** 2 * rt1 ** 2 * np.cos(
                                                                                                        theta)) ** 2.5),
                                                       (
                                                               -12 * ph ** 2 * np.pi ** 4 * rt1 ** 2 * rt2 ** 2 * theta * (
                                                               1 + np.cos(theta)) * np.sin(
                                                               theta)) / ((
                                                                                  ph ** 2 + 4 * np.pi ** 2 * rt1 ** 2) * (
                                                                                  8 * np.pi ** 2 * rt1 ** 2 + ph ** 2 * theta ** 2 + 8 * np.pi ** 2 * rt1 ** 2 * np.cos(
                                                                                  theta)) ** 2.5)),
                                                      (
                                                          (
                                                                  48 * np.pi ** 6 * rt1 ** 4 * rt2 ** 2 * np.cos(
                                                                  theta) ** 2 * (1 + np.cos(
                                                                  theta)) * np.sin(theta) ** 2) / ((
                                                                                                           ph ** 2 + 4 * np.pi ** 2 * rt1 ** 2) * (
                                                                                                           8 * np.pi ** 2 * rt1 ** 2 + ph ** 2 * theta ** 2 + 8 * np.pi ** 2 * rt1 ** 2 * np.cos(
                                                                                                           theta)) ** 2.5),
                                                          (np.pi ** 2 * rt2 ** 2 * np.cos(theta) * (
                                                                  -1 + (
                                                                  4 * np.pi ** 2 * rt1 ** 2 * np.cos(
                                                                  theta) ** 2) / (
                                                                          ph ** 2 + 4 * np.pi ** 2 * rt1 ** 2)) * (
                                                                   2 * np.pi ** 2 * rt1 ** 2 + ph ** 2 * theta ** 2 + 2 * np.pi ** 2 * rt1 ** 2 * (
                                                                   4 * np.cos(
                                                                   theta) + 3 * np.cos(
                                                                   2 * theta)))) / (
                                                                  8 * np.pi ** 2 * rt1 ** 2 + ph ** 2 * theta ** 2 + 8 * np.pi ** 2 * rt1 ** 2 * np.cos(
                                                                  theta)) ** 2.5, (
                                                                  6 * ph ** 2 * np.pi ** 4 * rt1 ** 2 * rt2 ** 2 * theta * np.sin(
                                                                  2 * theta)) / ((
                                                                                         ph ** 2 + 4 * np.pi ** 2 * rt1 ** 2) * (
                                                                                         8 * np.pi ** 2 * rt1 ** 2 + ph ** 2 * theta ** 2 + 8 * np.pi ** 2 * rt1 ** 2 * np.cos(
                                                                                         theta)) ** 2.5)),
                                                      ((
                                                               12 * ph ** 2 * np.pi ** 4 * rt1 ** 2 * rt2 ** 2 * theta * np.cos(
                                                               theta) * (1 + np.cos(
                                                               theta)) * np.sin(
                                                               theta)) / ((
                                                                                  ph ** 2 + 4 * np.pi ** 2 * rt1 ** 2) * (
                                                                                  8 * np.pi ** 2 * rt1 ** 2 + ph ** 2 * theta ** 2 + 8 * np.pi ** 2 * rt1 ** 2 * np.cos(
                                                                                  theta)) ** 2.5), (
                                                               -6 * ph ** 2 * np.pi ** 4 * rt1 ** 2 * rt2 ** 2 * theta * np.cos(
                                                               theta) * np.sin(2 * theta)) / ((
                                                                                                      ph ** 2 + 4 * np.pi ** 2 * rt1 ** 2) * (
                                                                                                      8 * np.pi ** 2 * rt1 ** 2 + ph ** 2 * theta ** 2 + 8 * np.pi ** 2 * rt1 ** 2 * np.cos(
                                                                                                      theta)) ** 2.5),
                                                       (8 * np.pi ** 4 * rt1 ** 2 * rt2 ** 2 * (
                                                               4 * np.pi ** 2 * rt1 ** 2 - ph ** 2 * theta ** 2 + 4 * np.pi ** 2 * rt1 ** 2 * np.cos(
                                                               theta))) / ((
                                                                                   ph ** 2 + 4 * np.pi ** 2 * rt1 ** 2) * (
                                                                                   8 * np.pi ** 2 * rt1 ** 2 + ph ** 2 * theta ** 2 + 8 * np.pi ** 2 * rt1 ** 2 * np.cos(
                                                                                   theta)) ** 2.5))))
sij_1hlx_fun = lambda theta, ph, rt1: (_sij_1hlx_fun(theta, ph, rt1), 1)
sij_2hlx_fun = lambda theta, ph, rt1: (_sij_2hlx_fun(theta, ph, rt1), 2)
dij_2hlx_fun = lambda theta, ph, rt1, rt2: _dij_2hlx_fun(theta, ph, rt1, rt2)


def dbg_intp_c33(ph, rt1, rt2, ch, sij_fun=sij_2hlx_fun, epsabs=1e-200,
                 epsrel=1e-08):
    s33_warper = lambda theta: sij_fun(theta, ph, rt1)[0][2, 2]
    tFij = Fn1Mat_fun(0, ph, rt1, np.nan)
    maxtheta = ch * np.pi  # (2 \pi n_1) / 2
    mintheta = optimize.brentq(natu_cut_th1_fun, 0, np.pi, args=(ph, rt1, rt2))
    S = np.sqrt(4 * np.pi ** 2 * rt1 ** 2 + ph ** 2)
    # noinspection PyTupleAssignmentBalance,PyTypeChecker
    tS33, _ = integrate.quad_vec(s33_warper, mintheta, maxtheta, epsabs=epsabs, epsrel=epsrel)
    tM33 = tS33 * S / (2 * np.pi) + tFij[2, 2] / (4 * np.pi)
    return tM33, tS33 * S / (2 * np.pi), tFij[2, 2] / (4 * np.pi)


def asymtoticM_lighthill_000(ph, rt1, rt2, ch, sij_fun=sij_2hlx_fun,
                             epsabs=1e-200, epsrel=1e-08) -> np.ndarray:
    sij_warper = lambda theta: sij_fun(theta, ph, rt1)[0]
    tFij = Fn1Mat_fun(0, ph, rt1, np.nan)
    maxtheta = ch * np.pi  # (2 \pi n_1) / 2
    mintheta = optimize.brentq(natu_cut_th1_fun, 0, np.pi, args=(ph, rt1, rt2))
    S = np.sqrt(4 * np.pi ** 2 * rt1 ** 2 + ph ** 2)
    # noinspection PyTupleAssignmentBalance,PyTypeChecker
    tSij, _ = integrate.quad_vec(sij_warper, mintheta, maxtheta, epsabs=epsabs, epsrel=epsrel)
    tMij = tSij * S / (2 * np.pi) + tFij / (4 * np.pi)
    return tMij


def asymtoticM_lighthill(ph, rt1, rt2, ch, sij_fun=sij_2hlx_fun,
                         epsabs=1e-200, epsrel=1e-08,
                         use_doublets=False) -> np.ndarray:
    if use_doublets:
        err_msg = 'input sij_fun must be "sij_2hlx_fun". '
        assert sij_fun is sij_2hlx_fun
    sij_warper = lambda theta: sij_fun(theta, ph, rt1)[0]
    tFij = Fn1Mat_fun(0, ph, rt1, np.nan)
    maxtheta = ch * np.pi  # (2 \pi n_1) / 2
    mintheta = optimize.brentq(natu_cut_th1_fun, 0, np.pi, args=(ph, rt1, rt2))
    S = np.sqrt(4 * np.pi ** 2 * rt1 ** 2 + ph ** 2)
    # noinspection PyTupleAssignmentBalance,PyTypeChecker
    tSij, _ = integrate.quad_vec(sij_warper, mintheta, maxtheta, epsabs=epsabs, epsrel=epsrel)
    tMij = tSij * S / (2 * np.pi) + tFij / (4 * np.pi)
    if use_doublets:
        # the effect of the doublet for the 1th helix is in the tFij.
        dij_warper = lambda theta: _dij_2hlx_fun(theta, ph, rt1, rt2)
        tDij, _ = integrate.quad_vec(dij_warper, 0, maxtheta, epsabs=epsabs, epsrel=epsrel)
        tMij = tMij + tDij * S / (2 * np.pi)
    return tMij


def asymtotic_AtBtCtCij_lighthill(ph, rt1, rt2, ch, sij_fun=sij_2hlx_fun,
                                  epsabs=1e-200, epsrel=1e-08, ifprint=False,
                                  use_doublets=False):
    tMij = asymtoticM_lighthill(ph, rt1, rt2, ch, sij_fun=sij_fun,
                                epsabs=epsabs, epsrel=epsrel, use_doublets=use_doublets)
    nhlx = sij_fun(np.nan, np.nan, np.nan)[1]
    S = np.sqrt(4 * np.pi ** 2 * rt1 ** 2 + ph ** 2)
    maxtheta = ch * np.pi  # (2 \pi n_1) / 2
    # translation
    u0 = np.array((0, 0, 1))
    ftr = np.linalg.solve(tMij, u0)
    At = ftr[2] * (S * 2 * maxtheta / (2 * np.pi)) * nhlx
    Bt1 = -ftr[1] * rt1 * (S * 2 * maxtheta / (2 * np.pi)) * nhlx
    # rotation
    u0 = np.cross((0, 0, 1), (rt1, 0, 0))
    frt = np.linalg.solve(tMij, u0)
    Bt2 = -frt[2] * (S * 2 * maxtheta / (2 * np.pi)) * nhlx
    Ct = frt[1] * rt1 * (S * 2 * maxtheta / (2 * np.pi)) * nhlx
    Bt = (Bt1 + Bt2) / 2
    if ifprint:
        print('At=%f, Bt=%f, Ct=%f, rel_err of Bt is %e' % (At, Bt, Ct, (Bt1 - Bt2) / Bt))
    tCij = tMij.copy()
    tCij[2, 2] = tCij[2, 2] - (S * np.log(maxtheta) / (ph * np.pi))
    return At, Bt, Ct, ftr, frt, tCij


def asymtotic_AtBtCt_lighthill(ph, rt1, rt2, ch, sij_fun=sij_2hlx_fun,
                               epsabs=1e-200, epsrel=1e-08, ifprint=False):
    At, Bt, Ct, ftr, frt, tCij = \
        asymtotic_AtBtCtCij_lighthill(ph, rt1, rt2, ch, sij_fun=sij_fun,
                                      epsabs=epsabs, epsrel=epsrel, ifprint=ifprint)
    return At, Bt, Ct, ftr, frt


# my asymptotic theory, fit c22, c23, and c33.
_fun_p31 = lambda theta, ph, rt1, c22, c33, c23: \
    (c33 * ph * np.pi + np.sqrt(4 * np.pi ** 2 * rt1 ** 2 + ph ** 2) * np.log(theta)) / (
            (c33 - c23 ** 2 / c22) * ph * np.pi + np.sqrt(
            4 * np.pi ** 2 * rt1 ** 2 + ph ** 2) * np.log(theta))


def iterate_fit_c22c33c23(ph, rt1, tpsi2, tpsi3, tpsi6, chmin=10, chmax=np.inf, rtol=1e-10,
                          atol=1e-100, max_iterate=100):
    def fun_At_fit(tx, tp2):
        return 2 * ph * tx - ph * np.pi / S * tp2

    def fun_Bt_fit(tx, tp1):
        return 2 * ph * rt1 * tx * tp1 - ph * np.pi / S * p2

    def fun_Ct_fit(tx, tp30):
        tp31 = _fun_p31(tx, ph, rt1, c22, c33, c23)
        return 2 * S * rt1 ** 2 / np.pi * tx * tp30 * tp31

    def psi20psi60_tx_ty(tpsi):
        tx = tpsi.index.values * np.pi
        ty = tpsi.values
        idxi = np.isfinite(tx) & np.isfinite(ty) & (chmin * np.pi <= tx) & (tx <= chmax * np.pi)
        fitx = tx[idxi] / ty[idxi]
        fity = np.log(tx[idxi])
        return fitx, fity

    def psi30_tx_ty(tpsi):
        tx = tpsi.index.values * np.pi
        ty = tpsi.values
        idxi = np.isfinite(tx) & np.isfinite(ty) & (chmin * np.pi <= tx) & (tx <= chmax * np.pi)
        fitx = tx[idxi]
        fity = ty[idxi]
        return fitx, fity

    tpsi6 = np.abs(tpsi6)
    S = np.sqrt(4 * np.pi ** 2 * rt1 ** 2 + ph ** 2)

    txAt, tyAt = psi20psi60_tx_ty(tpsi2)
    txCt, tyCt = psi30_tx_ty(tpsi3)
    txBt, tyBt = psi20psi60_tx_ty(tpsi6)

    c22, c33, c23 = 1, 1, 0  # ini guess
    do_iterate = True
    n_iterate = 0
    while do_iterate and n_iterate < max_iterate:
        p30, _ = optimize.curve_fit(fun_Ct_fit, txCt, tyCt,
                                    bounds=((0, ), (np.inf, )))
        p2, _ = optimize.curve_fit(fun_At_fit, txAt, tyAt,
                                   bounds=((-np.inf, ), (np.inf, )))
        p1, _ = optimize.curve_fit(fun_Bt_fit, txBt, tyBt,
                                   bounds=((0, ), (np.inf, )))
        tc22 = 1 / p30
        tc23 = p1 * tc22
        tc33 = p2 + tc23 ** 2 / tc22

        do_iterate = not np.allclose(np.hstack((c22, c33, c23)), np.hstack((tc22, tc33, tc23)),
                                     rtol=rtol, atol=atol)
        (c22, c33, c23) = (tc22, tc33, tc23)
        n_iterate = n_iterate + 1
    return c22, c33, c23


def iterate_fit_c22c33c23_v2(ph, rt1, tpsi2, tpsi3, tpsi6, chmin=10, chmax=np.inf, rtol=1e-10,
                              atol=1e-100, max_iterate=100):
    def fun_At(tx, tp2, tq1):
        return 2 * ph * tx / (ph * np.pi * tp2 / S + np.log(tx)) + tq1

    def fun_Bt(tx, tp1, tq2):
        return 2 * ph * rt1 * tx * tp1 / (ph * np.pi * p2 / S + np.log(tx)) + tq2

    def fun_Ct(tx, tp30, tq3):
        tp31 = _fun_p31(tx, ph, rt1, c22, c33, c23)
        return 2 * S * rt1 ** 2 * tx / np.pi * tp30 * tp31 + tq3

    def psi_tx_ty(tpsi):
        tx = tpsi.index.values * np.pi
        ty = tpsi.values
        idxi = np.isfinite(tx) & np.isfinite(ty) & (chmin * np.pi <= tx) & (tx <= chmax * np.pi)
        fitx = tx[idxi]
        fity = ty[idxi]
        return fitx, fity

    S = np.sqrt(4 * np.pi ** 2 * rt1 ** 2 + ph ** 2)
    txAt, tyAt = psi_tx_ty(tpsi2)
    txCt, tyCt = psi_tx_ty(tpsi3)
    txBt, tyBt = psi_tx_ty(tpsi6)
    assert np.logical_or(np.all(tyBt < 0), np.all(tyBt > 0), )
    tyBt = np.abs(tyBt)

    c22, c33, c23 = 1, 1, 0  # ini guess
    do_iterate = True
    n_iterate = 0
    while do_iterate and n_iterate < max_iterate:
        (p30, q3), _ = optimize.curve_fit(fun_Ct, txCt, tyCt,
                                          bounds=((0, -np.inf), (np.inf, np.inf)))
        (p2, q1), _ = optimize.curve_fit(fun_At, txAt, tyAt,
                                         bounds=((-np.inf, -np.inf), (np.inf, np.inf)))
        (p1, q2), _ = optimize.curve_fit(fun_Bt, txBt, tyBt,
                                         bounds=((0, -np.inf), (np.inf, np.inf)))
        tc22 = 1 / p30
        tc23 = p1 * tc22
        tc33 = p2 + tc23 ** 2 / tc22
        # print(tc22, tc33, tc23)
        # print(p2, q1, p1, q2, p30, q3)

        do_iterate = not np.allclose(np.hstack((c22, c33, c23)), np.hstack((tc22, tc33, tc23)),
                                     rtol=rtol, atol=atol)
        (c22, c33, c23) = (tc22, tc33, tc23)
        n_iterate = n_iterate + 1
    return c22, c33, c23, q1, q2, q3

def iterate_fit_c22c33c23_v3(ph, rt1, tpsi2, tpsi3, tpsi6, chmin=10, chmax=np.inf, rtol=1e-10,
                              atol=1e-100, max_iterate=100):
    def fun_At(tx, tp2, tq1):
        return 2 * ph * tx / (ph * np.pi * tp2 / S + np.log(tx)) + tq1

    def fun_Bt(tx, tp1, tq2):
        return 2 * ph * rt1 * tx * tp1 / (ph * np.pi * p2 / S + np.log(tx)) + tq2

    def fun_Ct(tx, tp30, tq3):
        # tp31 = _fun_p31(tx, ph, rt1, c22, c33, c23)
        tp31 = 1
        return 2 * S * rt1 ** 2 * tx / np.pi * tp30 * tp31 + tq3

    def psi_tx_ty(tpsi):
        tx = tpsi.index.values * np.pi
        ty = tpsi.values
        idxi = np.isfinite(tx) & np.isfinite(ty) & (chmin * np.pi <= tx) & (tx <= chmax * np.pi)
        fitx = tx[idxi]
        fity = ty[idxi]
        return fitx, fity

    S = np.sqrt(4 * np.pi ** 2 * rt1 ** 2 + ph ** 2)
    txAt, tyAt = psi_tx_ty(tpsi2)
    txCt, tyCt = psi_tx_ty(tpsi3)
    txBt, tyBt = psi_tx_ty(tpsi6)
    assert np.logical_or(np.all(tyBt < 0), np.all(tyBt > 0), )
    tyBt = np.abs(tyBt)

    (p2, q1), _ = optimize.curve_fit(fun_At, txAt, tyAt,
                                     bounds=((0, -np.inf), (np.inf, np.inf)))
    (p1, q2), _ = optimize.curve_fit(fun_Bt, txBt, tyBt,
                                     bounds=((0, -np.inf), (np.inf, np.inf)))
    (p30, q3), _ = optimize.curve_fit(fun_Ct, txCt, tyCt,
                                      bounds=((0, -np.inf), (np.inf, np.inf)))
    c22 = 1 / p30
    c23 = p1 * c22
    c33 = p2 + c23 ** 2 / c22
    return c22, c33, c23, q1, q2, q3


def fit_AtBtCt_c22c23c33(theta_fit, ph, rt1, c22, c33, c23):
    S = np.sqrt(4 * np.pi ** 2 * rt1 ** 2 + ph ** 2)
    tp1 = c23 / c22
    tp2 = c33 - c23 ** 2 / c22
    tp30 = 1 / c22
    tp31 = _fun_p31(theta_fit, ph, rt1, c22, c33, c23)
    fit_At = 2 * ph * theta_fit / (ph * np.pi / S * tp2 + np.log(theta_fit))
    fit_Bt = 2 * ph * theta_fit * rt1 / (ph * np.pi / S * tp2 + np.log(theta_fit)) * tp1
    fit_Ct = 2 * S * rt1 ** 2 * theta_fit / np.pi * tp30 * tp31
    return fit_At, fit_Bt, fit_Ct, (tp1, tp2, tp30, tp31)


def fit_AtBtCt_c22c23c33_v2(theta_fit, ph, rt1, c22, c33, c23, q1=0, q2=0, q3=0):
    S = np.sqrt(4 * np.pi ** 2 * rt1 ** 2 + ph ** 2)
    tp1 = c23 / c22
    tp2 = c33 - c23 ** 2 / c22
    tp30 = 1 / c22
    tp31 = _fun_p31(theta_fit, ph, rt1, c22, c33, c23)
    fit_At = 2 * ph * theta_fit / (ph * np.pi / S * tp2 + np.log(theta_fit)) + q1
    fit_Bt = 2 * ph * theta_fit * rt1 * tp1 / (ph * np.pi / S * tp2 + np.log(theta_fit)) + q2
    fit_Ct = 2 * S * rt1 ** 2 * theta_fit / np.pi * tp30 * tp31 + q3
    return fit_At, fit_Bt, fit_Ct, (tp1, tp2, tp30, tp31)

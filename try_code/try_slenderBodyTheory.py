# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 

@author: zhangji
"""

import os
import importlib
from time import time
import numpy as np
from src import slenderBodyTheory as slb

PWD = os.getcwd()
np.set_printoptions(linewidth=130, precision=5)

importlib.reload(slb)

ph = 1
rt1 = 1
rt2 = 0.01
# ch_list = [1, 2, 4, 8, 12, 16, 24, 32, 40]
ch_list = [6, ]
intsij_nth_ch = 2
gmres_maxiter = 30
intsij_limit = 1000
intsij_epsabs = 1e-10
intsij_epsrel = 1e-6
ifprint = True
hlx_node_fun_list = (slb.x1_fun, slb.x2_fun,)
FnMat_fun_list = (slb.Fn1Mat_fun, slb.Fn2Mat_fun,)
T_fun_list = (slb.T1_fun, slb.T2_fun,)

At_list = []
Bt_list = []
Ct_list = []
ftr_list = []
frt_list = []
for chi in ch_list:
    nthi = int(intsij_nth_ch * chi)
    print('-->ch=%5.2f, nth=%d' % (chi, nthi))
    # tAt, tBt, tCt, tftr, tfrt = \
    #     slb.Lighthill_AtBtCt_nhlx(ph, rt1, rt2, chi, nth=nthi, gmres_maxiter=gmres_maxiter,
    #                              hlx_node_fun_list=hlx_node_fun_list, FnMat_fun_list=FnMat_fun_list,
    #                              intsij_epsabs=intsij_epsabs, intsij_epsrel=intsij_epsrel,
    #                              intsij_limit=intsij_limit,
    #                              ifprint=ifprint)
    tAt, tBt, tCt, tftr, tfrt = \
        slb.KRJ_AtBtCt_nhlx(ph, rt1, rt2, chi, nth=nthi, gmres_maxiter=gmres_maxiter,
                            hlx_node_fun_list=hlx_node_fun_list, T_fun_list=T_fun_list,
                            intsij_epsabs=intsij_epsabs, intsij_epsrel=intsij_epsrel,
                            intsij_limit=intsij_limit, ifprint=ifprint)
    At_list.append(tAt)
    Bt_list.append(tBt)
    Ct_list.append(tCt)
    ftr_list.append(tftr)
    frt_list.append(tfrt)
    if not ifprint:
        print('ch=%5.2f, At=%f, Bt=%f, Ct=%f' % (chi, tAt, tBt, tCt))

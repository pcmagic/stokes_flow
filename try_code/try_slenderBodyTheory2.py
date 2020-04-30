import os
import importlib
from time import time
import numpy as np
import scipy as sp
import pandas as pd
import re
from scanf import scanf
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from scipy.optimize import leastsq, curve_fit
from IPython.display import display, HTML
from scipy import interpolate, integrate, optimize, sparse
from codeStore import support_fun as spf
from src import slenderBodyTheory as slb
from src import slenderBodyTheory as slbt

importlib.reload(slbt)
gmres_maxiter = 30000
intsij_limit = 1000
intsij_epsabs = 1e-10
intsij_epsrel = 1e-6
ifprint = True
use_tqdm_notebook = True
hlx_node_fun = slbt.x1_fun
FnMat_fun = slbt.Fn1Mat_fun
T_fun = slbt.T1_fun
hlx_node_fun_list = (slbt.x1_fun, slbt.x2_fun,)
FnMat_fun_list = (slbt.Fn1Mat_fun, slbt.Fn2Mat_fun,)
T_fun_list = (slbt.T1_fun, slbt.T2_fun,)

ph, ch, rt1, rt2 = 10, 1, 1, 0.01
n_segment = 200 * ch
# tAt, tBt, tCt, tftr, tfrt = slbt.KRJ_AtBtCt_1hlx(ph, rt1, rt2, ch, nth=n_segment,
#                                                       gmres_maxiter=gmres_maxiter,
#                                                       hlx_node_fun=hlx_node_fun,
#                                                       T_fun=T_fun,
#                                                       intsij_epsabs=intsij_epsabs,
#                                                       intsij_epsrel=intsij_epsrel,
#                                                       intsij_limit=intsij_limit,
#                                                       intsij_workers=1,
#                                                       ifprint=ifprint)
slb.KRJ_AtBtCt_nhlx(ph, rt1, rt2, ch, nth=n_segment, gmres_maxiter=gmres_maxiter,
                    hlx_node_fun_list=hlx_node_fun_list, T_fun_list=T_fun_list,
                    intsij_epsabs=intsij_epsabs, intsij_epsrel=intsij_epsrel,
                    intsij_limit=intsij_limit,
                    ifprint=ifprint, use_tqdm_notebook=use_tqdm_notebook)

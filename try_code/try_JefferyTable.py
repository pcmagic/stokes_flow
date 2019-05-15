import os
import glob
import natsort
import numpy as np
import scipy as sp
from scipy.optimize import leastsq, curve_fit
from scipy import interpolate, integrate
from scipy import spatial
# from scipy.interpolate import interp1d
from scipy.io import loadmat, savemat
# import scipy.misc
import importlib
from IPython.display import display, HTML

import matplotlib
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
from matplotlib import animation, rc
import matplotlib.ticker as mtick
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, zoomed_inset_axes
from mpl_toolkits.mplot3d import Axes3D, axes3d
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from time import time
from src.support_class import *
from src import jeffery_model as jm
from codeStore import support_fun as spf
from codeStore import support_fun_table as spf_tb

# t_theta, t_phi, ini_psi = np.array((0, 0, 0)) * np.pi
max_t = 10
update_fun = 'RK23'
rtol = 1e-3
atol = 1e-6
eval_dt = 0.1
save_every = np.ceil(1 / eval_dt) / 100
# norm = np.array((np.sin(t_theta) * np.cos(t_phi), np.sin(t_theta) * np.sin(t_phi), np.cos(t_theta)))

# Table_t, Table_X, Table_P, Table_P2, Table_theta, Table_phi, Table_psi, Table_eta \
#     = spf_tb.do_calculate_helix_RK(norm, ini_psi, max_t, update_fun=integrate.RK45, rtol=1e-2, atol=1e-3)
#
# Table_t, Table_X, Table_P, Table_P2, Table_theta, Table_phi, Table_psi, Table_eta \
#     = spf_tb.do_calculate_ecoli_RK(norm, ini_psi, max_t, update_fun=integrate.RK45, rtol=1e-2, atol=1e-3)

# Table_t, Table_X, Table_P, Table_P2, Table_theta, Table_phi, Table_psi, Table_eta \
#     = spf_tb.do_calculate_helix_Petsc(norm, ini_psi, max_t, update_fun='3bs', rtol=1e-2, atol=1e-3)

# Table_t, Table_X, Table_P, Table_P2, Table_theta, Table_phi, Table_psi, Table_eta \
#     = spf_tb.do_calculate_ecoli_Petsc(norm, ini_psi, max_t, update_fun=update_fun,
#                                       rtol=1e-2, atol=1e-3, eval_dt=eval_dt, save_every=save_every)

norm = np.array((0, 0, 1))
ini_psi = 0
update_fun = '1fe'
Table_t, Table_dt, Table_X, Table_P, Table_P2, Table_theta, Table_phi, Table_psi, Table_eta \
    = spf_tb.do_calculate_ellipse_Petsc4n(norm, ini_psi, max_t, update_fun=update_fun,
                                          rtol=1e-2, atol=1e-3)

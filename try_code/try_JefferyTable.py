# import numpy as np

# # !/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on 20181219

# @author: zhangji

# Trajection of a ellipse, Jeffery equation.
# """

# %pylab inline
# pylab.rcParams['figure.figsize'] = (25, 11)
# fontsize = 40

# import numpy as np
# import scipy as sp
# from scipy.optimize import leastsq, curve_fit
# from scipy import interpolate
# from scipy.interpolate import interp1d
# from scipy.io import loadmat, savemat
# # import scipy.misc

# import matplotlib
# from matplotlib import pyplot as plt
# from matplotlib import animation, rc
# import matplotlib.ticker as mtick
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes, zoomed_inset_axes
# from mpl_toolkits.mplot3d import Axes3D, axes3d

# from sympy import symbols, simplify, series, exp
# from sympy.matrices import Matrix
# from sympy.solvers import solve

# from IPython.display import display, HTML
# from tqdm import tqdm_notebook as tqdm
# import pandas as pd
# import re
# from scanf import scanf
# import os
# import glob

# from codeStore import support_fun as spf
# from src.support_class import *
# from src import stokes_flow as sf

# rc('animation', html='html5')
# PWD = os.getcwd()
# font = {'size': 20}
# matplotlib.rc('font', **font)
# np.set_printoptions(linewidth=90, precision=5)

import os
import glob
import natsort
import numpy as np
import scipy as sp
from scipy.optimize import leastsq, curve_fit
from scipy import interpolate
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

# %matplotlib notebook

rc('animation', html='html5')
fontsize = 40
PWD = os.getcwd()

# %matplotlib notebook

importlib.reload(jm)
eval_dt = 0.1
max_iter = 1001
update_order = 3
ellipse_speed = 0
planeShearRate = np.array((1, 0, 0))
talpha = 1 / 0.3
tnorm = np.array((1, 0, 0))
lateral_norm = np.array((0, 1, 0))
tcenter = np.array((0, 0, 0))

# Jeffery result
ellipse_kwargs = {'name':         'ellipse',
                  'center':       tcenter,
                  'norm':         tnorm / np.linalg.norm(tnorm),
                  'lateral_norm': lateral_norm / np.linalg.norm(lateral_norm),
                  'speed':        ellipse_speed,
                  'lbd':          (talpha ** 2 - 1) / (talpha ** 2 + 1)}
fileHandle = 'ShearJefferyProblem'
ellipse_obj = jm.JefferyObj(**ellipse_kwargs)
ellipse_obj.set_update_para(fix_x=False, fix_y=False, fix_z=False, update_order=update_order)
problem = jm.ShearJefferyProblem(name=fileHandle, planeShearRate=planeShearRate)
problem.add_obj(ellipse_obj)
t0 = time()
for idx in range(1, max_iter + 1):
    problem.update_location(eval_dt, print_handle='%d / %d' % (idx, max_iter))
t1 = time()
Jeffery_X = np.vstack(ellipse_obj.center_hist)
Jeffery_U = np.vstack(ellipse_obj.U_hist)
Jeffery_P = np.vstack(ellipse_obj.norm_hist)
Jeffery_t = np.arange(max_iter) * eval_dt + eval_dt
print('%s: run %d loops using %f' % (fileHandle, max_iter, (t1 - t0)))

# Table result
ellipse_kwargs = {'name':         'ellipse',
                  'center':       tcenter,
                  'norm':         tnorm / np.linalg.norm(tnorm),
                  'lateral_norm': lateral_norm / np.linalg.norm(lateral_norm),
                  'speed':        ellipse_speed,
                  'lbd':          (talpha ** 2 - 1) / (talpha ** 2 + 1)}
fileHandle = 'ShearTableProblem'
ellipse_obj = jm.TableObj(**ellipse_kwargs)
ellipse_obj.set_update_para(fix_x=False, fix_y=False, fix_z=False, update_order=update_order)
ellipse_obj.load_table('ellipse_alpha3')
problem = jm.ShearTableProblem(name=fileHandle, planeShearRate=planeShearRate)
problem.add_obj(ellipse_obj)
t0 = time()
for idx in range(1, max_iter + 1):
    problem.update_location(eval_dt, print_handle='%d / %d' % (idx, max_iter))
t1 = time()
Table_X = np.vstack(ellipse_obj.center_hist)
Table_U = np.vstack(ellipse_obj.U_hist)
Table_P = np.vstack(ellipse_obj.norm_hist)
Table_t = np.arange(max_iter) * eval_dt + eval_dt
print('%s: run %d loops using %f' % (fileHandle, max_iter, (t1 - t0)))

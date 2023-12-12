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
# from tqdm.notebook import tqdm as tqdm_notebook
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

from tqdm import tqdm_notebook
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
import pandas as pd
import pickle

import matplotlib
from matplotlib import pyplot as plt
import matplotlib.colors as colors
from matplotlib import animation, rc
import matplotlib.ticker as mtick
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, zoomed_inset_axes
from mpl_toolkits.mplot3d import Axes3D, axes3d
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib import cm

from time import time
from datetime import datetime
from src.support_class import *
from src import jeffery_model as jm
from codeStore import support_fun_bck as spf
from codeStore import support_fun_table as spf_tb
# %matplotlib notebook
from src.objComposite import *

rc('animation', html='html5')
fontsize = 40
PWD = os.getcwd()

pickle_name = 'ecoC01B01_tau1c' + '_kwargs'
problem_kwargs = spf_tb.load_problem_kwargs(pickle_name)
ecoli_comp = create_ecoli_2part(**problem_kwargs)
ecoli_comp.show_u_nodes()
ecoli_comp.get_obj_list()[0].show_u_nodes()
ecoli_comp.get_obj_list()[1].show_u_nodes()
# tnode1, tnode2 = get_ecoli_nodes_split_at(0, 0, 0,
#                                           now_center=np.zeros(3), **problem_kwargs)

# fig = plt.figure(figsize=(5, 5))
# fig.patch.set_facecolor('white')
# ax0 = fig.add_subplot(111, projection='3d')
# for spine in ax0.spines.values():
#     spine.set_visible(False)
# ax0.plot(tnode1[:, 0], tnode1[:, 1], tnode1[:, 2])
# ax0.plot(tnode2[:, 0], tnode2[:, 1], tnode2[:, 2])
# ax0.set_xlabel('X')
# ax0.set_ylabel('Y')
# ax0.set_zlabel('Z')
# plt.show()

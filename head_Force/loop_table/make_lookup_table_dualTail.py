#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
import re
import pandas as pd
from scanf import scanf
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
import pickle

import matplotlib
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
from matplotlib import animation, rc
import matplotlib.ticker as mtick
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, zoomed_inset_axes
from mpl_toolkits.mplot3d import Axes3D, axes3d
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from time import time
from src.support_class import *
from src import jeffery_model as jm
from codeStore import support_fun as spf

# %matplotlib notebook

rc('animation', html='html5')
fontsize = 40
figsize = (30, 16)
PWD = os.getcwd()


# In[59]:


def read_data_loopup_table(psi_dir_list, tcenter, ignore_first=0):
    ecoli_U_list = []
    ecoli_norm_list = []
    ecoli_center_list = []
    ecoli_nodes_list = []
    ecoli_idx_list = []
    norm_phi_list = []
    norm_psi1_list = []
    norm_psi2_list = []
    norm_theta_list = []
    i0 = -1
    t1 = []
    for psi_dir in psi_dir_list:
        print(psi_dir)
        file_handle = os.path.basename(psi_dir)
        mat_names = natsort.natsorted(glob.glob('%s/%s_th*' % (psi_dir, file_handle)))
        for mati in mat_names[ignore_first:]:
            i0 = i0 + 1
            mat_contents = loadmat(mati)
            ecoli_U = mat_contents['ecoli_U'].flatten()
            ecoli_norm = mat_contents['ecoli_norm'].flatten()
            ecoli_center = mat_contents['ecoli_center'].flatten()
            planeShearRate = mat_contents['planeShearRate'].flatten()
            rel_U_list = mat_contents['rel_U']
            norm_phi = mat_contents['norm_phi'].flatten()
            norm_psi1 = mat_contents['norm_psi1'].flatten()
            norm_psi2 = mat_contents['norm_psi2'].flatten()
            norm_theta = mat_contents['norm_theta'].flatten()
            ecoli_U_list.append(ecoli_U)
            ecoli_norm_list.append(ecoli_norm)
            ecoli_center_list.append(ecoli_center)
            ecoli_idx_list.append(i0)
            norm_phi_list.append(norm_phi)
            norm_psi1_list.append(norm_psi1)
            norm_psi2_list.append(norm_psi2)
            norm_theta_list.append(norm_theta)
    ecoli_U = np.vstack(ecoli_U_list)
    ecoli_norm = np.vstack(ecoli_norm_list)
    ecoli_center = np.vstack(ecoli_center_list)
    ecoli_idx = np.hstack(ecoli_idx_list)
    norm_phi = np.hstack(norm_phi_list)
    norm_psi1 = np.hstack(norm_psi1_list)
    norm_psi2 = np.hstack(norm_psi2_list)
    norm_theta = np.hstack(norm_theta_list)
    norm_tpp = np.vstack((norm_theta, norm_phi, norm_psi1, norm_psi2)).T    
    return ecoli_U, ecoli_norm, ecoli_center, ecoli_idx, norm_tpp, planeShearRate, rel_U_list

importlib.reload(spf)
sm = 'pf'
ecoli_name = 'ecoD01_all'
job_dir = 'dualTail_1c'
ksp_max_it = 300
main_fun_noIter = 1
planeShearRatex = 1
ch = 3
nth = 20
rh1 = 0.1
rh2 = 0.03
ph = 2/3
n_tail = 1
rel_tail1 = 193.66659814
rel_tail1 = 0
rel_tail2 = 0
write_pbs_head = spf.write_pbs_head_newturb
norm_psi1_list = np.linspace(0, 2 * np.pi, 10, endpoint=False)
norm_psi2_list = np.linspace(0, 2 * np.pi, 10, endpoint=False)
n_norm_theta = 24
n_norm_phi = 48
PWD = os.getcwd()

n_pbs = 0
t_name = os.path.join(job_dir, 'run2_all.sh')
with open(t_name, 'w') as frun0:
    for norm_psi1 in norm_psi1_list: 
        t_run_name = 'run2_psi1-%4.2f.sh' % norm_psi1
        frun0.write('bash %s \n' % t_run_name)
        t_name = os.path.join(job_dir, t_run_name)
        with open(t_name, 'w') as frun:
            # create .pbs file
            frun.write('t_dir=$PWD \n')
            for norm_psi2 in norm_psi2_list:
                job_name = '%s_psi1-%4.2f_psi2-%4.2f' % (ecoli_name, norm_psi1, norm_psi2)
                t_path = os.path.join(job_dir, job_name)
                t_name = os.path.join(t_path, '%s_2.pbs' % job_name)
                psi_dir_list = [t_path, ]
                ecoli_U, ecoli_norm, ecoli_center, ecoli_idx, norm_tpp, planeShearRate, rel_U_list                   = read_data_loopup_table(psi_dir_list, np.zeros(3), ignore_first=0)
                norm_theta = norm_tpp[:, 0]
                th_idx = np.argmin(np.linspace(0, np.pi, n_norm_theta) < norm_theta.max())
                print(th_idx, norm_theta.max(), np.linspace(0, np.pi, n_norm_theta)[th_idx])
    #             print(rel_U_list)
                with open(t_name, 'w') as fpbs:
                    write_pbs_head(fpbs, job_name)                           
                    fpbs.write('mpirun -n 24 python ')
                    fpbs.write(' ../../../loop_table_dualTail_ecoli.py ')
                    fpbs.write(' -f %s ' % job_name)
                    fpbs.write(' -pickProblem %d ' % 0)
                    fpbs.write(' -save_singleEcoli_vtk %d ' % 0)
                    fpbs.write(' -rh1 %f ' % rh1)
                    fpbs.write(' -rh2 %f ' % rh2)
                    fpbs.write(' -ch %f ' % ch)
                    fpbs.write(' -nth %d ' % nth)
                    fpbs.write(' -eh %f ' % -1)
                    fpbs.write(' -ph %f ' % ph)
                    fpbs.write(' -hfct %f ' % 1)
                    fpbs.write(' -n_tail %d ' % n_tail)
                    fpbs.write(' -with_cover %d ' % 2)
                    fpbs.write(' -left_hand %d ' % 0)
                    fpbs.write(' -rs1 %f ' % 1.5)
                    fpbs.write(' -rs2 %f ' % 0.5)
                    fpbs.write(' -ds %f ' % 0.07)
                    fpbs.write(' -es %f ' % -1)
                    fpbs.write(' -with_T_geo %d ' % 0)
                    fpbs.write(' -dist_hs %f ' % 0.5)
                    fpbs.write(' -ksp_max_it %d ' % ksp_max_it)
                    fpbs.write(' -plot_geo %d ' % 0)
                    fpbs.write(' -rel_wsz %f ' % 0)
                    fpbs.write(' -ffweight %f ' % 2)
                    fpbs.write(' -sm %s ' % sm)
                    fpbs.write(' -zoom_factor %f ' % 1)
                    fpbs.write(' -planeShearRatex %f ' % planeShearRatex)
                    fpbs.write(' -n_norm_theta %d ' % n_norm_theta)
                    fpbs.write(' -n_norm_phi %d ' % n_norm_phi)
                    fpbs.write(' -norm_psi1 %f ' % norm_psi1)
                    fpbs.write(' -norm_psi2 %f ' % norm_psi2)
                    fpbs.write(' -rel_tail1 %f ' % rel_tail1)
                    fpbs.write(' -rel_tail2 %f ' % rel_tail2)
                    fpbs.write(' -th_idx %d ' % th_idx)
                    fpbs.write(' -main_fun_noIter %d ' % main_fun_noIter)
                    fpbs.write(' > %s.txt \n\n' % job_name)
                # write to .sh file
                frun.write('cd $t_dir/%s\n' % job_name)
                frun.write('qsub %s_2.pbs\n\n' % job_name)
                n_pbs = n_pbs + 1
            frun.write('\n')
print('n_pbs = ', n_pbs)


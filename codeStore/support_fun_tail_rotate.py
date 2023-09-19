import os
import importlib
import pickle
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
from scipy import interpolate, integrate, optimize, sparse, signal
from codeStore import support_fun as spf
from src import slenderBodyTheory as slb
from src.geo import *
from src.objComposite import *
from tqdm.notebook import tqdm_notebook
from itertools import compress
from codeStore.support_fun_head_tail import *
import vtk
from vtk.util import numpy_support as VN

np.set_printoptions(linewidth=130, precision=5)


def total_force_part(f_geo, x_fgeo, tidx1):
    tidx1 = np.hstack(tidx1)
    tf = f_geo[tidx1]
    tr = x_fgeo[tidx1]
    F1 = np.sum(tf, axis=0)
    T1 = np.sum(np.cross(tr, tf), axis=0)
    #     print(np.hstack((F1, T1)))
    return np.hstack((F1, T1))


def AtBtCt_txt(filename):
    with open(filename, 'r') as myinput:
        FILE_DATA = myinput.read()

    text_headle = ' geometry zoom factor is'
    temp1 = spf.read_array(text_headle, FILE_DATA, array_length=1)
    t_zf = temp1

    text_headle = 'tran tail resultant is \['
    temp1 = spf.read_array(text_headle, FILE_DATA, array_length=6)
    psi2 = temp1[2] / t_zf ** 1
    psi61 = temp1[5]
    text_headle = 'rota tail resultant is \['
    temp1 = spf.read_array(text_headle, FILE_DATA, array_length=6)
    psi62 = temp1[2]
    psi3 = temp1[5] / t_zf ** 3
    # psi6.append((psi61 + psi62) / 2 / t_zf ** 2)
    psi6 = psi62 / t_zf ** 2

    text_headle = ', velocity nodes:'
    temp1 = spf.read_array(text_headle, FILE_DATA, array_length=1)
    n_nodes = temp1
    return psi2, -1 * psi6, psi3, n_nodes


def load_case_data(fileHandle, foldername):
    fVTUname = '%s_Prb_force_t00000.vtu' % fileHandle
    uVTUname = '%s_Prb_velocity_t00000.vtu' % fileHandle
    picklename = '%s_pick.bin' % fileHandle
    txtname = '%s.txt' % fileHandle[:-5] if fileHandle[-5:] in ('_rota', '_tran') \
        else '%s.txt' % fileHandle

    tname = os.path.join(foldername, txtname)
    tAt, tBt, tCt, tn_nodes = AtBtCt_txt(tname)

    tname = os.path.join(foldername, fVTUname)
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(tname)
    reader.Update()
    data = reader.GetOutput()  # vtkUnstructuredGrid
    f_geo = VN.vtk_to_numpy(data.GetPointData().GetArray('force'))
    x_fgeo = np.array([data.GetPoint(i) for i in range(data.GetNumberOfPoints())])
    x_fgeo[:, 2] = x_fgeo[:, 2] - x_fgeo[:, 2].mean()

    tname = os.path.join(foldername, uVTUname)
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(tname)
    reader.Update()
    data = reader.GetOutput()  # vtkUnstructuredGrid
    u_geo = VN.vtk_to_numpy(data.GetPointData().GetArray('velocity'))
    x_ugeo = np.array([data.GetPoint(i) for i in range(data.GetNumberOfPoints())])
    x_ugeo[:, 2] = x_ugeo[:, 2] - x_ugeo[:, 2].mean()

    tname = os.path.join(foldername, picklename)
    with open(tname, 'rb') as myinput:
        unpick = pickle.Unpickler(myinput)
        problem = unpick.load()
    problem_kwargs = problem.get_kwargs()
    return tAt, tBt, tCt, tn_nodes, f_geo, x_fgeo, u_geo, x_ugeo, problem_kwargs


def generate_geo(problem_kwargs, x_ugeo, plot_geo=True, rot_theta=None):
    center = problem_kwargs['center']
    ph = problem_kwargs['ph']
    ch = problem_kwargs['ch']
    rt1 = problem_kwargs['rh11']
    rt2 = problem_kwargs['rh2']

    # def fun_theta(theta, tgeo, x_ugeo):
    #     tgeo1 = tgeo.copy()
    #     tgeo1.node_rotation(norm=np.array((0, 0, 1)), theta=theta, rotation_origin=center)

    #     tnorm = np.linalg.norm(tgeo1.get_nodes() - x_ugeo)
    #     # print(theta, tnorm)
    #     return tnorm

    def fun_theta(theta, tgeo0, x_ugeo):
        tgeo1 = tgeo0.copy()
        tgeo1.node_rotation(norm=np.array((0, 0, 1)), theta=theta, rotation_origin=center)

        mid_idx = len(tgeo1.body_idx_list) // 2
        tT = tgeo1.frenetFrame[0][mid_idx]
        tN = tgeo1.frenetFrame[1][mid_idx]
        tB = tgeo1.frenetFrame[2][mid_idx]
        tfnodes = x_ugeo[tgeo1.body_idx_list[mid_idx]]
        tnode_line = tfnodes.mean(axis=0)
        tfnodes_local = np.dot((tfnodes - tnode_line), np.vstack((tN, tB, tT)).T)
        return tfnodes_local[:, 2].max()

    tail_obj_list = create_ecoli_tail(moveh=np.zeros(3), **problem_kwargs)
    if rot_theta is None:
        tge0 = tail_obj_list[0].get_u_geo()
        assert tge0.get_n_nodes() == x_ugeo.shape[0]
        theta = optimize.minimize(fun_theta, np.zeros(1),
                                  args=(tge0, x_ugeo)).x
        print('optimize minimize theta: %.15f' % theta)
    else:
        theta = rot_theta
    for ti in tail_obj_list:
        ti.node_rotation(norm=np.array((0, 0, 1)), theta=theta, rotation_origin=center)
    uobj0 = tail_obj_list[0]
    # use_idx0 = 0
    # uobj0 = tail_obj_list[1]
    # use_idx0 = tail_obj_list[0].get_u_geo().get_n_nodes()
    ugeo0 = uobj0.get_u_geo()
    ds = np.mean(np.linalg.norm(ugeo0.axisNodes[:-1] - ugeo0.axisNodes[1:], axis=-1))

    if plot_geo:
        # check, make sure the generated geos are correct.
        ugeo0_nodes = ugeo0.get_nodes()
        ugeo0_axisNodes = uobj0.get_f_geo().axisNodes
        fig = plt.figure(figsize=(8, 8), dpi=200)
        fig.patch.set_facecolor('white')
        ax0 = fig.add_subplot(1, 1, 1, projection='3d')
        ax0.plot(*x_ugeo.T)
        ax0.plot(*ugeo0_nodes.T)
        ax0.plot(*ugeo0_axisNodes.T)
        ax0.set_title('$\\lambda=%.2f, n_1=%.2f, r_{t1}=%.2f, r_{t2}=%.2f$' % (ph, ch, rt1, rt2))
        spf.set_axes_equal(ax0)
    return uobj0, ds


def get_slice_ft(ugeo0, fgeo0, f_geo, x_fgeo, problem_kwargs, tfct=0.05):
    center = problem_kwargs['center']
    ph = problem_kwargs['ph']
    ch = problem_kwargs['ch']
    rt1 = problem_kwargs['rh11']
    rt2 = problem_kwargs['rh2']

    slice_ft = []
    tx = []
    # cover start
    tidx = fgeo0.cover_strat_idx
    t_ft = total_force_part(f_geo, x_fgeo, tidx)
    slice_ft.append(t_ft)
    tx.append(ugeo0.axisNodes[0, 2] - ch * tfct)
    # body slice
    for tidx in fgeo0.body_idx_list:
        t_ft = total_force_part(f_geo, x_fgeo, tidx)
        slice_ft.append(t_ft)
    tx.append(ugeo0.axisNodes[:, 2])
    # cover end
    tidx = fgeo0.cover_end_idx
    t_ft = total_force_part(f_geo, x_fgeo, tidx)
    slice_ft.append(t_ft)
    tx.append(ugeo0.axisNodes[-1, 2] + ch * tfct)
    slice_ft = np.vstack(slice_ft)
    tx = np.hstack(tx)
    return tx, slice_ft

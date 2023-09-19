# coding=utf-8
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from src import stokes_flow as sf
from src import geo
from tqdm import tqdm
from scipy import integrate
from src.support_class import *
from src import forceSphere2d as fs2


# from scipy import interpolate, optimize, sparse
# from time import time
# from multiprocessing import cpu_count, Pool
# from numba import jit
# import numexpr as ne
# from numpy import linalg as LA


def delta(i, j):  # delta symbol
    return int(i == j)


def eijk(i, j, k):  # Levi-Civita symbol
    temp = np.zeros((3, 3, 3))
    temp[0, 1, 2] = temp[1, 2, 0] = temp[2, 0, 1] = 1
    temp[0, 2, 1] = temp[2, 1, 0] = temp[1, 0, 2] = -1
    return temp[i, j, k]


def regularized_stokeslets_matrix_3d(vnodes: np.ndarray,  # nodes contain velocity information
                                     fnodes: np.ndarray,  # nodes contain force information
                                     delta: float):  # correction factor
    # Solve m matrix using regularized Stokeslets method
    # U = M * F.
    # Cortez R. The method of regularized Stokeslets[J]. SIAM Journal on Scientific Computing, 2001, 23(4): 1204-1225.
    
    comm = PETSc.COMM_WORLD.tompi4py()
    size = comm.Get_size()
    rank = comm.Get_rank()
    if rank == 0:
        n_vnode = vnodes.shape
        if n_vnode[0] < n_vnode[1]:
            vnodes = vnodes.transpose()
            n_vnode = vnodes.shape
        n_fnode = fnodes.shape
        if n_fnode[0] < n_fnode[1]:
            fnodes = fnodes.transpose()
            n_fnode = fnodes.shape
        
        split = np.array_split(vnodes, size)
        split_size_in = [len(split[i]) * 3 for i in range(len(split))]
        split_disp_in = np.insert(np.cumsum(split_size_in), 0, [0])[0:-1]
        split_size_out = [len(split[i]) * 3 * n_fnode[0] * 3 for i in range(len(split))]
        split_disp_out = np.insert(np.cumsum(split_size_out), 0, [0])[0:-1]
    else:
        fnodes = None
        vnodes = None
        split = None
        split_disp_in = None
        split_size_in = None
        split_disp_out = None
        split_size_out = None
        n_fnode = None
        n_vnode = None
    
    split_size_in = comm.bcast(split_size_in, root=0)
    split_disp_in = comm.bcast(split_disp_in, root=0)
    split_size_out = comm.bcast(split_size_out, root=0)
    split_disp_out = comm.bcast(split_disp_out, root=0)
    vnodes_local = np.zeros(split_size_in[rank], dtype='float64')
    comm.Scatterv([vnodes, split_size_in, split_disp_in, MPI.DOUBLE], vnodes_local, root=0)
    vnodes_local = vnodes_local.reshape((3, -1)).T
    n_vnode_local = len(vnodes_local)
    n_fnode_local = comm.bcast(n_fnode, root=0)
    if rank == 0:
        fnodes_local = fnodes
    else:
        fnodes_local = np.zeros(n_fnode_local, dtype='float64')
    comm.Bcast(fnodes_local, root=0)
    
    m_local = np.zeros((n_vnode_local * 3, n_fnode_local[0] * 3))
    for i0 in range(n_vnode_local):
        delta_xi = fnodes_local - vnodes_local[i0]
        temp1_local = delta_xi ** 2
        delta_2 = np.square(delta)  # delta_2 = e^2
        delta_r2 = temp1_local.sum(axis=1) + delta_2  # delta_r2 = r^2+e^2
        delta_r3 = delta_r2 * np.sqrt(delta_r2)  # delta_r3 = (r^2+e^2)^1.5
        temp2 = (delta_r2 + delta_2) / delta_r3  # temp2 = (r^2+2*e^2)/(r^2+e^2)^1.5
        m_local[3 * i0, 0::3] = temp2 + np.square(delta_xi[:, 0]) / delta_r3  # Mxx
        m_local[3 * i0 + 1, 1::3] = temp2 + np.square(delta_xi[:, 1]) / delta_r3  # Myy
        m_local[3 * i0 + 2, 2::3] = temp2 + np.square(delta_xi[:, 2]) / delta_r3  # Mzz
        m_local[3 * i0 + 1, 0::3] = m_local[3 * i0, 1::3] = delta_xi[:, 0] * delta_xi[:,
                                                                             1] / delta_r3  # Mxy
        m_local[3 * i0 + 2, 0::3] = m_local[3 * i0, 2::3] = delta_xi[:, 0] * delta_xi[:,
                                                                             2] / delta_r3  # Mxz
        m_local[3 * i0 + 2, 1::3] = m_local[3 * i0 + 1, 2::3] = delta_xi[:, 1] * delta_xi[:,
                                                                                 2] / delta_r3  # Myz
    
    if rank == 0:
        m = np.zeros((n_vnode[0] * 3, n_fnode[0] * 3))
    else:
        m = None
    comm.Gatherv(m_local, [m, split_size_out, split_disp_out, MPI.DOUBLE], root=0)
    return m  # ' regularized Stokeslets matrix, U = M * F '


def light_stokeslets_matrix_3d(u_nodes: np.ndarray, f_nodes: np.ndarray) -> np.ndarray:
    # from src.geo import geo
    # temp_geo1 = geo() # velocity nodes
    # temp_geo1.set_nodes(u_nodes, deltalength=0)
    # temp_obj1 = sf.StokesFlowObj()
    # temp_obj1.set_data(temp_geo1, temp_geo1, np.zeros(u_nodes.size))
    # temp_geo2 = geo() # force nodes
    # temp_geo2.set_nodes(f_nodes, deltalength=0)
    # temp_obj2 = sf.StokesFlowObj()
    # temp_obj2.set_data(temp_geo2, temp_geo2, np.zeros(3))
    # m = stokeslets_matrix_3d(temp_obj1, temp_obj2)
    
    m = np.ones((u_nodes.size, f_nodes.size))
    for i0, u_node in enumerate(u_nodes):
        dxi = (u_node - f_nodes).T
        dx0 = dxi[0]
        dx1 = dxi[1]
        dx2 = dxi[2]
        dr2 = np.sum(dxi ** 2, axis=0)
        dr1 = np.sqrt(dr2)
        dr3 = dr1 * dr2
        temp1 = 1 / (dr1 * (8 * np.pi))  # 1/r^1
        temp2 = 1 / (dr3 * (8 * np.pi))  # 1/r^3
        i1 = i0 * 3
        m[i1 + 0, 0::3] = temp2 * dx0 * dx0 + temp1
        m[i1 + 0, 1::3] = temp2 * dx0 * dx1
        m[i1 + 0, 2::3] = temp2 * dx0 * dx2
        m[i1 + 1, 0::3] = temp2 * dx1 * dx0
        m[i1 + 1, 1::3] = temp2 * dx1 * dx1 + temp1
        m[i1 + 1, 2::3] = temp2 * dx1 * dx2
        m[i1 + 2, 0::3] = temp2 * dx2 * dx0
        m[i1 + 2, 1::3] = temp2 * dx2 * dx1
        m[i1 + 2, 2::3] = temp2 * dx2 * dx2 + temp1
    return m


def light_rotlets_matrix_3d(u_nodes: np.ndarray, f_nodes: np.ndarray) -> np.ndarray:
    # Hydromechanics of low-Reynolds-number flow Part 1. Rotation of axisymmetric prolate bodies
    m = np.zeros((u_nodes.size, f_nodes.size))
    for i0, u_node in enumerate(u_nodes):
        dxi = (u_node - f_nodes).T
        dx0 = dxi[0]
        dx1 = dxi[1]
        dx2 = dxi[2]
        dr2 = np.sum(dxi ** 2, axis=0)
        dr1 = np.sqrt(dr2)
        dr3 = dr1 * dr2
        temp2 = 1 / (dr3 * (8 * np.pi))  # 1/r^3
        i1 = i0 * 3
        m[i1 + 0, 1::3] = temp2 * dx2
        m[i1 + 0, 2::3] = temp2 * -dx1
        m[i1 + 1, 0::3] = temp2 * -dx2
        m[i1 + 1, 2::3] = temp2 * dx0
        m[i1 + 2, 0::3] = temp2 * dx1
        m[i1 + 2, 1::3] = temp2 * -dx0
    return m


def stokeslets_matrix_3d_mij(u_node, f_nodes, i1):
    t_m = np.ones((3, f_nodes.size))
    mypi = np.pi
    dxi = (u_node - f_nodes).T
    dx0 = dxi[0]
    dx1 = dxi[1]
    dx2 = dxi[2]
    dr2 = np.sum(dxi ** 2, axis=0)
    dr1 = np.sqrt(dr2)
    dr3 = dr1 * dr2
    temp1 = 1 / (dr1 * (8 * mypi))  # 1/r^1
    temp2 = 1 / (dr3 * (8 * mypi))  # 1/r^3
    t_m[0, 0::3] = temp2 * dx0 * dx0 + temp1
    t_m[0, 1::3] = temp2 * dx0 * dx1
    t_m[0, 2::3] = temp2 * dx0 * dx2
    t_m[1, 0::3] = temp2 * dx1 * dx0
    t_m[1, 1::3] = temp2 * dx1 * dx1 + temp1
    t_m[1, 2::3] = temp2 * dx1 * dx2
    t_m[2, 0::3] = temp2 * dx2 * dx0
    t_m[2, 1::3] = temp2 * dx2 * dx1
    t_m[2, 2::3] = temp2 * dx2 * dx2 + temp1
    return t_m, i1


# def stokeslets_matrix_3d_mij2(myinput):
#   u_node, f_nodes, i1 = myinput
#   t_m = np.ones((3, f_nodes.size))
#   mypi = np.pi
#   dxi = (u_node - f_nodes).T
#   dx0 = dxi[0]
#   dx1 = dxi[1]
#   dx2 = dxi[2]
#   dr2 = np.sum(dxi ** 2, axis=0)
#   dr1 = np.sqrt(dr2)
#   dr3 = dr1 * dr2
#   temp1 = 1 / (dr1 * (8 * mypi)) # 1/r^1
#   temp2 = 1 / (dr3 * (8 * mypi)) # 1/r^3
#   t_m[0, 0::3] = temp2 * dx0 * dx0 + temp1
#   t_m[0, 1::3] = temp2 * dx0 * dx1
#   t_m[0, 2::3] = temp2 * dx0 * dx2
#   t_m[1, 0::3] = temp2 * dx1 * dx0
#   t_m[1, 1::3] = temp2 * dx1 * dx1 + temp1
#   t_m[1, 2::3] = temp2 * dx1 * dx2
#   t_m[2, 0::3] = temp2 * dx2 * dx0
#   t_m[2, 1::3] = temp2 * dx2 * dx1
#   t_m[2, 2::3] = temp2 * dx2 * dx2 + temp1
#   return t_m, i1


def stokeslets_matrix_3d_set(m, t_m, i1):
    print(i1)
    i2 = i1 * 3
    m[i2:i2 + 3, :] = t_m[:]
    return True


def stokeslets_matrix_3d(obj1: 'sf.StokesFlowObj',  # object contain velocity information
                         obj2: 'sf.StokesFlowObj'):  # object contain force information
    # Solve m matrix: (delta(i,i)/r + (x_i*x_j)/r^3
    
    u_nodes = obj1.get_u_nodes()
    n_unode = obj1.get_n_u_node()
    f_nodes = obj2.get_f_nodes()
    n_fnode = obj2.get_n_f_node()
    n_unknown = obj2.get_n_unknown()
    assert n_unknown == 3
    m = np.ones((n_unode * n_unknown, n_fnode * n_unknown))
    
    # def stokeslets_matrix_3d_mij(input, output):
    #   for u_node, f_nodes, i1, m in iter(input.get, 'STOP'):
    #     t_m = np.ones((3, f_nodes.size))
    #     mypi = np.pi
    #     dxi = (u_node - f_nodes).T
    #     dx0 = dxi[0]
    #     dx1 = dxi[1]
    #     dx2 = dxi[2]
    #     dr2 = np.sum(dxi ** 2, axis=0)
    #     dr1 = np.sqrt(dr2)
    #     dr3 = dr1 * dr2
    #     temp1 = 1 / (dr1 * (8 * mypi)) # 1/r^1
    #     temp2 = 1 / (dr3 * (8 * mypi)) # 1/r^3
    #     t_m[0, 0::3] = temp2 * dx0 * dx0 + temp1
    #     t_m[0, 1::3] = temp2 * dx0 * dx1
    #     t_m[0, 2::3] = temp2 * dx0 * dx2
    #     t_m[1, 0::3] = temp2 * dx1 * dx0
    #     t_m[1, 1::3] = temp2 * dx1 * dx1 + temp1
    #     t_m[1, 2::3] = temp2 * dx1 * dx2
    #     t_m[2, 0::3] = temp2 * dx2 * dx0
    #     t_m[2, 1::3] = temp2 * dx2 * dx1
    #     t_m[2, 2::3] = temp2 * dx2 * dx2 + temp1
    #     output.put((t_m, i1))
    #   output.put('STOP')
    #   return True
    # def stokeslets_matrix_3d_set(input, m):
    #   for t_m, i1 in iter(input.get, 'STOP'):
    #     i2 = i1 * 3
    #     m[i2:i2 + 3, :] = t_m[:]
    #   return True
    #
    # from multiprocessing import Process, Queue, current_process, cpu_count
    # NUMBER_OF_PROCESSES = cpu_count() - 1
    # mij_queue = Queue()
    # set_queue = Queue()
    # # set_p = Process(target=stokeslets_matrix_3d_set, args=((set_queue, m)))
    # # set_p.start()
    # for i0 in range(n_unode):
    #   mij_queue.put((u_nodes[i0], f_nodes, i0, m))
    # for i in range(NUMBER_OF_PROCESSES):
    #   Process(target=stokeslets_matrix_3d_mij, args=(mij_queue, set_queue)).start()
    # for i0 in range(n_unode):
    #   t_m, i1 = set_queue.get()
    #   i2 = i1 * 3
    #   m[i2:i2 + 3, :] = t_m[:]
    # for i in range(NUMBER_OF_PROCESSES):
    #   mij_queue.put('STOP')
    # # set_p.join()
    
    # mij_args = [(u_nodes[i0], f_nodes, i0) for i0 in range(n_unode)]
    # with Pool(1) as pool:
    #   mij_list = pool.imap_unordered(stokeslets_matrix_3d_mij2, mij_args)
    #   for t_m, i1 in mij_list:
    #     i2 = i1 * 3
    #     m[i2:i2 + 3, :] = t_m[:]
    
    # pass
    # mij_args = [(u_nodes[i0], f_nodes, i0) for i0 in range(n_unode)]
    # with Pool() as pool:
    #   for t_m, i1 in pool.starmap(stokeslets_matrix_3d_mij, mij_args):
    #     i2 = i1 * 3
    #     m[i2:i2+3, :] = t_m[:]
    
    # for i0, u_node in enumerate(u_nodes):
    #   t_m, i1 = stokeslets_matrix_3d_mij(u_node, f_nodes, i0)
    #   i2 = i1 * 3
    #   m[i2:i2+3, :] = t_m[:]
    mypi = np.pi
    for i0, u_node in enumerate(u_nodes):
        dxi = (u_node - f_nodes).T
        dx0 = dxi[0]
        dx1 = dxi[1]
        dx2 = dxi[2]
        dr2 = np.sum(dxi ** 2, axis=0)
        dr1 = np.sqrt(dr2)
        dr3 = dr1 * dr2
        temp1 = 1 / (dr1 * (8 * mypi))  # 1/r^1
        temp2 = 1 / (dr3 * (8 * mypi))  # 1/r^3
        i1 = i0 * 3
        m[i1 + 0, 0::3] = temp2 * dx0 * dx0 + temp1
        m[i1 + 0, 1::3] = temp2 * dx0 * dx1
        m[i1 + 0, 2::3] = temp2 * dx0 * dx2
        m[i1 + 1, 0::3] = temp2 * dx1 * dx0
        m[i1 + 1, 1::3] = temp2 * dx1 * dx1 + temp1
        m[i1 + 1, 2::3] = temp2 * dx1 * dx2
        m[i1 + 2, 0::3] = temp2 * dx2 * dx0
        m[i1 + 2, 1::3] = temp2 * dx2 * dx1
        m[i1 + 2, 2::3] = temp2 * dx2 * dx2 + temp1
    return m  # Stokeslets matrix, U = M * F


def light_stokeslets_matrix_3d_petsc(u_nodes: np.ndarray, f_nodes: np.ndarray):
    from src.geo import base_geo
    
    temp_geo1 = base_geo()  # velocity nodes
    temp_geo1.set_nodes(u_nodes, deltalength=0)
    temp_geo2 = base_geo()  # force nodes
    temp_geo2.set_nodes(f_nodes, deltalength=0)
    temp_obj1 = sf.StokesFlowObj()
    temp_obj1.set_data(temp_geo2, temp_geo1)
    m = stokeslets_matrix_3d_petsc(temp_obj1, temp_obj1)
    return m


def stokeslets_matrix_3d_petsc(obj1: 'sf.StokesFlowObj',  # object contain velocity information
                               obj2: 'sf.StokesFlowObj'):  # object contain force information
    # Solve m matrix: (delta(i,i)/r + (x_i*x_j)/r^3
    
    vnodes = obj1.get_u_nodes()
    n_vnode = obj1.get_n_velocity()
    fnodes = obj2.get_f_nodes()
    n_fnode = obj2.get_n_force()
    
    n_unknown = obj2.get_n_unknown()
    m = PETSc.Mat().create(comm=PETSc.COMM_WORLD)
    m.setSizes(((None, n_vnode), (None, n_fnode)))
    m.setType('dense')
    m.setFromOptions()
    m.setUp()
    m_start, m_end = m.getOwnershipRange()
    for i0 in range(m_start, m_end):
        delta_xi = fnodes - vnodes[i0 // 3]
        temp1 = delta_xi ** 2
        delta_r2 = temp1.sum(axis=1)  # delta_r2 = r^2
        delta_r3 = delta_r2 * np.sqrt(delta_r2)  # delta_r3 = r^3
        temp2 = 1 / np.sqrt(delta_r2)
        temp3 = 1 / delta_r3
        if i0 % 3 == 0:  # x axis
            m[i0, 0::n_unknown] = (temp2 + np.square(delta_xi[:, 0]) * temp3) / (8 * np.pi)  # Mxx
            m[i0, 1::n_unknown] = delta_xi[:, 0] * delta_xi[:, 1] * temp3 / (8 * np.pi)  # Mxy
            m[i0, 2::n_unknown] = delta_xi[:, 0] * delta_xi[:, 2] * temp3 / (8 * np.pi)  # Mxz
        elif i0 % 3 == 1:  # y axis
            m[i0, 0::n_unknown] = delta_xi[:, 0] * delta_xi[:, 1] * temp3 / (8 * np.pi)  # Mxy
            m[i0, 1::n_unknown] = (temp2 + np.square(delta_xi[:, 1]) * temp3) / (8 * np.pi)  # Myy
            m[i0, 2::n_unknown] = delta_xi[:, 1] * delta_xi[:, 2] * temp3 / (8 * np.pi)  # Myz
        else:  # z axis
            m[i0, 0::n_unknown] = delta_xi[:, 0] * delta_xi[:, 2] * temp3 / (8 * np.pi)  # Mxz
            m[i0, 1::n_unknown] = delta_xi[:, 1] * delta_xi[:, 2] * temp3 / (8 * np.pi)  # Myz
            m[i0, 2::n_unknown] = (temp2 + np.square(delta_xi[:, 2]) * temp3) / (8 * np.pi)  # Mzz
    m.assemble()
    
    return m  # Stokeslets matrix, U = M * F


def regularized_stokeslets_matrix_3d_petsc_mij(t_u_node: np.ndarray,  # velocity node
                                               f_nodes: np.ndarray,  # force nodes
                                               delta_2,  # delta_2 = e^2
                                               i0, **kwargs):
    mypi = np.pi
    dxi = (t_u_node - f_nodes).T
    dx0 = dxi[0]
    dx1 = dxi[1]
    dx2 = dxi[2]
    dr2 = np.sum(dxi ** 2, axis=0) + delta_2  # dr2 = r^2+e^2
    dr1 = np.sqrt(dr2)  # dr1 = (r^2+e^2)^0.5
    dr3 = dr1 * dr2  # dr3 = (r^2+e^2)^1.5
    temp1 = (dr2 + delta_2) / (dr3 * (8 * mypi))  # (r^2+2*e^2)/(r^2+e^2)^1.5
    temp2 = 1 / (dr3 * (8 * mypi))  # 1/(r^2+e^2)^1.5
    m00 = temp2 * dx0 * dx0 + temp1
    m01 = temp2 * dx0 * dx1
    m02 = temp2 * dx0 * dx2
    m10 = temp2 * dx1 * dx0
    m11 = temp2 * dx1 * dx1 + temp1
    m12 = temp2 * dx1 * dx2
    m20 = temp2 * dx2 * dx0
    m21 = temp2 * dx2 * dx1
    m22 = temp2 * dx2 * dx2 + temp1
    return m00, m01, m02, m10, m11, m12, m20, m21, m22, i0


def regularized_stokeslets_matrix_3d_petsc(obj1: 'sf.StokesFlowObj',  # object contain velocity information
                                           obj2: 'sf.StokesFlowObj',  # object contain force information
                                           m, **kwargs):
    # Solve m matrix using regularized Stokeslets method
    # U = M * F.
    # Cortez R. The method of regularized Stokeslets [J]. SIAM Journal on Scientific Computing, 2001, 23(4): 1204-1225.
    epsilon = kwargs['epsilon']
    delta = epsilon * obj2.get_f_geo().get_deltaLength()
    delta_2 = delta ** 2  # correction factor
    u_nodes = obj1.get_u_nodes()
    f_nodes = obj2.get_f_nodes()
    _, u_glbIdx_all = obj1.get_u_geo().get_glbIdx()
    _, f_glbIdx_all = obj2.get_f_geo().get_glbIdx()
    u_dmda = obj1.get_u_geo().get_dmda()
    
    for i0 in range(u_dmda.getRanges()[0][0], u_dmda.getRanges()[0][1]):
        m00, m01, m02, m10, m11, m12, m20, m21, m22, i1 = \
            regularized_stokeslets_matrix_3d_petsc_mij(u_nodes[i0], f_nodes, delta_2, i0)
        u_glb = u_glbIdx_all[i1 * 3]
        m.setValues(u_glb + 0, f_glbIdx_all[0::3], m00, addv=False)
        m.setValues(u_glb + 0, f_glbIdx_all[1::3], m01, addv=False)
        m.setValues(u_glb + 0, f_glbIdx_all[2::3], m02, addv=False)
        m.setValues(u_glb + 1, f_glbIdx_all[0::3], m10, addv=False)
        m.setValues(u_glb + 1, f_glbIdx_all[1::3], m11, addv=False)
        m.setValues(u_glb + 1, f_glbIdx_all[2::3], m12, addv=False)
        m.setValues(u_glb + 2, f_glbIdx_all[0::3], m20, addv=False)
        m.setValues(u_glb + 2, f_glbIdx_all[1::3], m21, addv=False)
        m.setValues(u_glb + 2, f_glbIdx_all[2::3], m22, addv=False)
        # if i0 % 1000==0:
        #   m.assemble()
    m.assemble()
    
    return m  # ' regularized Stokeslets matrix, U = M * F '


def check_regularized_stokeslets_matrix_3d(**kwargs):
    err_msg = 'the reguralized Stokeslets method needs parameter, epsilon. '
    assert 'epsilon' in kwargs, err_msg


def regularized_stokeslets_plane_matrix_3d_petsc_mij(t_u_node: np.ndarray,  # velocity node
                                                     f_nodes: np.ndarray,  # force nodes
                                                     e2,  # e2 = e^2
                                                     i0, **kwargs):
    # ZhengPeiyan version
    delta_xi = (f_nodes - t_u_node).T
    zwall = f_nodes[:, 2]
    dx = delta_xi[0]
    dy = delta_xi[1]
    dzs = delta_xi[2]
    dz = dzs - 2 * zwall
    r2sk = dx ** 2 + dzs ** 2 + dy ** 2  # r squared
    # rsk = r2sk ** 0.5
    rske = (r2sk + e2) ** 0.5
    H2sk = 1 / (rske ** 3)
    H1sk = 1 / rske + e2 * H2sk
    # D2sk = -6 / (rske ** 5)
    # D1sk = 2 / (rske ** 3) + e2 * D2sk
    r2k = dx ** 2 + dz ** 2 + dy ** 2  # r squared
    rk = r2k ** 0.5
    rke = (r2k + e2) ** 0.5
    H2k = 1 / (rke ** 3)
    H1k = 1 / rke + e2 * H2k
    D2k = -6 / (rke ** 5)
    D1k = 2 / (rke ** 3) + e2 * D2k
    dH2k = -3 * rk / (rke ** 5)
    dH1k = -rk / (rke ** 3) + e2 * dH2k
    m00 = H1sk + dx ** 2 * H2sk - H1k - dx ** 2 * H2k - zwall * zwall * (
            -1 * D1k - dx ** 2 * D2k) - 2 * zwall * (
                  dH1k / rk + H2k) * zwall - 2 * zwall * (
                  -dz * H2k - dz * dx ** 2 * dH2k / rk)  # Mxx
    m20 = dx * dzs * H2sk - dx * dz * H2k - zwall * zwall * (-dz * dx * D2k) - 2 * zwall * (
            dH1k / rk + H2k) * dx - 2 * zwall * (-dx * dH1k / rk - dx * dz ** 2 * dH2k / rk)  # Mzx
    m02 = dx * dzs * H2sk - dx * dz * H2k - zwall * zwall * (dx * dz * D2k) - 2 * zwall * (
            dx * H2k + dz ** 2 * dx * dH2k / rk)  # Mxz
    m22 = H1sk + dzs ** 2 * H2sk - H1k - dz ** 2 * H2k - zwall * zwall * (
            1 * D1k + dz ** 2 * D2k) - 2 * zwall * (
                  dz * H2k + dz * H2k + dz * dH1k / rk + dz ** 3 * dH2k / rk)  # Mzz
    m01 = dx * dy * H2sk - dx * dy * H2k - zwall * zwall * (-dx * dy * D2k) - 2 * zwall * (
            -dz * dx * dy * dH2k / rk)  # Mxy
    m21 = dzs * dy * H2sk - dz * dy * H2k - zwall * zwall * (-dz * dy * D2k) - 2 * zwall * (
            dH1k / rk + H2k) * dy - 2 * zwall * (-dy * dH1k / rk - dz * dy * dz * dH2k / rk)  # Mzy
    m10 = dx * dy * H2sk - dx * dy * H2k - zwall * zwall * (-dx * dy * D2k) - 2 * zwall * (
            -dx * dy * dz * dH2k / rk)  # Myx
    m12 = dzs * dy * H2sk - dz * dy * H2k - zwall * zwall * (dy * dz * D2k) - 2 * zwall * (
            dy * H2k + dy * dz * dz * dH2k / rk)  # Myz
    m11 = H1sk + dy * dy * H2sk - H1k - dy * dy * H2k - zwall * zwall * (
            -D1k - dy ** 2 * D2k) - 2 * zwall * (
                  dH1k / rk + H2k) * zwall - 2 * zwall * (
                  -dz * H2k - dy ** 2 * dz * dH2k / rk)  # Myy
    # # PETSc.Sys.Print('DBG version')
    # m11 = H1sk + dy * dy * H2sk - H1k - dy * dy * H2k - zwall * zwall * (-D1k - dy ** 2 * D2k) - 2 * zwall * (
    #   dH1k / rk + H2k) * zwall + 2 * zwall * (-dz * H2k - dy ** 2 * dz * dH2k / rk) # Myy
    m00 = m00 / (8 * np.pi)
    m01 = m01 / (8 * np.pi)
    m02 = m02 / (8 * np.pi)
    m10 = m10 / (8 * np.pi)
    m11 = m11 / (8 * np.pi)
    m12 = m12 / (8 * np.pi)
    m20 = m20 / (8 * np.pi)
    m21 = m21 / (8 * np.pi)
    m22 = m22 / (8 * np.pi)
    return m00, m01, m02, m10, m11, m12, m20, m21, m22, i0


# def regularized_stokeslets_plane_matrix_3d_petsc_mij(t_u_node: np.ndarray, # velocity node
#                           f_nodes: np.ndarray, # force nodes
#                           e2, # e2 = e^2
#                           i0, **kwargs):
#   # Ding version
#   xgridb = t_u_node[0]
#   ygridb = t_u_node[1]
#   zgridb = t_u_node[2]
#   xwall = f_nodes[:, 0]
#   ywall = f_nodes[:, 1]
#   zwall = f_nodes[:, 2]
#
#   dx = xgridb - xwall
#   dys = ygridb - ywall
#   dy = ygridb + ywall
#   dz = zgridb - zwall
#
#   r2sk = dx ** 2 + dys ** 2 + dz ** 2 # r squared
#   # rsk = r2sk ** 0.5
#   rske = (r2sk + e2) ** 0.5
#   H2sk = 1 / (rske ** 3)
#   H1sk = 1 / rske + e2 * H2sk
#   D2sk = -6 / (rske ** 5)
#   # D1sk = 2 / (rske ** 3) + e2 * D2sk
#
#   r2k = dx ** 2 + dy ** 2 + dz ** 2
#   rk = r2k ** 0.5
#   rke = (r2k + e2) ** 0.5
#   H2k = 1 / (rke ** 3)
#   H1k = 1 / rke + e2 * H2k
#   D2k = -6 / (rke ** 5)
#   D1k = 2 / (rke ** 3) + e2 * D2k
#   dH2k = -3 * rk / (rke ** 5)
#   dH1k = -rk / (rke ** 3) + e2 * dH2k
#
#   m00 = H1sk + dx ** 2 * H2sk - H1k - dx ** 2 * H2k - ywall * ywall * (-1 * D1k - dx ** 2 * D2k) - 2 * ywall * (
#     dH1k / rk + H2k) * dy + 2 * ywall * (-dy * H2k - dy * dx ** 2 * dH2k / rk)
#   m10 = dx * dys * H2sk - dx * dy * H2k - ywall * ywall * (-dy * dx * D2k) + 2 * ywall * (
#     dH1k / rk + H2k) * dx + 2 * ywall * (-dx * dH1k / rk - dx * dy ** 2 * dH2k / rk)
#   m01 = dx * dys * H2sk - dx * dy * H2k - ywall * ywall * (dx * dy * D2k) + 2 * ywall * (
#     dx * H2k + dy ** 2 * dx * dH2k / rk)
#   m11 = H1sk + dys ** 2 * H2sk - H1k - dy ** 2 * H2k - ywall * ywall * (1 * D1k + dy ** 2 * D2k) + 2 * ywall * (
#     dy * H2k + dy * H2k + dy * dH1k / rk + dy ** 3 * dH2k / rk)
#   m02 = dx * dz * H2sk - dx * dz * H2k - ywall * ywall * (-dx * dz * D2k) + 2 * ywall * (
#     -dy * dx * dz * dH2k / rk)
#   m12 = dys * dz * H2sk - dy * dz * H2k - ywall * ywall * (-dy * dz * D2k) + 2 * ywall * (
#     dH1k / rk + H2k) * dz + 2 * ywall * (-dz * dH1k / rk - dy * dz * dy * dH2k / rk)
#   m20 = dx * dz * H2sk - dx * dz * H2k - ywall * ywall * (-dx * dz * D2k) + 2 * ywall * (
#     -dx * dz * dy * dH2k / rk)
#   m21 = dys * dz * H2sk - dy * dz * H2k - ywall * ywall * (dz * dy * D2k) + 2 * ywall * (
#     dz * H2k + dz * dy * dy * dH2k / rk)
#   m22 = H1sk + dz * dz * H2sk - H1k - dz * dz * H2k - ywall * ywall * (-D1k - dz ** 2 * D2k) - 2 * ywall * (
#     dH1k / rk + H2k) * dy + 2 * ywall * (-dy * H2k - dz ** 2 * dy * dH2k / rk)
#   m00 = m00 / (8 * np.pi)
#   m01 = m01 / (8 * np.pi)
#   m02 = m02 / (8 * np.pi)
#   m10 = m10 / (8 * np.pi)
#   m11 = m11 / (8 * np.pi)
#   m12 = m12 / (8 * np.pi)
#   m20 = m20 / (8 * np.pi)
#   m21 = m21 / (8 * np.pi)
#   m22 = m22 / (8 * np.pi)
#   return m00, m01, m02, m10, m11, m12, m20, m21, m22, i0


def regularized_stokeslets_plane_matrix_3d_petsc(obj1: 'sf.StokesFlowObj',
                                                 # object contain velocity information
                                                 obj2: 'sf.StokesFlowObj',
                                                 # object contain force information
                                                 m, **kwargs):
    # Solve m matrix using regularized Stokeslets method
    # U = M * F.
    # Cortez R. The method of regularized Stokeslets [J]. SIAM Journal on Scientific Computing, 2001, 23(4): 1204-1225.
    # delta_2 = kwargs['delta'] ** 2  # correction factor
    epsilon = kwargs['epsilon']
    delta = epsilon * obj2.get_f_geo().get_deltaLength()
    delta_2 = delta ** 2
    u_nodes = obj1.get_u_nodes()
    f_nodes = obj2.get_f_nodes()
    _, u_glbIdx_all = obj1.get_u_geo().get_glbIdx()
    _, f_glbIdx_all = obj2.get_f_geo().get_glbIdx()
    u_dmda = obj1.get_u_geo().get_dmda()
    
    for i0 in range(u_dmda.getRanges()[0][0], u_dmda.getRanges()[0][1]):
        m00, m01, m02, m10, m11, m12, m20, m21, m22, i1 = \
            regularized_stokeslets_plane_matrix_3d_petsc_mij(u_nodes[i0], f_nodes, delta_2, i0)
        u_glb = u_glbIdx_all[i1 * 3]
        m.setValues(u_glb + 0, f_glbIdx_all[0::3], m00, addv=False)
        m.setValues(u_glb + 0, f_glbIdx_all[1::3], m01, addv=False)
        m.setValues(u_glb + 0, f_glbIdx_all[2::3], m02, addv=False)
        m.setValues(u_glb + 1, f_glbIdx_all[0::3], m10, addv=False)
        m.setValues(u_glb + 1, f_glbIdx_all[1::3], m11, addv=False)
        m.setValues(u_glb + 1, f_glbIdx_all[2::3], m12, addv=False)
        m.setValues(u_glb + 2, f_glbIdx_all[0::3], m20, addv=False)
        m.setValues(u_glb + 2, f_glbIdx_all[1::3], m21, addv=False)
        m.setValues(u_glb + 2, f_glbIdx_all[2::3], m22, addv=False)
    m.assemble()
    
    return m  # ' regularized Stokeslets matrix, U = M * F '


def check_regularized_stokeslets_plane_matrix_3d(**kwargs):
    # err_msg = 'the reguralized Stokeslets method needs parameter, delta. '
    # assert 'delta' in kwargs, err_msg
    err_msg = 'delta is been droped in current version. '
    assert 'delta' not in kwargs, err_msg


def legendre_regularized_stokeslets_matrix_3d_mij(t_u_node: np.ndarray,  # velocity node
                                                  f_nodes: np.ndarray,  # force nodes
                                                  f_ds, i0, **kwargs):
    # Hosseini, Bamdad, Nilima Nigam, and John M. Stockie. "On regularizations of the Dirac delta distribution." Journal of Computational Physics 305 (2016): 423-447.
    h1 = {
        'm=2,k=0': lambda r, e: (1 / 8) * e ** (-6) * np.pi ** (-1) * (
                12 * e ** 5 + (-160) * e ** 3 * r ** 2 + 375 * e ** 2 * r ** 3 + (
            -324) * e * r ** 4 + 98 * r ** 5),
        'm=2,k=1': lambda r, e: (1 / 8) * e ** (-8) * np.pi ** (-1) * (
                12 * e ** 7 + (-112) * e ** 5 * r ** 2 + 756 * e ** 3 * r ** 4 + (
            -1372) * e ** 2 * r ** 5 + 960 * e * r ** 6 + (-243) * r ** 7),
        'm=2,k=2': lambda r, e: (1 / 8) * e ** (-10) * np.pi ** (-1) * (
                12 * e ** 9 + (-96) * e ** 7 * r ** 2 + 2058 * e ** 4 * r ** 5 + (
            -5760) * e ** 3 * r ** 6 + 6561 * e ** 2 * r ** 7 + (
                    -3500) * e * r ** 8 + 726 * r ** 9),
        'm=2,k=3': lambda r, e: (1 / 8) * e ** (-12) * np.pi ** (-1) * (
                12 * e ** 11 + (-88) * e ** 9 * r ** 2 + 6336 * e ** 5 * r ** 6 + (
            -24057) * e ** 4 * r ** 7 + 38500 * e ** 3 * r ** 8 + (
                    -31944) * e ** 2 * r ** 9 + 13608 * e * r ** 10 + (
                    -2366) * r ** 11),
        'm=2,k=4': lambda r, e: (1 / 40) * e ** (-14) * np.pi ** (-1) * (
                60 * e ** 13 + (-416) * e ** 11 * r ** 2 + 104247 * e ** 6 * r ** 7 + (
            -500500) * e ** 5 * r ** 8 + 1038180 * e ** 4 * r ** 9 + (
                    -1179360) * e ** 3 * r ** 10 + 768950 * e ** 2 * r ** 11 + (
                    -271656) * e * r ** 12 + 40500 * r ** 13),
        'm=3,k=0': lambda r, e: (1 / 24) * e ** (-7) * np.pi ** (-1) * (
                56 * e ** 6 + (-1680) * e ** 4 * r ** 2 + 6125 * e ** 3 * r ** 3 + (
            -9072) * e ** 2 * r ** 4 + 6174 * e * r ** 5 + (-1600) * r ** 6),
        'm=3,k=1': lambda r, e: (1 / 8) * e ** (-9) * np.pi ** (-1) * (
                18 * e ** 8 + (-336) * e ** 6 * r ** 2 + 4536 * e ** 4 * r ** 4 + (
            -12348) * e ** 3 * r ** 5 + 14400 * e ** 2 * r ** 6 + (
                    -8019) * e * r ** 7 + 1750 * r ** 8),
        'm=3,k=2': lambda r, e: (1 / 40) * e ** (-11) * np.pi ** (-1) * (88 * e ** 10 + r ** 2 * (
                (-1320) * e ** 8 + r ** 3 * (67914 * e ** 5 + r * (
                (-264000) * e ** 4 + r * (
                441045 * e ** 3 + r * ((-385000) * e ** 2 + (173030 * e + (-31752) * r) * r)))))),
        'm=3,k=3': lambda r, e: (1 / 120) * e ** (-13) * np.pi ** (-1) * (
                260 * e ** 12 + (-3432) * e ** 10 * r ** 2 + 686400 * e ** 6 * r ** 6 + (
            -3440151) * e ** 5 * r ** 7 + 7507500 * e ** 4 * r ** 8 + (
                    -8997560) * e ** 3 * r ** 9 + 6191640 * e ** 2 * r ** 10 + (
                    -2306850) * e * r ** 11 + 362208 * r ** 12),
        'm=4,k=0': lambda r, e: (1 / 24) * e ** (-8) * np.pi ** (-1) * (
                80 * e ** 7 + (-4704) * e ** 5 * r ** 2 + 24500 * e ** 4 * r ** 3 + (
            -54432) * e ** 3 * r ** 4 + 61740 * e ** 2 * r ** 5 + (
                    -35200) * e * r ** 6 + 8019 * r ** 7),
        'm=4,k=1': lambda r, e: (1 / 8) * e ** (-10) * np.pi ** (-1) * (
                25 * e ** 9 + (-840) * e ** 7 * r ** 2 + 20412 * e ** 5 * r ** 4 + (
            -77175) * e ** 4 * r ** 5 + 132000 * e ** 3 * r ** 6 + (
                    -120285) * e ** 2 * r ** 7 + 56875 * e * r ** 8 + (
                    -11011) * r ** 9),
        'm=4,k=2': lambda r, e: (1 / 8) * e ** (-12) * np.pi ** (-1) * (
                24 * e ** 11 + (-616) * e ** 9 * r ** 2 + 67914 * e ** 6 * r ** 5 + (
            -348480) * e ** 5 * r ** 6 + 793881 * e ** 4 * r ** 7 + (
                    -1001000) * e ** 3 * r ** 8 + 726726 * e ** 2 * r ** 9 + (
                    -285768) * e * r ** 10 + 47320 * r ** 11),
        'm=4,k=3': lambda r, e: (1 / 120) * e ** (-14) * np.pi ** (-1) * (
                350 * e ** 13 + (-7644) * e ** 11 * r ** 2 + 3775200 * e ** 7 * r ** 6 + (
            -24081057) * e ** 6 * r ** 7 + 68318250 * e ** 5 * r ** 8 + (
                    -110220110) * e ** 4 * r ** 9 + 108353700 * e ** 3 * r ** 10 + (
                    -64591800) * e ** 2 * r ** 11 + 21551376 * e * r ** 12 + (-3098250) * r ** 13),
        'm=5,k=0': lambda r, e: (1 / 40) * e ** (-9) * np.pi ** (-1) * (
                180 * e ** 8 + (-18816) * e ** 6 * r ** 2 + 132300 * e ** 5 * r ** 3 + (
            -408240) * e ** 4 * r ** 4 + 679140 * e ** 3 * r ** 5 + (
                    -633600) * e ** 2 * r ** 6 + 312741 * e * r ** 7 + (
                    -63700) * r ** 8),
        'm=5,k=1': lambda r, e: (1 / 8) * e ** (-11) * np.pi ** (-1) * (
                33 * e ** 10 + (-1848) * e ** 8 * r ** 2 + 74844 * e ** 6 * r ** 4 + (
            -373527) * e ** 5 * r ** 5 + 871200 * e ** 4 * r ** 6 + (
                    -1146717) * e ** 3 * r ** 7 + 875875 * e ** 2 * r ** 8 + (
                    -363363) * e * r ** 9 + 63504 * r ** 10),
        'm=5,k=2': lambda r, e: (1 / 200) * e ** (-13) * np.pi ** (-1) * (
                780 * e ** 12 + (-32032) * e ** 10 * r ** 2 + 6936930 * e ** 7 * r ** 5 + (
            -45302400) * e ** 6 * r ** 6 + 134165889 * e ** 5 * r ** 7 + (
                    -227727500) * e ** 4 * r ** 8 + 236185950 * e ** 3 * r ** 9 + (
                    -148599360) * e ** 2 * r ** 10 + 52288600 * e * r ** 11 + (-7916832) * r ** 12)
        }
    h2 = {
        'm=2,k=0': lambda r, e: (1 / 8) * e ** (-6) * np.pi ** (-1) * (
                80 * e ** 3 + (-225) * e ** 2 * r + 216 * e * r ** 2 + (-70) * r ** 3),
        'm=2,k=1': lambda r, e: (1 / 8) * e ** (-8) * np.pi ** (-1) * (
                56 * e ** 5 + (-504) * e ** 3 * r ** 2 + 980 * e ** 2 * r ** 3 + (
            -720) * e * r ** 4 + 189 * r ** 5),
        'm=2,k=2': lambda r, e: (1 / 8) * e ** (-10) * np.pi ** (-1) * (
                48 * e ** 7 + (-1470) * e ** 4 * r ** 3 + 4320 * e ** 3 * r ** 4 + (
            -5103) * e ** 2 * r ** 5 + 2800 * e * r ** 6 + (-594) * r ** 7),
        'm=2,k=3': lambda r, e: (1 / 8) * e ** (-12) * np.pi ** (-1) * (
                44 * e ** 9 + (-4752) * e ** 5 * r ** 4 + 18711 * e ** 4 * r ** 5 + (
            -30800) * e ** 3 * r ** 6 + 26136 * e ** 2 * r ** 7 + (
                    -11340) * e * r ** 8 + 2002 * r ** 9),
        'm=2,k=4': lambda r, e: (1 / 40) * e ** (-14) * np.pi ** (-1) * (
                208 * e ** 11 + (-81081) * e ** 6 * r ** 5 + 400400 * e ** 5 * r ** 6 + (
            -849420) * e ** 4 * r ** 7 + 982800 * e ** 3 * r ** 8 + (
                    -650650) * e ** 2 * r ** 9 + 232848 * e * r ** 10 + (
                    -35100) * r ** 11),
        'm=3,k=0': lambda r, e: (1 / 8) * e ** (-7) * np.pi ** (-1) * (
                280 * e ** 4 + (-1225) * e ** 3 * r + 2016 * e ** 2 * r ** 2 + (
            -1470) * e * r ** 3 + 400 * r ** 4),
        'm=3,k=1': lambda r, e: (1 / 8) * e ** (-9) * np.pi ** (-1) * (
                168 * e ** 6 + (-3024) * e ** 4 * r ** 2 + 8820 * e ** 3 * r ** 3 + (
            -10800) * e ** 2 * r ** 4 + 6237 * e * r ** 5 + (-1400) * r ** 6),
        'm=3,k=2': lambda r, e: (1 / 8) * e ** (-11) * np.pi ** (-1) * (
                132 * e ** 8 + r ** 3 * ((-9702) * e ** 5 + r * (
                39600 * e ** 4 + r * (
                (-68607) * e ** 3 + r * (61600 * e ** 2 + r * ((-28314) * e + 5292 * r)))))),
        'm=3,k=3': lambda r, e: (1 / 40) * e ** (-13) * np.pi ** (-1) * (
                572 * e ** 10 + (-171600) * e ** 6 * r ** 4 + 891891 * e ** 5 * r ** 5 + (
            -2002000) * e ** 4 * r ** 6 + 2453880 * e ** 3 * r ** 7 + (
                    -1719900) * e ** 2 * r ** 8 + 650650 * e * r ** 9 + (
                    -103488) * r ** 10),
        'm=4,k=0': lambda r, e: (1 / 8) * e ** (-8) * np.pi ** (-1) * (
                784 * e ** 5 + (-4900) * e ** 4 * r + 12096 * e ** 3 * r ** 2 + (
            -14700) * e ** 2 * r ** 3 + 8800 * e * r ** 4 + (-2079) * r ** 5),
        'm=4,k=1': lambda r, e: (1 / 8) * e ** (-10) * np.pi ** (-1) * (
                420 * e ** 7 + (-13608) * e ** 5 * r ** 2 + 55125 * e ** 4 * r ** 3 + (
            -99000) * e ** 3 * r ** 4 + 93555 * e ** 2 * r ** 5 + (
                    -45500) * e * r ** 6 + 9009 * r ** 7),
        'm=4,k=2': lambda r, e: (1 / 8) * e ** (-12) * np.pi ** (-1) * (
                308 * e ** 9 + (-48510) * e ** 6 * r ** 3 + 261360 * e ** 5 * r ** 4 + (
            -617463) * e ** 4 * r ** 5 + 800800 * e ** 3 * r ** 6 + (
                    -594594) * e ** 2 * r ** 7 + 238140 * e * r ** 8 + (
                    -40040) * r ** 9),
        'm=4,k=3': lambda r, e: (1 / 40) * e ** (-14) * np.pi ** (-1) * (
                1274 * e ** 11 + (-943800) * e ** 7 * r ** 4 + 6243237 * e ** 6 * r ** 5 + (
            -18218200) * e ** 5 * r ** 6 + 30060030 * e ** 4 * r ** 7 + (
                    -30098250) * e ** 3 * r ** 8 + 18218200 * e ** 2 * r ** 9 + (
                    -6157536) * e * r ** 10 + 895050 * r ** 11),
        'm=5,k=0': lambda r, e: (1 / 40) * e ** (-9) * np.pi ** (-1) * (
                9408 * e ** 6 + (-79380) * e ** 5 * r + 272160 * e ** 4 * r ** 2 + (
            -485100) * e ** 3 * r ** 3 + 475200 * e ** 2 * r ** 4 + (
                    -243243) * e * r ** 5 + 50960 * r ** 6),
        'm=5,k=1': lambda r, e: (1 / 8) * e ** (-11) * np.pi ** (-1) * (
                924 * e ** 8 + (-49896) * e ** 6 * r ** 2 + 266805 * e ** 5 * r ** 3 + (
            -653400) * e ** 4 * r ** 4 + 891891 * e ** 3 * r ** 5 + (
                    -700700) * e ** 2 * r ** 6 + 297297 * e * r ** 7 + (
                    -52920) * r ** 8),
        'm=5,k=2': lambda r, e: (1 / 200) * e ** (-13) * np.pi ** (-1) * (
                16016 * e ** 10 + (-4954950) * e ** 7 * r ** 3 + 33976800 * e ** 6 * r ** 4 + (
            -104351247) * e ** 5 * r ** 5 + 182182000 * e ** 4 * r ** 6 + (
                    -193243050) * e ** 3 * r ** 7 + 123832800 * e ** 2 * r ** 8 + (
                    -44244200) * e * r ** 9 + 6785856 * r ** 10)
        }
    
    epsilon = kwargs['epsilon']
    e = f_ds * epsilon  # correction factor
    m = kwargs['legendre_m']
    k = kwargs['legendre_k']
    key = "m=%d,k=%d" % (m, k)
    
    delta_xi = f_nodes - t_u_node
    temp1 = delta_xi ** 2
    delta_r = np.sqrt(temp1.sum(axis=1))
    mi = np.zeros(f_nodes.shape[0])
    mi_all = []
    INDEX1 = delta_r < e
    INDEX2 = ~INDEX1
    th1 = h1[key](delta_r[INDEX1], e[INDEX1])
    th2 = h2[key](delta_r[INDEX1], e[INDEX1])
    tr1 = 1 / (8 * np.pi * delta_r[INDEX2])
    tr3 = (8 * np.pi * delta_r[INDEX2] ** 3)
    # m00
    mi[INDEX1] = delta_xi[INDEX1, 0] * delta_xi[INDEX1, 0] * th2 + th1
    mi[INDEX2] = delta_xi[INDEX2, 0] * delta_xi[INDEX2, 0] / tr3 + tr1
    mi_all.append(mi.copy())
    # m01
    mi[INDEX1] = delta_xi[INDEX1, 0] * delta_xi[INDEX1, 1] * th2
    mi[INDEX2] = delta_xi[INDEX2, 0] * delta_xi[INDEX2, 1] / tr3
    mi_all.append(mi.copy())
    # m02
    mi[INDEX1] = delta_xi[INDEX1, 0] * delta_xi[INDEX1, 2] * th2
    mi[INDEX2] = delta_xi[INDEX2, 0] * delta_xi[INDEX2, 2] / tr3
    mi_all.append(mi.copy())
    # m10
    mi[INDEX1] = delta_xi[INDEX1, 1] * delta_xi[INDEX1, 0] * th2
    mi[INDEX2] = delta_xi[INDEX2, 1] * delta_xi[INDEX2, 0] / tr3
    mi_all.append(mi.copy())
    # m11
    mi[INDEX1] = delta_xi[INDEX1, 1] * delta_xi[INDEX1, 1] * th2 + th1
    mi[INDEX2] = delta_xi[INDEX2, 1] * delta_xi[INDEX2, 1] / tr3 + tr1
    mi_all.append(mi.copy())
    # m12
    mi[INDEX1] = delta_xi[INDEX1, 1] * delta_xi[INDEX1, 2] * th2
    mi[INDEX2] = delta_xi[INDEX2, 1] * delta_xi[INDEX2, 2] / tr3
    mi_all.append(mi.copy())
    # m20
    mi[INDEX1] = delta_xi[INDEX1, 2] * delta_xi[INDEX1, 0] * th2
    mi[INDEX2] = delta_xi[INDEX2, 2] * delta_xi[INDEX2, 0] / tr3
    mi_all.append(mi.copy())
    # m21
    mi[INDEX1] = delta_xi[INDEX1, 2] * delta_xi[INDEX1, 1] * th2
    mi[INDEX2] = delta_xi[INDEX2, 2] * delta_xi[INDEX2, 1] / tr3
    mi_all.append(mi.copy())
    # m22
    mi[INDEX1] = delta_xi[INDEX1, 2] * delta_xi[INDEX1, 2] * th2 + th1
    mi[INDEX2] = delta_xi[INDEX2, 2] * delta_xi[INDEX2, 2] / tr3 + tr1
    mi_all.append(mi.copy())
    return mi_all, i0


def legendre_regularized_stokeslets_matrix_3d(obj1: 'sf.StokesFlowObj',
                                              # object contain velocity information
                                              obj2: 'sf.StokesFlowObj',
                                              # object contain force information
                                              m, **kwargs):
    u_nodes = obj1.get_u_nodes()
    f_nodes = obj2.get_f_nodes()
    f_ds = obj2.get_f_geo().get_deltaLength()
    if np.array(f_ds).size == 1:
        f_ds = f_ds * np.ones(obj2.get_n_f_node())
    _, u_glbIdx_all = obj1.get_u_geo().get_glbIdx()
    _, f_glbIdx_all = obj2.get_f_geo().get_glbIdx()
    u_dmda = obj1.get_u_geo().get_dmda()
    
    for i0 in range(u_dmda.getRanges()[0][0], u_dmda.getRanges()[0][1]):
        [m00, m01, m02, m10, m11, m12, m20, m21, m22], i1 = \
            legendre_regularized_stokeslets_matrix_3d_mij(u_nodes[i0], f_nodes, f_ds, i0, **kwargs)
        u_glb = u_glbIdx_all[i1 * 3]
        m.setValues(u_glb + 0, f_glbIdx_all[0::3], m00, addv=False)
        m.setValues(u_glb + 0, f_glbIdx_all[1::3], m01, addv=False)
        m.setValues(u_glb + 0, f_glbIdx_all[2::3], m02, addv=False)
        m.setValues(u_glb + 1, f_glbIdx_all[0::3], m10, addv=False)
        m.setValues(u_glb + 1, f_glbIdx_all[1::3], m11, addv=False)
        m.setValues(u_glb + 1, f_glbIdx_all[2::3], m12, addv=False)
        m.setValues(u_glb + 2, f_glbIdx_all[0::3], m20, addv=False)
        m.setValues(u_glb + 2, f_glbIdx_all[1::3], m21, addv=False)
        m.setValues(u_glb + 2, f_glbIdx_all[2::3], m22, addv=False)
        # if i0 % 1000==0:
        #   m.assemble()
    m.assemble()
    
    return m  # ' regularized Stokeslets matrix, U = M * F '


def check_legendre_regularized_stokeslets_matrix_3d(**kwargs):
    err_msg = 'the regularized Stokeslets method needs parameter, epsilon. '
    assert 'epsilon' in kwargs, err_msg
    err_msg = 'the regularized Stokeslets method needs parameter, Legendre_m. '
    assert 'legendre_m' in kwargs, err_msg
    err_msg = 'the regularized Stokeslets method needs parameter, Legendre_k. '
    assert 'legendre_k' in kwargs, err_msg
    return True


# @jit
def two_para_regularized_stokeslets_matrix_3d(obj1: 'sf.StokesFlowObj',
                                              # object contain velocity information
                                              obj2: 'sf.StokesFlowObj',
                                              # object contain force information
                                              **kwargs):
    # Solve m matrix using regularized Stokeslets method
    # U = M * F.
    # Cortez R. The method of regularized Stokeslets [J]. SIAM Journal on Scientific Computing, 2001, 23(4): 1204-1225.
    # Ong, Benjamin, Andrew Christlieb, and Bryan Quaife. "A New Family of Regularized Kernels for the Harmonic Oscillator." arXiv preprint arXiv:1407.1108 (2014).
    
    h1 = {
        '0':  lambda r, e: (1 / 8) * np.pi ** (-1) * r ** (-3) * (e ** 2 + r ** 2) ** (-1 / 2) *
                           ((-1) * e ** 2 * r + r ** 3 + e ** 2 * (e ** 2 + r ** 2) ** (
                                   1 / 2) * np.log(
                                   r + (e ** 2 + r ** 2) ** (1 / 2))),
        '1':  lambda r, e: (1 / 8) * np.pi ** (-1) * (e ** 2 + r ** 2) ** (-3 / 2) * (
                2 * e ** 2 + r ** 2),
        '2':  lambda r, e: (1 / 32) * np.pi ** (-1) * (e ** 2 + r ** 2) ** (-5 / 2) * (
                10 * e ** 4 + 11 * e ** 2 * r ** 2 + 4 * r ** 4),
        '10': lambda r, e: (1 / 5242880) * np.pi ** (-1) * (e ** 2 + r ** 2) ** (-21 / 2) *
                           (
                                   3233230 * e ** 20 + 19076057 * e ** 18 * r ** 2 + 64849356 * e ** 16 * r ** 4 +
                                   143370656 * e ** 14 * r ** 6 + 218213632 * e ** 12 * r ** 8 + 234420480 * e ** 10 * r ** 10 +
                                   178275328 * e ** 8 * r ** 12 + 94244864 * e ** 6 * r ** 14 + 33030144 * e ** 4 * r ** 16 +
                                   6914048 * e ** 2 * r ** 18 + 655360 * r ** 20),
        '20': lambda r, e: (1 / 5497558138880) * np.pi ** (-1) * (e ** 2 + r ** 2) ** (-41 / 2) * (
                4709756401350 * e ** 40 + 56046101176065 * e ** 38 * r ** 2 + 403962534767220 * e ** 36 * r ** 4 + 2014699224028920 * e ** 34 * r ** 6 +
                7460058409602240 * e ** 32 * r ** 8 + 21397820731584000 * e ** 30 * r ** 10 + 48864841525217280 * e ** 28 * r ** 12 +
                90479597908439040 * e ** 26 * r ** 14 + 137493374367498240 * e ** 24 * r ** 16 + 172768642426798080 * e ** 22 * r ** 18 +
                180210608563814400 * e ** 20 * r ** 20 + 156131403845074944 * e ** 18 * r ** 22 + 112062880559398912 * e ** 16 * r ** 24 +
                66227949547814912 * e ** 14 * r ** 26 + 31896562611257344 * e ** 12 * r ** 28 + 12321355873648640 * e ** 10 * r ** 30 +
                3726846201954304 * e ** 8 * r ** 32 + 850444326797312 * e ** 6 * r ** 34 + 137705241444352 * e ** 4 * r ** 36 +
                14104672600064 * e ** 2 * r ** 38 + 687194767360 * r ** 40)
        }
    h2 = {
        '0':  lambda r, e: (1 / 8) * np.pi ** (-1) * r ** (-5) * (e ** 2 + r ** 2) ** (-1 / 2) *
                           (3 * e ** 2 * r + r ** 3 + (-3) * e ** 2 * (e ** 2 + r ** 2) ** (
                                   1 / 2) * np.log(
                                   r + (e ** 2 + r ** 2) ** (1 / 2))),
        '1':  lambda r, e: (1 / 8) * np.pi ** (-1) * (e ** 2 + r ** 2) ** (-3 / 2),
        '2':  lambda r, e: (1 / 32) * np.pi ** (-1) * (e ** 2 + r ** 2) ** (-5 / 2) * (
                7 * e ** 2 + 4 * r ** 2),
        '10': lambda r, e: (1 / 5242880) * np.pi ** (-1) * (e ** 2 + r ** 2) ** (-21 / 2) *
                           (
                                   7436429 * e ** 18 + 38244492 * e ** 16 * r ** 2 + 101985312 * e ** 14 * r ** 4 +
                                   173065984 * e ** 12 * r ** 6 + 199691520 * e ** 10 * r ** 8 + 159753216 * e ** 8 * r ** 10 +
                                   87707648 * e ** 6 * r ** 12 + 31653888 * e ** 4 * r ** 14 + 6782976 * e ** 2 * r ** 16 + 655360 * r ** 18),
        '20': lambda r, e: (1 / 5497558138880) * np.pi ** (-1) * (e ** 2 + r ** 2) ** (-41 / 2) * (
                20251952525805 * e ** 38 + 219878341708740 * e ** 36 * r ** 2 + 1319270050252440 * e ** 34 * r ** 4 +
                5436991722252480 * e ** 32 * r ** 6 + 16729205299238400 * e ** 30 * r ** 8 + 40150092718172160 * e ** 28 * r ** 10 +
                77151158556487680 * e ** 26 * r ** 12 + 120657450975559680 * e ** 24 * r ** 14 + 155131008397148160 * e ** 22 * r ** 16 +
                164873535494553600 * e ** 20 * r ** 18 + 145088711235207168 * e ** 18 * r ** 20 + 105519062716514304 * e ** 16 * r ** 22 +
                63068865071939584 * e ** 14 * r ** 24 + 30673691201241088 * e ** 12 * r ** 26 + 11950788779704320 * e ** 10 * r ** 28 +
                3642145151909888 * e ** 8 * r ** 30 + 836709021384704 * e ** 6 * r ** 32 + 136296492171264 * e ** 4 * r ** 34 +
                14035953123328 * e ** 2 * r ** 36 + 687194767360 * r ** 38)
        }
    
    e = kwargs['delta']  # correction factor
    n = str(kwargs['twoPara_n'])
    vnodes = obj1.get_u_nodes()
    n_vnode = obj1.get_n_velocity()
    fnodes = obj2.get_f_nodes()
    n_fnode = obj2.get_n_force()
    
    n_unknown = obj2.get_n_unknown()
    m = PETSc.Mat().create(comm=PETSc.COMM_WORLD)
    m.setSizes(((None, n_vnode), (None, n_fnode)))
    m.setType('dense')
    m.setFromOptions()
    m.setUp()
    m_start, m_end = m.getOwnershipRange()
    for i0 in range(m_start, m_end):
        delta_xi = fnodes - vnodes[i0 // 3]
        temp1 = delta_xi ** 2
        delta_r = np.sqrt(temp1.sum(axis=1))
        i = i0 % 3
        for j in range(3):
            m[i0, j::n_unknown] = delta(i, j) * h1[n](delta_r, e) + delta_xi[:, i] * delta_xi[:,
                                                                                     j] * h2[n](
                    delta_r, e)
    m.assemble()
    
    return m  # ' regularized Stokeslets matrix, U = M * F '


def check_two_para_regularized_stokeslets_matrix_3d(**kwargs):
    err_msg = 'the regularized Stokeslets method needs parameter, delta. '
    assert 'delta' in kwargs, err_msg
    err_msg = 'the regularized Stokeslets method needs parameter, twoPara_n. '
    assert 'twoPara_n' in kwargs, err_msg


def surf_force_matrix_3d_debug(obj1: 'sf.surf_forceObj',  # object contain velocity information
                               obj2: 'sf.surf_forceObj',  # object contain force information
                               **kwargs):
    # Solve m matrix using regularized Stokeslets method
    # U = M * F.
    # Cortez R. The method of regularized Stokeslets [J]. SIAM Journal on Scientific Computing, 2001, 23(4): 1204-1225.
    
    delta = kwargs['delta']  # correction factor
    d_radia = kwargs['d_radia']  # the radial of the integral surface.
    vnodes = obj1.get_u_nodes()
    n_vnode = obj1.get_n_velocity()
    fnodes = obj2.get_f_nodes()
    n_fnode = obj2.get_n_force()
    
    n_unknown = obj2.get_n_unknown()
    m = PETSc.Mat().create(comm=PETSc.COMM_WORLD)
    m.setSizes(((None, n_vnode), (None, n_fnode)))
    m.setType('dense')
    m.setFromOptions()
    m.setUp()
    m_start, m_end = m.getOwnershipRange()
    for i0 in range(m_start, m_end):
        delta_xi = fnodes - vnodes[i0 // 3]
        temp1 = delta_xi ** 2
        delta_2 = np.square(delta)  # delta_2 = e^2
        delta_r2 = temp1.sum(axis=1) + delta_2  # delta_r2 = r^2+e^2
        delta_r3 = delta_r2 * np.sqrt(delta_r2)  # delta_r3 = (r^2+e^2)^1.5
        temp2 = (delta_r2 + delta_2) / delta_r3  # temp2 = (r^2+2*e^2)/(r^2+e^2)^1.5
        if i0 % 3 == 0:  # x axis
            m[i0, 0::n_unknown] = (temp2 + np.square(delta_xi[:, 0]) / delta_r3) / (
                    8 * np.pi)  # Mxx
            m[i0, 1::n_unknown] = delta_xi[:, 0] * delta_xi[:, 1] / delta_r3 / (8 * np.pi)  # Mxy
            m[i0, 2::n_unknown] = delta_xi[:, 0] * delta_xi[:, 2] / delta_r3 / (8 * np.pi)  # Mxz
        elif i0 % 3 == 1:  # y axis
            m[i0, 0::n_unknown] = delta_xi[:, 0] * delta_xi[:, 1] / delta_r3 / (8 * np.pi)  # Mxy
            m[i0, 1::n_unknown] = (temp2 + np.square(delta_xi[:, 1]) / delta_r3) / (
                    8 * np.pi)  # Myy
            m[i0, 2::n_unknown] = delta_xi[:, 1] * delta_xi[:, 2] / delta_r3 / (8 * np.pi)  # Myz
        else:  # z axis
            m[i0, 0::n_unknown] = delta_xi[:, 0] * delta_xi[:, 2] / delta_r3 / (8 * np.pi)  # Mxz
            m[i0, 1::n_unknown] = delta_xi[:, 1] * delta_xi[:, 2] / delta_r3 / (8 * np.pi)  # Myz
            m[i0, 2::n_unknown] = (temp2 + np.square(delta_xi[:, 2]) / delta_r3) / (
                    8 * np.pi)  # Mzz
    if obj1 is obj2:  # self-interaction
        for i0 in range(m_start, m_end):
            norm = obj1.get_norm()[i0 // 3, :]
            if i0 % 3 == 0:  # x axis
                m[i0, i0 + 0] = (
                                        3 * np.cos(norm[0]) ** 2 + 1 / 2 * (
                                        5 + np.cos(2 * norm[1])) * np.sin(
                                        norm[0]) ** 2) / (
                                        8 * np.pi * d_radia)  # Mxx
                m[i0, i0 + 1] = (np.cos(norm[0]) * np.sin(norm[0]) * np.sin(norm[1]) ** 2) / (
                        8 * np.pi * d_radia)  # Mxy
                m[i0, i0 + 2] = (-np.cos(norm[1]) * np.sin(norm[0]) * np.sin(norm[1])) / (
                        8 * np.pi * d_radia)  # Mxz
            elif i0 % 3 == 1:  # y axis
                m[i0, i0 - 1] = (np.cos(norm[0]) * np.sin(norm[0]) * np.sin(norm[1]) ** 2) / (
                        8 * np.pi * d_radia)  # Mxy
                m[i0, i0 + 0] = (1 / 8 * (22 - 2 * np.cos(2 * norm[0]) + np.cos(
                        2 * (norm[0] - norm[1])) + 2 * np.cos(
                        2 * norm[1]) + np.cos(2 * (norm[0] + norm[1])))) / (
                                        8 * np.pi * d_radia)  # Myy
                m[i0, i0 + 1] = (np.cos(norm[0]) * np.cos(norm[1]) * np.sin(norm[1])) / (
                        8 * np.pi * d_radia)  # Myz
            else:  # z axis
                m[i0, i0 - 2] = (-np.cos(norm[1]) * np.sin(norm[0]) * np.sin(norm[1])) / (
                        8 * np.pi * d_radia)  # Mxz
                m[i0, i0 - 1] = (np.cos(norm[0]) * np.cos(norm[1]) * np.sin(norm[1])) / (
                        8 * np.pi * d_radia)  # Myz
                m[i0, i0 + 0] = (1 / 2 * (5 - np.cos(2 * norm[1]))) / (8 * np.pi * d_radia)  # Mzz
    m.assemble()
    return m  # ' regularized Stokeslets matrix, U = M * F '


def surf_force_matrix_3d(obj1: 'sf.surf_forceObj',  # object contain velocity information
                         obj2: 'sf.surf_forceObj',  # object contain force information
                         **kwargs):
    # Solve m matrix using surface force distribution method
    # U = M * F.
    # details see my notes, 面力分布法
    # Zhang Ji, 20160928
    
    d_radia = kwargs['d_radia']  # the radial of the integral surface.
    vnodes = obj1.get_u_nodes()
    n_vnode = obj1.get_n_velocity()
    fnodes = obj2.get_f_nodes()
    n_fnode = obj2.get_n_force()
    
    n_unknown = obj2.get_n_unknown()
    m = PETSc.Mat().create(comm=PETSc.COMM_WORLD)
    m.setSizes(((None, n_vnode), (None, n_fnode)))
    m.setType('dense')
    m.setFromOptions()
    m.setUp()
    m_start, m_end = m.getOwnershipRange()
    for i0 in range(m_start, m_end):  # interaction between different nodes.
        delta_xi = fnodes - vnodes[i0 // 3]  # [delta_x, delta_y, delta_z]
        temp1 = delta_xi ** 2
        delta_r2 = temp1.sum(axis=1)  # r^2
        if obj1 is obj2:  # self-interaction will be solved later
            delta_r2[i0 // 3] = 1
        delta_r1 = delta_r2 ** 0.5  # r^1
        delta_r3 = delta_r2 * delta_r1  # r^3
        temp2 = 1 / delta_r1  # 1/r
        if i0 % 3 == 0:  # x axis
            m[i0, 0::n_unknown] = (temp2 + delta_xi[:, 0] * delta_xi[:, 0] / delta_r3) / (
                    8 * np.pi)  # Mxx
            m[i0, 1::n_unknown] = (delta_xi[:, 0] * delta_xi[:, 1] / delta_r3) / (8 * np.pi)  # Mxy
            m[i0, 2::n_unknown] = (delta_xi[:, 0] * delta_xi[:, 2] / delta_r3) / (8 * np.pi)  # Mxz
        elif i0 % 3 == 1:  # y axis
            m[i0, 0::n_unknown] = (delta_xi[:, 0] * delta_xi[:, 1] / delta_r3) / (8 * np.pi)  # Mxy
            m[i0, 1::n_unknown] = (temp2 + delta_xi[:, 1] * delta_xi[:, 1] / delta_r3) / (
                    8 * np.pi)  # Myy
            m[i0, 2::n_unknown] = (delta_xi[:, 1] * delta_xi[:, 2] / delta_r3) / (8 * np.pi)  # Myz
        else:  # z axis
            m[i0, 0::n_unknown] = (delta_xi[:, 0] * delta_xi[:, 2] / delta_r3) / (8 * np.pi)  # Mxz
            m[i0, 1::n_unknown] = (delta_xi[:, 1] * delta_xi[:, 2] / delta_r3) / (8 * np.pi)  # Myz
            m[i0, 2::n_unknown] = (temp2 + delta_xi[:, 2] * delta_xi[:, 2] / delta_r3) / (
                    8 * np.pi)  # Mzz
    if obj1 is obj2:  # self-interaction
        for i0 in range(m_start, m_end):
            norm = obj1.get_norm()[i0 // 3, :]
            if i0 % 3 == 0:  # x axis
                m[i0, i0 + 0] = (
                                        3 * np.cos(norm[0]) ** 2 + 1 / 2 * (
                                        5 + np.cos(2 * norm[1])) * np.sin(
                                        norm[0]) ** 2) / (
                                        8 * np.pi * d_radia)  # Mxx
                m[i0, i0 + 1] = (np.cos(norm[0]) * np.sin(norm[0]) * np.sin(norm[1]) ** 2) / (
                        8 * np.pi * d_radia)  # Mxy
                m[i0, i0 + 2] = (-np.cos(norm[1]) * np.sin(norm[0]) * np.sin(norm[1])) / (
                        8 * np.pi * d_radia)  # Mxz
            elif i0 % 3 == 1:  # y axis
                m[i0, i0 - 1] = (np.cos(norm[0]) * np.sin(norm[0]) * np.sin(norm[1]) ** 2) / (
                        
                        8 * np.pi * d_radia)  # Mxy
                m[i0, i0 + 0] = (1 / 8 * (
                        22 - 2 * np.cos(2 * norm[0]) + np.cos(2 * (norm[0] - norm[1])) + 2 * np.cos(
                        2 * norm[1]) + np.cos(
                        2 * (norm[0] + norm[1])))) / (8 * np.pi * d_radia)  # Myy
                m[i0, i0 + 1] = (np.cos(norm[0]) * np.cos(norm[1]) * np.sin(norm[1])) / (
                        8 * np.pi * d_radia)  # Myz
            else:  # z axis
                m[i0, i0 - 2] = (-np.cos(norm[1]) * np.sin(norm[0]) * np.sin(norm[1])) / (
                        8 * np.pi * d_radia)  # Mxz
                m[i0, i0 - 1] = (np.cos(norm[0]) * np.cos(norm[1]) * np.sin(norm[1])) / (
                        8 * np.pi * d_radia)  # Myz
                m[i0, i0 + 0] = (1 / 2 * (5 - np.cos(2 * norm[1]))) / (8 * np.pi * d_radia)  # Mzz
    m.assemble()
    # import matplotlib.pyplot as plt
    # M = m.getDenseArray()
    # fig, ax = plt.subplots()
    # cax = ax.matshow(M, origin='lower')
    # fig.colorbar(cax)
    # plt.show()
    return m  # ' regularized Stokeslets matrix, U = M * F '


def check_surf_force_matrix_3d(**kwargs):
    if not ('d_radia' in kwargs):
        err_msg = 'the surface force method needs parameter, d_radia, the radial of the integral surface. '
        raise ValueError(err_msg)


def point_source_matrix_3d_petsc(obj1: 'sf.StokesFlowObj',  # object contain velocity information
                                 obj2: 'sf.StokesFlowObj',  # object contain force information
                                 **kwargs):
    # Solve m matrix using regularized Stokeslets method
    # U = M * F.
    # Cortez R. The method of regularized Stokeslets[J]. SIAM Journal on Scientific Computing, 2001, 23(4): 1204-1225.
    
    vnodes = obj1.get_u_nodes()
    n_vnode = obj1.get_n_velocity()
    fnodes = obj2.get_f_nodes()
    n_fnode = obj2.get_n_force()
    
    m = PETSc.Mat().create(comm=PETSc.COMM_WORLD)
    n_unknown = obj2.get_n_unknown()
    m.setSizes(((None, n_vnode), (None, n_fnode)))
    m.setType('dense')
    m.setFromOptions()
    m.setUp()
    m_start, m_end = m.getOwnershipRange()
    for i0 in range(m_start, m_end):  # interaction between different nodes.
        delta_xi = fnodes - vnodes[i0 // 3]  # [delta_x, delta_y, delta_z]
        temp1 = delta_xi ** 2
        delta_r2 = temp1.sum(axis=1)  # r^2
        delta_r1 = delta_r2 ** 0.5  # r^1
        delta_r3 = delta_r2 * delta_r1  # r^3
        temp2 = 1 / delta_r1  # 1/r
        if i0 % 3 == 0:  # velocity x axis
            m[i0, 0::n_unknown] = (temp2 + delta_xi[:, 0] * delta_xi[:, 0] / delta_r3) / (
                    8 * np.pi)  # Mxx
            m[i0, 1::n_unknown] = (delta_xi[:, 0] * delta_xi[:, 1] / delta_r3) / (8 * np.pi)  # Mxy
            m[i0, 2::n_unknown] = (delta_xi[:, 0] * delta_xi[:, 2] / delta_r3) / (8 * np.pi)  # Mxz
            m[i0, 3::n_unknown] = delta_xi[:, 0] / delta_r3 / (4 * np.pi)  # Mx_source
        elif i0 % 3 == 1:  # velocity y axis
            m[i0, 0::n_unknown] = (delta_xi[:, 0] * delta_xi[:, 1] / delta_r3) / (8 * np.pi)  # Mxy
            m[i0, 1::n_unknown] = (temp2 + delta_xi[:, 1] * delta_xi[:, 1] / delta_r3) / (
                    8 * np.pi)  # Myy
            m[i0, 2::n_unknown] = (delta_xi[:, 1] * delta_xi[:, 2] / delta_r3) / (8 * np.pi)  # Myz
            m[i0, 3::n_unknown] = delta_xi[:, 1] / delta_r3 / (4 * np.pi)  # My_source
        elif i0 % 3 == 2:  # velocity z axis
            m[i0, 0::n_unknown] = (delta_xi[:, 0] * delta_xi[:, 2] / delta_r3) / (8 * np.pi)  # Mxz
            m[i0, 1::n_unknown] = (delta_xi[:, 1] * delta_xi[:, 2] / delta_r3) / (8 * np.pi)  # Myz
            m[i0, 2::n_unknown] = (temp2 + delta_xi[:, 2] * delta_xi[:, 2] / delta_r3) / (
                    8 * np.pi)  # Mzz
            m[i0, 3::n_unknown] = delta_xi[:, 2] / delta_r3 / (4 * np.pi)  # Mz_source
    m.assemble()
    
    return m  # ' regularized Stokeslets matrix, U = M * F '


def check_point_source_matrix_3d_petsc(**kwargs):
    pass


def point_source_dipole_matrix_3d_petsc(obj1: 'sf.StokesFlowObj',
                                        # object contain velocity information
                                        obj2: 'sf.StokesFlowObj',
                                        # object contain force information
                                        **kwargs):
    # Solve m matrix using regularized Stokeslets method
    # U = M * F.
    
    vnodes = obj1.get_u_nodes()
    n_vnode = obj1.get_n_velocity()
    fnodes = obj2.get_f_nodes()
    n_fnode = obj2.get_n_force()
    ps_ds_para = kwargs['ps_ds_para']  # weight factor of dipole for ps_ds method
    
    m = PETSc.Mat().create(comm=PETSc.COMM_WORLD)
    n_unknown = obj2.get_n_unknown()
    m.setSizes(((None, n_vnode), (None, n_fnode)))
    m.setType('dense')
    m.setFromOptions()
    m.setUp()
    m_start, m_end = m.getOwnershipRange()
    for i0 in range(m_start, m_end):  # interaction between different nodes.
        delta_xi = fnodes - vnodes[i0 // 3]  # [delta_x, delta_y, delta_z]
        temp1 = delta_xi ** 2
        delta_r2 = temp1.sum(axis=1)  # r^2
        delta_r1 = delta_r2 ** 0.5  # r^1
        delta_r3 = delta_r2 * delta_r1  # r^3
        delta_r5 = delta_r2 * delta_r3  # r^5
        temp2 = 1 / (delta_r1 * (8 * np.pi))  # 1/r^1/(8*np.pi)
        temp3 = 1 / (delta_r3 * (8 * np.pi))  # 1/r^3/(8*np.pi)
        temp4 = ps_ds_para * -1 / (delta_r3 * (4 * np.pi))  # -1/r^3/(4*np.pi)
        temp5 = ps_ds_para * 3 / (delta_r5 * (4 * np.pi))  # 3/r^5/(4*np.pi)
        i = i0 % 3
        for j in range(3):
            m[i0, j::n_unknown] = delta(i, j) * temp2 + delta_xi[:, i] * delta_xi[:, j] * temp3
        for j in range(3):
            m[i0, (j + 3)::n_unknown] = delta(i, j) * temp4 + delta_xi[:, i] * delta_xi[:,
                                                                               j] * temp5
    m.assemble()
    
    return m  # ' regularized Stokeslets matrix, U = M * F '


def check_point_source_dipole_matrix_3d_petsc(**kwargs):
    pass


def point_force_matrix_3d_petsc_mij(u_nodes: np.ndarray,  # velocity node
                                    f_nodes: np.ndarray,  # force nodes
                                    i0, **kwargs):
    mypi = np.pi
    dxi = (u_nodes - f_nodes).T
    dx0 = dxi[0]
    dx1 = dxi[1]
    dx2 = dxi[2]
    dr2 = np.sum(dxi ** 2, axis=0)
    dr1 = np.sqrt(dr2)
    dr3 = dr1 * dr2
    temp1 = 1 / (dr1 * (8 * mypi))  # 1/r^1
    temp2 = 1 / (dr3 * (8 * mypi))  # 1/r^3
    m00 = temp2 * dx0 * dx0 + temp1
    m01 = temp2 * dx0 * dx1
    m02 = temp2 * dx0 * dx2
    m10 = temp2 * dx1 * dx0
    m11 = temp2 * dx1 * dx1 + temp1
    m12 = temp2 * dx1 * dx2
    m20 = temp2 * dx2 * dx0
    m21 = temp2 * dx2 * dx1
    m22 = temp2 * dx2 * dx2 + temp1
    return m00, m01, m02, m10, m11, m12, m20, m21, m22, i0


def point_force_matrix_3d_petsc(obj1: 'sf.StokesFlowObj',  # object contain velocity information
                                obj2: 'sf.StokesFlowObj',  # object contain force information
                                m, **kwargs):
    # Solve m matrix using point force Stokeslets method
    # U = M * F.
    u_nodes = obj1.get_u_nodes()
    f_nodes = obj2.get_f_nodes()
    _, u_glbIdx_all = obj1.get_u_geo().get_glbIdx()
    _, f_glbIdx_all = obj2.get_f_geo().get_glbIdx()
    u_dmda = obj1.get_u_geo().get_dmda()
    
    for i0 in range(u_dmda.getRanges()[0][0], u_dmda.getRanges()[0][1]):
        m00, m01, m02, m10, m11, m12, m20, m21, m22, i1 = point_force_matrix_3d_petsc_mij(
                u_nodes[i0], f_nodes, i0, **kwargs)
        u_glb = u_glbIdx_all[i1 * 3]
        m.setValues(u_glb + 0, f_glbIdx_all[0::3], m00, addv=False)
        m.setValues(u_glb + 0, f_glbIdx_all[1::3], m01, addv=False)
        m.setValues(u_glb + 0, f_glbIdx_all[2::3], m02, addv=False)
        m.setValues(u_glb + 1, f_glbIdx_all[0::3], m10, addv=False)
        m.setValues(u_glb + 1, f_glbIdx_all[1::3], m11, addv=False)
        m.setValues(u_glb + 1, f_glbIdx_all[2::3], m12, addv=False)
        m.setValues(u_glb + 2, f_glbIdx_all[0::3], m20, addv=False)
        m.setValues(u_glb + 2, f_glbIdx_all[1::3], m21, addv=False)
        m.setValues(u_glb + 2, f_glbIdx_all[2::3], m22, addv=False)
    m.assemble()
    return True  # ' point_force_matrix, U = M * F '


def check_point_force_matrix_3d_petsc(**kwargs):
    return True


def pf_sphere_image_petsc_mij(t_u_node: np.ndarray,  # velocity node
                              f_nodes: np.ndarray,  # force nodes
                              i0, **kwargs):
    Ra = kwargs['pf_sphere_image_Ra']
    img_f_nodes = (f_nodes.T * (Ra / np.linalg.norm(f_nodes, axis=-1)) ** 2).T
    dxi = (t_u_node - f_nodes).T
    disx = dxi[0]
    disy = dxi[1]
    disz = dxi[2]
    r2sk = disx ** 2 + disy ** 2 + disz ** 2
    rsk = r2sk ** 0.5
    r3sk = r2sk ** 1.5
    
    H1sk = 1.0 / r3sk
    MSxx = (r2sk + disx ** 2) * H1sk
    MSyx = disy * disx * H1sk
    MSxy = MSyx
    MSyy = (r2sk + disy ** 2) * H1sk
    MSxz = disx * disz * H1sk
    MSyz = disy * disz * H1sk
    MSzx = MSxz
    MSzy = MSyz
    MSzz = (r2sk + disz ** 2) * H1sk
    
    img_dxi = (t_u_node - img_f_nodes).T
    disxi = img_dxi[0]
    disyi = img_dxi[1]
    diszi = img_dxi[2]
    r2ski = disxi ** 2 + disyi ** 2 + diszi ** 2
    rski = r2ski ** 0.5
    r3ski = r2ski ** 1.5
    r5ski = r2ski ** 2.5
    H1ski = 1 / rski
    H2ski = 1 / r2ski
    H3ski = 1 / r3ski
    H5ski = 1 / r5ski
    
    Xf = np.linalg.norm(f_nodes, axis=1)
    Xfi = np.linalg.norm(img_f_nodes, axis=1)
    xf = np.linalg.norm(t_u_node)
    Dx1 = np.sum(img_dxi.T * img_f_nodes, axis=1)
    Dx2 = np.sum(t_u_node * img_f_nodes, axis=1)
    
    A = 0.5 * (Xf ** 2 - Ra ** 2) / Xf ** 3
    B = r2ski * (rski - Xfi) * Xfi
    C = 3 * Ra / Xfi
    Det = 1 / (Xfi * (Xfi * rski + Dx2 - Xfi ** 2))
    E = 1 / (xf * Xfi * (xf * Xfi + Dx2))
    A1 = (Xf ** 2 - Ra ** 2) / Xf
    A2 = xf ** 2 - Ra ** 2.5
    
    Pxx = A * (-3 * f_nodes[:, 0] * disxi * H3ski / Ra + Ra * H3ski - 3 * Ra * disxi ** 2 * H5ski
               - 2 * f_nodes[:, 0] * img_f_nodes[:, 0] * H3ski / Ra
               + 6 * f_nodes[:, 0] * H5ski * disxi * Dx1 / Ra
               + C * (img_f_nodes[:, 0] * disxi * r2ski + disxi ** 2 * Xfi ** 2 + B)
               * H3ski * Det - C * H2ski * Det ** 2 * Xfi * (Xfi * disxi + img_f_nodes[:, 0] * rski)
               * (img_f_nodes[:, 0] * r2ski - disxi * Xfi ** 2 + (
                    t_u_node[0] - 2 * img_f_nodes[:, 0])
                  * rski * Xfi) - C * E * (t_u_node[0] * img_f_nodes[:, 0] + xf * Xfi)
               + C * E ** 2 * (Xfi * t_u_node[0] + xf * img_f_nodes[:, 0]) * (
                       Xfi * t_u_node[0] + xf * img_f_nodes[:, 0]) * xf * Xfi)
    Pyy = A * (-3 * f_nodes[:, 1] * disyi * H3ski / Ra + Ra * H3ski - 3 * Ra * disyi ** 2 * H5ski
               - 2 * f_nodes[:, 1] * img_f_nodes[:, 1] * H3ski / Ra
               + 6 * f_nodes[:, 1] * H5ski * disyi * Dx1 / Ra
               + C * (img_f_nodes[:, 1] * disyi * r2ski + disyi ** 2 * Xfi ** 2 + B)
               * H3ski * Det - C * H2ski * Det ** 2 * Xfi * (Xfi * disyi + img_f_nodes[:, 1] * rski)
               * (img_f_nodes[:, 1] * r2ski - disyi * Xfi ** 2
                  + (t_u_node[1] - 2 * img_f_nodes[:, 1]) * rski * Xfi) - C * E * (
                       t_u_node[1] * img_f_nodes[:, 1] + xf * Xfi) + C * E ** 2 * (
                       Xfi * t_u_node[1] + xf * img_f_nodes[:, 1]) * (
                       Xfi * t_u_node[1] + xf * img_f_nodes[:, 1]) * xf * Xfi)
    Pzz = A * (-3 * f_nodes[:,
                    2] * diszi * H3ski / Ra + Ra * H3ski - 3 * Ra * diszi ** 2 * H5ski - 2 * f_nodes[
                                                                                             :,
                                                                                             2] * img_f_nodes[
                                                                                                  :,
                                                                                                  2] * H3ski / Ra + 6 * f_nodes[
                                                                                                                        :,
                                                                                                                        2] * H5ski * diszi * Dx1 / Ra + C * (
                       img_f_nodes[:,
                       2] * diszi * r2ski + diszi ** 2 * Xfi ** 2 + B) * H3ski * Det - C * H2ski * Det ** 2 * Xfi * (
                       Xfi * diszi + img_f_nodes[:, 2] * rski) * (
                       img_f_nodes[:, 2] * r2ski - diszi * Xfi ** 2 + (
                       t_u_node[2] - 2 * img_f_nodes[:, 2]) * rski * Xfi) - C * E * (
                       t_u_node[2] * img_f_nodes[:, 2] + xf * Xfi) + C * E ** 2 * (
                       Xfi * t_u_node[2] + xf * img_f_nodes[:, 2]) * (
                       Xfi * t_u_node[2] + xf * img_f_nodes[:, 2]) * xf * Xfi)
    
    Pxy = A * (-3 * f_nodes[:,
                    1] * disxi * H3ski / Ra - 3 * Ra * disxi * disyi * H5ski - 2 * f_nodes[:,
                                                                                   1] * img_f_nodes[
                                                                                        :,
                                                                                        0] * H3ski / Ra + 6 * f_nodes[
                                                                                                              :,
                                                                                                              1] * H5ski * disxi * Dx1 / Ra + C * (
                       img_f_nodes[:,
                       1] * disxi * r2ski + disxi * disyi * Xfi ** 2) * H3ski * Det - C * H2ski * Det ** 2 * Xfi * (
                       Xfi * disxi + img_f_nodes[:, 0] * rski) * (
                       img_f_nodes[:, 1] * r2ski - disyi * Xfi ** 2 + (
                       t_u_node[1] - 2 * img_f_nodes[:, 1]) * rski * Xfi) - C * E * (
                       t_u_node[0] * img_f_nodes[:, 1]) + C * E ** 2 * (
                       Xfi * t_u_node[0] + xf * img_f_nodes[:, 0]) * (
                       Xfi * t_u_node[1] + xf * img_f_nodes[:, 1]) * xf * Xfi)
    Pyx = A * (-3 * f_nodes[:,
                    0] * disyi * H3ski / Ra - 3 * Ra * disxi * disyi * H5ski - 2 * f_nodes[:,
                                                                                   1] * img_f_nodes[
                                                                                        :,
                                                                                        0] * H3ski / Ra + 6 * f_nodes[
                                                                                                              :,
                                                                                                              0] * H5ski * disyi * Dx1 / Ra + C * (
                       img_f_nodes[:,
                       0] * disyi * r2ski + disxi * disyi * Xfi ** 2) * H3ski * Det - C * H2ski * Det ** 2 * Xfi * (
                       Xfi * disyi + img_f_nodes[:, 1] * rski) * (
                       img_f_nodes[:, 0] * r2ski - disxi * Xfi ** 2 + (
                       t_u_node[0] - 2 * img_f_nodes[:, 0]) * rski * Xfi) - C * E * (
                       t_u_node[1] * img_f_nodes[:, 0]) + C * E ** 2 * (
                       Xfi * t_u_node[0] + xf * img_f_nodes[:, 0]) * (
                       Xfi * t_u_node[1] + xf * img_f_nodes[:, 1]) * xf * Xfi)
    
    Pxz = A * (-3 * f_nodes[:,
                    2] * disxi * H3ski / Ra - 3 * Ra * disxi * diszi * H5ski - 2 * f_nodes[:,
                                                                                   2] * img_f_nodes[
                                                                                        :,
                                                                                        0] * H3ski / Ra + 6 * f_nodes[
                                                                                                              :,
                                                                                                              2] * H5ski * disxi * Dx1 / Ra + C * (
                       img_f_nodes[:,
                       2] * disxi * r2ski + disxi * diszi * Xfi ** 2) * H3ski * Det - C * H2ski * Det ** 2 * Xfi * (
                       Xfi * disxi + img_f_nodes[:, 0] * rski) * (
                       img_f_nodes[:, 2] * r2ski - diszi * Xfi ** 2 + (
                       t_u_node[2] - 2 * img_f_nodes[:, 2]) * rski * Xfi) - C * E * (
                       t_u_node[0] * img_f_nodes[:, 2]) + C * E ** 2 * (
                       Xfi * t_u_node[0] + xf * img_f_nodes[:, 0]) * (
                       Xfi * t_u_node[2] + xf * img_f_nodes[:, 2]) * xf * Xfi)
    Pzx = A * (-3 * f_nodes[:,
                    0] * diszi * H3ski / Ra - 3 * Ra * disxi * diszi * H5ski - 2 * f_nodes[:,
                                                                                   2] * img_f_nodes[
                                                                                        :,
                                                                                        0] * H3ski / Ra + 6 * f_nodes[
                                                                                                              :,
                                                                                                              0] * H5ski * diszi * Dx1 / Ra + C * (
                       img_f_nodes[:,
                       0] * diszi * r2ski + disxi * diszi * Xfi ** 2) * H3ski * Det - C * H2ski * Det ** 2 * Xfi * (
                       Xfi * diszi + img_f_nodes[:, 2] * rski) * (
                       img_f_nodes[:, 0] * r2ski - disxi * Xfi ** 2 + (
                       t_u_node[0] - 2 * img_f_nodes[:, 0]) * rski * Xfi) - C * E * (
                       t_u_node[2] * img_f_nodes[:, 0]) + C * E ** 2 * (
                       Xfi * t_u_node[0] + xf * img_f_nodes[:, 0]) * (
                       Xfi * t_u_node[2] + xf * img_f_nodes[:, 2]) * xf * Xfi)
    
    Pyz = A * (-3 * f_nodes[:,
                    2] * disyi * H3ski / Ra - 3 * Ra * diszi * disyi * H5ski - 2 * f_nodes[:,
                                                                                   1] * img_f_nodes[
                                                                                        :,
                                                                                        2] * H3ski / Ra + 6 * f_nodes[
                                                                                                              :,
                                                                                                              2] * H5ski * disyi * Dx1 / Ra + C * (
                       img_f_nodes[:,
                       2] * disyi * r2ski + diszi * disyi * Xfi ** 2) * H3ski * Det - C * H2ski * Det ** 2 * Xfi * (
                       Xfi * disyi + img_f_nodes[:, 1] * rski) * (
                       img_f_nodes[:, 2] * r2ski - diszi * Xfi ** 2 + (
                       t_u_node[2] - 2 * img_f_nodes[:, 2]) * rski * Xfi) - C * E * (
                       t_u_node[1] * img_f_nodes[:, 2]) + C * E ** 2 * (
                       Xfi * t_u_node[2] + xf * img_f_nodes[:, 2]) * (
                       Xfi * t_u_node[1] + xf * img_f_nodes[:, 1]) * xf * Xfi)
    Pzy = A * (-3 * f_nodes[:,
                    1] * diszi * H3ski / Ra - 3 * Ra * disyi * diszi * H5ski - 2 * f_nodes[:,
                                                                                   2] * img_f_nodes[
                                                                                        :,
                                                                                        1] * H3ski / Ra + 6 * f_nodes[
                                                                                                              :,
                                                                                                              1] * H5ski * diszi * Dx1 / Ra + C * (
                       img_f_nodes[:,
                       1] * diszi * r2ski + disyi * diszi * Xfi ** 2) * H3ski * Det - C * H2ski * Det ** 2 * Xfi * (
                       Xfi * diszi + img_f_nodes[:, 2] * rski) * (
                       img_f_nodes[:, 1] * r2ski - disyi * Xfi ** 2 + (
                       t_u_node[1] - 2 * img_f_nodes[:, 1]) * rski * Xfi) - C * E * (
                       t_u_node[2] * img_f_nodes[:, 1]) + C * E ** 2 * (
                       Xfi * t_u_node[1] + xf * img_f_nodes[:, 1]) * (
                       Xfi * t_u_node[2] + xf * img_f_nodes[:, 2]) * xf * Xfi)
    
    m00 = MSxx - Ra * H1ski / Xf - Ra ** 3 * disxi * disxi * H3ski / Xf ** 3 - A1 * (
            img_f_nodes[:, 0] * img_f_nodes[:, 0] * H1ski / Ra ** 3 - Ra * H3ski * (
            img_f_nodes[:, 0] * disxi + img_f_nodes[:, 0] * disxi) / Xf ** 2 + 2 * img_f_nodes[:,
                                                                                   0] * img_f_nodes[
                                                                                        :,
                                                                                        0] * Dx1 * H3ski / Ra ** 3) - A2 * Pxx
    m11 = MSyy - Ra * H1ski / Xf - Ra ** 3 * disyi * disyi * H3ski / Xf ** 3 - A1 * (
            img_f_nodes[:, 1] * img_f_nodes[:, 1] * H1ski / Ra ** 3 - Ra * H3ski * (
            img_f_nodes[:, 1] * disyi + img_f_nodes[:, 1] * disyi) / Xf ** 2 + 2 * img_f_nodes[:,
                                                                                   1] * img_f_nodes[
                                                                                        :,
                                                                                        1] * Dx1 * H3ski / Ra ** 3) - A2 * Pyy
    m22 = MSzz - Ra * H1ski / Xf - Ra ** 3 * diszi * diszi * H3ski / Xf ** 3 - A1 * (
            img_f_nodes[:, 2] * img_f_nodes[:, 2] * H1ski / Ra ** 3 - Ra * H3ski * (
            img_f_nodes[:, 2] * diszi + img_f_nodes[:, 2] * diszi) / Xf ** 2 + 2 * img_f_nodes[:,
                                                                                   2] * img_f_nodes[
                                                                                        :,
                                                                                        2] * Dx1 * H3ski / Ra ** 3) - A2 * Pzz
    
    m01 = MSxy - Ra ** 3 * disxi * disyi * H3ski / Xf ** 3 - A1 * (
            img_f_nodes[:, 0] * img_f_nodes[:, 1] * H1ski / Ra ** 3 - Ra * H3ski * (
            img_f_nodes[:, 0] * disyi + img_f_nodes[:, 1] * disxi) / Xf ** 2 + 2 * img_f_nodes[:,
                                                                                   1] * img_f_nodes[
                                                                                        :,
                                                                                        0] * Dx1 * H3ski / Ra ** 3) - A2 * Pxy
    m02 = MSxz - Ra ** 3 * disxi * diszi * H3ski / Xf ** 3 - A1 * (
            img_f_nodes[:, 0] * img_f_nodes[:, 2] * H1ski / Ra ** 3 - Ra * H3ski * (
            img_f_nodes[:, 0] * diszi + img_f_nodes[:, 2] * disxi) / Xf ** 2 + 2 * img_f_nodes[:,
                                                                                   2] * img_f_nodes[
                                                                                        :,
                                                                                        0] * Dx1 * H3ski / Ra ** 3) - A2 * Pxz
    m12 = MSyz - Ra ** 3 * disyi * diszi * H3ski / Xf ** 3 - A1 * (
            img_f_nodes[:, 2] * img_f_nodes[:, 1] * H1ski / Ra ** 3 - Ra * H3ski * (
            img_f_nodes[:, 2] * disyi + img_f_nodes[:, 1] * diszi) / Xf ** 2 + 2 * img_f_nodes[:,
                                                                                   1] * img_f_nodes[
                                                                                        :,
                                                                                        2] * Dx1 * H3ski / Ra ** 3) - A2 * Pyz
    
    m10 = MSyx - Ra ** 3 * disxi * disyi * H3ski / Xf ** 3 - A1 * (
            img_f_nodes[:, 0] * img_f_nodes[:, 1] * H1ski / Ra ** 3 - Ra * H3ski * (
            img_f_nodes[:, 0] * disyi + img_f_nodes[:, 1] * disxi) / Xf ** 2 + 2 * img_f_nodes[:,
                                                                                   1] * img_f_nodes[
                                                                                        :,
                                                                                        0] * Dx1 * H3ski / Ra ** 3) - A2 * Pyx
    m20 = MSzx - Ra ** 3 * disxi * diszi * H3ski / Xf ** 3 - A1 * (
            img_f_nodes[:, 0] * img_f_nodes[:, 2] * H1ski / Ra ** 3 - Ra * H3ski * (
            img_f_nodes[:, 0] * diszi + img_f_nodes[:, 2] * disxi) / Xf ** 2 + 2 * img_f_nodes[:,
                                                                                   2] * img_f_nodes[
                                                                                        :,
                                                                                        0] * Dx1 * H3ski / Ra ** 3) - A2 * Pzx
    m21 = MSzy - Ra ** 3 * disyi * diszi * H3ski / Xf ** 3 - A1 * (
            img_f_nodes[:, 2] * img_f_nodes[:, 1] * H1ski / Ra ** 3 - Ra * H3ski * (
            img_f_nodes[:, 2] * disyi + img_f_nodes[:, 1] * diszi) / Xf ** 2 + 2 * img_f_nodes[:,
                                                                                   1] * img_f_nodes[
                                                                                        :,
                                                                                        2] * Dx1 * H3ski / Ra ** 3) - A2 * Pzy
    return m00, m01, m02, m10, m11, m12, m20, m21, m22, i0


def pf_sphere_image_petsc(obj1: 'sf.StokesFlowObj',  # object contain velocity information
                          obj2: 'sf.StokesFlowObj',  # object contain force information
                          m, **kwargs):
    # Solve m matrix using point force Stokeslets method
    # U = M * F.
    u_nodes = obj1.get_u_nodes()
    f_nodes = obj2.get_f_nodes()
    _, u_glbIdx_all = obj1.get_u_geo().get_glbIdx()
    _, f_glbIdx_all = obj2.get_f_geo().get_glbIdx()
    u_dmda = obj1.get_u_geo().get_dmda()
    
    for i0 in range(u_dmda.getRanges()[0][0], u_dmda.getRanges()[0][1]):
        m00, m01, m02, m10, m11, m12, m20, m21, m22, i1 = point_force_matrix_3d_petsc_mij(
                u_nodes[i0], f_nodes, i0, **kwargs)
        u_glb = u_glbIdx_all[i1 * 3]
        m.setValues(u_glb + 0, f_glbIdx_all[0::3], m00, addv=False)
        m.setValues(u_glb + 0, f_glbIdx_all[1::3], m01, addv=False)
        m.setValues(u_glb + 0, f_glbIdx_all[2::3], m02, addv=False)
        m.setValues(u_glb + 1, f_glbIdx_all[0::3], m10, addv=False)
        m.setValues(u_glb + 1, f_glbIdx_all[1::3], m11, addv=False)
        m.setValues(u_glb + 1, f_glbIdx_all[2::3], m12, addv=False)
        m.setValues(u_glb + 2, f_glbIdx_all[0::3], m20, addv=False)
        m.setValues(u_glb + 2, f_glbIdx_all[1::3], m21, addv=False)
        m.setValues(u_glb + 2, f_glbIdx_all[2::3], m22, addv=False)
    m.assemble()
    return True  # ' point_force_matrix, U = M * F '


def check_pf_sphere_image_petsc(**kwargs):
    err_msg = 'the image of Stokeslets outside sphere needs parameter, pf_sphere_image_Ra. '
    assert 'pf_sphere_image_Ra' in kwargs, err_msg
    return True


def two_plate_matrix_3d_petsc(obj1: 'sf.StokesFlowObj',  # object contain velocity information
                              obj2: 'sf.StokesFlowObj',  # object contain force information
                              m, **kwargs):
    return two_plane_matrix_3d_petsc(obj1, obj2, m)


def two_plane_matrix_3d_petsc(obj1: 'sf.StokesFlowObj',  # object contain velocity information
                              obj2: 'sf.StokesFlowObj',  # object contain force information
                              m, **kwargs):
    # see Liron, N., & Mochon, S. (1976). Stokes flow for a stokeslet between two parallel flat plates. Journal of Engineering Mathematics, 10(4), 287-303.
    # U = M * F.
    from src.stokesTwoPlate import tank
    u_nodes = obj1.get_u_nodes()
    f_nodes = obj2.get_f_nodes()
    _, u_glbIdx_all = obj1.get_u_geo().get_glbIdx()
    _, f_glbIdx_all = obj2.get_f_geo().get_glbIdx()
    f_dmda = obj2.get_f_geo().get_dmda()
    Height = kwargs['twoPlateHeight']
    INDEX = kwargs['INDEX']
    greenFun = tank(Height=Height)
    
    for i0 in tqdm(range(f_dmda.getRanges()[0][0], f_dmda.getRanges()[0][1]), desc=INDEX):
        greenFun.set_locF(f_nodes[i0, 0], f_nodes[i0, 1], f_nodes[i0, 2])
        m00, m01, m02, m10, m11, m12, m20, m21, m22 = greenFun.get_Ufunc_series(u_nodes)
        m00 = m00 / (4 * np.pi)
        m01 = m01 / (4 * np.pi)
        m02 = m02 / (4 * np.pi)
        m10 = m10 / (4 * np.pi)
        m11 = m11 / (4 * np.pi)
        m12 = m12 / (4 * np.pi)
        m20 = m20 / (4 * np.pi)
        m21 = m21 / (4 * np.pi)
        m22 = m22 / (4 * np.pi)
        f_glb = f_glbIdx_all[i0 * 3]
        m.setValues(u_glbIdx_all[0::3], f_glb + 0, m00, addv=False)
        m.setValues(u_glbIdx_all[0::3], f_glb + 1, m01, addv=False)
        m.setValues(u_glbIdx_all[0::3], f_glb + 2, m02, addv=False)
        m.setValues(u_glbIdx_all[1::3], f_glb + 0, m10, addv=False)
        m.setValues(u_glbIdx_all[1::3], f_glb + 1, m11, addv=False)
        m.setValues(u_glbIdx_all[1::3], f_glb + 2, m12, addv=False)
        m.setValues(u_glbIdx_all[2::3], f_glb + 0, m20, addv=False)
        m.setValues(u_glbIdx_all[2::3], f_glb + 1, m21, addv=False)
        m.setValues(u_glbIdx_all[2::3], f_glb + 2, m22, addv=False)
    m.assemble()
    return True  # ' point_force_matrix, U = M * F '


def check_two_plate_matrix_3d_petsc(**kwargs):
    return check_two_plane_matrix_3d_petsc(**kwargs)


def check_two_plane_matrix_3d_petsc(**kwargs):
    err_msg = 'The height of two plate system is necessary. '
    assert 'twoPlateHeight' in kwargs, err_msg


def check_pass(**kwargs):
    pass


def dual_potential_matrix_3d_petsc_mij(t_u_node: np.ndarray,  # velocity node
                                       f_nodes: np.ndarray,  # force nodes
                                       i0, **kwargs):
    dxi = (t_u_node - f_nodes).T
    dr2 = np.sum(dxi ** 2, axis=0)
    dr1 = np.sqrt(dr2)
    dx0 = dxi[0] / dr1
    dx1 = dxi[1] / dr1
    dx2 = dxi[2] / dr1
    m00 = np.zeros_like(dx0)
    m01 = -dx2
    m02 = dx1
    m03 = -dx0 / dr2
    m10 = -dx2
    m11 = np.zeros_like(dx0)
    m12 = -dx0
    m13 = -dx1 / dr2
    m20 = -dx1
    m21 = dx0
    m22 = np.zeros_like(dx0)
    m23 = -dx2 / dr2
    m30 = dx0
    m31 = dx1
    m32 = dx2
    m33 = np.zeros_like(dx0)
    # return m00, m01, m02, m03, m10, m11, m12, m13, m20, m21, m22, m23, m30, m31, m32, m33, i0
    return m30, m31, m32, m33, m20, m21, m22, m23, m10, m11, m12, m13, m00, m01, m02, m03, i0


def dual_potential_matrix_3d_petsc(obj1: 'sf.StokesFlowObj',  # object contain velocity information
                                   obj2: 'sf.StokesFlowObj',  # object contain force information
                                   m, **kwargs):
    # Solve m matrix using dual_potential method
    # Young, D. L., et al. "Method of fundamental solutions for multidimensional Stokes equations by the dual-potential formulation." European Journal of Mechanics-B/Fluids 25.6 (2006): 877-893.
    # U = M * F.
    err_msg = 'dof of fgeo of obj %s should be 4. ' % str(obj1)
    assert obj1.get_f_geo().get_dmda().getDof() == 4, err_msg
    err_msg = 'dof of ugeo of obj %s should be 4. ' % str(obj1)
    assert obj1.get_u_geo().get_dmda().getDof() == 4, err_msg
    err_msg = 'dof of fgeo of obj %s should be 4. ' % str(obj2)
    assert obj2.get_f_geo().get_dmda().getDof() == 4, err_msg
    err_msg = 'dof of ugeo of obj %s should be 4. ' % str(obj2)
    assert obj2.get_u_geo().get_dmda().getDof() == 4, err_msg
    
    u_nodes = obj1.get_u_nodes()
    f_nodes = obj2.get_f_nodes()
    _, u_glbIdx_all = obj1.get_u_geo().get_glbIdx()
    _, f_glbIdx_all = obj2.get_f_geo().get_glbIdx()
    u_dmda = obj1.get_u_geo().get_dmda()
    
    # t_n_node = u_dmda.getRanges()[0][1]
    for i0 in range(u_dmda.getRanges()[0][0], u_dmda.getRanges()[0][1]):
        m00, m01, m02, m03, m10, m11, m12, m13, m20, m21, m22, m23, m30, m31, m32, m33, i1 \
            = dual_potential_matrix_3d_petsc_mij(u_nodes[i0], f_nodes, i0)
        u_glb = u_glbIdx_all[i1 * 4]
        # u_glb = u_glbIdx_all[(t_n_node - i1 - 1) * 4]
        m.setValues(u_glb + 0, f_glbIdx_all[0::4], m00, addv=False)
        m.setValues(u_glb + 0, f_glbIdx_all[1::4], m01, addv=False)
        m.setValues(u_glb + 0, f_glbIdx_all[2::4], m02, addv=False)
        m.setValues(u_glb + 0, f_glbIdx_all[3::4], m03, addv=False)
        m.setValues(u_glb + 1, f_glbIdx_all[0::4], m10, addv=False)
        m.setValues(u_glb + 1, f_glbIdx_all[1::4], m11, addv=False)
        m.setValues(u_glb + 1, f_glbIdx_all[2::4], m12, addv=False)
        m.setValues(u_glb + 1, f_glbIdx_all[3::4], m13, addv=False)
        m.setValues(u_glb + 2, f_glbIdx_all[0::4], m20, addv=False)
        m.setValues(u_glb + 2, f_glbIdx_all[1::4], m21, addv=False)
        m.setValues(u_glb + 2, f_glbIdx_all[2::4], m22, addv=False)
        m.setValues(u_glb + 2, f_glbIdx_all[3::4], m23, addv=False)
        m.setValues(u_glb + 3, f_glbIdx_all[0::4], m30, addv=False)
        m.setValues(u_glb + 3, f_glbIdx_all[1::4], m31, addv=False)
        m.setValues(u_glb + 3, f_glbIdx_all[2::4], m32, addv=False)
        m.setValues(u_glb + 3, f_glbIdx_all[3::4], m33, addv=False)
    m.assemble()
    return True  # ' point_force_matrix, U = M * F '


def check_dual_potential_matrix_3d_petsc(**kwargs):
    return True


# method of fundamental solution, infinite helix Stokeslets, cut off at maxtheta, 3d case
def pf_infhelix_3d_petsc_mij(t_u_node, f_geo: 'geo.infgeo_1d', i0):
    def tmij(unode, fnodes):  # solve deltaij/r * 1/*(8*np.pi) + (xi*xj)/r**3 * 1/*(8*np.pi)
        dxi = (unode - fnodes).T
        dx0 = dxi[0]
        dx1 = dxi[1]
        dx2 = dxi[2]
        dr2 = np.sum(dxi ** 2, axis=0)
        dr1 = np.sqrt(dr2)
        dr3 = dr1 * dr2
        temp1 = 1 / (dr1 * (8 * np.pi))  # 1/r^1
        temp2 = 1 / (dr3 * (8 * np.pi))  # 1/r^3
        tm00 = temp2 * dx0 * dx0 + temp1
        tm01 = temp2 * dx0 * dx1
        tm02 = temp2 * dx0 * dx2
        tm10 = temp2 * dx1 * dx0
        tm11 = temp2 * dx1 * dx1 + temp1
        tm12 = temp2 * dx1 * dx2
        tm20 = temp2 * dx2 * dx0
        tm21 = temp2 * dx2 * dx1
        tm22 = temp2 * dx2 * dx2 + temp1
        # return np.dstack((np.vstack((tm00, tm01, tm02)).T,
        #          np.vstack((tm10, tm11, tm12)).T,
        #          np.vstack((tm20, tm21, tm22)).T))
        # return np.dstack(((tm00, tm01, tm02),(tm10, tm11, tm12),(tm20, tm21, tm22)))
        return (tm00, tm01, tm02), (tm10, tm11, tm12), (tm20, tm21, tm22)
    
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = 0, 0, 0, 0, 0, 0, 0, 0, 0
    for ni in np.arange(-f_geo.get_max_period(), f_geo.get_max_period()):
        for thi in np.linspace(0, 2 * np.pi, f_geo.get_nSegment(), endpoint=False):
            th = ni * 2 * np.pi + thi
            f_nodes = f_geo.coord_x123(th)
            rm = f_geo.rot_matrix(th)  # rotation matrix, local->global
            tm = tmij(t_u_node, f_nodes)
            # # dbg code
            # dr1 = np.sqrt(np.sum((t_u_node - f_nodes).T ** 2, axis=0))
            # wgf = 1 / np.exp(0.1 * dr1)
            wgf = 1
            m00 = m00 + wgf * (tm[0][0] * rm[0][0] + tm[0][1] * rm[1][0] + tm[0][2] * rm[2][0])
            m01 = m01 + wgf * (tm[0][0] * rm[0][1] + tm[0][1] * rm[1][1] + tm[0][2] * rm[2][1])
            m02 = m02 + wgf * (tm[0][0] * rm[0][2] + tm[0][1] * rm[1][2] + tm[0][2] * rm[2][2])
            m10 = m10 + wgf * (tm[1][0] * rm[0][0] + tm[1][1] * rm[1][0] + tm[1][2] * rm[2][0])
            m11 = m11 + wgf * (tm[1][0] * rm[0][1] + tm[1][1] * rm[1][1] + tm[1][2] * rm[2][1])
            m12 = m12 + wgf * (tm[1][0] * rm[0][2] + tm[1][1] * rm[1][2] + tm[1][2] * rm[2][2])
            m20 = m20 + wgf * (tm[2][0] * rm[0][0] + tm[2][1] * rm[1][0] + tm[2][2] * rm[2][0])
            m21 = m21 + wgf * (tm[2][0] * rm[0][1] + tm[2][1] * rm[1][1] + tm[2][2] * rm[2][1])
            m22 = m22 + wgf * (tm[2][0] * rm[0][2] + tm[2][1] * rm[1][2] + tm[2][2] * rm[2][2])
    return m00, m01, m02, m10, m11, m12, m20, m21, m22, i0


def pf_infhelix_3d_petsc(obj1: 'sf.StokesFlowObj',  # object contain velocity information
                         obj2: 'sf.StokesFlowObj',  # object contain force information
                         m, **kwargs):
    # Solve m matrix using point force Stokeslets method
    # U = M * F.
    err_msg = 'dof of fgeo of obj %s should be 3. ' % str(obj1)
    assert obj1.get_f_geo().get_dmda().getDof() == 3, err_msg
    err_msg = 'dof of ugeo of obj %s should be 3. ' % str(obj1)
    assert obj1.get_u_geo().get_dmda().getDof() == 3, err_msg
    err_msg = 'dof of fgeo of obj %s should be 3. ' % str(obj2)
    assert obj2.get_f_geo().get_dmda().getDof() == 3, err_msg
    err_msg = 'dof of ugeo of obj %s should be 3. ' % str(obj2)
    assert obj2.get_u_geo().get_dmda().getDof() == 3, err_msg
    
    u_nodes = obj1.get_u_nodes()
    f_geo = obj2.get_f_geo()
    _, u_glbIdx_all = obj1.get_u_geo().get_glbIdx()
    _, f_glbIdx_all = obj2.get_f_geo().get_glbIdx()
    u_dmda = obj1.get_u_geo().get_dmda()
    INDEX = kwargs['INDEX']
    
    for i0 in tqdm(range(u_dmda.getRanges()[0][0], u_dmda.getRanges()[0][1]), desc=INDEX):
        m00, m01, m02, m10, m11, m12, m20, m21, m22, i1 \
            = pf_infhelix_3d_petsc_mij(u_nodes[i0], f_geo, i0)
        u_glb = u_glbIdx_all[i1 * 3]
        m.setValues(u_glb + 0, f_glbIdx_all[0::3], m00, addv=False)
        m.setValues(u_glb + 0, f_glbIdx_all[1::3], m01, addv=False)
        m.setValues(u_glb + 0, f_glbIdx_all[2::3], m02, addv=False)
        m.setValues(u_glb + 1, f_glbIdx_all[0::3], m10, addv=False)
        m.setValues(u_glb + 1, f_glbIdx_all[1::3], m11, addv=False)
        m.setValues(u_glb + 1, f_glbIdx_all[2::3], m12, addv=False)
        m.setValues(u_glb + 2, f_glbIdx_all[0::3], m20, addv=False)
        m.setValues(u_glb + 2, f_glbIdx_all[1::3], m21, addv=False)
        m.setValues(u_glb + 2, f_glbIdx_all[2::3], m22, addv=False)
    m.assemble()
    return True


def check_pf_infhelix_3d_petsc(**kwargs):
    pass


# rotational symmetric stokeslets matrix in bulk fluid.
#  assert the geometric symmetric along z axis, and the slice of nodes are at XY plate.
def point_force_ring_3d_petsc_int(obj1: 'sf.StokesFlowObj',  # contain velocity information
                                  obj2: 'sf.StokesFlowObj',  # contain force information
                                  m, int_epsabs=1e-20, int_epsrel=1e-05,
                                  int_limit=1000, **kwargs):
    def sij1(ru, zu, rf, zf, theta):
        _sij = lambda ru, zu, rf, zf, theta: np.array(
                (((3 * rf ** 2 + 4 * ru ** 2 + 2 * (
                        zf - zu) ** 2 - 8 * rf * ru * np.cos(theta) + rf ** 2 * np.cos(
                        2 * theta)) / (
                          16. * np.pi * (rf ** 2 + ru ** 2 + (
                          zf - zu) ** 2 - 2 * rf * ru * np.cos(
                          theta)) ** 1.5), (rf * (
                        -ru + rf * np.cos(theta)) * np.sin(theta)) / (8. * np.pi * (
                        rf ** 2 + ru ** 2 + (zf - zu) ** 2 - 2 * rf * ru * np.cos(theta)) ** 1.5),
                  -((zf - zu) * (ru - rf * np.cos(theta))) / (
                          8. * np.pi * (rf ** 2 + ru ** 2 + (
                          zf - zu) ** 2 - 2 * rf * ru * np.cos(
                          theta)) ** 1.5)),
                 ((rf * (
                         -ru + rf * np.cos(theta)) * np.sin(theta)) / (8. * np.pi * (
                         rf ** 2 + ru ** 2 + (zf - zu) ** 2 - 2 * rf * ru * np.cos(theta)) ** 1.5),
                  (
                          3 * rf ** 2 + 2 * (
                          ru ** 2 + (
                          zf - zu) ** 2) - rf * (
                                  4 * ru * np.cos(
                                  theta) + rf * np.cos(
                                  2 * theta))) / (
                          16. * np.pi * (
                          rf ** 2 + ru ** 2 + (
                          zf - zu) ** 2 - 2 * rf * ru * np.cos(
                          theta)) ** 1.5),
                  (rf * (
                          zf - zu) * np.sin(
                          theta)) / (
                          8. * np.pi * (
                          rf ** 2 + ru ** 2 + (
                          zf - zu) ** 2 - 2 * rf * ru * np.cos(
                          theta)) ** 1.5)),
                 (-((zf - zu) * (ru - rf * np.cos(theta))) / (
                         8. * np.pi * (rf ** 2 + ru ** 2 + (
                         zf - zu) ** 2 - 2 * rf * ru * np.cos(
                         theta)) ** 1.5),
                  (rf * (zf - zu) * np.sin(theta)) / (
                          8. * np.pi * (rf ** 2 + ru ** 2 + (
                          zf - zu) ** 2 - 2 * rf * ru * np.cos(
                          theta)) ** 1.5), (
                          rf ** 2 + ru ** 2 + 2 * (
                          zf - zu) ** 2 - 2 * rf * ru * np.cos(
                          theta)) / (8. * np.pi * (
                         rf ** 2 + ru ** 2 + (
                         zf - zu) ** 2 - 2 * rf * ru * np.cos(
                         theta)) ** 1.5))))
        tsij = _sij(ru, zu, rf, zf, theta)
        # noinspection PyTypeChecker
        return np.hstack(np.transpose(tsij, (2, 0, 1)))
    
    def sij2(ru, zu, rf, zf, theta):
        _sij = lambda ru, zu, rf, zf, theta: np.array(
                (((3 * rf ** 2 + 4 * ru ** 2 + 2 * (zf - zu) ** 2 - 8 * rf * ru * np.cos(
                        theta) + rf ** 2 * np.cos(2 * theta)) / (16. * np.pi * (
                        rf ** 2 + ru ** 2 + (zf - zu) ** 2 - 2 * rf * ru * np.cos(
                        theta)) ** 1.5), (rf * (-ru + rf * np.cos(theta)) * np.sin(theta)) / (
                          8. * np.pi * (
                          rf ** 2 + ru ** 2 + (zf - zu) ** 2 - 2 * rf * ru * np.cos(
                          theta)) ** 1.5), -((zf - zu) * (ru - rf * np.cos(theta))) / (
                          8. * np.pi * (
                          rf ** 2 + ru ** 2 + (zf - zu) ** 2 - 2 * rf * ru * np.cos(
                          theta)) ** 1.5)),
                 ((rf * (-ru + rf * np.cos(theta)) * np.sin(theta)) / (8. * np.pi * (
                         rf ** 2 + ru ** 2 + (zf - zu) ** 2 - 2 * rf * ru * np.cos(
                         theta)) ** 1.5), (3 * rf ** 2 + 2 * (ru ** 2 + (zf - zu) ** 2) - rf * (
                         4 * ru * np.cos(theta) + rf * np.cos(2 * theta))) / (16. * np.pi * (
                         rf ** 2 + ru ** 2 + (zf - zu) ** 2 - 2 * rf * ru * np.cos(
                         theta)) ** 1.5), (rf * (zf - zu) * np.sin(theta)) / (8. * np.pi * (
                         rf ** 2 + ru ** 2 + (zf - zu) ** 2 - 2 * rf * ru * np.cos(
                         theta)) ** 1.5)),
                 (-((zf - zu) * (ru - rf * np.cos(theta))) / (
                         8. * np.pi * (rf ** 2 + ru ** 2 + (zf - zu) ** 2 - 2 * rf * ru * np.cos(
                         theta)) ** 1.5), (rf * (zf - zu) * np.sin(theta)) / (8. * np.pi * (
                         rf ** 2 + ru ** 2 + (zf - zu) ** 2 - 2 * rf * ru * np.cos(
                         theta)) ** 1.5), (rf ** 2 + ru ** 2 + 2 * (
                         zf - zu) ** 2 - 2 * rf * ru * np.cos(theta)) / (8. * np.pi * (
                         rf ** 2 + ru ** 2 + (zf - zu) ** 2 - 2 * rf * ru * np.cos(
                         theta)) ** 1.5))))
        tsij = _sij(ru, zu, rf, zf, theta)
        # noinspection PyTypeChecker
        return np.hstack(np.transpose(tsij, (2, 0, 1)))
    
    def sij3(ru, zu, rf, zf, theta):
        _sij = lambda ru, zu, rf, zf, theta: np.array(
                ((((3 * rf ** 2 + 4 * ru ** 2 + 2 * (zf - zu) ** 2) * np.cos(theta) + rf * (
                        -3 * ru - 5 * ru * np.cos(2 * theta) + rf * np.cos(3 * theta))) / (
                          16. * np.pi * (
                          rf ** 2 + ru ** 2 + (zf - zu) ** 2 - 2 * rf * ru * np.cos(
                          theta)) ** 1.5), ((2 * rf ** 2 + 2 * ru ** 2 + (
                        zf - zu) ** 2 - 5 * rf * ru * np.cos(theta) + rf ** 2 * np.cos(
                        2 * theta)) * np.sin(theta)) / (8. * np.pi * (
                        rf ** 2 + ru ** 2 + (zf - zu) ** 2 - 2 * rf * ru * np.cos(
                        theta)) ** 1.5), -((zf - zu) * (ru - rf * np.cos(theta))) / (8. * np.pi * (
                        rf ** 2 + ru ** 2 + (zf - zu) ** 2 - 2 * rf * ru * np.cos(
                        theta)) ** 1.5)),
                 (((-rf ** 2 - ru ** 2 - (zf - zu) ** 2 + rf * ru * np.cos(
                         theta) + rf ** 2 * np.cos(2 * theta)) * np.sin(theta)) / (8. * np.pi * (
                         rf ** 2 + ru ** 2 + (zf - zu) ** 2 - 2 * rf * ru * np.cos(
                         theta)) ** 1.5), ((3 * rf ** 2 + 2 * (ru ** 2 + (zf - zu) ** 2)) * np.cos(
                         theta) - rf * (ru * (3 + np.cos(2 * theta)) + rf * np.cos(3 * theta))) / (
                          16. * np.pi * (rf ** 2 + ru ** 2 + (
                          zf - zu) ** 2 - 2 * rf * ru * np.cos(
                          theta)) ** 1.5),
                  (rf * (zf - zu) * np.sin(theta)) / (8. * np.pi * (
                          rf ** 2 + ru ** 2 + (
                          zf - zu) ** 2 - 2 * rf * ru * np.cos(
                          theta)) ** 1.5)),
                 (((zf - zu) * (-(ru * np.cos(theta)) + rf * np.cos(2 * theta))) / (8. * np.pi * (
                         rf ** 2 + ru ** 2 + (zf - zu) ** 2 - 2 * rf * ru * np.cos(
                         theta)) ** 1.5),
                  ((zf - zu) * (-ru + 2 * rf * np.cos(theta)) * np.sin(theta)) / (8. * np.pi * (
                          rf ** 2 + ru ** 2 + (zf - zu) ** 2 - 2 * rf * ru * np.cos(
                          theta)) ** 1.5),
                  (rf ** 2 + ru ** 2 + 2 * (zf - zu) ** 2 - 2 * rf * ru * np.cos(theta)) / (
                          8. * np.pi * (
                          rf ** 2 + ru ** 2 + (zf - zu) ** 2 - 2 * rf * ru * np.cos(
                          theta)) ** 1.5))))
        tsij = _sij(ru, zu, rf, zf, theta)
        # noinspection PyTypeChecker
        return np.hstack(np.transpose(tsij, (2, 0, 1)))
    
    def sij4(ru, zu, rf, zf, theta):
        _sij = lambda ru, zu, rf, zf, theta: np.array(
                (((2 * (2 * rf ** 2 + 2 * ru ** 2 + (zf - zu) ** 2) * np.cos(theta) - rf * ru * (
                        5 + 3 * np.cos(2 * theta))) / (16. * np.pi * (
                        rf ** 2 + ru ** 2 + (zf - zu) ** 2 - 2 * rf * ru * np.cos(
                        theta)) ** 1.5), -((rf ** 2 + 2 * ru ** 2 + (
                        zf - zu) ** 2 - 3 * rf * ru * np.cos(theta)) * np.sin(theta)) / (
                          8. * np.pi * (
                          rf ** 2 + ru ** 2 + (zf - zu) ** 2 - 2 * rf * ru * np.cos(
                          theta)) ** 1.5), -((zf - zu) * (ru - rf * np.cos(theta))) / (
                          8. * np.pi * (
                          rf ** 2 + ru ** 2 + (zf - zu) ** 2 - 2 * rf * ru * np.cos(
                          theta)) ** 1.5)),
                 (((2 * rf ** 2 + ru ** 2 + (
                         zf - zu) ** 2 - 3 * rf * ru * np.cos(theta)) * np.sin(theta)) / (
                          8. * np.pi * (rf ** 2 + ru ** 2 + (
                          zf - zu) ** 2 - 2 * rf * ru * np.cos(
                          theta)) ** 1.5), (2 * (
                         rf ** 2 + ru ** 2 + (zf - zu) ** 2) * np.cos(theta) - rf * ru * (
                                                    1 + 3 * np.cos(
                                                    2 * theta))) / (
                          16. * np.pi * (rf ** 2 + ru ** 2 + (
                          zf - zu) ** 2 - 2 * rf * ru * np.cos(
                          theta)) ** 1.5),
                  (rf * (zf - zu) * np.sin(theta)) / (8. * np.pi * (
                          rf ** 2 + ru ** 2 + (
                          zf - zu) ** 2 - 2 * rf * ru * np.cos(
                          theta)) ** 1.5)),
                 (((zf - zu) * (rf - ru * np.cos(theta))) / (8. * np.pi * (
                         rf ** 2 + ru ** 2 + (zf - zu) ** 2 - 2 * rf * ru * np.cos(
                         theta)) ** 1.5), (ru * (zf - zu) * np.sin(theta)) / (8. * np.pi * (
                         rf ** 2 + ru ** 2 + (zf - zu) ** 2 - 2 * rf * ru * np.cos(
                         theta)) ** 1.5),
                  (rf ** 2 + ru ** 2 + 2 * (zf - zu) ** 2 - 2 * rf * ru * np.cos(theta)) / (
                          8. * np.pi * (
                          rf ** 2 + ru ** 2 + (zf - zu) ** 2 - 2 * rf * ru * np.cos(
                          theta)) ** 1.5))))
        tsij = _sij(ru, zu, rf, zf, theta)
        # noinspection PyTypeChecker
        return np.hstack(np.transpose(tsij, (2, 0, 1)))
    
    u_nodes = obj1.get_u_nodes()
    f_nodes = obj2.get_f_nodes()
    err_msg = '#The geometric symmetric along z axis, and the slice of nodes are at XY plate. '
    assert np.allclose(u_nodes[:, 1], 0), err_msg
    assert np.allclose(f_nodes[:, 1], 0), err_msg
    ru = u_nodes[:, 0]
    zu = u_nodes[:, 2]
    _, u_glbIdx_all = obj1.get_u_geo().get_glbIdx()
    _, f_glbIdx_all = obj2.get_f_geo().get_glbIdx()
    # u_geo = obj1.get_u_geo()
    f_geo = obj2.get_f_geo()
    f_dmda = f_geo.get_dmda()
    
    warpper_sij = lambda theta: sij4(ru, zu, rf, zf, theta)
    for i0 in range(f_dmda.getRanges()[0][0], f_dmda.getRanges()[0][1]):
        rf = f_nodes[i0, 0]
        zf = f_nodes[i0, 2]
        tsij = integrate.quad_vec(warpper_sij, 0, 2 * np.pi, epsabs=int_epsabs, epsrel=int_epsrel,
                                  limit=int_limit, )[0]
        # print(rf)
        f_glb = f_glbIdx_all[i0 * 3]
        m.setValues(u_glbIdx_all, f_glb + 0, tsij[0] * rf, addv=False)
        m.setValues(u_glbIdx_all, f_glb + 1, tsij[1] * rf, addv=False)
        m.setValues(u_glbIdx_all, f_glb + 2, tsij[2] * rf, addv=False)
    m.assemble()
    return m


def point_force_ring_3d_petsc_sum(obj1: 'sf.StokesFlowObj',  # contain velocity information
                                  obj2: 'sf.StokesFlowObj',  # contain force information
                                  m, **kwargs):
    # Solve m matrix using point force Stokeslets method
    # U = M * F.
    n_c = kwargs['n_c']
    u_nodes = obj1.get_u_nodes()
    f_nodes = obj2.get_f_nodes()
    rf = f_nodes[:, 0]
    zf = f_nodes[:, 2]
    _, u_glbIdx_all = obj1.get_u_geo().get_glbIdx()
    _, f_glbIdx_all = obj2.get_f_geo().get_glbIdx()
    u_dmda = obj1.get_u_geo().get_dmda()
    
    for i0 in range(u_dmda.getRanges()[0][0], u_dmda.getRanges()[0][1]):
        mij = [[0, 0, 0], [0, 0, 0], [0, 0, 0], ]
        for thetai in np.linspace(0, 2 * np.pi, n_c, endpoint=False):
            use_nodes = np.vstack((rf * np.cos(thetai), rf * np.sin(thetai), zf)).T
            tmij = point_force_matrix_3d_petsc_mij(u_nodes[i0], use_nodes, i0)
            mij[0][0] = mij[0][0] + tmij[0] * np.cos(thetai) - tmij[1] * np.sin(-thetai)
            mij[0][1] = mij[0][1] + tmij[0] * np.sin(-thetai) + tmij[1] * np.cos(thetai)
            mij[0][2] = mij[0][2] + tmij[2]
            mij[1][0] = mij[1][0] + tmij[3] * np.cos(thetai) - tmij[4] * np.sin(-thetai)
            mij[1][1] = mij[1][1] + tmij[3] * np.sin(-thetai) + tmij[4] * np.cos(thetai)
            mij[1][2] = mij[1][2] + tmij[5]
            mij[2][0] = mij[2][0] + tmij[6] * np.cos(thetai) - tmij[7] * np.sin(-thetai)
            mij[2][1] = mij[2][1] + tmij[6] * np.sin(-thetai) + tmij[7] * np.cos(thetai)
            mij[2][2] = mij[2][2] + tmij[8]
        u_glb = u_glbIdx_all[i0 * 3]
        m.setValues(u_glb + 0, f_glbIdx_all[0::3], mij[0][0], addv=False)
        m.setValues(u_glb + 0, f_glbIdx_all[1::3], mij[0][1], addv=False)
        m.setValues(u_glb + 0, f_glbIdx_all[2::3], mij[0][2], addv=False)
        m.setValues(u_glb + 1, f_glbIdx_all[0::3], mij[1][0], addv=False)
        m.setValues(u_glb + 1, f_glbIdx_all[1::3], mij[1][1], addv=False)
        m.setValues(u_glb + 1, f_glbIdx_all[2::3], mij[1][2], addv=False)
        m.setValues(u_glb + 2, f_glbIdx_all[0::3], mij[2][0], addv=False)
        m.setValues(u_glb + 2, f_glbIdx_all[1::3], mij[2][1], addv=False)
        m.setValues(u_glb + 2, f_glbIdx_all[2::3], mij[2][2], addv=False)
    m.assemble()
    return True  # ' point_force_matrix, U = M * F '


def point_force_ring_3d_petsc(obj1: 'sf.StokesFlowObj',  # contain velocity information
                              obj2: 'sf.StokesFlowObj',  # contain force information
                              m, int_epsabs=1e-20, int_epsrel=1e-05,
                              int_limit=1000, **kwargs):
    return point_force_ring_3d_petsc_int(obj1, obj2, m, int_epsabs, int_epsrel, int_limit, **kwargs)
    # return point_force_ring_3d_petsc_sum(obj1, obj2, m, **kwargs)


def self_repeat_3d_petsc(obj1: 'sf.StokesFlowObj',  # object contain velocity information
                         obj2: 'sf.StokesFlowObj',  # object contain force information
                         m, **kwargs):
    # Solve m matrix using point force Stokeslets method
    # U = M * F.
    f_geo = obj2.get_f_geo()
    if isinstance(f_geo, geo.SelfRepeat_body_geo):
        u_nodes = obj1.get_u_nodes()
        _, u_glbIdx_all = obj1.get_u_geo().get_glbIdx()
        _, f_glbIdx_all = obj2.get_f_geo().get_glbIdx()
        u_dmda = obj1.get_u_geo().get_dmda()
        repeat_n = f_geo.repeat_n
        for repeat_i in range(repeat_n):
            f_nodes = f_geo.get_bodyi_nodes(repeat_i)
            for i0 in range(u_dmda.getRanges()[0][0], u_dmda.getRanges()[0][1]):
                m00, m01, m02, m10, m11, m12, m20, m21, m22, i1 = \
                    point_force_matrix_3d_petsc_mij(u_nodes[i0], f_nodes, i0)
                u_glb = u_glbIdx_all[i1 * 3]
                m.setValues(u_glb + 0, f_glbIdx_all[0::3], m00, addv=True)
                m.setValues(u_glb + 0, f_glbIdx_all[1::3], m01, addv=True)
                m.setValues(u_glb + 0, f_glbIdx_all[2::3], m02, addv=True)
                m.setValues(u_glb + 1, f_glbIdx_all[0::3], m10, addv=True)
                m.setValues(u_glb + 1, f_glbIdx_all[1::3], m11, addv=True)
                m.setValues(u_glb + 1, f_glbIdx_all[2::3], m12, addv=True)
                m.setValues(u_glb + 2, f_glbIdx_all[0::3], m20, addv=True)
                m.setValues(u_glb + 2, f_glbIdx_all[1::3], m21, addv=True)
                m.setValues(u_glb + 2, f_glbIdx_all[2::3], m22, addv=True)
        m.assemble()
    else:
        point_force_matrix_3d_petsc(obj1, obj2, m, **kwargs)
    return True  # ' point_force_matrix, U = M * F '


def self_rotate_3d_petsc_old(obj1: 'sf.StokesFlowObj',  # object contain velocity information
                             obj2: 'sf.StokesFlowObj',  # object contain force information
                             m, **kwargs):
    n_tail = kwargs['n_tail']
    
    u_nodes = obj1.get_u_nodes()
    f_geo = obj2.get_f_geo()
    _, u_glbIdx_all = obj1.get_u_geo().get_glbIdx()
    _, f_glbIdx_all = obj2.get_f_geo().get_glbIdx()
    u_dmda = obj1.get_u_geo().get_dmda()
    delta_rot = 2 * np.pi / n_tail
    tf_geo = f_geo.copy()
    tf_geo.node_rotation(tf_geo.get_geo_norm(), -1 * delta_rot)
    
    for i0 in range(u_dmda.getRanges()[0][0], u_dmda.getRanges()[0][1]):
        u_glb = u_glbIdx_all[i0 * 3]
        for thetai in np.linspace(0, 2 * np.pi, n_tail, endpoint=False):
            rot_M = get_rot_matrix(f_geo.get_geo_norm(), thetai)
            tf_geo.node_rotation(tf_geo.get_geo_norm(), delta_rot)
            f_nodes = tf_geo.get_nodes()
            m00, m01, m02, m10, m11, m12, m20, m21, m22, i1 = \
                point_force_matrix_3d_petsc_mij(u_nodes[i0], f_nodes, i0)
            t1 = m00 * rot_M[0][0] + m01 * rot_M[1][0] + m02 * rot_M[2][0]
            m.setValues(u_glb + 0, f_glbIdx_all[0::3], t1, addv=True)
            t1 = m00 * rot_M[0][1] + m01 * rot_M[1][1] + m02 * rot_M[2][1]
            m.setValues(u_glb + 0, f_glbIdx_all[1::3], t1, addv=True)
            t1 = m00 * rot_M[0][2] + m01 * rot_M[1][2] + m02 * rot_M[2][2]
            m.setValues(u_glb + 0, f_glbIdx_all[2::3], t1, addv=True)
            t1 = m10 * rot_M[0][0] + m11 * rot_M[1][0] + m12 * rot_M[2][0]
            m.setValues(u_glb + 1, f_glbIdx_all[0::3], t1, addv=True)
            t1 = m10 * rot_M[0][1] + m11 * rot_M[1][1] + m12 * rot_M[2][1]
            m.setValues(u_glb + 1, f_glbIdx_all[1::3], t1, addv=True)
            t1 = m10 * rot_M[0][2] + m11 * rot_M[1][2] + m12 * rot_M[2][2]
            m.setValues(u_glb + 1, f_glbIdx_all[2::3], t1, addv=True)
            t1 = m20 * rot_M[0][0] + m21 * rot_M[1][0] + m22 * rot_M[2][0]
            m.setValues(u_glb + 2, f_glbIdx_all[0::3], t1, addv=True)
            t1 = m20 * rot_M[0][1] + m21 * rot_M[1][1] + m22 * rot_M[2][1]
            m.setValues(u_glb + 2, f_glbIdx_all[1::3], t1, addv=True)
            t1 = m20 * rot_M[0][2] + m21 * rot_M[1][2] + m22 * rot_M[2][2]
            m.setValues(u_glb + 2, f_glbIdx_all[2::3], t1, addv=True)
    m.assemble()
    return True


def self_rotate_3d_petsc(obj1: 'sf.StokesFlowObj',  # object contain velocity information
                         obj2: 'sf.StokesFlowObj',  # object contain force information
                         m, **kwargs):
    # _mij_dict = {
    #     'rs_selfRotate':    regularized_stokeslets_matrix_3d_petsc_mij,
    #     'lg_rs_selfRotate': legendre_regularized_stokeslets_matrix_3d_mij,
    #     'pf_selfRotate':    point_force_matrix_3d_petsc_mij,
    # }
    
    def _wrapper_regularized_stokeslets_matrix_3d_petsc_mij(u_nodes, f_nodes, i0, **kwargs):
        kwargs['i0'] = i0
        t1 = regularized_stokeslets_matrix_3d_petsc_mij(u_nodes, f_nodes, **kwargs)
        return t1
    
    def _wrapper_legendre_regularized_stokeslets_matrix_3d_mij(u_nodes, f_nodes, i0, **kwargs):
        kwargs['i0'] = i0
        t1 = legendre_regularized_stokeslets_matrix_3d_mij(u_nodes, f_nodes, **kwargs)
        [m00, m01, m02, m10, m11, m12, m20, m21, m22], i1 = t1
        return m00, m01, m02, m10, m11, m12, m20, m21, m22, i1
    
    def _wrapper_point_force_matrix_3d_petsc_mij(u_nodes, f_nodes, i0, **kwargs):
        kwargs['i0'] = i0
        t1 = point_force_matrix_3d_petsc_mij(u_nodes, f_nodes, **kwargs)
        return t1
    
    problem_norm = kwargs['problem_norm']
    problem_n_copy = kwargs['problem_n_copy']
    problem_center = kwargs['problem_center']
    matrix_method = kwargs['matrix_method']
    
    u_nodes = obj1.get_u_nodes()
    f_geo = obj2.get_f_geo()
    _, u_glbIdx_all = obj1.get_u_geo().get_glbIdx()
    _, f_glbIdx_all = obj2.get_f_geo().get_glbIdx()
    u_dmda = obj1.get_u_geo().get_dmda()
    delta_rot = 2 * np.pi / problem_n_copy
    tf_geo = f_geo.copy()
    tf_geo.node_rotation(problem_norm, -1 * delta_rot, rotation_origin=problem_center)
    
    if matrix_method == 'rs_selfRotate':
        epsilon = kwargs['epsilon']
        delta = epsilon * obj2.get_f_geo().get_deltaLength()
        delta_2 = delta ** 2  # correction factor
        kwargs['delta_2'] = delta_2
        mij_fun = _wrapper_regularized_stokeslets_matrix_3d_petsc_mij
    elif matrix_method == 'lg_rs_selfRotate':
        f_ds = obj2.get_f_geo().get_deltaLength()
        if np.array(f_ds).size == 1:
            f_ds = f_ds * np.ones(obj2.get_n_f_node())
        kwargs['f_ds'] = f_ds
        mij_fun = _wrapper_legendre_regularized_stokeslets_matrix_3d_mij
    elif matrix_method == 'pf_selfRotate':
        mij_fun = _wrapper_point_force_matrix_3d_petsc_mij
    else:
        raise Exception('matrix_method do NOT sufficient self_rotate_3d_petsc function. ')
    
    for i0 in range(u_dmda.getRanges()[0][0], u_dmda.getRanges()[0][1]):
        u_glb = u_glbIdx_all[i0 * 3]
        for thetai in np.linspace(0, 2 * np.pi, problem_n_copy, endpoint=False):
            rot_M = get_rot_matrix(problem_norm, thetai)
            tf_geo.node_rotation(problem_norm, delta_rot, rotation_origin=problem_center)
            f_nodes = tf_geo.get_nodes()
            m00, m01, m02, m10, m11, m12, m20, m21, m22, i1 = \
                mij_fun(u_nodes[i0], f_nodes, i0, **kwargs)
            t1 = m00 * rot_M[0][0] + m01 * rot_M[1][0] + m02 * rot_M[2][0]
            m.setValues(u_glb + 0, f_glbIdx_all[0::3], t1, addv=True)
            t1 = m00 * rot_M[0][1] + m01 * rot_M[1][1] + m02 * rot_M[2][1]
            m.setValues(u_glb + 0, f_glbIdx_all[1::3], t1, addv=True)
            t1 = m00 * rot_M[0][2] + m01 * rot_M[1][2] + m02 * rot_M[2][2]
            m.setValues(u_glb + 0, f_glbIdx_all[2::3], t1, addv=True)
            t1 = m10 * rot_M[0][0] + m11 * rot_M[1][0] + m12 * rot_M[2][0]
            m.setValues(u_glb + 1, f_glbIdx_all[0::3], t1, addv=True)
            t1 = m10 * rot_M[0][1] + m11 * rot_M[1][1] + m12 * rot_M[2][1]
            m.setValues(u_glb + 1, f_glbIdx_all[1::3], t1, addv=True)
            t1 = m10 * rot_M[0][2] + m11 * rot_M[1][2] + m12 * rot_M[2][2]
            m.setValues(u_glb + 1, f_glbIdx_all[2::3], t1, addv=True)
            t1 = m20 * rot_M[0][0] + m21 * rot_M[1][0] + m22 * rot_M[2][0]
            m.setValues(u_glb + 2, f_glbIdx_all[0::3], t1, addv=True)
            t1 = m20 * rot_M[0][1] + m21 * rot_M[1][1] + m22 * rot_M[2][1]
            m.setValues(u_glb + 2, f_glbIdx_all[1::3], t1, addv=True)
            t1 = m20 * rot_M[0][2] + m21 * rot_M[1][2] + m22 * rot_M[2][2]
            m.setValues(u_glb + 2, f_glbIdx_all[2::3], t1, addv=True)
    m.assemble()
    return True


def check_self_rotate_3d_petsc(**kwargs):
    err_msg = 'the self_rotate method needs parameter, problem_norm. '
    assert 'problem_norm' in kwargs, err_msg
    err_msg = 'the self_rotate method needs parameter, problem_n_copy. '
    assert 'problem_n_copy' in kwargs, err_msg
    
    matrix_method = kwargs['matrix_method']
    if matrix_method == 'rs_selfRotate':
        check_regularized_stokeslets_matrix_3d(**kwargs)
    elif matrix_method == 'lg_rs_selfRotate':
        check_legendre_regularized_stokeslets_matrix_3d(**kwargs)
    elif matrix_method == 'pf_selfRotate':
        check_point_force_matrix_3d_petsc(**kwargs)
    else:
        raise Exception('matrix_method do NOT sufficient self_rotate_3d_petsc function. ')
    
    return True


def forceSphere_2d_petsc(obj1: 'sf.StokesFlowObj',  # object contain velocity information
                         obj2: 'sf.StokesFlowObj',  # object contain force information
                         m, **kwargs):
    err_msg = 'current version, only support a single object during a simulation. '
    assert obj1 == obj2, err_msg
    
    problem = obj1.get_problem()  # type: sf.ForceSphere2DProblem
    ugeo = obj1.get_u_geo()  # type: geo.sphere_particle_2d
    NS = ugeo.get_n_nodes()  # NS: Total number of spheres. (总的球体数)
    sphere_R = ugeo.get_sphere_R()
    sphere_X = ugeo.get_nodes()
    diag_err = kwargs['diag_err']  # Avoiding errors introduced by nan values. (避免nan值引入的误差)
    rs2 = kwargs['rs2']
    sdis = kwargs['sdis']
    length = kwargs['length']
    width = kwargs['width']
    mu = kwargs['mu']
    
    Minf_petsc = m
    Rlub_petsc = problem.get_Rlub_petsc()
    Rtol_petsc = problem.get_Rtol_petsc()
    ptc_lub_list = problem.get_ptc_lub_list()
    lamb_inter_list = problem.get_lamb_inter_list()
    u_dmda = obj1.get_u_geo().get_dmda()
    fs2.M_R_petsc(Minf_petsc, Rlub_petsc, Rtol_petsc, u_dmda,
                  sphere_R, sphere_X, rs2, sdis, length, width, ptc_lub_list, lamb_inter_list,
                  mu=mu, diag_err=diag_err)
    # print()
    # print('ptc_lub_list')
    # print(ptc_lub_list)
    # print()
    # print('Minf_petsc')
    # Minf_petsc.view()
    # print()
    # print('Rlub_petsc')
    # Rlub_petsc.view()
    # print()
    # print('Rtol_petsc')
    # Rtol_petsc.view()
    return True


def check_forceSphere_2d_petsc(**kwargs):
    pass
    return True

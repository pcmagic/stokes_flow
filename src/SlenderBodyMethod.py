# coding=utf-8
# implimentation of slender body theories for lighthill and KRJ (KeKeller-Rubinow-Johnson)
# and, maybe, my improvement.

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from src import stokes_flow as sf
from src import geo
from tqdm import tqdm
from src.StokesFlowMethod import point_force_matrix_3d_petsc_mij
from scipy import interpolate, integrate, optimize, sparse


def _mij(dxi, dr2, t_m):
    dx0 = dxi[0]
    dx1 = dxi[1]
    dx2 = dxi[2]
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


def stokeslets_matrix_mij(u_node, f_s, f_node_fun: 'geo.slb_geo.xc_fun') -> np.ndarray:
    # local M matrix mij, with size 3*3
    f_node = f_node_fun(f_s)
    t_m = np.zeros((3, 3))
    dxi = u_node - f_node
    dr2 = np.sum(dxi ** 2, axis=0)
    t_m = _mij(dxi, dr2, t_m)
    return t_m


def stokeslets_matrix_mij2(u_nodes, f_s, fidx, f_node_fun: 'geo.slb_geo.xc_fun') -> np.ndarray:
    # mj = S(:, j), along u
    f_node = f_node_fun(f_s)
    t_m = np.zeros((3, u_nodes.size))
    dxi = (u_nodes - f_node).T
    dr2 = np.sum(dxi ** 2, axis=0)
    if np.isfinite(fidx):
        dr2[fidx] = np.inf
    t_m = _mij(dxi, dr2, t_m)
    return t_m


def mij_theta(u_node, f_s, f_theta, f_node_fun: 'geo.slb_geo.xs_fun') -> np.ndarray:
    # local M matrix mij, with size 3*3
    f_node = f_node_fun(f_s, f_theta)
    t_m = np.zeros((3, 3))
    dxi = (u_node - f_node).T
    dr2 = np.sum(dxi ** 2, axis=0)
    t_m = _mij(dxi, dr2, t_m)
    return t_m


# Slender Body Theory, part 1, nonlocal part,
#     version 3, locally integration of tsij, loop along j (force points).
def SLB_matrix_nonlocal_petsc(obj1: 'sf.StokesFlowObj',  # contain velocity information
                              obj2: 'sf.StokesFlowObj',  # contain force information
                              m, slb_epsabs=1e-200, slb_epsrel=1e-08,
                              slb_limit=10000, **kwargs):
    u_nodes = obj1.get_u_nodes()
    _, u_glbIdx_all = obj1.get_u_geo().get_glbIdx()
    _, f_glbIdx_all = obj2.get_f_geo().get_glbIdx()
    u_geo = obj1.get_u_geo()  # type: geo.slb_geo
    f_geo = obj2.get_f_geo()  # type: geo.slb_geo
    s_list = u_geo.s_list
    ds = f_geo.get_deltaLength()
    f_dmda = f_geo.get_dmda()
    i0_fct = 1 if obj1 is obj2 else np.nan

    warpper_mij2 = lambda s_f: stokeslets_matrix_mij2(u_nodes, s_f, i1, f_geo.xc_fun)
    for i0 in range(f_dmda.getRanges()[0][0], f_dmda.getRanges()[0][1]):
        i1 = i0_fct * i0
        s0 = s_list[i0]
        s_a = s0 - ds / 2
        s_b = s0 + ds / 2
        tsij = integrate.quad_vec(warpper_mij2, s_a, s_b, epsabs=slb_epsabs, epsrel=slb_epsrel,
                                  limit=slb_limit, )[0]
        f_glb = f_glbIdx_all[i0 * 3]
        m.setValues(u_glbIdx_all, f_glb + 0, tsij[0], addv=False)
        m.setValues(u_glbIdx_all, f_glb + 1, tsij[1], addv=False)
        m.setValues(u_glbIdx_all, f_glb + 2, tsij[2], addv=False)
    m.assemble()
    return m


# Slender Body Theory, part 1, nonlocal part,
#     version 3, locally integration of tsij, loop along j (force points).
def mdf_SLB_matrix_nonlocal_petsc(obj1: 'sf.StokesFlowObj',  # contain velocity information
                                  obj2: 'sf.StokesFlowObj',  # contain force information
                                  m, slb_epsabs=1e-200, slb_epsrel=1e-08,
                                  slb_limit=10000, **kwargs):
    assert 1 == 2
    u_nodes = obj1.get_u_nodes()
    _, u_glbIdx_all = obj1.get_u_geo().get_glbIdx()
    _, f_glbIdx_all = obj2.get_f_geo().get_glbIdx()
    u_geo = obj1.get_u_geo()  # type: geo.slb_geo
    f_geo = obj2.get_f_geo()  # type: geo.slb_geo
    s_list = u_geo.s_list
    ds = f_geo.get_deltaLength()
    f_dmda = f_geo.get_dmda()
    i0_fct = 1 if obj1 is obj2 else np.nan

    warpper_mij2 = lambda s_f: stokeslets_matrix_mij2(u_nodes, s_f, i1, f_geo.xc_fun)
    for i0 in range(f_dmda.getRanges()[0][0], f_dmda.getRanges()[0][1]):
        i1 = i0_fct * i0
        s0 = s_list[i0]
        s_a = s0 - ds / 2
        s_b = s0 + ds / 2
        tsij = integrate.quad_vec(warpper_mij2, s_a, s_b, epsabs=slb_epsabs, epsrel=slb_epsrel,
                                  limit=slb_limit, )[0]
        f_glb = f_glbIdx_all[i0 * 3]
        m.setValues(u_glbIdx_all, f_glb + 0, tsij[0], addv=False)
        m.setValues(u_glbIdx_all, f_glb + 1, tsij[1], addv=False)
        m.setValues(u_glbIdx_all, f_glb + 2, tsij[2], addv=False)
    m.assemble()
    return m


# Lighthill Slender Body Theory, this version assert mesh size > local part size.
#     part 2, local part, version 3, locally integration of tsij.
def Lighthill_matrix_local_petsc(obj1: 'sf.StokesFlowObj',  # contain velocity information
                                obj2: 'sf.StokesFlowObj',  # contain force information
                                m, slb_epsabs=1e-200, slb_epsrel=1e-08,
                                slb_limit=10000, **kwargs):
    u_nodes = obj1.get_u_nodes()
    _, u_glbIdx_all = obj1.get_u_geo().get_glbIdx()
    # noinspection PyTypeChecker
    u_geo = obj1.get_u_geo()  # type: geo.slb_geo
    s_list = u_geo.s_list
    ds = u_geo.get_deltaLength()
    u_dmda = u_geo.get_dmda()

    # part 2, local part.
    warpper_mij = lambda f_node: stokeslets_matrix_mij(u_node, f_node, u_geo.xc_fun)
    for i0 in range(u_dmda.getRanges()[0][0], u_dmda.getRanges()[0][1]):
        u_node = u_nodes[i0]
        u_glb = u_glbIdx_all[i0 * 3] + np.array((0, 1, 2), dtype='int32')
        s0 = s_list[i0]
        s_a = s0 - ds / 2
        s_b = s0 + ds / 2
        use_cut = u_geo.natu_cut(s0)
        t1 = u_geo.fn_matrix(s0) / (4 * np.pi)
        t2 = integrate.quad_vec(warpper_mij, s0 + use_cut, s_b,
                                epsabs=slb_epsabs, epsrel=slb_epsrel, limit=slb_limit, )[0]
        t3 = integrate.quad_vec(warpper_mij, s_a, s0 - use_cut,
                                epsabs=slb_epsabs, epsrel=slb_epsrel, limit=slb_limit, )[0]
        m[u_glb, u_glb] = t1 + (t2 + t3)
    m.assemble()
    return m


# KRJ Slender Body Theory, part 2, local part.
def KRJ_matrix_local_petsc(obj1: 'sf.StokesFlowObj',  # contain velocity information
                           obj2: 'sf.StokesFlowObj',  # contain force information
                           m, slb_epsabs=1e-200, slb_epsrel=1e-08,
                           slb_limit=10000, dbg_Lsbt=False, **kwargs):
    tfct = 100  # ignore local singularity of self part.

    u_nodes = obj1.get_u_nodes()
    _, u_glbIdx_all = obj1.get_u_geo().get_glbIdx()
    # noinspection PyTypeChecker
    u_geo = obj1.get_u_geo()  # type: geo.slb_geo
    s_list = u_geo.s_list
    ds = u_geo.get_deltaLength()
    u_dmda = u_geo.get_dmda()

    l_min = s_list[0] - ds / 2
    l_max = s_list[-1] + ds / 2
    warpper_mij_self = lambda s_f: - tc / np.abs(s0 - s_f) + 8 * np.pi \
                                   * stokeslets_matrix_mij(u_node, s_f, u_geo.xc_fun)
    for i0 in range(u_dmda.getRanges()[0][0], u_dmda.getRanges()[0][1]):
        u_node = u_nodes[i0]
        u_glb = u_glbIdx_all[i0 * 3] + np.array((0, 1, 2), dtype='int32')
        s0 = s_list[i0]
        s_a = s0 - ds / 2
        s_b = s0 + ds / 2
        t = u_geo.t_fun(s0)
        rt2_use = u_geo.rt2 * u_geo.rho_r(s0)
        Lsbt = np.log(4 * (-l_max * l_min + (l_max + l_min) * s0 - s0 ** 2) / rt2_use ** 2)
        ta = np.eye(3)
        tb = np.outer(t, t)
        tc = ta + tb
        t1 = Lsbt * tc + ta - 3 * tb
        t2 = np.log((l_min - s0) / (s_a - s0))
        t3 = np.log((l_max - s0) / (s_b - s0))
        tint = (t2 + t3) * tc
        # tint_self = integrate.quad_vec(warpper_mij_self, s_a, s0 - ds / tfct,
        #                                epsabs=slb_epsabs, epsrel=slb_epsrel,
        #                                limit=slb_limit, )[0] + \
        #             integrate.quad_vec(warpper_mij_self, s0 + ds / tfct, s_b,
        #                                epsabs=slb_epsabs, epsrel=slb_epsrel,
        #                                limit=slb_limit, )[0]
        tint_self = 0
        m[u_glb, u_glb] = (t1 - tint - tint_self) / (8 * np.pi)
    m.assemble()
    return m


# modified KRJ Slender Body Theory, warpper function
def mod_KRJ_matrix_local_petsc(obj1: 'sf.StokesFlowObj',  # contain velocity information
                               obj2: 'sf.StokesFlowObj',  # contain force information
                               m, slb_epsabs=1e-200, slb_epsrel=1e-08,
                               slb_limit=10000, **kwargs):
    m = KRJ_matrix_local_petsc(obj1, obj2, m, slb_epsabs=slb_epsabs,
                               slb_epsrel=slb_epsrel, slb_limit=slb_limit, **kwargs)
    m = KRJ_matrix_neighbor_petsc(obj1, obj2, m, slb_epsabs=slb_epsabs,
                                  slb_epsrel=slb_epsrel, slb_limit=slb_limit, **kwargs)
    return m


# modified KRJ Slender Body Theory, part 3, neighbor part.
def KRJ_matrix_neighbor_petsc(obj1: 'sf.StokesFlowObj',  # contain velocity information
                              obj2: 'sf.StokesFlowObj',  # contain force information
                              m, slb_epsabs=1e-200, slb_epsrel=1e-08,
                              slb_limit=10000, neighbor_range=1, **kwargs):
    u_nodes = obj1.get_u_nodes()
    _, u_glbIdx_all = obj1.get_u_geo().get_glbIdx()
    _, f_glbIdx_all = obj2.get_f_geo().get_glbIdx()
    u_geo = obj1.get_u_geo()  # type: geo.slb_geo
    f_geo = obj2.get_f_geo()  # type: geo.slb_geo
    s_list = u_geo.s_list
    ds = f_geo.get_deltaLength()
    u_dmda = u_geo.get_dmda()

    _mij_int_theta = lambda s_f, theta_f: mij_theta(u_node, s_f, theta_f, f_geo.xs_fun)[idxa, idxb]
    for i0 in tqdm(range(u_dmda.getRanges()[0][0], u_dmda.getRanges()[0][1])):
        u_glb = u_glbIdx_all[i0 * 3] + np.array((0, 1, 2), dtype='int32')
        u_node = u_nodes[i0]
        t1 = np.max((i0 - neighbor_range, 0))
        t2 = np.min((i0 + neighbor_range, f_geo.get_n_nodes() - 1))
        neib_idx = np.arange(t1, t2 + 1, dtype='int')
        for i3 in neib_idx:
            if i0 == i3:
                continue
            f_glb = f_glbIdx_all[i3 * 3] + np.array((0, 1, 2), dtype='int32')
            tint_self = np.zeros((3, 3))
            s0 = s_list[i3]
            s_a = s0 - ds / 2
            s_b = s0 + ds / 2
            for idxa in range(3):
                for idxb in range(idxa, 3):
                    ti = integrate.dblquad(_mij_int_theta, s_a, s_b, -np.pi, np.pi,
                                           epsabs=slb_epsabs, epsrel=slb_epsrel)[0]
                    tint_self[idxa, idxb] = ti / (2 * np.pi)
                    tint_self[idxb, idxa] = ti / (2 * np.pi)
            m[u_glb, f_glb] = tint_self
    m.assemble()
    return m

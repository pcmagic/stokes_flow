# coding=utf-8
"""
functions for solving stokes flow using regularised stokeslets (and its improved) method.
Zhang Ji, 20160409
"""
# import sys
# sys.path = ['..'] + sys.path
# from memory_profiler import profile
# from math import sin, cos
# import warnings
from pyvtk import *
import os
import matplotlib.pyplot as plt
import copy
import numpy as np
from scipy.io import savemat, loadmat
from evtk.hl import pointsToVTK, gridToVTK
from petsc4py import PETSc
import pickle
from time import time
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook
from src.support_class import *
from src.geo import *
from src.ref_solution import *
from src.StokesFlowMethod import light_stokeslets_matrix_3d
from src import forceSphere2d as fs2
import itertools


class StokesFlowProblem:
    def _init_kwargs(self, **kwargs):
        pass
    
    def __init__(self, **kwargs):
        self._obj_list = uniqueList()  # contain objects
        self._kwargs = kwargs  # kwargs associate with solving method,
        self._init_kwargs(**kwargs)
        self._force = np.zeros([0])  # force information
        self._force_petsc = PETSc.Vec().create(comm=PETSc.COMM_WORLD)
        self._velocity_petsc = PETSc.Vec().create(comm=PETSc.COMM_WORLD)  # velocity information
        self._re_velocity = np.zeros([0])  # resolved velocity information
        self._n_fnode = 0
        self._n_unode = 0
        self._f_pkg = PETSc.DMComposite().create(comm=PETSc.COMM_WORLD)
        self._u_pkg = PETSc.DMComposite().create(comm=PETSc.COMM_WORLD)
        self._M_petsc = PETSc.Mat().create(comm=PETSc.COMM_WORLD)  # M matrix
        self._M_destroyed = False  # weather Mat().destroy() have been called.
        self._finish_solve = False
        self._pick_M = False  # weather save M matrix and F vector in binary files.
        self._n_unknown = 3  # number of unknowns at each  node.
        self._pick_filename = '..'  # prefix filename of pick files. i.e. filename_F.bin, filename_M.bin, filename_pick.bin.
        self._residualNorm = 0.
        self._convergenceHistory = np.zeros([0])
        
        from src import StokesFlowMethod
        self._method_dict = {
            'rs':                                 StokesFlowMethod.regularized_stokeslets_matrix_3d_petsc,
            'rs_plane':                           StokesFlowMethod.regularized_stokeslets_plane_matrix_3d_petsc,
            'tp_rs':                              StokesFlowMethod.two_para_regularized_stokeslets_matrix_3d,
            'lg_rs':                              StokesFlowMethod.legendre_regularized_stokeslets_matrix_3d,
            'pf':                                 StokesFlowMethod.point_force_matrix_3d_petsc,
            'pf_dualPotential':                   StokesFlowMethod.dual_potential_matrix_3d_petsc,
            'rs_stokesletsInPipe':                StokesFlowMethod.regularized_stokeslets_matrix_3d_petsc,
            'pf_stokesletsInPipe':                StokesFlowMethod.point_force_matrix_3d_petsc,
            'pf_stokesletsInPipeforcefree':       StokesFlowMethod.point_force_matrix_3d_petsc,
            'pf_stokesletsTwoPlane':              StokesFlowMethod.two_plane_matrix_3d_petsc,
            'pf_infhelix':                        StokesFlowMethod.pf_infhelix_3d_petsc,
            'pf_stokesletsRingInPipe':            StokesFlowMethod.point_force_matrix_3d_petsc,
            'pf_stokesletsRingInPipeProblemSymz': StokesFlowMethod.point_force_matrix_3d_petsc,
            'pf_stokesletsRing':                  StokesFlowMethod.point_force_ring_3d_petsc,
            'pf_selfRepeat':                      StokesFlowMethod.self_repeat_3d_petsc,
            'pf_selfRotate':                      StokesFlowMethod.self_rotate_3d_petsc,
            'rs_selfRotate':                      StokesFlowMethod.self_rotate_3d_petsc,
            'lg_rs_selfRotate':                   StokesFlowMethod.self_rotate_3d_petsc,
            'pf_sphere':                          StokesFlowMethod.pf_sphere_image_petsc,
            'forceSphere2d':                      StokesFlowMethod.forceSphere_2d_petsc,
            'forceSphere2d_simp':                 StokesFlowMethod.forceSphere_2d_simp_petsc,
            }
        self._check_args_dict = {
            'rs':                                 StokesFlowMethod.check_regularized_stokeslets_matrix_3d,
            'rs_plane':                           StokesFlowMethod.check_regularized_stokeslets_plane_matrix_3d,
            'tp_rs':                              StokesFlowMethod.check_two_para_regularized_stokeslets_matrix_3d,
            'lg_rs':                              StokesFlowMethod.check_legendre_regularized_stokeslets_matrix_3d,
            'pf':                                 StokesFlowMethod.check_point_force_matrix_3d_petsc,
            'pf_dualPotential':                   StokesFlowMethod.check_dual_potential_matrix_3d_petsc,
            'rs_stokesletsInPipe':                StokesFlowMethod.check_regularized_stokeslets_matrix_3d,
            'pf_stokesletsInPipe':                StokesFlowMethod.check_point_force_matrix_3d_petsc,
            'pf_stokesletsInPipeforcefree':       StokesFlowMethod.check_point_force_matrix_3d_petsc,
            'pf_stokesletsTwoPlane':              StokesFlowMethod.check_two_plane_matrix_3d_petsc,
            'pf_infhelix':                        StokesFlowMethod.check_pf_infhelix_3d_petsc,
            'pf_stokesletsRingInPipe':            StokesFlowMethod.check_point_force_matrix_3d_petsc,
            'pf_stokesletsRingInPipeProblemSymz': StokesFlowMethod.check_point_force_matrix_3d_petsc,
            'pf_stokesletsRing':                  StokesFlowMethod.check_point_force_matrix_3d_petsc,
            'pf_selfRepeat':                      StokesFlowMethod.check_point_force_matrix_3d_petsc,
            'pf_selfRotate':                      StokesFlowMethod.check_self_rotate_3d_petsc,
            'rs_selfRotate':                      StokesFlowMethod.check_self_rotate_3d_petsc,
            'lg_rs_selfRotate':                   StokesFlowMethod.check_self_rotate_3d_petsc,
            'pf_sphere':                          StokesFlowMethod.check_pf_sphere_image_petsc,
            'forceSphere2d':                      StokesFlowMethod.check_forceSphere_2d_petsc,
            'forceSphere2d_simp':                 StokesFlowMethod.check_forceSphere_2d_simp_petsc,
            }
    
    def _check_add_obj(self, obj):
        pass
    
    def add_obj(self, obj):
        """
        Add a new object to the problem.

        :type obj: StokesFlowObj
        :param obj: added object
        :return: none.
        """
        self._check_add_obj(obj)
        self._obj_list.append(obj)
        obj.set_index(self.get_n_obj())
        obj.set_problem(self)
        obj.set_matrix_method(self.get_matrix_method())
        self._f_pkg.addDM(obj.get_f_geo().get_dmda())
        self._u_pkg.addDM(obj.get_u_geo().get_dmda())
        self._n_fnode += obj.get_n_f_node()
        self._n_unode += obj.get_n_u_node()
        return True
    
    def do_solve_process(self, obj_list, pick_M=False):
        obj_tube = list(tube_flatten((obj_list,)))
        fileHandle = self._kwargs['fileHandle']
        
        for obj in obj_tube:
            self.add_obj(obj)
        if self._kwargs['pickProblem']:
            self.pickmyself(fileHandle, ifcheck=True)
        self.print_info()
        self.create_matrix()
        residualNorm = self.solve()
        # # dbg
        # self.saveM_mat(fileHandle)
        if self._kwargs['pickProblem']:
            self.pickmyself(fileHandle, pick_M=pick_M)
        return residualNorm
    
    def __repr__(self):
        return type(self).__name__
    
    def __str__(self):
        return self.get_name()
    
    def _create_matrix_obj(self, obj1, m_petsc, INDEX='', *args):
        # obj1 contain velocity information, obj2 contain force information
        kwargs = self.get_kwargs()
        n_obj = len(self.get_all_obj_list())
        for i0, obj2 in enumerate(self.get_all_obj_list()):
            kwargs['INDEX'] = ' %d/%d, ' % (i0 + 1, n_obj) + INDEX
            self._check_args_dict[obj2.get_matrix_method()](**kwargs)
            self._method_dict[obj2.get_matrix_method()](obj1, obj2, m_petsc, **kwargs)
        m_petsc.assemble()
        return True
    
    def updata_matrix(self, obj1, obj2, INDEX=''):
        # obj1 contain velocity information, obj2 contain force information
        kwargs = self._kwargs
        kwargs['INDEX'] = INDEX
        self._method_dict[obj2.get_matrix_method()](obj1, obj2, self._M_petsc, **kwargs)
        self._M_petsc.assemble()
        return True
    
    def _create_U(self):
        comm = PETSc.COMM_WORLD.tompi4py()
        rank = comm.Get_rank()
        velocity = self._u_pkg.createGlobalVector()
        velocity.zeroEntries()
        for i0, obj0 in enumerate(self.get_obj_list()):
            u0 = obj0.get_velocity()
            _, u_glbIdx_all = obj0.get_u_geo().get_glbIdx()
            if rank == 0:
                velocity[u_glbIdx_all] = u0[:]
        velocity.assemble()
        self._velocity_petsc = velocity
        return True
    
    def _create_F(self):
        self._force_petsc = self._f_pkg.createGlobalVector()
        self._force_petsc.set(0)
        return True
    
    def _set_glbIdx(self):
        # global index
        f_isglb = self._f_pkg.getGlobalISs()
        u_isglb = self._u_pkg.getGlobalISs()
        # for i0, obj0 in enumerate(self.get_obj_list()):
        #     obj0.get_f_geo().set_glbIdx(f_isglb[i0].getIndices())
        #     obj0.get_u_geo().set_glbIdx(u_isglb[i0].getIndices())
        for obj0, t_f_isglb, t_u_isglb in zip(self.get_all_obj_list(), f_isglb, u_isglb):
            obj0.get_f_geo().set_glbIdx(t_f_isglb.getIndices())
            obj0.get_u_geo().set_glbIdx(t_u_isglb.getIndices())
        return True
    
    def create_F_U(self):
        # create f and u DMComposite
        self._f_pkg.setFromOptions()
        self._f_pkg.setUp()
        self._u_pkg.setFromOptions()
        self._u_pkg.setUp()
        #
        self._set_glbIdx()
        self._create_U()
        self._create_F()
        return True
    
    def _check_create_empty_M(self):
        kwargs = self._kwargs
        solve_method = kwargs['solve_method']
        err_msg = 'at least one object is necessary. '
        assert len(self._obj_list) > 0, err_msg
        err_msg = 'unequal force and velocity degrees of freedom, only lsqr method is accept. '
        for obj1 in self.get_all_obj_list():
            assert obj1.get_n_force() == obj1.get_n_velocity() or solve_method == 'lsqr', err_msg
        return True
    
    def create_empty_M(self):
        self._check_create_empty_M()
        
        # create matrix
        self._M_petsc.setSizes((self._velocity_petsc.getSizes(), self._force_petsc.getSizes()))
        self._M_petsc.setType('dense')
        self._M_petsc.setFromOptions()
        self._M_petsc.setUp()
        return self._M_petsc
    
    def create_matrix(self):
        t0 = time()
        self.create_F_U()
        if not self._M_petsc.isAssembled():
            self.create_empty_M()
            self._M_destroyed = False
        n_obj = len(self.get_all_obj_list())
        for i0, obj1 in enumerate(self.get_all_obj_list()):
            INDEX = ' %d/%d' % (i0 + 1, n_obj)
            self._create_matrix_obj(obj1, self._M_petsc, INDEX)
        # self._M_petsc.view()
        t1 = time()
        PETSc.Sys.Print('  %s: create matrix use: %fs' % (str(self), (t1 - t0)))
        return True
    
    def set_matrix(self, m_petsc):
        self.create_F_U()
        self._M_petsc = m_petsc
        return True
    
    def create_part_matrix(self, uobj_old, fobj_old, uobj_new, fobj_new, M):
        # assuming the problem have n+1 objects named obj_0, obj_1, ... obj_n. After create the M matrix of the problem
        #   we may need to create a mini M matrix associated with part of objects {obj_k, k in [0, n]}, this method get
        #   values of mini_M matrix from the main M matrix of the problem, to save the creating time.
        """
        :type uobj_old: StokesFlowObj
        :type uobj_new: StokesFlowObj
        :type fobj_old: StokesFlowObj
        :type fobj_new: StokesFlowObj
        :param uobj_old:
        :param fobj_old:
        :param uobj_new:
        :param fobj_new:
        :param M:
        :return:
        """
        err_msg = 'uobj_old and uobj_new are not same. '
        assert (uobj_old.get_u_geo().get_nodes() == uobj_new.get_u_geo().get_nodes()).all(), err_msg
        err_msg = 'fobj_old and fobj_new are not same. '
        assert (fobj_old.get_f_geo().get_nodes() == fobj_new.get_f_geo().get_nodes()).all(), err_msg
        u_glbIdx_old, u_glbIdx_all_old = uobj_old.get_u_geo().get_glbIdx()
        f_glbIdx_old, f_glbIdx_all_old = fobj_old.get_f_geo().get_glbIdx()
        _, u_glbIdx_all_new = uobj_new.get_u_geo().get_glbIdx()
        _, f_glbIdx_all_new = fobj_new.get_f_geo().get_glbIdx()
        
        t_Idx = np.searchsorted(u_glbIdx_all_old, u_glbIdx_old)
        u_glbIdx_new = u_glbIdx_all_new[t_Idx]
        temp0 = self._M_petsc.getValues(u_glbIdx_old, f_glbIdx_all_old)
        M.setValues(u_glbIdx_new, f_glbIdx_all_new, temp0, addv=False)
        M.assemble()
        return True
    
    def create_obj_matrix(self, objf: 'StokesFlowObj',  # force object
                          obju: 'StokesFlowObj',  # velocity object
                          copy_obj=True,
                          **kwargs):
        if copy_obj:
            obj1 = objf.copy()
            obj2 = obju.copy()
        else:
            obj1 = objf
            obj2 = obju
        t_f_pkg = PETSc.DMComposite().create(comm=PETSc.COMM_WORLD)
        t_u_pkg = PETSc.DMComposite().create(comm=PETSc.COMM_WORLD)
        t_f_pkg.addDM(obj1.get_f_geo().get_dmda())
        t_u_pkg.addDM(obj2.get_u_geo().get_dmda())
        t_f_pkg.setFromOptions()
        t_f_pkg.setUp()
        t_u_pkg.setFromOptions()
        t_u_pkg.setUp()
        f_isglb = t_f_pkg.getGlobalISs()
        u_isglb = t_u_pkg.getGlobalISs()
        obj1.get_f_geo().set_glbIdx(f_isglb[0].getIndices())
        obj2.get_u_geo().set_glbIdx(u_isglb[0].getIndices())
        t_velocity = t_u_pkg.createGlobalVector()
        t_force = t_f_pkg.createGlobalVector()
        m_petsc = PETSc.Mat().create(comm=PETSc.COMM_WORLD)
        m_petsc.setSizes((t_velocity.getSizes(), t_force.getSizes()))
        m_petsc.setType('dense')
        m_petsc.setFromOptions()
        m_petsc.setUp()
        self._method_dict[kwargs['matrix_method']](obj2, obj1, m_petsc, **kwargs)
        m_petsc.assemble()
        t_velocity.destroy()
        t_force.destroy()
        return m_petsc
    
    def solve(self, ini_guess=None):
        t0 = time()
        kwargs = self._kwargs
        solve_method = kwargs['solve_method']
        precondition_method = kwargs['precondition_method']
        
        if ini_guess is not None:
            err_msg = 'size of initial guess for force vector must equal to the number of M matrix rows. '
            assert self._force_petsc.getSize() == ini_guess.getSize(), err_msg
            self._force_petsc[:] = ini_guess[:]
        
        ksp = PETSc.KSP()
        ksp.create(comm=PETSc.COMM_WORLD)
        ksp.setType(solve_method)
        ksp.getPC().setType(precondition_method)
        ksp.setOperators(self._M_petsc)
        OptDB = PETSc.Options()
        ksp.setFromOptions()
        # Todo: dbg_GMRESRestart
        if not OptDB.getBool('debug', False):
            tolerance = ksp.getTolerances()
            ksp.setGMRESRestart(tolerance[-1])
        ksp.setInitialGuessNonzero(True)
        ksp.setUp()
        
        self._solve_force(ksp)
        self._residualNorm = self._resolve_velocity(ksp)
        ksp.destroy()
        
        t1 = time()
        PETSc.Sys.Print('  %s: solve matrix equation use: %fs, with residual norm %e' %
                        (str(self), (t1 - t0), self._residualNorm))
        return self._residualNorm
    
    def _solve_force(self, ksp):
        if self._kwargs['getConvergenceHistory']:
            ksp.setConvergenceHistory()
            ksp.solve(self._velocity_petsc, self._force_petsc)
            self._convergenceHistory = ksp.getConvergenceHistory()
        else:
            ksp.solve(self._velocity_petsc, self._force_petsc)
        
        # reorder force from petsc index to normal index, and separate to each object.
        t_force = self.vec_scatter(self._force_petsc, destroy=False)
        tmp = []
        for obj0 in self.get_obj_list():
            _, f_glbIdx_all = obj0.get_f_geo().get_glbIdx()
            obj0.set_force(t_force[f_glbIdx_all])
            tmp.append(t_force[f_glbIdx_all])
        self._force = np.hstack(tmp)
        return True
    
    def _resolve_velocity(self, ksp):
        re_velocity_petsc = self._M_petsc.createVecLeft()
        self._M_petsc.mult(self._force_petsc, re_velocity_petsc)
        self._re_velocity = self.vec_scatter(re_velocity_petsc)
        for obj0 in self.get_all_obj_list():
            _, u_glbIdx_all = obj0.get_u_geo().get_glbIdx()
            obj0.set_re_velocity(self._re_velocity[u_glbIdx_all])
        self._finish_solve = True
        return ksp.getResidualNorm()
    
    def solve_obj_u(self, obj: 'StokesFlowObj', INDEX=''):
        """
        solve velocity for given object.
        """
        self.check_finish_solve()
        kwargs = self._kwargs
        n_node_threshold = kwargs['n_node_threshold']
        
        # partition object into several parts if it contain too many nodes; and then solve in a loop.
        sub_obj_list = uniqueList()
        n_obj_nodes = obj.get_n_u_node()
        n_sub_obj = int(obj.get_n_u_node() / n_node_threshold) + 1
        obj_nodes = obj.get_u_nodes()
        for i0 in range(n_sub_obj):
            sub_obj1 = obj_dic[self.get_kwargs()['matrix_method']]()
            sub_geo1 = base_geo()
            id0 = i0 * n_obj_nodes // n_sub_obj
            id1 = (i0 + 1) * n_obj_nodes // n_sub_obj
            sub_geo1.set_dof(sub_obj1.get_n_unknown())
            sub_geo1.set_nodes(obj_nodes[id0:id1], resetVelocity=True,
                               deltalength=obj.get_u_geo().get_deltaLength())
            sub_obj_kwargs = {
                'name': '%s_sub_%d' % (str(obj), i0)
                }
            sub_obj1.set_data(sub_geo1, sub_geo1, **sub_obj_kwargs)
            sub_obj_list.append(sub_obj1)
        
        obj_u = obj.get_velocity().copy()
        n_obj = len(sub_obj_list)
        for i1, sub_obj1 in enumerate(sub_obj_list):
            sub_u_dmda = sub_obj1.get_u_geo().get_dmda()
            sub_u_pkg = PETSc.DMComposite().create(comm=PETSc.COMM_WORLD)
            sub_u_pkg.addDM(sub_u_dmda)
            sub_u_pkg.setFromOptions()
            sub_u_pkg.setUp()
            sub_u_isglb = sub_u_pkg.getGlobalISs()
            sub_obj_u_petsc = sub_u_dmda.createGlobalVector()
            sub_obj1.get_u_geo().set_glbIdx(sub_u_isglb[0].getIndices())
            m_petsc = PETSc.Mat().create(comm=PETSc.COMM_WORLD)
            m_petsc.setSizes((sub_obj_u_petsc.getSizes(), self._force_petsc.getSizes()))
            m_petsc.setType('dense')
            m_petsc.setFromOptions()
            m_petsc.setUp()
            INDEX = ' %d/%d, ' % (i1 + 1, n_obj) + INDEX
            self._create_matrix_obj(sub_obj1, m_petsc, INDEX)
            # sub_obj_u_petsc.set(0)
            m_petsc.mult(self._force_petsc, sub_obj_u_petsc)
            sub_obj_u = self.vec_scatter(sub_obj_u_petsc)
            id0 = i1 * n_obj_nodes // n_sub_obj * 3
            id1 = (i1 + 1) * n_obj_nodes // n_sub_obj * 3
            obj_u[id0 + 0:id1:3] = sub_obj_u[0::self.get_n_unknown()]
            obj_u[id0 + 1:id1:3] = sub_obj_u[1::self.get_n_unknown()]
            obj_u[id0 + 2:id1:3] = sub_obj_u[2::self.get_n_unknown()]
            m_petsc.destroy()
            sub_u_pkg.destroy()
            sub_obj1.get_u_geo().destroy_dmda()
        return obj_u
    
    def vtk_check(self, filename: str, obj: 'StokesFlowObj', ref_slt=None):
        self.check_finish_solve()
        obj_tube = list(tube_flatten((obj,)))
        err = []
        for obj in obj_tube:
            if isinstance(obj, StokesFlowObj):
                err.append(self._vtk_check(filename + '_' + str(obj) + '_check',
                                           obj, ref_slt=ref_slt, INDEX=str(obj)))
            else:
                err_msg = 'unknown obj type. '
                raise err_msg
        return tube_flatten((err,))
    
    def _vtk_check(self, filename: str, obj: "StokesFlowObj", ref_slt: "slt" = None, INDEX=''):
        """
        check velocity at the surface of objects.

        :type filename: str
        :param filename: output file name
        :type obj: StokesFlowObj
        :param obj: check object (those known exact velocity information. )
        :param ref_slt: reference solution function handle
        :return: none.
        """
        comm = PETSc.COMM_WORLD.tompi4py()
        rank = comm.Get_rank()
        
        obj_u = self.solve_obj_u(obj, INDEX)
        obj.set_re_velocity(obj_u)
        if ref_slt is None:
            u_exact = obj.get_velocity()
        else:
            u_exact = ref_slt.get_solution(obj.get_u_geo())
        
        if rank == 0:
            velocity_x = obj_u[0::3].copy()
            velocity_y = obj_u[1::3].copy()
            velocity_z = obj_u[2::3].copy()
            velocity_err = u_exact - obj_u
            velocity_err_x = velocity_err[0::3].copy()
            velocity_err_y = velocity_err[1::3].copy()
            velocity_err_z = velocity_err[2::3].copy()
            rel_err_x = np.abs(velocity_err_x / velocity_x)
            rel_err_y = np.abs(velocity_err_y / velocity_y)
            rel_err_z = np.abs(velocity_err_z / velocity_z)
            nodes = obj.get_u_nodes()
            pointsToVTK(filename, nodes[:, 0].copy(), nodes[:, 1].copy(), nodes[:, 2].copy(),
                        data={
                            "velocity_err": (velocity_err_x, velocity_err_y, velocity_err_z),
                            "velocity":     (velocity_x, velocity_y, velocity_z),
                            "rel_err":      (rel_err_x, rel_err_y, rel_err_z)
                            })
        
        errorall = np.sqrt(np.sum((obj_u - u_exact) ** 2) / np.sum(u_exact ** 2))
        errorx = np.sqrt(np.sum((obj_u[0::3] - u_exact[0::3]) ** 2) / np.sum(u_exact[0::3] ** 2))
        errory = np.sqrt(np.sum((obj_u[1::3] - u_exact[1::3]) ** 2) / np.sum(u_exact[1::3] ** 2))
        errorz = np.sqrt(np.sum((obj_u[2::3] - u_exact[2::3]) ** 2) / np.sum(u_exact[2::3] ** 2))
        error = np.hstack((errorall, errorx, errory, errorz))
        
        return error
    
    def check_vtk_velocity(self):
        field_range = self._kwargs['field_range']
        n_grid = self._kwargs['n_grid']
        
        n_range = field_range.shape
        if n_range[0] > n_range[1]:
            field_range = field_range.transpose()
            n_range = field_range.shape
        if n_range != (2, 3):
            err_msg = 'maximum and minimum coordinates for the rectangular velocity field are necessary, ' + \
                      'i.e. range = [[0,0,0],[10,10,10]]. '
            raise ValueError(err_msg)
        self.check_finish_solve()
        n_grid = n_grid.ravel()
        if n_grid.shape != (3,):
            err_msg = 'mesh number of each axis for the rectangular velocity field is necessary, ' + \
                      'i.e. n_grid = [100, 100, 100]. '
            raise ValueError(err_msg)
        
        return field_range, n_grid
    
    def vtk_velocity(self, filename: str):
        t0 = time()
        self.check_finish_solve()
        comm = PETSc.COMM_WORLD.tompi4py()
        rank = comm.Get_rank()
        field_range, n_grid = self.check_vtk_velocity()
        region_type = self._kwargs['region_type']
        
        myregion = region()
        full_region_x, full_region_y, full_region_z = \
            myregion.type[region_type](field_range, n_grid)
        
        # to handle big problem, solve velocity field at every splice along x axis.
        if rank == 0:
            u_x = np.zeros((n_grid[0], n_grid[1], n_grid[2]))
            u_y = np.zeros((n_grid[0], n_grid[1], n_grid[2]))
            u_z = np.zeros((n_grid[0], n_grid[1], n_grid[2]))
        else:
            u_x = None
            u_y = None
            u_z = None
        obj0 = StokesFlowObj()
        for i0 in range(full_region_x.shape[0]):
            temp_x = full_region_x[i0]
            temp_y = full_region_y[i0]
            temp_z = full_region_z[i0]
            temp_nodes = np.c_[temp_x.ravel(), temp_y.ravel(), temp_z.ravel()]
            temp_geo = base_geo()
            temp_geo.set_nodes(temp_nodes, resetVelocity=True, deltalength=0)
            obj0.set_data(temp_geo, temp_geo)
            u = self.solve_obj_u(obj0)
            if rank == 0:
                u_x[i0, :, :] = u[0::3].reshape((n_grid[1], n_grid[2]))
                u_y[i0, :, :] = u[1::3].reshape((n_grid[1], n_grid[2]))
                u_z[i0, :, :] = u[2::3].reshape((n_grid[1], n_grid[2]))
            else:
                u_x = None
                u_y = None
                u_z = None
        
        if rank == 0:
            # output data
            gridToVTK(filename, full_region_x, full_region_y, full_region_z,
                      pointData={
                          "velocity": (u_x, u_y, u_z)
                          })
        t1 = time()
        PETSc.Sys.Print('%s: write vtk files of surrounding velocity use: %fs'
                        % (str(self), (t1 - t0)))
        return True
    
    def vtk_self(self, filename, stp_idx=0):
        t0 = time()
        self.check_finish_solve()
        obj0 = obj_dic[self.get_kwargs()['matrix_method']]()
        obj0.combine(self.get_all_obj_list(), set_re_u=True, set_force=True)
        obj0.set_name('Prb')
        obj0.set_matrix_method(self.get_kwargs()['matrix_method'])
        # self.show_velocity()
        # obj0.show_velocity()
        obj0.vtk(filename, stp_idx)
        t1 = time()
        PETSc.Sys.Print('  %s: write self vtk files use: %fs' % (str(self), (t1 - t0)))
        return True
    
    def vtk_obj(self, filename, stp_idx=0):
        self.check_finish_solve()
        for obj1 in self._obj_list:
            obj1.vtk(filename, stp_idx)
        return True
    
    def vtk_tetra(self, filename: str, bgeo: base_geo):
        comm = PETSc.COMM_WORLD.tompi4py()
        rank = comm.Get_rank()
        t0 = time()
        self.check_finish_solve()
        
        bnodes = bgeo.get_nodes()
        belems, elemtype = bgeo.get_mesh()
        err_msg = 'mesh type is NOT tetrahedron. '
        assert elemtype == 'tetra', err_msg
        
        obj1 = StokesFlowObj()
        obj1.set_data(bgeo, bgeo)
        u = self.solve_obj_u(obj1)
        if rank == 0:
            u = np.array(u).reshape(bnodes.shape)
            vtk = VtkData(UnstructuredGrid(bnodes, tetra=belems, ),
                          PointData(Vectors(u, name='u')),
                          str(self))
            vtk.tofile(filename)
        
        t1 = time()
        PETSc.Sys.Print(
                'export to %s.vtk, element type is %s, contain %d nodes and %d elements using %fs. '
                % (filename, elemtype, bnodes.shape[0], belems.shape[0], (t1 - t0)))
        return True
    
    def saveM_ASCII(self, filename: str = '..', ):
        if filename[-4:] != '.txt':
            filename = filename + '.txt'
        err_msg = 'M matrix is been destroyed. '
        assert not self._M_destroyed, err_msg
        viewer = PETSc.Viewer().createASCII(filename, 'w', comm=PETSc.COMM_WORLD)
        viewer(self._M_petsc)
        viewer.destroy()
        PETSc.Sys.Print('%s: save M matrix to %s' % (str(self), filename))
        return True
    
    def saveM_HDF5(self, filename: str = '..', ):
        if filename[-3:] != '.h5':
            filename = filename + '.h5'
        err_msg = 'M matrix is been destroyed. '
        assert not self._M_destroyed, err_msg
        viewer = PETSc.Viewer().createHDF5(filename, 'w', comm=PETSc.COMM_WORLD)
        viewer(self._M_petsc)
        viewer.destroy()
        PETSc.Sys.Print('%s: save M matrix to %s' % (str(self), filename))
        return True
    
    def _save_M_mat_dict(self, M_dict, obj):
        t_name_all = str(obj) + '_Idx_all'
        t_name = str(obj) + '_Idx'
        u_glbIdx, u_glbIdx_all = obj.get_u_geo().get_glbIdx()
        M_dict[t_name_all] = u_glbIdx_all
        M_dict[t_name] = u_glbIdx
        return True
    
    def saveM_mat(self, filename: str = '..', M_name='M'):
        if filename[-4:] == '.mat':
            filename = filename[:-4]
        comm = PETSc.COMM_WORLD.tompi4py()
        rank = comm.Get_rank()
        size = comm.Get_size()
        err_msg = 'M matrix is been destroyed. '
        assert not self._M_destroyed, err_msg
        
        M_dict = {
            M_name: self._M_petsc.getDenseArray(),
            }
        for obj in self.get_obj_list():
            self._save_M_mat_dict(M_dict, obj)
        
        savemat(filename + '_rank%03d.mat' % rank,
                M_dict,
                oned_as='column')
        PETSc.Sys.Print(
                '%s: save M matrix to %s_rank(%03d~%03d).mat' % (str(self), filename, 0, size))
        return True
    
    def saveF_ASCII(self, filename: str = '..', ):
        if filename[-4:] != '.txt':
            filename = filename + '.txt'
        viewer = PETSc.Viewer().createASCII(filename, 'w', comm=PETSc.COMM_WORLD)
        viewer(self._force_petsc)
        viewer.destroy()
        PETSc.Sys.Print('%s: save force to %s' % (str(self), filename))
        return True
    
    def saveV_ASCII(self, filename: str = '..', ):
        if filename[-4:] != '.txt':
            filename = filename + '.txt'
        viewer = PETSc.Viewer().createASCII(filename, 'w', comm=PETSc.COMM_WORLD)
        viewer(self._velocity_petsc)
        viewer.destroy()
        PETSc.Sys.Print('%s: save velocity to %s' % (str(self), filename))
        return True
    
    def saveF_Binary(self, filename: str = '..', ):
        if filename[-4:] != '.bin':
            filename = filename + '.bin'
        viewer = PETSc.Viewer().createBinary(filename, 'w', comm=PETSc.COMM_WORLD)
        viewer(self._force_petsc)
        viewer.destroy()
        PETSc.Sys.Print('%s: save force to %s' % (str(self), filename))
        return True
    
    def saveV_Binary(self, filename: str = '..', ):
        if filename[-4:] != '.bin':
            filename = filename + '.bin'
        viewer = PETSc.Viewer().createBinary(filename, 'w', comm=PETSc.COMM_WORLD)
        viewer(self._velocity_petsc)
        viewer.destroy()
        PETSc.Sys.Print('%s: save velocity to %s' % (str(self), filename))
        return True
    
    def saveM_Binary(self, filename: str = '..', ):
        if filename[-4:] != '.bin':
            filename = filename + '.bin'
        err_msg = 'M matrix is been destroyed. '
        assert not self._M_destroyed, err_msg
        
        viewer = PETSc.Viewer().createBinary(filename, 'w', comm=PETSc.COMM_WORLD)
        viewer.pushFormat(viewer.Format.NATIVE)
        viewer(self._M_petsc)
        viewer.destroy()
        PETSc.Sys.Print('%s: save M matrix to %s' % (str(self), filename))
        return True
    
    def loadM_Binary(self, filename: str):
        if filename[-4:] != '.bin':
            filename = filename + '.bin'
        viewer = PETSc.Viewer().createBinary(filename, 'r')
        self._M_petsc = PETSc.Mat().create(comm=PETSc.COMM_WORLD)
        self._M_petsc.setSizes((self._velocity_petsc.getSizes(), self._force_petsc.getSizes()))
        self._M_petsc.setType('dense')
        self._M_petsc.setFromOptions()
        self._M_petsc = self._M_petsc.load(viewer)
        return True
    
    def mat_destroy(self):
        if not self._M_destroyed:
            self._M_petsc.destroy()
            self._M_destroyed = True
            self._M_petsc = None
            return True
        else:
            return False
    
    def destroy(self):
        self._force_petsc.destroy()
        self._velocity_petsc.destroy()
        self._f_pkg.destroy()
        self._u_pkg.destroy()
        self._f_pkg = None
        self._u_pkg = None
        self._force_petsc = None
        self._velocity_petsc = None
        return True
    
    def pickmyself_prepare(self):
        self.destroy()
        for obji in self.get_obj_list():
            obji.pickmyself_prepare()
        return True
    
    def pickmyself(self, filename: str, ifcheck=False, pick_M=False, unpick=True,
                   mat_destroy=True):
        t0 = time()
        comm = PETSc.COMM_WORLD.tompi4py()
        rank = comm.Get_rank()
        self._pick_filename = filename
        
        if (not ifcheck) and pick_M:
            self._pick_M = pick_M
            err_msg = 'self._finish_solve=%s' % self._finish_solve
            assert self._finish_solve, err_msg
            self.saveM_Binary(filename + '_M')
        
        self.pickmyself_prepare()
        if mat_destroy:
            self.mat_destroy()
        else:
            t_M_petsc = self._M_petsc
            self._M_destroyed = True
            self._M_petsc = None
        if rank == 0:
            with open(filename + '_pick.bin', 'wb') as output:
                pickler = pickle.Pickler(output, -1)
                pickler.dump(self)
        
        if unpick or ifcheck:
            self.unpick_myself()
        if not mat_destroy:
            self._M_destroyed = False
            self._M_petsc = t_M_petsc
        t1 = time()
        PETSc.Sys().Print('%s: pick the problem use: %fs' % (str(self), (t1 - t0)))
        return True
    
    def _unpick_addDM(self, obj1):
        self._f_pkg.addDM(obj1.get_f_geo().get_dmda())
        self._u_pkg.addDM(obj1.get_u_geo().get_dmda())
        return True
    
    def _unpick_set_force(self):
        f_numpy = []
        f_glbIdx = []
        for obj0 in self.get_obj_list():
            if isinstance(obj0, ForceFreeComposite):
                for sub_obj in obj0.get_obj_list():
                    _, f_glbIdx_all = sub_obj.get_f_geo().get_glbIdx()
                    f_numpy.append(sub_obj.get_force())
                    f_glbIdx.append(f_glbIdx_all)
                _, f_glbIdx_all = obj0.get_f_glbIdx()
                f_numpy.append(obj0.get_ref_U())
                f_glbIdx.append(f_glbIdx_all)
            else:
                _, f_glbIdx_all = obj0.get_f_geo().get_glbIdx()
                f_numpy.append(obj0.get_force())
                f_glbIdx.append(f_glbIdx_all)
        f_numpy = np.hstack(f_numpy)
        f_glbIdx = np.hstack(f_glbIdx)
        self._force_petsc[f_glbIdx] = f_numpy[:]
        self._force_petsc.assemble()
        return True
    
    def unpick_myself(self, check_MPISIZE=True):
        filename = self._pick_filename
        OptDB = PETSc.Options()
        kwargs = self._kwargs
        comm = PETSc.COMM_WORLD.tompi4py()
        MPISIZE = comm.Get_size()
        
        err_msg = 'call pickmyself() before unpick_myself(). i.e. store date first and reload them at restart mode. '
        assert filename != '..', err_msg
        if OptDB.getBool('check_MPISIZE', True) and check_MPISIZE:
            err_msg = 'problem was picked with MPI size %d, current MPI size %d is wrong. ' \
                      % (kwargs['MPISIZE'], MPISIZE,)
            assert kwargs['MPISIZE'] == MPISIZE, err_msg
        else:
            PETSc.Sys.Print('-->Warning, make sure the mpi size %d is correct. ' % MPISIZE)
        
        self._f_pkg = PETSc.DMComposite().create(comm=PETSc.COMM_WORLD)
        self._u_pkg = PETSc.DMComposite().create(comm=PETSc.COMM_WORLD)
        self._M_petsc = PETSc.Mat().create(comm=PETSc.COMM_WORLD)  # M matrix
        for obj1 in self.get_obj_list():
            obj1.unpick_myself()
            self._unpick_addDM(obj1)
        self._f_pkg.setFromOptions()
        self._u_pkg.setFromOptions()
        # Todo: setUp f_pkg and u_pkg at a appropriate time
        # self._f_pkg.setUp()
        # self._u_pkg.setUp()
        
        if self._finish_solve:
            self.create_F_U()
            self._unpick_set_force()
            # self._force_petsc.view()
        if self._finish_solve and self._pick_M:
            self.loadM_Binary(filename + '_M')
        PETSc.Sys.Print('Unpick the problem from %s. ' % filename)
        return True
    
    def view_log_M(self, **kwargs):
        m = self._M_petsc.getDenseArray()
        view_args = {
            'vmin':  -10,
            'vmax':  0,
            'title': 'log10_abs_' + kwargs['matrix_method'],
            'cmap':  'gray'
            }
        self._view_matrix(np.log10(np.abs(m) + 1e-100), **view_args)
    
    def view_M(self, **kwargs):
        m = self._M_petsc.getDenseArray()
        view_args = {
            'vmin':  None,
            'vmax':  None,
            'title': kwargs['matrix_method'],
            'cmap':  'gray'
            }
        self._view_matrix(m, **view_args)
    
    @staticmethod
    def _view_matrix(m, **kwargs):
        args = {
            'vmin':  None,
            'vmax':  None,
            'title': ' ',
            'cmap':  None
            }
        for key, value in args.items():
            if key in kwargs:
                args[key] = kwargs[key]
        
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        cax = ax.matshow(m,
                         origin='lower',
                         vmin=args['vmin'],
                         vmax=args['vmax'],
                         cmap=plt.get_cmap(args['cmap']))
        fig.colorbar(cax)
        plt.title(args['title'])
        plt.show()
    
    def check_finish_solve(self):
        err_msg = 'call solve() method first.'
        assert self._finish_solve, err_msg
        return True
    
    def print_info(self):
        OptDB = PETSc.Options()
        PETSc.Sys.Print()
        PETSc.Sys.Print('Information about %s' % str(self))
        
        for obj in self.get_obj_list():
            obj.print_info()
        PETSc.Sys.Print('%s: force nodes: %d, velocity nodes: %d'
                        % (str(self), self.get_n_f_node(), self.get_n_u_node()))
        if self._kwargs['plot_geo']:
            # self.show_f_u_nodes(linestyle='-')
            self.show_f_u_nodes(linestyle='')
        if OptDB.getBool('plot_vel', 0):
            length_factor = OptDB.getReal('vel_fct', 1)
            self.show_velocity(length_factor=length_factor)
        return True
    
    def get_M(self):
        err_msg = 'this method must be called before method vtk_velocity(), the latter one would destroy the M matrix. '
        assert not self._M_destroyed, err_msg
        M = self._M_petsc.getDenseArray().copy()
        return M
    
    def get_M_petsc(self):
        return self._M_petsc
    
    def get_n_f_node(self):
        return self._n_fnode
    
    def get_n_u_node(self):
        return self._n_unode
    
    def get_n_force(self):
        return self._force_petsc.getSizes()[1]
    
    def get_n_velocity(self):
        return self._velocity_petsc.getSizes()[1]
    
    def get_obj_list(self):
        return self._obj_list
    
    def get_all_obj_list(self):
        return self.get_obj_list()
    
    def get_n_obj(self):
        return len(self._obj_list)
    
    def dbg_get_U(self):
        return self.vec_scatter(self._velocity_petsc, destroy=False)
    
    def get_force(self):
        return self._force
    
    def get_force_x(self):
        return self._force[0::self._n_unknown]
    
    def get_force_y(self):
        return self._force[1::self._n_unknown]
    
    def get_force_z(self):
        return self._force[2::self._n_unknown]
    
    def get_total_force(self, center=np.zeros(3)):
        F = np.zeros(6)
        for obj0 in self.get_all_obj_list():
            assert isinstance(obj0, StokesFlowObj)
            F = F + obj0.get_total_force(center=center)
        return F
    
    def set_kwargs(self, **kwargs):
        self._kwargs = kwargs  # kwargs associate with solving method,
        self._init_kwargs(**kwargs)
        return True
    
    def get_force_petsc(self):
        return self._force_petsc
    
    def get_velocity_petsc(self):
        return self._velocity_petsc
    
    def get_n_unknown(self):
        return self._n_unknown
    
    def get_kwargs(self):
        return self._kwargs
    
    def get_matrix_method(self):
        return self._kwargs['matrix_method']
    
    def get_residualNorm(self):
        self.check_finish_solve()
        return self._residualNorm
    
    def get_convergenceHistory(self):
        self.check_finish_solve()
        return self._convergenceHistory
    
    def get_name(self):
        return self._kwargs['fileHandle']
    
    @staticmethod
    def vec_scatter(vec_petsc, destroy=True):
        scatter, temp = PETSc.Scatter().toAll(vec_petsc)
        scatter.scatterBegin(vec_petsc, temp, False, PETSc.Scatter.Mode.FORWARD)
        scatter.scatterEnd(vec_petsc, temp, False, PETSc.Scatter.Mode.FORWARD)
        vec = temp.getArray()
        if destroy:
            vec_petsc.destroy()
        return vec
    
    def show_velocity(self, length_factor=1, show_nodes=True):
        geo_list = uniqueList()
        for obj1 in self.get_obj_list():
            geo_list.append(obj1.get_u_geo())
        temp_geo = base_geo()
        temp_geo.combine(geo_list)
        temp_geo.show_velocity(length_factor=length_factor, show_nodes=show_nodes)
        return True
    
    def show_force(self, length_factor=1, show_nodes=True):
        geo_list = uniqueList()
        for obj1 in self.get_obj_list():
            geo_list.append(obj1.get_f_geo())
        temp_geo = base_geo()
        temp_geo.combine(geo_list)
        temp_geo.set_velocity(self._force)
        temp_geo.show_velocity(length_factor=length_factor, show_nodes=show_nodes)
        return True
    
    def show_f_nodes(self, linestyle='-'):
        geo_list = uniqueList()
        for obj1 in self.get_all_obj_list():
            geo_list.append(obj1.get_f_geo())
        temp_geo = base_geo()
        temp_geo.combine(geo_list)
        temp_geo.show_nodes(linestyle)
        return True
    
    def show_u_nodes(self, linestyle='-'):
        geo_list = uniqueList()
        for obj1 in self.get_all_obj_list():
            geo_list.append(obj1.get_u_geo())
        temp_geo = base_geo()
        temp_geo.combine(geo_list)
        temp_geo.show_nodes(linestyle)
        return True
    
    def show_f_u_nodes(self, linestyle='-'):
        f_geo_list = uniqueList()
        u_geo_list = uniqueList()
        for obj1 in self.get_all_obj_list():
            f_geo_list.append(obj1.get_f_geo())
            if obj1.get_f_geo() is not obj1.get_u_geo():
                u_geo_list.append(obj1.get_u_geo())
        f_geo = base_geo()
        f_geo.combine(f_geo_list)
        u_geo = base_geo()
        u_geo.combine(u_geo_list)
        temp_geo = geoComposit()
        temp_geo.append(u_geo)
        temp_geo.append(f_geo)
        temp_geo.show_nodes(linestyle)
        return True
    
    def update_location(self, eval_dt, print_handle=''):
        for obj0 in self.get_obj_list():
            obj0.update_location(eval_dt, print_handle)
        return True


class StokesFlowObj:
    # general class of object, contain general properties of objcet.
    def __init__(self):
        self._index = -1  # index of object
        self._f_geo = base_geo()  # global coordinates of force nodes
        self._u_geo = base_geo()  # global coordinates of velocity nodes
        self._re_velocity = np.zeros([0])  # resolved information
        self._force = np.zeros([0])  # force information
        self._type = 'uninitialized'  # object type
        self._name = '...'  # object name
        self._n_unknown = 3
        self._problem = None
        self._matrix_method = None
        # the following properties store the location history of the composite.
        # current such kind of obj don't move.
        # fix the center at u_geo.center()
        self._obj_norm_hist = []
        # self._locomotion_fct = np.ones(3)
        # self._center_hist = []
        # self._U_hist = []   # (ux,uy,uz,wx,wy,wz)
        # self._displace_hist = []
        # self._rotation_hist = []
    
    def __repr__(self):
        return self.get_obj_name()
    
    def __str__(self):
        return self.get_name()
    
    def print_info(self):
        PETSc.Sys.Print('  %s, father: %s, type: %s, index: %d, '
                        % (self.get_name(), self._problem.get_name(), self._type, self.get_index(),))
        PETSc.Sys.Print('    force nodes %d, velocity nodes %d' % (self.get_n_f_node(), self.get_n_u_node()))
        self.get_u_geo().print_info()
        self.get_f_geo().print_info()
        return True
    
    def save_mat(self, addInfo=''):
        comm = PETSc.COMM_WORLD.tompi4py()
        rank = comm.Get_rank()
        u_glbIdx, u_glbIdx_all = self.get_u_geo().get_glbIdx()
        f_glbIdx, f_glbIdx_all = self.get_f_geo().get_glbIdx()
        filename = addInfo + self._problem.get_name() + '_' + self.get_name() + '.mat'
        if rank == 0:
            savemat(filename,
                    {
                        'fnodes':       self.get_f_geo().get_nodes(),
                        'unodes':       self.get_u_geo().get_nodes(),
                        'u_glbIdx':     u_glbIdx,
                        'u_glbIdx_all': u_glbIdx_all,
                        'f_glbIdx':     f_glbIdx,
                        'f_glbIdx_all': f_glbIdx_all,
                        'force':        self._force,
                        're_velocity':  self._re_velocity,
                        'velocity':     self.get_u_geo().get_velocity(),
                        },
                    oned_as='column')
        PETSc.Sys.Print('%s: save information to %s' % (str(self), filename))
        return True
    
    def set_data(self, f_geo: base_geo, u_geo: base_geo, name='...', **kwargs):
        # err_msg = 'f_geo and u_geo need geo objects contain force and velocity nodes, respectively. '
        # assert isinstance(f_geo, base_geo) and isinstance(u_geo, base_geo), err_msg
        
        self._f_geo = f_geo
        self._u_geo = u_geo
        self._force = np.zeros(self.get_n_f_node() * self.get_n_unknown())
        self._re_velocity = np.zeros(self.get_u_nodes().size)
        self._name = name
        self._type = 'general obj'
        return True
    
    def set_velocity(self, velocity: np.array):
        return self.get_u_geo().set_velocity(velocity)
    
    def set_rigid_velocity(self, *args, **kwargs):
        return self.get_u_geo().set_rigid_velocity(*args, **kwargs)
    
    def get_problem(self) -> StokesFlowProblem:
        return self._problem
    
    def set_problem(self, problem: 'StokesFlowProblem'):
        self._problem = problem
        return True
    
    def get_matrix_method(self):
        return self._matrix_method
    
    def set_matrix_method(self, matrix_method, **kwargs):
        self._matrix_method = matrix_method
        return True
    
    def copy(self):
        """
        copy a new object.
        """
        problem = self._problem
        self._problem = None
        self.get_f_geo().destroy_dmda()
        if self.get_f_geo() is not self.get_u_geo():
            self.get_u_geo().destroy_dmda()
        
        obj2 = copy.deepcopy(self)  # type: StokesFlowObj
        self.set_problem(problem)
        obj2.set_problem(problem)
        obj2.set_index(-1)
        self.get_f_geo().set_dmda()
        if self.get_f_geo() is not self.get_u_geo():
            self.get_u_geo().set_dmda()
        obj2.get_f_geo().set_dmda()
        if obj2.get_f_geo() is not obj2.get_u_geo():
            obj2.get_u_geo().set_dmda()
        return obj2
    
    def combine(self, obj_list: uniqueList, set_re_u=False, set_force=False,
                geo_fun=base_geo):
        obj_list = list(tube_flatten((obj_list,)))
        fgeo_list = uniqueList()
        ugeo_list = uniqueList()
        for obj0 in obj_list:
            err_msg = 'some object(s) in obj_list are not StokesFlowObj object. %s' % \
                      type(obj0)
            assert isinstance(obj0, StokesFlowObj), err_msg
            fgeo_list.append(obj0.get_f_geo())
            ugeo_list.append(obj0.get_u_geo())
        
        fgeo = geo_fun()
        ugeo = geo_fun()
        fgeo.combine(fgeo_list)
        ugeo.combine(ugeo_list)
        self.set_data(fgeo, ugeo, name=self.get_name())
        
        if set_re_u:
            self.set_re_velocity(np.zeros([0]))
            for obj0 in obj_list:
                self.set_re_velocity(np.hstack((self.get_re_velocity(), obj0.get_re_velocity())))
        if set_force:
            self.set_force(np.zeros([0]))
            for obj0 in obj_list:
                self.set_force(np.hstack((self.get_force(), obj0.get_force())))
        return True
    
    def move(self, displacement):
        self.get_f_geo().move(displacement)
        if self.get_f_geo() is not self.get_u_geo():
            self.get_u_geo().move(displacement)
        return True
    
    def node_rotation(self, norm=np.array([0, 0, 1]), theta=0, rotation_origin=None):
        rotM = get_rot_matrix(norm, theta)
        self.node_rotM(rotM=rotM, rotation_origin=rotation_origin)
        return True
    
    def node_rotM(self, rotM, rotation_origin=None):
        self.get_f_geo().node_rotM(rotM, rotation_origin)
        if self.get_f_geo() is not self.get_u_geo():
            self.get_u_geo().node_rotM(rotM, rotation_origin)
        return True
    
    def zoom(self, factor, zoom_origin=None):
        self.get_f_geo().node_zoom(factor, zoom_origin=zoom_origin)
        if self.get_f_geo() is not self.get_u_geo():
            self.get_u_geo().node_zoom(factor, zoom_origin=zoom_origin)
        return True
    
    def get_index(self):
        return self._index
    
    def get_type(self):
        return self._type
    
    def get_obj_name(self):
        return self._type + ' (index %d)' % self._index
    
    def get_name(self):
        return self._name
    
    def set_name(self, name):
        self._name = name
        return True
    
    def set_index(self, new_index):
        self._index = new_index
        return True
    
    def get_f_nodes(self):
        return self._f_geo.get_nodes()
    
    def get_u_nodes(self):
        return self._u_geo.get_nodes()
    
    def get_force(self):
        return self._force
    
    def get_force_x(self):
        return self._force[0::self._n_unknown]
    
    def get_force_y(self):
        return self._force[1::self._n_unknown]
    
    def get_force_z(self):
        return self._force[2::self._n_unknown]
    
    def get_total_force(self, center=None):
        if center is None:
            center = self.get_u_geo().get_origin()
        
        f = self.get_force().reshape((-1, self.get_n_unknown()))
        r = self.get_f_geo().get_nodes() - center
        t = np.cross(r, f[:, :3])  # some solve methods may have additional degrees of freedoms.
        f_t = np.hstack((f, t)).sum(axis=0)
        return f_t
    
    def set_force(self, force):
        self._force = force
    
    def get_re_velocity(self):
        return self._re_velocity
    
    def get_velocity(self):
        return self.get_u_geo().get_velocity()
    
    def set_re_velocity(self, re_velocity):
        self._re_velocity = re_velocity
    
    def get_n_f_node(self):
        return self.get_f_nodes().shape[0]
    
    def get_n_u_node(self):
        return self.get_u_nodes().shape[0]
    
    def get_n_velocity(self):
        return self.get_u_nodes().shape[0] * self._n_unknown
    
    def get_n_force(self):
        return self.get_f_nodes().shape[0] * self._n_unknown
    
    def get_n_unknown(self):
        return self._n_unknown
    
    def get_f_geo(self) -> base_geo:
        return self._f_geo
    
    def get_u_geo(self) -> base_geo:
        return self._u_geo
    
    @staticmethod
    def vec_scatter(vec_petsc, destroy=True):
        scatter, temp = PETSc.Scatter().toAll(vec_petsc)
        scatter.scatter(vec_petsc, temp, False, PETSc.Scatter.Mode.FORWARD)
        vec = temp.getArray()
        if destroy:
            vec_petsc.destroy()
        return vec
    
    def vtk(self, filename, stp_idx=0):
        if str(self) == '...':
            return
        
        comm = PETSc.COMM_WORLD.tompi4py()
        rank = comm.Get_rank()
        if rank == 0:
            force_x = self._force[0::self._n_unknown].copy()
            force_y = self._force[1::self._n_unknown].copy()
            force_z = self._force[2::self._n_unknown].copy()
            velocity_x = self._re_velocity[0::self._n_unknown].copy()
            velocity_y = self._re_velocity[1::self._n_unknown].copy()
            velocity_z = self._re_velocity[2::self._n_unknown].copy()
            velocity_err_x = np.abs(self._re_velocity[0::self._n_unknown]
                                    - self.get_velocity()[0::3])
            velocity_err_y = np.abs(self._re_velocity[1::self._n_unknown]
                                    - self.get_velocity()[1::3])
            velocity_err_z = np.abs(self._re_velocity[2::self._n_unknown]
                                    - self.get_velocity()[2::3])
            
            if 'rs' in self.get_matrix_method():
                filename = '%s_%s_t%05d' % (filename, str(self), stp_idx)
                pointsToVTK(filename, self.get_f_nodes()[:, 0], self.get_f_nodes()[:, 1],
                            self.get_f_nodes()[:, 2],
                            data={
                                "force":        (force_x, force_y, force_z),
                                "velocity":     (velocity_x, velocity_y, velocity_z),
                                "velocity_err": (velocity_err_x,
                                                 velocity_err_y,
                                                 velocity_err_z),
                                })
            else:
                f_filename = '%s_%s_force_t%05d' % (filename, str(self), stp_idx)
                pointsToVTK(f_filename, self.get_f_nodes()[:, 0], self.get_f_nodes()[:, 1],
                            self.get_f_nodes()[:, 2],
                            data={
                                "force": (force_x, force_y, force_z),
                                })
                u_filename = '%s_%s_velocity_t%05d' % (filename, str(self), stp_idx)
                pointsToVTK(u_filename, self.get_u_nodes()[:, 0], self.get_u_nodes()[:, 1],
                            self.get_u_nodes()[:, 2],
                            data={
                                "velocity":     (velocity_x, velocity_y, velocity_z),
                                "velocity_err": (velocity_err_x,
                                                 velocity_err_y,
                                                 velocity_err_z),
                                })
        return True
    
    def show_velocity(self, length_factor=1, show_nodes=True):
        self.get_u_geo().show_velocity(length_factor, show_nodes)
        return True
    
    def show_re_velocity(self, length_factor=1, show_nodes=True):
        self.get_problem().check_finish_solve()
        tgeo = self.get_u_geo().copy()
        tgeo.set_velocity(self._re_velocity)
        tgeo.show_velocity(length_factor, show_nodes)
        return True
    
    def show_force(self, length_factor=1, show_nodes=True):
        self.get_problem().check_finish_solve()
        tgeo = self.get_f_geo().copy()
        tgeo.set_velocity(self._force)
        tgeo.show_velocity(length_factor, show_nodes)
        return True
    
    def show_f_nodes(self, linestyle='-'):
        self.get_f_geo().show_nodes(linestyle)
        return True
    
    def show_u_nodes(self, linestyle='-'):
        self.get_u_geo().show_nodes(linestyle)
        return True
    
    def show_f_u_nodes(self, linestyle='-'):
        temp_geo = geoComposit()
        temp_geo.append(self.get_u_geo())
        if self.get_u_geo() is not self.get_f_geo():
            temp_geo.append(self.get_f_geo())
        temp_geo.show_nodes(linestyle)
        return True
    
    def pickmyself_prepare(self):
        self.get_f_geo().pickmyself_prepare()
        self.get_u_geo().pickmyself_prepare()
        return True
    
    def unpick_myself(self):
        self.get_u_geo().set_dmda()
        self.get_f_geo().set_dmda()
        return True
    
    def update_location(self, eval_dt, print_handle=''):
        self._obj_norm_hist.append(self.get_u_geo().get_geo_norm())
        return True
    
    def get_obj_norm_hist(self):
        return self._obj_norm_hist


class StokesFlowRingObj(StokesFlowObj):
    def check_nodes(self, nodes):
        err_msg = 'nodes are distribute in the line (r, 0, z). '
        assert np.allclose(nodes[:, 1], 0), err_msg
        return True
    
    def set_data(self, f_geo: base_geo, u_geo: base_geo, name='...', **kwargs):
        self.check_nodes(f_geo.get_nodes())
        self.check_nodes(u_geo.get_nodes())
        super().set_data(f_geo, u_geo, name, **kwargs)
        self._type = 'stokes flow ring obj'
        return True
    
    def get_total_force_sum(self, center=None):
        n_c = self.get_problem().get_kwargs()['n_c']
        f_t = super().get_total_force(center) * n_c
        return f_t
    
    def get_total_force_int(self, center=None):
        if center is None:
            center = self.get_u_geo().get_origin()
        fnodes = self.get_f_nodes()
        rf = np.vstack((fnodes[:, 0], fnodes[:, 0], fnodes[:, 0],)).T
        f = self.get_force().reshape((-1, self.get_n_unknown())) * rf
        r = self.get_f_geo().get_nodes() - center
        t = np.cross(r, f[:, :3])  # some solve methods may have additional degrees of freedoms.
        f_t = np.hstack((f, t)).sum(axis=0) * 2 * np.pi
        return f_t
    
    def get_total_force(self, center=None):
        return self.get_total_force_int(center)
        # return self.get_total_force_sum(center)


class StokesletsRingObjFull(StokesFlowObj):
    def __init__(self):
        super().__init__()
        self._n_c = -1  # amount of copies along the symmetical axis
    
    def set_data(self, f_geo: '_revolve_geo', u_geo: '_revolve_geo', name='...', **kwargs):
        n_c = kwargs['n_c']
        self._n_c = n_c
        u_geo.create_full_geo(n_c)
        f_geo.create_full_geo(n_c)
        super().set_data(f_geo, u_geo, name, **kwargs)
        self._type = 'stokeslets ring obj'
        return True
    
    def show_slice_force(self, idx=0, length_factor=1, show_nodes=True):
        self.get_problem().check_finish_solve()
        n_c = self._n_c
        assert idx < n_c
        tnodes = self.get_f_nodes()[idx::n_c, :]
        tforce = self.get_force().reshape((-1, 3))[idx::n_c, :].flatten()
        tgeo = base_geo()
        tgeo.set_nodes(tnodes, self.get_f_geo().get_deltaLength())
        tgeo.set_velocity(tforce)
        tgeo.show_velocity(length_factor, show_nodes)
        return True


class StokesletsInPipeProblem(StokesFlowProblem):
    # pipe center line along z axis
    def __init__(self, **kwargs):
        from src.stokesletsInPipe import detail_light
        super().__init__(**kwargs)
        self._fpgeo = base_geo()  # force geo of pipe
        self._vpgeo = base_geo()  # velocity geo of pipe
        self._cpgeo = base_geo()  # check geo of pipe
        self._m_pipe = PETSc.Mat().create(comm=PETSc.COMM_WORLD)
        self._m_pipe_check = PETSc.Mat().create(comm=PETSc.COMM_WORLD)
        self._b_list = np.ones(0)
        
        # create a empty matrix, and a empty velocity vecters, to avoid use too much time to allocate memory.
        # for numerical part
        self._t_m = PETSc.Mat().create(
                comm=PETSc.COMM_WORLD)  # M matrix associated with u1 part ,velocity due to pipe boundary.
        self._t_u11 = uniqueList()  # a list contain three u1 component of f1, for interpolation
        self._t_u12 = uniqueList()  # a list contain three u1 component of f2, for interpolation
        self._t_u13 = uniqueList()  # a list contain three u1 component of f3, for interpolation
        self._set_f123()
        self._stokeslet_m = PETSc.Mat().create(
                comm=PETSc.COMM_WORLD)  # M matrix associated with stokeslet singularity
        self._t_u2 = PETSc.Vec().create(comm=PETSc.COMM_WORLD)
        # for theoretical part
        # DBG
        OptDB = PETSc.Options()
        dbg_threshold = OptDB.getReal('dbg_threshold', 10)
        PETSc.Sys.Print('--------------------> DBG: dbg_threshold = %f' % dbg_threshold)
        self._greenFun = detail_light(threshold=dbg_threshold)
        self._greenFun.solve_prepare_light()
        dbg_z_the_threshold = OptDB.getReal('dbg_z_the_threshold', np.inf)
        PETSc.Sys.Print('--------------------> DBG: dbg_z_the_threshold = %f' % dbg_z_the_threshold)
        self._z_the_threshold = dbg_z_the_threshold
        
        self._f1_list = []  # list of forces lists for each object at or outside pipe associated with force-nodes at x axis
        self._f2_list = []  # list of forces lists for each object at or outside pipe associated with force-nodes at y axis
        self._f3_list = []  # list of forces lists for each object at or outside pipe associated with force-nodes at z axis
        self._residualNorm_list = []  # residualNorm of f1, f2, and f3 of different b
        self._err_list = []  # relative velocity error solved using check geo.
        
        # # set values later
        # self._dp = np.nan
        # self._rp = np.nan
        # self._lp = np.nan
        # self._ep = np.nan
        # self._th = np.nan
        # self._with_cover = np.nan
        self._stokesletsInPipe_pipeFactor = np.nan
    
    def _set_f123(self):
        # set point source vector f1, f2, f3.
        fgeo = base_geo()
        fgeo.set_nodes((0, 0, 0), deltalength=0)
        t_f_pkg = PETSc.DMComposite().create(comm=PETSc.COMM_WORLD)
        t_f_pkg.addDM(fgeo.get_dmda())
        t_f_pkg.setFromOptions()
        t_f_pkg.setUp()
        f_isglb = t_f_pkg.getGlobalISs()
        fgeo.set_glbIdx(f_isglb[0].getIndices())
        f1_petsc = t_f_pkg.createGlobalVector()
        f2_petsc = t_f_pkg.createGlobalVector()
        f3_petsc = t_f_pkg.createGlobalVector()
        f1_petsc[:] = (1, 0, 0)
        f2_petsc[:] = (0, 1, 0)
        f3_petsc[:] = (0, 0, 1)
        f1_petsc.assemble()
        f2_petsc.assemble()
        f3_petsc.assemble()
        t_f_pkg.destroy()
        
        self._f123_petsc = [f1_petsc, f2_petsc, f3_petsc]
        self._stokeslet_geo = fgeo
        return True
    
    def _check_add_obj(self, obj):
        _, b, _ = obj.get_f_geo().get_polar_coord()
        b_list = self.get_b_list()
        b1 = np.max(b_list)
        err_msg = 'b is out of maximum %f' % b1
        assert all(b <= b1), err_msg
        return True
    
    def get_b_list(self):
        return self._b_list
    
    def get_residualNorm_list(self):
        return self._residualNorm_list
    
    def get_err_list(self):
        return self._err_list
    
    def set_b_list(self, b_list):
        self._b_list = b_list
        return True
    
    def get_n_b(self):
        return self._b_list.size
    
    def get_fpgeo(self):
        return self._fpgeo
    
    def get_vpgeo(self):
        return self._vpgeo
    
    def debug_solve_stokeslets_b(self, b, node):
        t_geo = base_geo()
        t_geo.set_nodes(node, deltalength=0)
        
        obj1 = StokesFlowObj()
        obj1.set_data(t_geo, t_geo)
        u_glbIdx = self._set_temp_var(obj1)
        return self._solve_stokeslets_b_num(b, node, use_cart=True, u_glbIdx_all=u_glbIdx)
    
    def debug_solve_u_pipe(self, pgeo, outputHandle, greenFun):
        return self._solve_u1_pipe(pgeo, outputHandle, greenFun)
    
    def debug_solve_stokeslets_fnode(self, fnode, geo1):
        unodes = geo1.get_nodes()
        obj1 = StokesFlowObj()
        obj1.set_data(geo1, geo1)
        t_u_glbIdx_all = self._set_temp_var(obj1)
        u_fx_petsc, u_fy_petsc, u_fz_petsc = self._solve_stokeslets_fnode(fnode, unodes,
                                                                          t_u_glbIdx_all)
        obj1.set_velocity(u_fx_petsc.getArray())
        obj1.show_velocity(show_nodes=False, length_factor=1)
        obj1.set_velocity(u_fy_petsc.getArray())
        obj1.show_velocity(show_nodes=False, length_factor=1)
        obj1.set_velocity(u_fz_petsc.getArray())
        obj1.show_velocity(show_nodes=False, length_factor=1)
        return True
    
    def _solve_u1_b_list(self, k, ugeo, use_cart=False):
        # solve velocity component due to boundary as a function of b and u_node location. Here, b is in self._b_list.
        # total u = u1 + u2, u1: force at (or outside) pipe boundary
        kwargs = self.get_kwargs()
        temp_m = self._t_m
        temp_obj1 = StokesFlowObj()
        temp_obj1.set_data(self._fpgeo, ugeo)
        self._method_dict[kwargs['matrix_method']](temp_obj1, temp_obj1, temp_m, **kwargs)
        temp_m.assemble()
        temp_m = self._t_m
        
        for i0, ID in enumerate(k):
            f1 = self._f1_list[ID]
            f2 = self._f2_list[ID]
            f3 = self._f3_list[ID]
            temp_m.mult(f1, self._t_u11[i0])
            temp_m.mult(f2, self._t_u12[i0])
            temp_m.mult(f3, self._t_u13[i0])
            
            if not use_cart:
                uphi, _, _ = ugeo.get_polar_coord()
                # Transform to polar coord
                ux1 = self._t_u11[i0][0::3].copy()
                uy1 = self._t_u11[i0][1::3].copy()
                uz1 = self._t_u11[i0][2::3].copy()
                uR1 = np.cos(uphi) * ux1 + np.sin(uphi) * uy1
                uPhi1 = -np.sin(uphi) * ux1 + np.cos(uphi) * uy1
                self._t_u11[i0][:] = np.dstack((uR1, uPhi1, uz1)).flatten()
                ux2 = self._t_u12[i0][0::3].copy()
                uy2 = self._t_u12[i0][1::3].copy()
                uz2 = self._t_u12[i0][2::3].copy()
                uR2 = np.cos(uphi) * ux2 + np.sin(uphi) * uy2
                uPhi2 = -np.sin(uphi) * ux2 + np.cos(uphi) * uy2
                self._t_u12[i0][:] = np.dstack((uR2, uPhi2, uz2)).flatten()
                ux3 = self._t_u13[i0][0::3].copy()
                uy3 = self._t_u13[i0][1::3].copy()
                uz3 = self._t_u13[i0][2::3].copy()
                uR3 = np.cos(uphi) * ux3 + np.sin(uphi) * uy3
                uPhi3 = -np.sin(uphi) * ux3 + np.cos(uphi) * uy3
                self._t_u13[i0][:] = np.dstack((uR3, uPhi3, uz3)).flatten()
            self._t_u11[i0].assemble()
            self._t_u12[i0].assemble()
            self._t_u13[i0].assemble()
        return True
    
    def _solve_stokeslets_b_num(self, b, unode_xyz, use_cart=False, u_glbIdx_all=[]):
        from src.StokesFlowMethod import point_force_matrix_3d_petsc
        # velocity due to stokesles.
        kwargs = self.get_kwargs()
        stokeslet_geo = self._stokeslet_geo
        stokeslet_node = np.hstack((b, 0, 0))
        stokeslet_geo.set_nodes(stokeslet_node, deltalength=0)
        ugeo = base_geo()
        ugeo.set_nodes(unode_xyz, deltalength=0)
        ugeo.set_glbIdx_all(u_glbIdx_all)
        obj1 = StokesFlowObj()
        obj1.set_data(stokeslet_geo, ugeo)
        stokeslet_m = self._stokeslet_m
        point_force_matrix_3d_petsc(obj1, obj1, stokeslet_m, **kwargs)
        stokeslet_m.assemble()
        # velocity due to boundary, lagrange interploation
        b_list = self.get_b_list()
        clsID = min(range(len(b_list)), key=lambda i: abs(b_list[i] - b))  # index of closest b
        u_petsc = []
        if b_list[clsID] == b_list[0]:  # top of the list
            k = [0, 1, 2]
            self._solve_u1_b_list(k, ugeo, use_cart)
        elif b_list[clsID] == b_list[-1]:  # botton of the list
            k = [-3, -2, -1]
            self._solve_u1_b_list(k, ugeo, use_cart)
        else:
            k = [clsID - 1, clsID, clsID + 1]
            self._solve_u1_b_list(k, ugeo, use_cart)
        l1 = ((b - b_list[k[1]]) * (b - b_list[k[2]])) / (
                (b_list[k[0]] - b_list[k[1]]) * (b_list[k[0]] - b_list[k[2]]))
        l2 = ((b - b_list[k[0]]) * (b - b_list[k[2]])) / (
                (b_list[k[1]] - b_list[k[0]]) * (b_list[k[1]] - b_list[k[2]]))
        l3 = ((b - b_list[k[0]]) * (b - b_list[k[1]])) / (
                (b_list[k[2]] - b_list[k[0]]) * (b_list[k[2]] - b_list[k[1]]))
        # ux
        t_u1 = self._t_u11[0] * l1 + self._t_u11[1] * l2 + self._t_u11[
            2] * l3  # velocity due to boundary.
        stokeslet_m.mult(self._f123_petsc[0], self._t_u2)
        u_petsc.append(t_u1 + self._t_u2)
        # uy
        t_u1 = self._t_u12[0] * l1 + self._t_u12[1] * l2 + self._t_u12[
            2] * l3  # velocity due to boundary.
        stokeslet_m.mult(self._f123_petsc[1], self._t_u2)
        u_petsc.append(t_u1 + self._t_u2)
        # uz
        t_u1 = self._t_u13[0] * l1 + self._t_u13[1] * l2 + self._t_u13[
            2] * l3  # velocity due to boundary.
        stokeslet_m.mult(self._f123_petsc[2], self._t_u2)
        u_petsc.append(t_u1 + self._t_u2)
        return u_petsc
    
    def _solve_stokeslets_b_the(self, b, unode_rpz, use_cart=False):
        comm = PETSc.COMM_WORLD.tompi4py()
        dmda_the = PETSc.DMDA().create(sizes=(unode_rpz.shape[0],), dof=3, stencil_width=0,
                                       comm=PETSc.COMM_WORLD)
        t_u_pkg = PETSc.DMComposite().create(comm=PETSc.COMM_WORLD)
        t_u_pkg.addDM(dmda_the)
        t_u_pkg.setFromOptions()
        t_u_pkg.setUp()
        
        u1 = t_u_pkg.createGlobalVector()
        u2 = t_u_pkg.createGlobalVector()
        u3 = t_u_pkg.createGlobalVector()
        u_isglb = t_u_pkg.getGlobalISs()
        u_glbIdx_all = np.hstack(comm.allgather(u_isglb[0].getIndices()))
        t_u_pkg.destroy()
        
        greenFun = self._greenFun
        greenFun.set_b(b=b)
        greenFun.solve_prepare_b()
        t_dmda_range = range(dmda_the.getRanges()[0][0], dmda_the.getRanges()[0][1])
        t_Vec_range = u1.getOwnershipRange()
        for i0 in t_dmda_range:
            R, phi, z = unode_rpz[i0]
            t_u_glbIdx = u_glbIdx_all[i0 * 3]
            sign_z = np.sign(z)
            abs_z = np.abs(z)
            uR1, uPhi1, uz1, uR2, uPhi2, uz2, uR3, uPhi3, uz3 = \
                greenFun.solve_u_light(R, phi, abs_z)
            u1[t_u_glbIdx:t_u_glbIdx + 3] = [uR1, uPhi1, sign_z * uz1]
            u2[t_u_glbIdx:t_u_glbIdx + 3] = [uR2, uPhi2, sign_z * uz2]
            u3[t_u_glbIdx:t_u_glbIdx + 3] = [sign_z * uR3, sign_z * uPhi3, uz3]
        
        if use_cart:
            phi = unode_rpz[t_dmda_range, 1]
            t_Vec_range_x = range(t_Vec_range[0] + 0, t_Vec_range[1], 3)
            t_Vec_range_y = range(t_Vec_range[0] + 1, t_Vec_range[1], 3)
            t_ux1 = np.cos(phi) * u1[t_Vec_range_x] - np.sin(phi) * u1[t_Vec_range_y]
            t_ux2 = np.cos(phi) * u2[t_Vec_range_x] - np.sin(phi) * u2[t_Vec_range_y]
            t_ux3 = np.cos(phi) * u3[t_Vec_range_x] - np.sin(phi) * u3[t_Vec_range_y]
            t_uy1 = np.sin(phi) * u1[t_Vec_range_x] + np.cos(phi) * u1[t_Vec_range_y]
            t_uy2 = np.sin(phi) * u2[t_Vec_range_x] + np.cos(phi) * u2[t_Vec_range_y]
            t_uy3 = np.sin(phi) * u3[t_Vec_range_x] + np.cos(phi) * u3[t_Vec_range_y]
            u1[t_Vec_range_x] = t_ux1
            u1[t_Vec_range_y] = t_uy1
            u2[t_Vec_range_x] = t_ux2
            u2[t_Vec_range_y] = t_uy2
            u3[t_Vec_range_x] = t_ux3
            u3[t_Vec_range_y] = t_uy3
        u1.assemble()
        u2.assemble()
        u3.assemble()
        u_petsc = (u1, u2, u3)
        # PETSc.Sys.Print(unode_rpz.size)
        return u_petsc
    
    def _solve_stokeslets_fnode(self, fnode, unodes, u_glbIdx_all=[]):
        fnode = fnode.reshape((1, 3))
        unodes = unodes.reshape((-1, 3))
        ugeo = base_geo()
        ugeo.set_nodes(unodes, resetVelocity=True, deltalength=0)
        uphi, urho, uz = ugeo.get_polar_coord()
        fgeo = base_geo()
        fgeo.set_nodes(fnode, resetVelocity=True, deltalength=0)
        fphi, frho, fz = fgeo.get_polar_coord()
        
        # calculate ux, uy, uz in local coordinate, definding by fnode.
        b = frho
        R = urho
        phi = uphi - fphi
        x = R * np.cos(phi)
        y = R * np.sin(phi)
        z = uz - fz
        t_node_xyz = np.vstack((x, y, z)).T
        u_fx_petsc, u_fy_petsc, u_fz_petsc = \
            self._solve_stokeslets_b_num(b, t_node_xyz, True, u_glbIdx_all)
        temp1 = np.abs(z) > self._z_the_threshold
        if any(temp1):
            theIdx = np.dstack((temp1, temp1, temp1)).flatten()
            t_node_rpz = np.vstack((R[temp1], phi[temp1], z[temp1])).T
            u_glbIdx_the = u_glbIdx_all[theIdx]
            u_fx_petsc_the, u_fy_petsc_the, u_fz_petsc_the = \
                self._solve_stokeslets_b_the(b, t_node_rpz, True)
            t_range = range(u_fx_petsc_the.getOwnershipRange()[0],
                            u_fx_petsc_the.getOwnershipRange()[1])
            temp2 = np.dstack((z, z, z)).flatten()
            temp3 = np.abs(temp2[u_glbIdx_the[t_range]])
            t_factor = np.abs((temp3 - self._z_the_threshold) /
                              (self._lp / 2 - self._z_the_threshold))
            u_fx_petsc[u_glbIdx_the[t_range]] = \
                u_fx_petsc_the.getArray() * t_factor + \
                u_fx_petsc[u_glbIdx_the[t_range]] * (1 - t_factor)
            u_fy_petsc[u_glbIdx_the[t_range]] = \
                u_fy_petsc_the.getArray() * t_factor + \
                u_fy_petsc[u_glbIdx_the[t_range]] * (1 - t_factor)
            u_fz_petsc[u_glbIdx_the[t_range]] = \
                u_fz_petsc_the.getArray() * t_factor + \
                u_fz_petsc[u_glbIdx_the[t_range]] * (1 - t_factor)
        
        u_fx_loc = u_fx_petsc.getArray()
        u_fy_loc = u_fy_petsc.getArray()
        u_fz_loc = u_fz_petsc.getArray()
        
        # shift to global coordinate
        theta = np.arctan2(fnode[0, 1], fnode[0, 0])
        T = np.array(((np.cos(theta), np.sin(theta), 0),
                      (-np.sin(theta), np.cos(theta), 0),
                      (0, 0, 1)))
        Tinv = np.array(((np.cos(theta), -np.sin(theta), 0),
                         (np.sin(theta), np.cos(theta), 0),
                         (0, 0, 1)))
        temp_loc = np.dstack(
                (u_fx_loc.reshape((-1, 3)), u_fy_loc.reshape((-1, 3)), u_fz_loc.reshape((-1, 3))))
        temp_glb = np.tensordot(Tinv, np.tensordot(temp_loc, T, axes=(2, 0)), axes=(1, 1))
        u_fx_glb = np.dstack((temp_glb[0, :, 0], temp_glb[1, :, 0], temp_glb[2, :, 0])).flatten()
        u_fy_glb = np.dstack((temp_glb[0, :, 1], temp_glb[1, :, 1], temp_glb[2, :, 1])).flatten()
        u_fz_glb = np.dstack((temp_glb[0, :, 2], temp_glb[1, :, 2], temp_glb[2, :, 2])).flatten()
        u_fx_petsc.setValues(range(u_fx_petsc.getOwnershipRange()[0],
                                   u_fx_petsc.getOwnershipRange()[1]), u_fx_glb)
        u_fy_petsc.setValues(range(u_fy_petsc.getOwnershipRange()[0],
                                   u_fy_petsc.getOwnershipRange()[1]), u_fy_glb)
        u_fz_petsc.setValues(range(u_fz_petsc.getOwnershipRange()[0],
                                   u_fz_petsc.getOwnershipRange()[1]), u_fz_glb)
        
        return u_fx_petsc, u_fy_petsc, u_fz_petsc
    
    def _check_f_accuracy(self, b, greenFun, waitBar=np.array((1, 1)), **kwargs):
        comm = PETSc.COMM_WORLD.tompi4py()
        rank = comm.Get_rank()
        fileHandle = self._kwargs['fileHandle']
        cpgeo = self._cpgeo
        fpgeo = self._fpgeo
        outputHandle = 'check'
        a_u11, a_u21, a_u31 = self._solve_u1_pipe(cpgeo, outputHandle, greenFun, waitBar)
        m_petsc = self._m_pipe_check
        
        c_u11_petsc = m_petsc.createVecLeft()
        # c_u11_petsc.set(0)
        m_petsc.mult(self._f1_list[-1], c_u11_petsc)
        c_u11 = self.vec_scatter(c_u11_petsc, destroy=True)
        c_u21_petsc = m_petsc.createVecLeft()
        # c_u21_petsc.set(0)
        m_petsc.mult(self._f2_list[-1], c_u21_petsc)
        c_u21 = self.vec_scatter(c_u21_petsc, destroy=True)
        c_u31_petsc = m_petsc.createVecLeft()
        # c_u31_petsc.set(0)
        m_petsc.mult(self._f3_list[-1], c_u31_petsc)
        c_u31 = self.vec_scatter(c_u31_petsc, destroy=True)
        
        err1 = np.sqrt(np.sum((a_u11 - c_u11) ** 2) / np.sum(a_u11 ** 2))
        err2 = np.sqrt(np.sum((a_u21 - c_u21) ** 2) / np.sum(a_u21 ** 2))
        err3 = np.sqrt(np.sum((a_u31 - c_u31) ** 2) / np.sum(a_u31 ** 2))
        PETSc.Sys().Print('      relative err: %f, %f, %f' % (err1, err2, err3))
        self._err_list.append((err1, err2, err3))
        
        f1 = self.vec_scatter(self._f1_list[-1], destroy=False)
        f2 = self.vec_scatter(self._f2_list[-1], destroy=False)
        f3 = self.vec_scatter(self._f3_list[-1], destroy=False)
        if rank == 0:
            savemat('%s_%s_b%.5f_u.mat' % (fileHandle, outputHandle, b),
                    {
                        'u11_num': a_u11,
                        'u21_num': a_u21,
                        'u31_num': a_u31,
                        'u11_ana': c_u11,
                        'u21_ana': c_u21,
                        'u31_ana': c_u31,
                        'nodes':   cpgeo.get_nodes(),
                        'kwargs':  self.get_kwargs(),
                        'fnodes':  fpgeo.get_nodes(),
                        'f1':      f1,
                        'f2':      f2,
                        'f3':      f3,
                        },
                    oned_as='column')
            t_filename = '%s_%s_b%.5f_u' % (fileHandle, outputHandle, b)
            a_u11 = np.asfortranarray(a_u11.reshape(-1, 3))
            a_u21 = np.asfortranarray(a_u21.reshape(-1, 3))
            a_u31 = np.asfortranarray(a_u31.reshape(-1, 3))
            c_u11 = np.asfortranarray(c_u11.reshape(-1, 3))
            c_u21 = np.asfortranarray(c_u21.reshape(-1, 3))
            c_u31 = np.asfortranarray(c_u31.reshape(-1, 3))
            e_u11 = a_u11 - c_u11
            e_u21 = a_u21 - c_u21
            e_u31 = a_u31 - c_u31
            pointsToVTK(t_filename, cpgeo.get_nodes()[:, 0], cpgeo.get_nodes()[:, 1],
                        cpgeo.get_nodes()[:, 2],
                        data={
                            "velocity_ana1": (a_u11[:, 0], a_u11[:, 1], a_u11[:, 2]),
                            "velocity_ana2": (a_u21[:, 0], a_u21[:, 1], a_u21[:, 2]),
                            "velocity_ana3": (a_u31[:, 0], a_u31[:, 1], a_u31[:, 2]),
                            "velocity_num1": (c_u11[:, 0], c_u11[:, 1], c_u11[:, 2]),
                            "velocity_num2": (c_u21[:, 0], c_u21[:, 1], c_u21[:, 2]),
                            "velocity_num3": (c_u31[:, 0], c_u31[:, 1], c_u31[:, 2]),
                            "velocity_err1": (e_u11[:, 0], e_u11[:, 1], e_u11[:, 2]),
                            "velocity_err2": (e_u21[:, 0], e_u21[:, 1], e_u21[:, 2]),
                            "velocity_err3": (e_u31[:, 0], e_u31[:, 1], e_u31[:, 2]),
                            })
            t_filename = '%s_%s_b%.5f_force' % (fileHandle, outputHandle, b)
            f1 = np.asfortranarray(f1.reshape(-1, 3))
            f2 = np.asfortranarray(f2.reshape(-1, 3))
            f3 = np.asfortranarray(f3.reshape(-1, 3))
            pointsToVTK(t_filename, fpgeo.get_nodes()[:, 0], fpgeo.get_nodes()[:, 1],
                        fpgeo.get_nodes()[:, 2],
                        data={
                            "force1": (f1[:, 0], f1[:, 1], f1[:, 2]),
                            "force2": (f2[:, 0], f2[:, 1], f2[:, 2]),
                            "force3": (f3[:, 0], f3[:, 1], f3[:, 2]),
                            })
            
            # t_filename = '%s_%s_b%.5f_velocity' % (fileHandle, outputHandle, b)
        return True
    
    def set_prepare(self, fileHandle, fullpath=False):
        fileHandle = check_file_extension(fileHandle, '_force_pipe.mat')
        if fullpath:
            mat_contents = loadmat(fileHandle)
        else:
            t_path = os.path.dirname(os.path.abspath(__file__))
            full_path = os.path.normpath(t_path + '/' + fileHandle)
            mat_contents = loadmat(full_path)
        
        self.set_b_list(mat_contents['b'].flatten())
        self._f1_list = [f1 for f1 in mat_contents['f1_list']]
        self._f2_list = [f2 for f2 in mat_contents['f2_list']]
        self._f3_list = [f3 for f3 in mat_contents['f3_list']]
        self._residualNorm_list = mat_contents['residualNorm'].tolist()
        self._err_list = mat_contents['err'].tolist()
        # self._dp = mat_contents['dp'][0, 0]
        # self._rp = mat_contents['rp'][0, 0]
        # self._lp = mat_contents['lp'][0, 0]
        # self._ep = mat_contents['ep'][0, 0]
        # self._th = mat_contents['th'][0, 0]
        # self._with_cover = mat_contents['with_cover'][0, 0]
        # self._stokesletsInPipe_pipeFactor = mat_contents['stokesletsInPipe_pipeFactor'][0, 0]
        
        kwargs = self.get_kwargs()
        # kwargs['dp'] = self._dp
        # kwargs['rp'] = self._rp
        # kwargs['lp'] = self._lp
        # kwargs['ep'] = self._ep
        # kwargs['th'] = self._th
        # kwargs['with_cover'] = self._with_cover
        # kwargs['stokesletsInPipe_pipeFactor'] = self._stokesletsInPipe_pipeFactor
        self._kwargs['unpickedPrb'] = True
        self._kwargs = kwargs
        
        self._pipe_geo_load(mat_contents)
        
        # PETSC version
        self._f_list_numpy2PETSC()
        return True
    
    def solve_prepare(self):
        from src.stokesletsInPipe import detail
        kwargs = self.get_kwargs()
        self._dp = kwargs['dp']
        self._rp = kwargs['rp']
        self._lp = kwargs['lp']
        self._ep = kwargs['ep']
        self._th = kwargs['th']
        self._with_cover = kwargs['with_cover']
        self._stokesletsInPipe_pipeFactor = kwargs['stokesletsInPipe_pipeFactor']
        self._b_list = np.linspace(kwargs['b0'], kwargs['b1'],
                                   kwargs['nb'])  # list of b (force location).
        
        PETSc.Sys.Print('                b_list: ')
        PETSc.Sys.Print(self.get_b_list())
        self._f1_list.clear()
        self._f2_list.clear()
        self._f3_list.clear()
        self._pipe_geo_generate(**kwargs)
        self._solve_m_pipe(**kwargs)
        
        ini_guess = (None, None, None,)
        for i0, b in enumerate(self.get_b_list()):
            greenFun = detail(threshold=self._th, b=b)
            greenFun.solve_prepare()
            waitBar = np.array((i0 + 1, self.get_n_b()))
            problem_u1, problem_u2, problem_u3 = self._solve_f_pipe(b, ini_guess, greenFun, waitBar,
                                                                    **kwargs)
            # # numpy based version
            # self._f1_list.append(self.vec_scatter(problem_u1.get_force_petsc()))
            # self._f2_list.append(self.vec_scatter(problem_u2.get_force_petsc()))
            # self._f3_list.append(self.vec_scatter(problem_u3.get_force_petsc()))
            # PETSC based version
            self._f1_list.append(problem_u1.get_force_petsc())
            self._f2_list.append(problem_u2.get_force_petsc())
            self._f3_list.append(problem_u3.get_force_petsc())
            self._residualNorm_list.append((problem_u1.get_residualNorm(),
                                            problem_u2.get_residualNorm(),
                                            problem_u3.get_residualNorm()))
            if kwargs['check_acc']:
                self._check_f_accuracy(b, greenFun, waitBar, **kwargs)
        self._m_pipe.destroy()
        self._m_pipe_check.destroy()
        return True
    
    def get_f_list(self):
        # PETSC version
        self._f_list_PETSC2numpy()
        return self._f1_list, self._f2_list, self._f3_list
    
    def _pipe_geo_generate(self, **kwargs):
        dp = self._dp
        rp = self._rp
        lp = self._lp
        ep = self._ep
        with_cover = self._with_cover
        stokesletsInPipe_pipeFactor = self._stokesletsInPipe_pipeFactor
        
        vpgeo = tunnel_geo()  # velocity node geo of pipe
        dth = 2 * np.arcsin(dp / 2 / rp)
        
        # debug
        # OptDB = PETSc.Options()
        # stokesletsInPipe_pipeFactor = OptDB.getReal('dbg_factor', 2.5)
        # PETSc.Sys.Print('--------------------> DBG: stokesletsInPipe_pipeFactor=%f' % stokesletsInPipe_pipeFactor)
        fpgeo = vpgeo.create_deltatheta(dth=dth, radius=rp, length=lp, epsilon=ep,
                                        with_cover=with_cover,
                                        factor=stokesletsInPipe_pipeFactor)
        t_pkg = PETSc.DMComposite().create(comm=PETSc.COMM_WORLD)
        t_pkg.addDM(fpgeo.get_dmda())
        t_pkg.setFromOptions()
        t_pkg.setUp()
        t_isglb = t_pkg.getGlobalISs()
        fpgeo.set_glbIdx(t_isglb[0].getIndices())
        # cbd_geo = geo()
        # cbd_geo.combine(geo_list=[vpgeo, fpgeo, ])
        # cbd_geo.show_nodes(linestyle='-')
        self._fpgeo = fpgeo
        self._vpgeo = vpgeo
        if self._kwargs['plot_geo']:
            fpgeo.show_nodes(linestyle='-')
            vpgeo.show_nodes(linestyle='-')
        
        if kwargs['check_acc']:
            cpgeo = tunnel_geo()
            # a simple method to control the # of nodes on the pipe boundary
            tmp_fun = lambda dth: cpgeo.create_deltatheta(dth=dth, radius=rp, length=lp, epsilon=0,
                                                          with_cover=2,
                                                          factor=1).get_n_nodes()
            dth1 = 0.1  # guess 1
            dth2 = 0.01  # guess 2
            dth_min = dth2  # memory limit
            tnode = 7000  # expect # of nodes
            for _ in np.arange(10):
                nnode1 = tmp_fun(dth1)
                nnode2 = tmp_fun(dth2)
                if np.abs(nnode2 - tnode) < tnode * 0.1:
                    break
                tdth = (tnode - nnode1) * (dth2 - dth1) / (nnode2 - nnode1) + dth1
                dth1 = dth2
                dth2 = np.max((tdth, (dth_min + dth1) / 2))
            cpgeo = tunnel_geo()
            cpgeo.create_deltatheta(dth=dth2, radius=rp, length=lp, epsilon=0, with_cover=2,
                                    factor=1)
            self._cpgeo = cpgeo
            if self._kwargs['plot_geo']:
                cpgeo.show_nodes(linestyle='-')
        
        # if kwargs['plot_geo']:
        #     temp_geo = geoComposit()
        #     temp_geo.append(vpgeo)
        #     temp_geo.append(fpgeo)
        #     temp_geo.show_nodes(linestyle='-')
        return True
    
    def _pipe_geo_load(self, mat_contents):
        vpgeo = base_geo()
        vpgeo.set_nodes(mat_contents['vp_nodes'], deltalength=0)
        fpgeo = base_geo()
        fpgeo.set_nodes(mat_contents['fp_nodes'], deltalength=0)
        t_pkg = PETSc.DMComposite().create(comm=PETSc.COMM_WORLD)
        t_pkg.addDM(fpgeo.get_dmda())
        t_pkg.setFromOptions()
        t_pkg.setUp()
        t_isglb = t_pkg.getGlobalISs()
        fpgeo.set_glbIdx(t_isglb[0].getIndices())
        t_pkg = PETSc.DMComposite().create(comm=PETSc.COMM_WORLD)
        t_pkg.addDM(vpgeo.get_dmda())
        t_pkg.setFromOptions()
        t_pkg.setUp()
        t_isglb = t_pkg.getGlobalISs()
        vpgeo.set_glbIdx(t_isglb[0].getIndices())
        self._fpgeo = fpgeo
        self._vpgeo = vpgeo
        # self._cpgeo = None
        return True
    
    def _solve_m_pipe(self, **kwargs):
        # generate geo and associated nodes: a finite length pipe with covers at both side.
        t0 = time()
        obj1 = StokesFlowObj()
        obj1.set_data(self._fpgeo, self._vpgeo)
        PETSc.Sys().Print(
                'Stokeslets in pipe prepare, contain %d nodes' % self._vpgeo.get_n_nodes())
        self._m_pipe = self.create_obj_matrix(obj1, obj1, copy_obj=False, **kwargs)
        
        if kwargs['check_acc']:
            obj2 = StokesFlowObj()
            obj2.set_data(self._fpgeo, self._cpgeo)
            PETSc.Sys().Print('Stokeslets in pipe check, contain %d nodes' %
                              self._cpgeo.get_n_nodes())
            self._m_pipe_check = self.create_obj_matrix(obj2, obj2, copy_obj=False, **kwargs)
            t1 = time()
            PETSc.Sys().Print('  create matrix use %fs:' % (t1 - t0))
        return True
    
    def _solve_f_pipe(self, b, ini_guess, greenFun, waitBar=np.array((1, 1)), **kwargs):
        # calculate force at each nodes at (or outside) the pipe boundary.
        vpgeo = self._vpgeo
        outputHandle = 'vpgeo'
        u11, u21, u31 = self._solve_u1_pipe(vpgeo, outputHandle, greenFun, waitBar)
        
        # for each direction, solve force at (or outside) nodes.
        fpgeo = self._fpgeo
        kwargs_u1 = kwargs.copy()
        kwargs_u1['deltaLength'] = self._dp
        kwargs_u1['epsilon'] = self._ep
        kwargs_u1['delta'] = self._dp * self._ep
        kwargs_u1['name'] = '  _%05d/%05d_u1' % (waitBar[0], waitBar[1])
        kwargs_u1['plot'] = False
        kwargs_u1['fileHandle'] = 'stokesletsInPipeProblem_u1'
        kwargs_u1['restart'] = False
        kwargs_u1['getConvergenceHistory'] = False
        kwargs_u1['pickProblem'] = False
        
        problem_u1 = StokesFlowProblem(**kwargs_u1)
        obj_u1 = StokesFlowObj()
        obj_u1_kwargs = {
            'name': 'stokesletsInPipeObj_u1'
            }
        vpgeo.set_velocity(u11)
        obj_u1.set_data(fpgeo, vpgeo, **obj_u1_kwargs)
        problem_u1.add_obj(obj_u1)
        problem_u1.set_matrix(self._m_pipe)
        problem_u1.solve(ini_guess=ini_guess[0])
        
        kwargs_u2 = kwargs_u1.copy()
        kwargs_u2['name'] = '  _%05d/%05d_u2' % (waitBar[0], waitBar[1])
        kwargs_u2['fileHandle'] = 'stokesletsInPipeProblem_u2'
        problem_u2 = StokesFlowProblem(**kwargs_u2)
        obj_u2 = StokesFlowObj()
        obj_u2_kwargs = {
            'name': 'stokesletsInPipeObj_u2'
            }
        vpgeo.set_velocity(u21)
        obj_u2.set_data(fpgeo, vpgeo, **obj_u2_kwargs)
        problem_u2.add_obj(obj_u2)
        problem_u2.set_matrix(self._m_pipe)
        problem_u2.solve(ini_guess=ini_guess[1])
        
        kwargs_u3 = kwargs_u1.copy()
        kwargs_u3['name'] = '  _%05d/%05d_u3' % (waitBar[0], waitBar[1])
        kwargs_u3['fileHandle'] = 'stokesletsInPipeProblem_u3'
        problem_u3 = StokesFlowProblem(**kwargs_u3)
        obj_u3 = StokesFlowObj()
        obj_u3_kwargs = {
            'name': 'stokesletsInPipeObj_u3'
            }
        vpgeo.set_velocity(u31)
        obj_u3.set_data(fpgeo, vpgeo, **obj_u3_kwargs)
        problem_u3.add_obj(obj_u3)
        problem_u3.set_matrix(self._m_pipe)
        problem_u3.solve(ini_guess=ini_guess[2])
        
        return problem_u1, problem_u2, problem_u3
    
    def _solve_u1_pipe(self, pgeo, outputHandle, greenFun, waitBar=np.array((1, 1))):
        t0 = time()
        from src.StokesFlowMethod import stokeslets_matrix_3d
        # 1 velocity at pipe
        iscover = pgeo.get_iscover()
        # uR1 = np.zeros(np.sum(np.logical_not(iscover)))
        # uR1 = np.zeros(np.sum(np.abs(iscover)).astype('int'))
        uR1 = np.zeros(np.sum(np.isclose(iscover, 0)))
        uPhi1 = np.zeros_like(uR1)
        uz1 = np.zeros_like(uR1)
        uR2 = np.zeros_like(uR1)
        uPhi2 = np.zeros_like(uR1)
        uz2 = np.zeros_like(uR1)
        uR3 = np.zeros_like(uR1)
        uPhi3 = np.zeros_like(uR1)
        uz3 = np.zeros_like(uR1)
        
        # 2 velocity at cover
        #  see Liron, N., and R. Shahar. "Stokes flow due to a Stokeslet in a pipe."
        #    Journal of Fluid Mechanics 86.04 (1978): 727-744.
        tuR1_list = []
        tuPhi1_list = []
        tuz1_list = []
        tuR2_list = []
        tuPhi2_list = []
        tuz2_list = []
        tuR3_list = []
        tuPhi3_list = []
        tuz3_list = []
        cover_start_list = pgeo.get_cover_start_list()
        n_cover_node = 0
        for t_nodes in cover_start_list:
            tR = t_nodes[0]
            tphi = t_nodes[1]
            tz = np.abs(t_nodes[2])
            sign_z = np.sign(t_nodes[2])
            n_cover_node = n_cover_node + tphi.size
            tuR1, tuPhi1, tuz1, tuR2, tuPhi2, tuz2, tuR3, tuPhi3, tuz3 = greenFun.solve_u(tR, tphi,
                                                                                          tz)
            tuR1_list.append(tuR1)
            tuPhi1_list.append(tuPhi1)
            tuz1_list.append(sign_z * tuz1)
            tuR2_list.append(tuR2)
            tuPhi2_list.append(tuPhi2)
            tuz2_list.append(sign_z * tuz2)
            tuR3_list.append(sign_z * tuR3)
            tuPhi3_list.append(sign_z * tuPhi3)
            tuz3_list.append(tuz3)
        uR1 = np.hstack((np.hstack(tuR1_list), uR1))
        uPhi1 = np.hstack((np.hstack(tuPhi1_list), uPhi1))
        uz1 = np.hstack((np.hstack(tuz1_list), uz1))
        uR2 = np.hstack((np.hstack(tuR2_list), uR2))
        uPhi2 = np.hstack((np.hstack(tuPhi2_list), uPhi2))
        uz2 = np.hstack((np.hstack(tuz2_list), uz2))
        uR3 = np.hstack((np.hstack(tuR3_list), uR3))
        uPhi3 = np.hstack((np.hstack(tuPhi3_list), uPhi3))
        uz3 = np.hstack((np.hstack(tuz3_list), uz3))
        
        tuR1_list = []
        tuPhi1_list = []
        tuz1_list = []
        tuR2_list = []
        tuPhi2_list = []
        tuz2_list = []
        tuR3_list = []
        tuPhi3_list = []
        tuz3_list = []
        cover_end_list = pgeo.get_cover_end_list()
        for t_nodes in cover_end_list:
            tR = t_nodes[0]
            tphi = t_nodes[1]
            tz = np.abs(t_nodes[2])
            sign_z = np.sign(t_nodes[2])
            n_cover_node = n_cover_node + tphi.size
            tuR1, tuPhi1, tuz1, tuR2, tuPhi2, tuz2, tuR3, tuPhi3, tuz3 = greenFun.solve_u(tR, tphi,
                                                                                          tz)
            tuR1_list.append(tuR1)
            tuPhi1_list.append(tuPhi1)
            tuz1_list.append(sign_z * tuz1)
            tuR2_list.append(tuR2)
            tuPhi2_list.append(tuPhi2)
            tuz2_list.append(sign_z * tuz2)
            tuR3_list.append(sign_z * tuR3)
            tuPhi3_list.append(sign_z * tuPhi3)
            tuz3_list.append(tuz3)
        uR1 = np.hstack((uR1, np.hstack(tuR1_list)))
        uPhi1 = np.hstack((uPhi1, np.hstack(tuPhi1_list)))
        uz1 = np.hstack((uz1, np.hstack(tuz1_list)))
        uR2 = np.hstack((uR2, np.hstack(tuR2_list)))
        uPhi2 = np.hstack((uPhi2, np.hstack(tuPhi2_list)))
        uz2 = np.hstack((uz2, np.hstack(tuz2_list)))
        uR3 = np.hstack((uR3, np.hstack(tuR3_list)))
        uPhi3 = np.hstack((uPhi3, np.hstack(tuPhi3_list)))
        uz3 = np.hstack((uz3, np.hstack(tuz3_list)))
        assert n_cover_node == np.sum(np.logical_not(np.isclose(iscover, 0))), 'something is wrong'
        
        pphi, _, _ = pgeo.get_polar_coord()
        ux1 = np.cos(pphi) * uR1 - np.sin(pphi) * uPhi1
        ux2 = np.cos(pphi) * uR2 - np.sin(pphi) * uPhi2
        ux3 = np.cos(pphi) * uR3 - np.sin(pphi) * uPhi3
        uy1 = np.sin(pphi) * uR1 + np.cos(pphi) * uPhi1
        uy2 = np.sin(pphi) * uR2 + np.cos(pphi) * uPhi2
        uy3 = np.sin(pphi) * uR3 + np.cos(pphi) * uPhi3
        u1 = np.vstack((ux1, uy1, uz1)).T
        u2 = np.vstack((ux2, uy2, uz2)).T
        u3 = np.vstack((ux3, uy3, uz3)).T
        
        # u2, stokeslets, singularity.
        b = greenFun.get_b()
        stokeslets_post = np.hstack((b, 0, 0)).reshape(1, 3)
        geo_stokeslets = base_geo()
        geo_stokeslets.set_nodes(stokeslets_post, deltalength=0, resetVelocity=True)
        obj_stokeslets = StokesFlowObj()
        obj_stokeslets.set_data(geo_stokeslets, geo_stokeslets)
        obj_p = StokesFlowObj()
        obj_p.set_data(pgeo, pgeo)
        m2 = stokeslets_matrix_3d(obj_p, obj_stokeslets)
        f12 = np.array((1, 0, 0))
        f22 = np.array((0, 1, 0))
        f32 = np.array((0, 0, 1))
        u12 = np.dot(m2, f12)
        u22 = np.dot(m2, f22)
        u32 = np.dot(m2, f32)
        u11 = u1.flatten() - u12
        u21 = u2.flatten() - u22
        u31 = u3.flatten() - u32
        
        t1 = time()
        PETSc.Sys().Print('  _%05d/%05d_b=%f:    calculate %s boundary condation use: %fs' % (
            waitBar[0], waitBar[1], b, outputHandle, t1 - t0))
        
        # debug
        # length_factor = 0.3
        # pgeo.set_velocity(u1)
        # pgeo.show_velocity(length_factor=length_factor, show_nodes=False)
        # pgeo.set_velocity(u2)
        # pgeo.show_velocity(length_factor=length_factor, show_nodes=False)
        # pgeo.set_velocity(u3)
        # pgeo.show_velocity(length_factor=length_factor, show_nodes=False)
        return u11, u21, u31
    
    def _set_temp_var(self, obj1):
        # create a empty matrix, and a empty velocity vectors, to avoid use too much time to allocate memory.
        comm = PETSc.COMM_WORLD.tompi4py()
        ugeo = obj1.get_u_geo().copy()
        kwargs = self.get_kwargs()
        t_u_pkg = PETSc.DMComposite().create(comm=PETSc.COMM_WORLD)
        t_u_pkg.addDM(ugeo.get_dmda())
        t_u_pkg.setFromOptions()
        t_u_pkg.setUp()
        self._t_u11 = [t_u_pkg.createGlobalVector(), t_u_pkg.createGlobalVector(),
                       t_u_pkg.createGlobalVector()]
        self._t_u12 = [t_u_pkg.createGlobalVector(), t_u_pkg.createGlobalVector(),
                       t_u_pkg.createGlobalVector()]
        self._t_u13 = [t_u_pkg.createGlobalVector(), t_u_pkg.createGlobalVector(),
                       t_u_pkg.createGlobalVector()]
        self._t_u2 = t_u_pkg.createGlobalVector()
        
        stokeslet_m = PETSc.Mat().create(comm=PETSc.COMM_WORLD)
        stokeslet_m.setSizes((self._t_u11[0].getSizes(), self._f123_petsc[0].getSizes()))
        stokeslet_m.setType('dense')
        stokeslet_m.setFromOptions()
        stokeslet_m.setUp()
        self._stokeslet_m = stokeslet_m
        
        u_isglb = t_u_pkg.getGlobalISs()
        u_glbIdx = np.hstack(comm.allgather(u_isglb[0].getIndices()))
        temp_m = PETSc.Mat().create(comm=PETSc.COMM_WORLD)
        temp_m.setSizes((self._t_u11[0].getSizes(), self._f1_list[0].getSizes()))
        temp_m.setType('dense')
        temp_m.setFromOptions()
        temp_m.setUp()
        self._t_m = temp_m
        t_u_pkg.destroy()
        return u_glbIdx
    
    def _create_matrix_obj(self, obj1, m, INDEX='', *args):
        # set stokeslets using numerical solution.
        t_u_glbIdx_all = self._set_temp_var(
                obj1)  # index of stokeslets, maybe different from index of m matrix.
        _, u_glbIdx_all = obj1.get_u_geo().get_glbIdx()
        unodes = obj1.get_u_nodes()
        n_obj = len(self.get_all_obj_list())
        for i0, obj2 in enumerate(self.get_all_obj_list()):
            f_nodes = obj2.get_f_nodes()
            _, f_glbIdx_all = obj2.get_f_geo().get_glbIdx()
            # f_dmda = obj2.get_f_geo().get_dmda()
            f_desc = INDEX + ' %d/%d, ' % (i0 + 1, n_obj)
            for i0 in tqdm(range(obj2.get_n_f_node()), desc=f_desc, leave=False):
                # for i0 in range(obj2.get_n_f_node( )):
                t_f_node = f_nodes[i0]
                f_glb = f_glbIdx_all[i0 * 3]
                u1, u2, u3 = self._solve_stokeslets_fnode(t_f_node, unodes, t_u_glbIdx_all)
                u_range = u1.getOwnershipRange()
                u_glbIdx = u_glbIdx_all[u_range[0]:u_range[1]]
                m.setValues(u_glbIdx, f_glb + 0, u1, addv=False)
                m.setValues(u_glbIdx, f_glb + 1, u2, addv=False)
                m.setValues(u_glbIdx, f_glb + 2, u3, addv=False)
        m.assemble()
        return True
    
    def _f_list_numpy2PETSC(self):
        t_f1_list = uniqueList()  # list of forces lists for each object at or outside pipe associated with force-nodes at x axis
        t_f2_list = uniqueList()  # list of forces lists for each object at or outside pipe associated with force-nodes at y axis
        t_f3_list = uniqueList()  # list of forces lists for each object at or outside pipe associated with force-nodes at z axis
        
        f_pkg = PETSc.DMComposite().create(comm=PETSc.COMM_WORLD)
        f_pkg.addDM(self._fpgeo.get_dmda())
        f_pkg.setFromOptions()
        f_pkg.setUp()
        for f1 in self._f1_list:
            f1_petsc = f_pkg.createGlobalVector()
            f1_petsc.setFromOptions()
            f1_petsc.setUp()
            f1_petsc[:] = f1[:]
            f1_petsc.assemble()
            t_f1_list.append(f1_petsc)
        for f2 in self._f2_list:
            f2_petsc = f_pkg.createGlobalVector()
            f2_petsc.setFromOptions()
            f2_petsc.setUp()
            f2_petsc[:] = f2[:]
            f2_petsc.assemble()
            t_f2_list.append(f2_petsc)
        for f3 in self._f3_list:
            f3_petsc = f_pkg.createGlobalVector()
            f3_petsc.setFromOptions()
            f3_petsc.setUp()
            f3_petsc[:] = f3[:]
            f3_petsc.assemble()
            t_f3_list.append(f3_petsc)
        self._f1_list = t_f1_list
        self._f2_list = t_f2_list
        self._f3_list = t_f3_list
        f_pkg.destroy()
        return True
    
    def _f_list_PETSC2numpy(self):
        t_f1_list = []
        t_f2_list = []
        t_f3_list = []
        # t_f1_list = uniqueList()  # list of forces lists for each object at or outside pipe associated with force-nodes at x axis
        # t_f2_list = uniqueList()  # list of forces lists for each object at or outside pipe associated with force-nodes at y axis
        # t_f3_list = uniqueList()  # list of forces lists for each object at or outside pipe associated with force-nodes at z axis
        
        for f1_petsc in self._f1_list:  # each obj
            f1 = self.vec_scatter(f1_petsc)
            t_f1_list.append(f1)
        for f2_petsc in self._f2_list:  # each obj
            f2 = self.vec_scatter(f2_petsc)
            t_f2_list.append(f2)
        for f3_petsc in self._f3_list:  # each obj
            f3 = self.vec_scatter(f3_petsc)
            t_f3_list.append(f3)
        self._f1_list = t_f1_list
        self._f2_list = t_f2_list
        self._f3_list = t_f3_list
        return True
    
    def pickmyself_prepare(self):
        super().pickmyself_prepare()
        self._f_list_PETSC2numpy()
        return True
    
    def destroy(self):
        super().destroy()
        t1 = (self._m_pipe, self._m_pipe_check, self._t_m, self._stokeslet_m, self._t_u2)
        for ti in itertools.chain(self._t_u11, self._t_u12, self._t_u13,
                                  self._f123_petsc, t1):
            if not ti is None:
                ti.destroy()
        
        self._m_pipe = None
        self._m_pipe_check = None
        self._t_m = None
        self._stokeslet_m = None
        self._t_u2 = None
        self._t_u11 = [[] for _ in self._t_u11]
        self._t_u12 = [[] for _ in self._t_u12]
        self._t_u13 = [[] for _ in self._t_u13]
        self._f123_petsc = [[] for _ in self._f123_petsc]
        self._cpgeo.pickmyself_prepare()
        self._fpgeo.pickmyself_prepare()
        self._vpgeo.pickmyself_prepare()
        self._stokeslet_geo.pickmyself_prepare()
        return True
    
    def unpick_myself(self, check_MPISIZE=True):
        super().unpick_myself(check_MPISIZE=check_MPISIZE)
        
        fileHandle = self.get_kwargs()['forcepipe']
        t_path = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.normpath(t_path + '/' + fileHandle)
        mat_contents = loadmat(full_path)
        self._pipe_geo_load(mat_contents)
        self._kwargs['unpickedPrb'] = True
        self._f_list_numpy2PETSC()
        
        # create a empty matrix, and a empty velocity vecters,
        #   to avoid use too much time to allocate memory.
        self._set_f123()
        # this property changes it's name.
        if not hasattr(self, '_stokesletsInPipe_pipeFactor'):
            self._stokesletsInPipe_pipeFactor = self._factor
        return True


class StokesletsRingProblem(StokesFlowProblem):
    # using the symmetric of head, nodes are distribute in the line (r, 0, z).
    def check_nodes(self, nodes):
        err_msg = 'nodes are distribute in the line (r, 0, z). '
        assert np.allclose(nodes[:, 1], 0), err_msg
        return True
    
    def add_obj(self, obj):
        assert isinstance(obj, StokesFlowRingObj)
        super().add_obj(obj)
        self.check_nodes(obj.get_u_nodes())
        self.check_nodes(obj.get_f_nodes())


class StokesletsRingInPipeProblem(StokesletsRingProblem):
    # using the symmetric of pipe and head, nodes are distribute in the line (r, 0, z).
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._th = kwargs['stokeslets_threshold']
        if kwargs['use_tqdm_notebook']:
            self._tqdm = tqdm_notebook
        else:
            self._tqdm = tqdm
    
    def check_nodes(self, nodes):
        super().check_nodes(nodes)
        err_msg = 'r=%f out of the range [0, 1). ' % np.max(nodes[:, 0])
        assert np.max(nodes[:, 0]) < 1 and np.min(nodes[:, 0]) >= 0, err_msg
        
        err_msg = 'r=%f out of the range [0, 1). ' % np.min(nodes[:, 0])
        assert np.min(nodes[:, 0]) >= 0, err_msg
        return True
    
    def _create_matrix_obj(self, obj1, m, INDEX='', *args):
        # set stokeslets using theoretical solution from Liron1977
        from src.stokesletsInPipe import StokesletsRinginPipe_light
        
        _, u_glbIdx_all = obj1.get_u_geo().get_glbIdx()
        u_nodes = obj1.get_u_nodes()
        n_obj = len(self.get_all_obj_list())
        green_fun = StokesletsRinginPipe_light(threshold=self._th)
        green_fun.solve_prepare_light()
        b_use = -np.inf
        
        for i00, obj2 in enumerate(self.get_all_obj_list()):
            use_matrix_method = 'pf_stokesletsRingInPipe'
            err_msg = 'keyword \'matrix_method\' should be \'%s\' ' % use_matrix_method
            assert obj2.get_matrix_method() == use_matrix_method, err_msg
            f_nodes = obj2.get_f_nodes()
            f_dmda_range = obj2.get_f_geo().get_dmda().getRanges()[0]
            _, f_glbIdx_all = obj2.get_f_geo().get_glbIdx()
            f_desc = INDEX + ' %d/%d, ' % (i00 + 1, n_obj)
            for i0 in self._tqdm(range(f_dmda_range[0], f_dmda_range[1]), desc=f_desc, leave=True):
                t_f_node = f_nodes[i0]
                tb = t_f_node[0]
                if not np.isclose(tb, b_use):
                    green_fun.set_b(tb)
                    green_fun.solve_prepare_b()
                    b_use = tb
                f_glb = f_glbIdx_all[i0 * 3]
                for i1, t_u_node in enumerate(u_nodes):
                    tru = t_u_node[0]
                    u_glb = u_glbIdx_all[i1 * 3]
                    t_z = t_u_node[2] - t_f_node[2]
                    abs_z = np.abs(t_z)
                    sign_z = np.sign(t_z)
                    tm = np.array(green_fun.solve_u_light(tru, abs_z)).reshape(3, 3).T
                    tsign = np.array(((1, 1, sign_z), (1, 1, sign_z), (sign_z, sign_z, 1)))
                    rows = (u_glb + 0, u_glb + 1, u_glb + 2)
                    cols = (f_glb + 0, f_glb + 1, f_glb + 2)
                    m.setValues(rows, cols, tsign * tm * tb, addv=False)
        m.assemble()
        return True


class StokesletsRingInPipeProblemSymz(StokesletsRingInPipeProblem):
    # assert another symmetry in z drection
    def check_nodes(self, nodes):
        super().check_nodes(nodes)
        err_msg = 'assert additional symmetry along z so z>=0. '
        assert np.all(nodes[:, 2] >= 0), err_msg
        return True
    
    def _create_matrix_obj(self, obj1, m, INDEX='', *args):
        # set stokeslets using theoretical solution from Liron1977
        from src.stokesletsInPipe import StokesletsRinginPipe_light
        
        _, u_glbIdx_all = obj1.get_u_geo().get_glbIdx()
        u_nodes = obj1.get_u_nodes()
        n_obj = len(self.get_all_obj_list())
        green_fun = StokesletsRinginPipe_light(threshold=self._th)
        green_fun.solve_prepare_light()
        b_use = -np.inf
        
        for i00, obj2 in enumerate(self.get_all_obj_list()):
            use_matrix_method = 'pf_stokesletsRingInPipeProblemSymz'
            err_msg = 'keyword \'matrix_method\' should be \'%s\' ' % use_matrix_method
            assert obj2.get_matrix_method() == use_matrix_method, err_msg
            f_nodes = obj2.get_f_nodes()
            f_dmda_range = obj2.get_f_geo().get_dmda().getRanges()[0]
            _, f_glbIdx_all = obj2.get_f_geo().get_glbIdx()
            f_desc = INDEX + ' %d/%d, ' % (i00 + 1, n_obj)
            for i0 in self._tqdm(range(f_dmda_range[0], f_dmda_range[1]), desc=f_desc, leave=True):
                t_f_node = f_nodes[i0]
                tb = t_f_node[0]
                if not np.isclose(tb, b_use):
                    green_fun.set_b(tb)
                    green_fun.solve_prepare_b()
                    b_use = tb
                f_glb = f_glbIdx_all[i0 * 3]
                for i1, t_u_node in enumerate(u_nodes):
                    u_glb = u_glbIdx_all[i1 * 3]
                    tru = t_u_node[0]
                    
                    # part 1 of force node, z >= 0
                    t_z = t_u_node[2] - t_f_node[2]
                    abs_z = np.abs(t_z)
                    sign_z = np.sign(t_z)
                    tm1 = np.array(green_fun.solve_u_light(tru, abs_z)).reshape(3, 3).T
                    tsign1 = np.array(((1, 1, sign_z), (1, 1, sign_z), (sign_z, sign_z, 1)))
                    # symmetric part of force node, z <= 0,
                    #   thus sign_z == 1, fR'=-fR, fphi'=fphi, fz'=fz.
                    t_z = t_u_node[2] + t_f_node[2]
                    tm2 = np.array(green_fun.solve_u_light(tru, t_z)).reshape(3, 3).T
                    tsign2 = np.array(((-1, 1, 1), (-1, 1, 1), (-1, 1, 1)))
                    tm = (tsign1 * tm1 + tsign2 * tm2) * tb
                    
                    rows = (u_glb + 0, u_glb + 1, u_glb + 2)
                    cols = (f_glb + 0, f_glb + 1, f_glb + 2)
                    m.setValues(rows, cols, tm, addv=False)
        m.assemble()
        return True


class SelfRepeatObj(StokesFlowObj):
    def set_data(self, f_geo: SelfRepeat_body_geo, u_geo: base_geo, name='...', **kwargs):
        assert isinstance(f_geo, SelfRepeat_body_geo)
        super().set_data(f_geo, u_geo, name, **kwargs)
        self._type = 'self repeat obj'
        return True
    
    def get_total_force(self, center=None):
        repeat_n = self.get_f_geo().repeat_n
        f_t = super().get_total_force(center=center)
        return f_t * repeat_n


class SelfRepeatHlxProblem(StokesFlowProblem):
    # Todo: check the directions of the geometry and the rigid body velocity.
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self._full_obj_list = []
        self._full_obj_list = uniqueList()
    
    @property
    def full_obj_list(self):
        return self._full_obj_list
    
    def add_obj(self, obj_pair):
        part_obj, full_obj = obj_pair
        assert isinstance(part_obj, SelfRepeatObj)
        
        ugeo = full_obj.get_u_geo()
        fgeo = full_obj.get_f_geo()
        assert isinstance(ugeo, SelfRepeat_FatHelix)
        assert isinstance(fgeo, SelfRepeat_FatHelix)
        
        self._full_obj_list.append(full_obj)
        return super().add_obj(part_obj)


class SelfRotateObj(StokesFlowObj):
    def set_data(self, *args, **kwargs):
        super().set_data(*args, **kwargs)
        self._type = 'self rotate obj'
        return True
    
    def get_total_force(self, center=None):
        f_t = super().get_total_force(center=center)
        f = f_t[:3]
        t = f_t[3:]
        problem_n_copy = self.get_problem().get_kwargs()['problem_n_copy']
        problem_norm = self.get_problem().get_kwargs()['problem_norm']
        
        F, T = 0, 0
        for thetai in np.linspace(0, 2 * np.pi, problem_n_copy, endpoint=False):
            rot_M = get_rot_matrix(problem_norm, thetai)
            # PETSc.Sys.Print(np.dot(rot_M, f))
            F = F + np.dot(rot_M, f)
            T = T + np.dot(rot_M, t)
        return np.hstack((F, T))


class SelfRotateProblem(StokesFlowProblem):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._problem_center = kwargs['problem_center']
        self._problem_norm = kwargs['problem_norm']
        self._problem_n_copy = kwargs['problem_n_copy']
    
    def set_rigid_velocity(self, u, w):
        problem_norm = self.get_kwargs()['problem_norm']
        problem_center = self.get_kwargs()['problem_center']
        U = np.hstack((u * problem_norm, w * problem_norm))
        for tobj in self.get_obj_list():
            tobj.set_rigid_velocity(U, problem_center)
        return True
    
    def add_obj(self, obj):
        assert isinstance(obj, SelfRotateObj)
        return super().add_obj(obj)
    
    def show_all_u_nodes(self, linestyle='-'):
        problem_norm = self.get_kwargs()['problem_norm']
        problem_n_copy = self.get_kwargs()['problem_n_copy']
        problem_center = self.get_kwargs()['problem_center']
        
        geo_list = uniqueList()
        for obj1 in self.get_all_obj_list():
            tugeo = obj1.get_u_geo()
            if isinstance(obj1, SelfRotateObj):
                for thetai in np.linspace(0, 2 * np.pi, problem_n_copy, endpoint=False):
                    tugeo2 = tugeo.copy()
                    tugeo2.node_rotation(problem_norm, thetai, problem_center)
                    geo_list.append(tugeo2)
            else:
                geo_list.append(tugeo)
        temp_geo = base_geo()
        temp_geo.combine(geo_list)
        temp_geo.show_nodes(linestyle)
        return True
    
    # def show_f_nodes(self, linestyle='-'):
    #     err_msg='not finish yet'
    #     assert 1==2, err_msg
    #
    # def show_f_u_nodes(self, linestyle='-'):
    #     err_msg='not finish yet'
    #     assert 1==2, err_msg
    #
    # def show_force(self, length_factor=1, show_nodes=True):
    #     err_msg='not finish yet'
    #     assert 1==2, err_msg
    #
    # def show_velocity(self, length_factor=1, show_nodes=True):
    #     err_msg='not finish yet'
    #     assert 1==2, err_msg


class ForceFreeComposite:
    def __init__(self, center: np.array, norm: np.array, name='...', *args):
        self._obj_list = uniqueList()
        self._rel_U_list = []  # (ux,uy,uz,wx,wy,wz)
        # self._rel_U_list = uniqueList()  # (ux,uy,uz,wx,wy,wz)
        self._index = -1  # index of object
        self._problem = None
        self._center = None
        self.set_center(center)
        self._norm = None
        self.set_norm(norm)
        self._psi = 0  # rotate of the Composite about the norm axis
        self._n_fnode = 0
        self._n_unode = 0
        self._f_glbIdx = np.array([])  # global indices
        self._f_glbIdx_all = np.array([])  # global indices for all process.
        self._u_glbIdx = np.array([])  # global indices
        self._u_glbIdx_all = np.array([])  # global indices for all process.
        self._type = 'ForceFreeComposite'  # object type
        self._name = name  # object name
        self._ref_U = np.zeros(6)  # ux, uy, uz, omega_x, omega_y, omega_z
        # self._sum_force = np.inf * np.ones(6)  # [F, T]==0 to satisfy the force free equations.
        self._min_ds = np.inf  # min deltalength of objects in the composite
        self._f_dmda = None
        self._u_dmda = None
        self.set_dmda()
        # the following properties store the location history of the composite.
        self._update_fun = Adams_Moulton_Methods
        self._update_order = 1
        self._locomotion_fct = np.ones(3)
        self._center_hist = []
        self._norm_hist = []
        self._ref_U_hist = []  # (ux,uy,uz,wx,wy,wz)
        self._displace_hist = []
        self._rotation_hist = []
    
    def __repr__(self):
        return self.get_obj_name()
    
    def __str__(self):
        t_str = self.get_name() + ': {'
        for subobj in self.get_obj_list():
            t_str = t_str + subobj.get_name() + '; '
        t_str = t_str + '}'
        return t_str
    
    def add_obj(self, obj, rel_U):
        self._obj_list.append(obj)
        obj.set_index(self.get_n_obj())
        obj.set_problem(self)
        obj.set_rigid_velocity(rel_U, self.get_center())
        self._rel_U_list.append(rel_U)
        self._n_fnode += obj.get_n_f_node()
        self._n_unode += obj.get_n_u_node()
        self._min_ds = np.min((self._min_ds, obj.get_u_geo().get_deltaLength()))
        return True
    
    def set_rel_U_list(self, rel_U_list):
        err_msg = 'wrong rel_U_list shape. '
        assert len(self.get_obj_list()) == len(rel_U_list), err_msg
        err_msg = 'wrong rel_U shape. '
        for sub_obj, rel_U in zip(self.get_obj_list(), rel_U_list):
            assert rel_U.size == 6, err_msg
            sub_obj.set_rigid_velocity(rel_U, self.get_center())
        self._rel_U_list = rel_U_list
        return True
    
    def get_f_dmda(self):
        return self._f_dmda
    
    def get_u_dmda(self):
        return self._u_dmda
    
    def set_dmda(self):
        # additional degrees of freedom for force free.
        self._f_dmda = PETSc.DMDA().create(sizes=(6,), dof=1, stencil_width=0,
                                           comm=PETSc.COMM_WORLD)
        self._f_dmda.setFromOptions()
        self._f_dmda.setUp()
        self._u_dmda = PETSc.DMDA().create(sizes=(6,), dof=1, stencil_width=0,
                                           comm=PETSc.COMM_WORLD)
        self._u_dmda.setFromOptions()
        self._u_dmda.setUp()
        return True
    
    def destroy_dmda(self):
        self._f_dmda.destroy()
        self._u_dmda.destroy()
        self._f_dmda = None
        self._u_dmda = None
        return True
    
    def get_n_obj(self):
        return len(self._obj_list)
    
    def get_obj_list(self):
        return self._obj_list
    
    def get_n_f_node(self):
        return self._n_fnode
    
    def get_n_u_node(self):
        return self._n_unode
    
    def get_rel_U_list(self):
        return self._rel_U_list
    
    def get_center(self):
        return self._center
    
    def set_center(self, center):
        err_msg = 'center=[x, y, z] has 3 components. '
        assert center.size == 3, err_msg
        self._center = center
        return True
    
    def get_norm(self):
        return self._norm
    
    def get_psi(self):
        return self._psi
    
    def set_norm(self, norm):
        err_msg = 'norm=[x, y, z] has 3 components and ||norm|| > 0. '
        assert norm.size == 3 and np.linalg.norm(norm) > 0, err_msg
        self._norm = norm / np.linalg.norm(norm)
        return True
    
    def get_index(self):
        return self._index
    
    def get_min_ds(self):
        return self._min_ds
    
    def get_type(self):
        return self._type
    
    def get_obj_name(self):
        return self._type + ' (index %d)' % self._index
    
    def get_name(self):
        return self._name
    
    def set_name(self, name):
        self._name = name
        return True
    
    def set_index(self, new_index):
        self._index = new_index
        return True
    
    def set_f_glbIdx(self, indices):
        comm = PETSc.COMM_WORLD.tompi4py()
        self._f_glbIdx = indices
        self._f_glbIdx_all = np.hstack(comm.allgather(indices))
        return True
    
    def get_f_glbIdx(self):
        return self._f_glbIdx, self._f_glbIdx_all
    
    def set_u_glbIdx(self, indices):
        comm = PETSc.COMM_WORLD.tompi4py()
        self._u_glbIdx = indices
        self._u_glbIdx_all = np.hstack(comm.allgather(indices))
        return True
    
    def get_u_glbIdx(self):
        return self._u_glbIdx, self._u_glbIdx_all
    
    def get_combined_obj(self):
        obj0 = StokesFlowObj()
        obj0.combine(self.get_obj_list(), set_re_u=True, set_force=True)
        return obj0
    
    def set_problem(self, problem: 'StokesFlowProblem'):
        self._problem = problem
        return True
    
    def clear_obj_list(self):
        self._obj_list = uniqueList()
        self._rel_U_list = []
        # self._rel_U_list = uniqueList()
        return True
    
    def print_info(self):
        PETSc.Sys.Print('  %s: father %s, type %s, index %d, force nodes %d, velocity nodes %d'
                        % (self.get_name(), self._problem.get_name(), self._type, self.get_index(),
                           self.get_n_f_node(), self.get_n_u_node()))
        PETSc.Sys.Print('  %s: norm %s, center %s' %
                        (self.get_name(), str(self.get_norm()), str(self.get_center())))
        for obj, rel_U in zip(self.get_obj_list(), self.get_rel_U_list()):
            PETSc.Sys.Print('  %s: relative velocity %s' % (obj.get_name(), str(rel_U)))
        for obj in self._obj_list:
            obj.print_info()
        return True
    
    def copy(self):
        composite2 = copy.copy(self)
        composite2.set_problem(self._problem)
        composite2.set_index(-1)
        composite2.set_dmda()
        composite2.clear_obj_list()
        for sub_obj, rel_U in zip(self.get_obj_list(), self.get_rel_U_list()):
            obj1 = sub_obj.copy()
            composite2.add_obj(obj1, rel_U)
        return composite2
    
    def move(self, displacement):
        for subobj in self.get_obj_list():
            subobj.move(displacement=displacement)
        self._center = self._center + displacement
        return True
    
    def node_rotation(self, norm=np.array([0, 0, 1]), theta=np.zeros(1), rotation_origin=None):
        rotM = get_rot_matrix(norm, theta)
        self.node_rotM(rotM=rotM, rotation_origin=rotation_origin)
        
        # dbg, current version have no effect.
        t_norm = self._norm.copy()
        t1 = np.dot(t_norm, norm) / (np.linalg.norm(t_norm) * np.linalg.norm(norm))
        self._psi = t1 * theta + self._psi
        return True
    
    def node_rotM(self, rotM, rotation_origin=None):
        rotation_origin = self._center if rotation_origin is None else rotation_origin
        for subobj, rel_U in zip(self.get_obj_list(), self.get_rel_U_list()):
            subobj.node_rotM(rotM=rotM, rotation_origin=rotation_origin)
            subobj.set_rigid_velocity(rel_U, self.get_center())
        
        t_origin = self._center
        self._center = np.dot(rotM, (self._center - rotation_origin)) + rotation_origin
        self._norm = np.dot(rotM, self._norm) / np.linalg.norm(self._norm)
        
        t1 = []
        for rel_U0 in self.get_rel_U_list():
            tU = np.dot(rotM, rel_U0[:3])
            tW = np.dot(rotM, rel_U0[3:])
            t1.append(np.hstack((tU, tW)))
        self._rel_U_list = t1
        
        ref_U0 = self.get_ref_U()
        tU = np.dot(rotM, ref_U0[:3])
        tW = np.dot(rotM, ref_U0[3:])
        self.set_ref_U(np.hstack((tU, tW)))
        return True
    
    def set_ref_U(self, U):
        self._ref_U = U
        return True
    
    def get_ref_U(self):
        return self._ref_U
    
    def get_ref_Ux(self):
        return self._ref_U[0]
    
    def get_ref_Uy(self):
        return self._ref_U[1]
    
    def get_ref_Uz(self):
        return self._ref_U[2]
    
    def get_ref_Omegax(self):
        return self._ref_U[3]
    
    def get_ref_Omegay(self):
        return self._ref_U[4]
    
    def get_ref_Omegaz(self):
        return self._ref_U[5]
    
    # def set_total_force(self, sum_force):
    #     self._sum_force = sum_force
    #     return True
    
    def get_total_force(self):
        sum_F = np.sum(
                [tobj.get_total_force(center=self.get_center()) for tobj in self.get_obj_list()],
                axis=0)
        return sum_F
    
    def show_velocity(self, length_factor=1, show_nodes=True):
        geo_list = uniqueList()
        for obj1 in self.get_obj_list():
            geo_list.append(obj1.get_u_geo())
        temp_geo = base_geo()
        temp_geo.combine(geo_list)
        temp_geo.show_velocity(length_factor=length_factor, show_nodes=show_nodes)
        return True
    
    def show_f_nodes(self, linestyle='-'):
        geo_list = uniqueList()
        for obj1 in self.get_obj_list():
            geo_list.append(obj1.get_f_geo())
        temp_geo = base_geo()
        temp_geo.combine(geo_list)
        temp_geo.show_nodes(linestyle)
        return True
    
    def get_f_nodes(self):
        geo_list = uniqueList()
        for obj1 in self.get_obj_list():
            geo_list.append(obj1.get_f_geo())
        temp_geo = base_geo()
        temp_geo.combine(geo_list)
        return temp_geo.get_nodes()
    
    def show_u_nodes(self, linestyle='-'):
        geo_list = uniqueList()
        for obj1 in self.get_obj_list():
            geo_list.append(obj1.get_u_geo())
        temp_geo = base_geo()
        temp_geo.combine(geo_list)
        temp_geo.show_nodes(linestyle)
        return True
    
    def get_u_nodes(self):
        geo_list = uniqueList()
        for obj1 in self.get_obj_list():
            geo_list.append(obj1.get_u_geo())
        temp_geo = base_geo()
        temp_geo.combine(geo_list)
        return temp_geo.get_nodes()
    
    def png_u_nodes(self, filename, linestyle='-'):
        geo_list = uniqueList()
        for obj1 in self.get_obj_list():
            geo_list.append(obj1.get_u_geo())
        temp_geo = base_geo()
        temp_geo.combine(geo_list)
        temp_geo.png_nodes(filename, linestyle)
        return True
    
    def show_f_u_nodes(self, linestyle='-'):
        f_geo_list = uniqueList()
        u_geo_list = uniqueList()
        for obj1 in self.get_obj_list():
            f_geo_list.append(obj1.get_f_geo())
            if obj1.get_f_geo() is not obj1.get_u_geo():
                u_geo_list.append(obj1.get_u_geo())
        f_geo = base_geo()
        f_geo.combine(f_geo_list)
        u_geo = base_geo()
        u_geo.combine(u_geo_list)
        temp_geo = geoComposit()
        temp_geo.append(u_geo)
        temp_geo.append(f_geo)
        temp_geo.show_nodes(linestyle)
        return True
    
    def save_mat(self):
        addInfo = self._problem.get_name() + '_'
        for subobj in self.get_obj_list():
            subobj.save_mat(addInfo)
        
        comm = PETSc.COMM_WORLD.tompi4py()
        rank = comm.Get_rank()
        u_glbIdx, u_glbIdx_all = self.get_u_glbIdx()
        f_glbIdx, f_glbIdx_all = self.get_f_glbIdx()
        t_nodes = np.full(6, np.nan).reshape((2, 3))
        filename = self._problem.get_name() + '_' + self.get_name() + '.mat'
        if rank == 0:
            savemat(filename,
                    {
                        'fnodes':       t_nodes,
                        'unodes':       t_nodes,
                        'u_glbIdx':     u_glbIdx,
                        'u_glbIdx_all': u_glbIdx_all,
                        'f_glbIdx':     f_glbIdx,
                        'f_glbIdx_all': f_glbIdx_all,
                        'force':        np.zeros(6),
                        # 're_velocity':  self._sum_force,
                        'velocity':     self._ref_U,
                        },
                    oned_as='column')
        PETSc.Sys.Print('%s: save information to %s' % (str(self), filename))
        return True
    
    def vtk(self, filename, stp_idx=0):
        for obj0 in self.get_obj_list():
            obj0.vtk(filename, stp_idx)
        return True
    
    def vtk_self(self, filename, stp_idx=0, matrix_method=None):
        if matrix_method is None:
            matrix_method = self.get_obj_list()[0].get_matrix_method()
        
        t0 = time()
        obj0 = obj_dic[matrix_method]()
        obj0.combine(self.get_obj_list(), set_re_u=True, set_force=True)
        obj0.set_name('comp')
        obj0.set_matrix_method(matrix_method)
        # self.show_velocity()
        # obj0.show_velocity()
        obj0.vtk(filename, stp_idx)
        t1 = time()
        PETSc.Sys.Print('%s: write self vtk files use: %fs' % (str(self), (t1 - t0)))
        return True
    
    def pickmyself_prepare(self):
        self.destroy_dmda()
        for sub_obj in self.get_obj_list():
            sub_obj.pickmyself_prepare()
        return True
    
    def unpick_myself(self):
        self.set_dmda()
        for sub_obj in self.get_obj_list():
            sub_obj.unpick_myself()
        return True
    
    def set_update_para(self, fix_x=False, fix_y=False, fix_z=False,
                        update_fun=Adams_Moulton_Methods,
                        update_order=1):
        # for a cutoff infinity symmetric problem,
        #   each time step set the obj in the center of the cutoff region to improve the accuracy.
        self._locomotion_fct = np.array((not fix_x, not fix_y, not fix_z), dtype=np.float)
        self._update_fun = update_fun
        self._update_order = update_order
        return self._locomotion_fct
    
    def update_location(self, eval_dt, print_handle=''):
        fct = self._locomotion_fct
        ref_U = self.get_ref_U()
        self._ref_U_hist.append(ref_U)
        norm = self.get_norm()
        PETSc.Sys.Print('  -->', str(self), print_handle)
        PETSc.Sys.Print('      ref_U', ref_U)
        PETSc.Sys.Print('      norm', norm)
        tU = np.dot(ref_U[:3], norm) / np.dot(norm, norm)
        tW = np.dot(ref_U[3:], norm) / np.dot(norm, norm)
        PETSc.Sys.Print('      |ref_U|',
                        np.hstack((np.linalg.norm(ref_U[:3]), np.linalg.norm(ref_U[3:]))))
        PETSc.Sys.Print('      ref_U projection on norm', np.hstack((tU, tW)))
        # # dbg
        # rel_U_head = self.get_rel_U_list()[0]
        # rel_U_tail = self.get_rel_U_list()[1]
        # PETSc.Sys.Print('      U_head', rel_U_head + ref_U)
        # PETSc.Sys.Print('      U_tail', rel_U_tail + ref_U)
        
        order = np.min((len(self.get_ref_U_hist()), self._update_order))
        fct_list = self.get_ref_U_hist()[-1:-(order + 1):-1]
        dst_fct_list = [fct[:3] for fct in fct_list]
        rot_fct_list = [fct[3:] for fct in fct_list]
        distance_true = self._update_fun(order, dst_fct_list, eval_dt)
        rotation = self._update_fun(order, rot_fct_list, eval_dt)
        # distance_true = ref_U[:3] * eval_dt
        # rotation = ref_U[3:] * eval_dt
        distance = distance_true * fct
        self.move(distance)
        self.node_rotation(norm=rotation, theta=np.linalg.norm(rotation))
        self._center_hist.append(self._center)
        self._norm_hist.append(self._norm)
        self._displace_hist.append(distance_true)
        self._rotation_hist.append(rotation)
        
        for sub_obj, rel_U in zip(self.get_obj_list(), self.get_rel_U_list()):
            distance = rel_U[:3] * eval_dt
            rotation = rel_U[3:] * eval_dt
            sub_obj.move(distance)
            sub_obj.node_rotation(norm=rotation, theta=np.linalg.norm(rotation))
            sub_obj.update_location(eval_dt)
        return True
    
    def get_locomotion_fct(self):
        return self._locomotion_fct
    
    def get_center_hist(self):
        return self._center_hist
    
    def get_norm_hist(self):
        return self._norm_hist
    
    def get_ref_U_hist(self):
        return self._ref_U_hist
    
    def get_displace_hist(self):
        return self._displace_hist
    
    def get_rotation_hist(self):
        return self._rotation_hist
    
    def get_update_fun(self):
        return self._update_fun
    
    def get_update_order(self):
        return self._update_order
    
    def get_locomotion_fct(self):
        return self._locomotion_fct


class GivenTorqueComposite(ForceFreeComposite):
    """
    [ M R ] [ F    ] = [ Uref + Wref×ri + Urel (-Ubi) ]
    [ R 0 ] [ Wref ]   [ Tgiven                        ]
    """
    
    def __init__(self, center: np.array, norm: np.array, givenT=np.zeros(3), givenU=np.zeros(3),
                 name='...', *args):
        super().__init__(center, norm, name, *args)
        self._type = 'GivenTorqueComposite'  # object type
        self._givenT = np.zeros(3)  # given Torque.
        self.set_givenT(givenT)
        self._givenU = np.zeros(3)  # given velocity.
        self.set_givenU(givenU)
    
    def set_dmda(self):
        # additional degrees of freedom for force free.
        self._f_dmda = PETSc.DMDA().create(sizes=(3,), dof=1, stencil_width=0,
                                           comm=PETSc.COMM_WORLD)
        self._f_dmda.setFromOptions()
        self._f_dmda.setUp()
        self._u_dmda = PETSc.DMDA().create(sizes=(3,), dof=1, stencil_width=0,
                                           comm=PETSc.COMM_WORLD)
        self._u_dmda.setFromOptions()
        self._u_dmda.setUp()
        return True
    
    def update_location(self, eval_dt, print_handle=''):
        fct = self._locomotion_fct
        ref_U = self.get_ref_U()
        self._ref_U_hist.append(ref_U)
        norm = self.get_norm()
        PETSc.Sys.Print('  -->', str(self), print_handle)
        PETSc.Sys.Print('      ref_U', ref_U)
        PETSc.Sys.Print('      norm', norm)
        tU = np.dot(ref_U[:3], norm) / np.dot(norm, norm)
        tW = np.dot(ref_U[3:], norm) / np.dot(norm, norm)
        PETSc.Sys.Print('      |ref_U|',
                        np.hstack((np.linalg.norm(ref_U[:3]), np.linalg.norm(ref_U[3:]))))
        PETSc.Sys.Print('      ref_U projection on norm', np.hstack((tU, tW)))
        
        order = np.min((len(self.get_ref_U_hist()), self._update_order))
        fct_list = self.get_ref_U_hist()[-1:-(order + 1):-1]
        dst_fct_list = [fct[:3] for fct in fct_list]
        rot_fct_list = [fct[3:] for fct in fct_list]
        distance_true = self._update_fun(order, dst_fct_list, eval_dt)
        rotation = self._update_fun(order, rot_fct_list, eval_dt)
        # distance_true = ref_U[:3] * eval_dt
        # rotation = ref_U[3:] * eval_dt
        distance = distance_true * fct
        self.move(distance)
        self.node_rotation(norm=rotation, theta=np.linalg.norm(rotation))
        self._center_hist.append(self._center)
        self._norm_hist.append(self._norm)
        self._displace_hist.append(distance_true)
        self._rotation_hist.append(rotation)
        
        for sub_obj, rel_U in zip(self.get_obj_list(), self.get_rel_U_list()):
            distance = rel_U[:3] * eval_dt
            rotation = rel_U[3:] * eval_dt
            sub_obj.move(distance)
            sub_obj.node_rotation(norm=rotation, theta=np.linalg.norm(rotation))
            sub_obj.update_location(eval_dt)
        return True
    
    def set_ref_U(self, W_ref):
        # in this case, U->W_ref is the reference spin.
        W_ref = np.array(W_ref).flatten()
        err_msg = 'in %s composite, W_ref=[wx, wy, wz] has 3 components. ' % repr(self)
        assert W_ref.size == 3, err_msg
        
        # U_ref=[ux, uy, uz] is a rigid body motion. U_ref = givenU * norm + u_b, where u_b is the background flow.
        # therefore, u_b is also a rigid body motion. here use the background velocity at composite center.
        givenU = self.get_givenU()
        problem = self._problem
        if isinstance(problem, _GivenFlowProblem):
            # # dbg
            # PETSc.Sys.Print(givenU)
            # PETSc.Sys.Print(problem.get_given_flow_at(self.get_center()))
            givenU = givenU + problem.get_given_flow_at(self.get_center())
        self._ref_U = np.hstack((givenU, W_ref))
        return True
    
    def get_givenT(self):
        return self._givenT
    
    def set_givenT(self, givenT):
        givenT = np.array(givenT).flatten()
        err_msg = 'givenT=[tx, ty, tz] has 3 components. '
        assert givenT.size == 3, err_msg
        self._givenT = givenT
        return True
    
    def get_givenU(self):
        return self._givenU
    
    def set_givenU(self, givenU):
        givenU = np.array(givenU).flatten()
        err_msg = 'givenU=[ux, uy, uz] has 3 components. '
        assert givenU.size == 3, err_msg
        self._givenU = givenU
        return True


class GivenVelocityComposite(ForceFreeComposite):
    """
    currently, only work for two parts currently.
    [ Mhh  Mht  Rh  0  ] [ Fh ]        = [ Uref + Urel (-Ubi) ]
    [ Mth  Mtt  0   Rt ] [ Ft ]          [ Uref + Urel (-Ubi) ]
    [ I    I    0   0  ] [ Wrel_head ]   [ 0     ]
    [ Rh   Rt   0   0  ] [ wrel_tail ]   [ 0     ]
    Wref_head == Wref_head == 0
    """
    
    def __init__(self, center: np.array, norm: np.array, givenU=np.zeros(3), name='...', *args):
        super().__init__(center, norm, name, *args)
        self._type = 'GivenVelocityComposite'  # object type
        self._givenU = np.zeros(3)  # given velocity.
        self.set_givenU(givenU)
    
    def add_obj(self, obj, rel_U):
        # rel_w(x,y,z) are solved later.
        err_msg = 'rel_U=[rel_ux, rel_uy, rel_uz, 0, 0, 0] for the GivenVelocityComposite'
        assert np.all(rel_U[3:] == np.zeros(3)), err_msg
        super().add_obj(obj, rel_U)
        return True
    
    def update_location(self, eval_dt, print_handle=''):
        fct = self._locomotion_fct
        ref_U = self.get_ref_U()
        self._ref_U_hist.append(ref_U)
        norm = self.get_norm()
        # currently, only wrok for two part composite.
        err_msg = 'current version: len(self.get_obj_list()) == 2'
        assert len(self.get_obj_list()) == 2, err_msg
        U_rel_head = self.get_rel_U_list()[0]
        U_rel_tail = self.get_rel_U_list()[1]
        PETSc.Sys.Print('  -->', str(self), print_handle)
        PETSc.Sys.Print('      ref_U', ref_U)
        PETSc.Sys.Print('      norm', norm)
        tU = np.dot(ref_U[:3], norm) / np.dot(norm, norm)
        tW = np.dot(ref_U[3:], norm) / np.dot(norm, norm)
        PETSc.Sys.Print('      |ref_U|',
                        np.hstack((np.linalg.norm(ref_U[:3]), np.linalg.norm(ref_U[3:]))))
        PETSc.Sys.Print('      ref_U projection on norm', np.hstack((tU, tW)))
        PETSc.Sys.Print('      U_rel_head', U_rel_head)
        PETSc.Sys.Print('      U_rel_tail', U_rel_tail)
        PETSc.Sys.Print('      Wrel_motor', (- U_rel_head + U_rel_tail)[3:])
        PETSc.Sys.Print('      U_head', U_rel_head + ref_U)
        PETSc.Sys.Print('      U_tail', U_rel_tail + ref_U)
        
        order = np.min((len(self.get_ref_U_hist()), self._update_order))
        fct_list = self.get_ref_U_hist()[-1:-(order + 1):-1]
        dst_fct_list = [fct[:3] for fct in fct_list]
        rot_fct_list = [fct[3:] for fct in fct_list]
        distance_true = self._update_fun(order, dst_fct_list, eval_dt)
        rotation = self._update_fun(order, rot_fct_list, eval_dt)
        # distance_true = ref_U[:3] * eval_dt
        # rotation = ref_U[3:] * eval_dt
        distance = distance_true * fct
        self.move(distance)
        self.node_rotation(norm=rotation, theta=np.linalg.norm(rotation))
        self._center_hist.append(self._center)
        self._norm_hist.append(self._norm)
        self._displace_hist.append(distance_true)
        self._rotation_hist.append(rotation)
        
        for sub_obj, rel_U in zip(self.get_obj_list(), self.get_rel_U_list()):
            distance = rel_U[:3] * eval_dt
            rotation = rel_U[3:] * eval_dt
            sub_obj.move(distance)
            sub_obj.node_rotation(norm=rotation, theta=np.linalg.norm(rotation))
            sub_obj.update_location(eval_dt)
        return True
    
    def set_ref_U(self, W_ref):
        # in this case, U->W_ref is the reference spin of head and tail.
        # currently, the the composite can only handle two part problem.
        W_ref = np.array(W_ref).flatten()
        
        # W_ref = [W_head_x, W_head_y, W_head_z, W_tail_x, W_tail_y, W_tail_z] is two rigid body spins.
        givenU = self.get_givenU()
        problem = self._problem
        if isinstance(problem, _GivenFlowProblem):
            givenU = givenU + problem.get_given_flow_at(self.get_center())
        self._ref_U = np.hstack((givenU, (0, 0, 0)))
        
        # reset the rel_U list
        # currently, only wrok for two part composite.
        err_msg = 'current version: len(self.get_obj_list()) == 2'
        assert len(self.get_obj_list()) == 2, err_msg
        rel_U_list = []
        # head
        tobj = self.get_obj_list()[0]
        rel_U = self.get_rel_U_list()[0]
        t_rel_U = np.hstack((rel_U[:3], W_ref[:3]))
        tobj.set_rigid_velocity(t_rel_U, self.get_center())
        rel_U_list.append(t_rel_U)
        # tail
        tobj = self.get_obj_list()[1]
        rel_U = self.get_rel_U_list()[1]
        t_rel_U = np.hstack((rel_U[:3], W_ref[3:]))
        tobj.set_rigid_velocity(t_rel_U, self.get_center())
        rel_U_list.append(t_rel_U)
        self._rel_U_list = rel_U_list
        return True
    
    def get_givenU(self):
        return self._givenU
    
    def set_givenU(self, givenU):
        givenU = np.array(givenU).flatten()
        err_msg = 'givenU=[ux, uy, uz] has 3 components. '
        assert givenU.size == 3, err_msg
        self._givenU = givenU
        return True


class ForceFree1DInfComposite(ForceFreeComposite):
    def set_dmda(self):
        # additional degrees of freedom for force free.
        self._f_dmda = PETSc.DMDA().create(sizes=(2,), dof=1, stencil_width=0,
                                           comm=PETSc.COMM_WORLD)
        self._f_dmda.setFromOptions()
        self._f_dmda.setUp()
        self._u_dmda = PETSc.DMDA().create(sizes=(2,), dof=1, stencil_width=0,
                                           comm=PETSc.COMM_WORLD)
        self._u_dmda.setFromOptions()
        self._u_dmda.setUp()
        return True


class GivenForceComposite(ForceFreeComposite):
    def __init__(self, center: np.array, norm: np.array, name='...', givenF=np.zeros(6), *args):
        self._givenF = np.zeros(6)  # given external force and torque.
        super().__init__(center=center, norm=norm, name=name, *args)
        self.set_givenF(givenF)
    
    def get_givenF(self):
        return self._givenF
    
    def set_givenF(self, givenF):
        err_msg = 'givenF=[fx, fy, fz, tx, ty, tz] has 6 components. '
        assert givenF.size == 6, err_msg
        self._givenF = givenF
        return True
    
    # def node_rotation(self, norm=np.array([0, 0, 1]), theta=np.zeros(1), rotation_origin=None):
    #     rotation_origin = self._center if rotation_origin is None else rotation_origin
    #     for subobj, rel_U in zip(self.get_obj_list(), self.get_rel_U_list()):
    #         subobj.node_rotation(norm=norm, theta=theta, rotation_origin=rotation_origin)
    #         subobj.set_rigid_velocity(rel_U, self.get_center())
    #
    #     rotation = get_rot_matrix(norm, theta)
    #     t_origin = self._center
    #     t_norm = self._norm.copy()
    #     self._center = np.dot(rotation, (self._center - rotation_origin)) + rotation_origin
    #     self._norm = np.dot(rotation, (self._norm + t_origin - rotation_origin)) \
    #                  + rotation_origin - self._center
    #     self._norm = self._norm / np.linalg.norm(self._norm)
    #
    #     rel_U_list = []
    #     for rel_U0 in self.get_rel_U_list():
    #         tU = np.dot(rotation, (rel_U0[:3] + t_origin - rotation_origin)) \
    #              + rotation_origin - self._center
    #         tW = np.dot(rotation, (rel_U0[3:] + t_origin - rotation_origin)) \
    #              + rotation_origin - self._center
    #         rel_U_list.append(np.hstack((tU, tW)))
    #     self._rel_U_list = rel_U_list
    #
    #     ref_U0 = self.get_ref_U()
    #     tU = np.dot(rotation, (ref_U0[:3] + t_origin - rotation_origin)) \
    #          + rotation_origin - self._center
    #     tW = np.dot(rotation, (ref_U0[3:] + t_origin - rotation_origin)) \
    #          + rotation_origin - self._center
    #     self.set_ref_U(np.hstack((tU, tW)))
    #
    #     # dbg, current version have no effect.
    #     self._psi = self._psi + np.dot(t_norm, norm) / (
    #             np.linalg.norm(t_norm) * np.linalg.norm(norm)) * theta
    #     return True
    
    def core_show_givenF(self, arrowFactor=1):
        comm = PETSc.COMM_WORLD.tompi4py()
        rank = comm.Get_rank()
        center = self.get_center()
        givenF = self.get_givenF()
        
        geo_list = uniqueList()
        for obj1 in self.get_obj_list():
            geo_list.append(obj1.get_u_geo())
        temp_geo = base_geo()
        temp_geo.combine(geo_list)
        
        fig = temp_geo.core_show_nodes()
        if rank == 0:
            temp1 = arrowFactor * givenF[:3] / np.sqrt(
                    np.sum(givenF[:3] ** 2))  # normalized, for show.
            temp2 = arrowFactor * givenF[3:] / np.sqrt(
                    np.sum(givenF[3:] ** 2))  # normalized, for show.
            ax = fig.gca()
            ax.quiver(center[0], center[1], center[2], temp1[0], temp1[1], temp1[2], color='r')
            ax.quiver(center[0], center[1], center[2], temp2[0], temp2[1], temp2[2], color='k')
        return fig
    
    def show_givenF(self, arrowFactor=1):
        comm = PETSc.COMM_WORLD.tompi4py()
        rank = comm.Get_rank()
        self.core_show_givenF(arrowFactor=arrowFactor)
        if rank == 0:
            plt.grid()
            plt.get_current_fig_manager().window.showMaximized()
            plt.show()
        return True
    
    def png_givenF(self, finename, arrowFactor=1):
        comm = PETSc.COMM_WORLD.tompi4py()
        rank = comm.Get_rank()
        finename = check_file_extension(finename, '.png')
        
        fig = self.core_show_givenF(arrowFactor=arrowFactor)
        if rank == 0:
            fig.set_size_inches(18.5, 10.5)
            fig.savefig(finename, dpi=100)
            plt.close()
        return True


class GivenForce1DInfComposite(GivenForceComposite, ForceFree1DInfComposite):
    def _nothing(self):
        pass


class ForceFreeProblem(StokesFlowProblem):
    # _ffweigth # type: np.ndarray
    def _init_kwargs(self, **kwargs):
        super()._init_kwargs(**kwargs)
        ffweightx = kwargs['ffweightx'] / kwargs['zoom_factor']
        ffweighty = kwargs['ffweighty'] / kwargs['zoom_factor']
        ffweightz = kwargs['ffweightz'] / kwargs['zoom_factor']
        ffweightT = kwargs['ffweightT'] / kwargs['zoom_factor']
        self._ffweigth = ...  # type: np.ndarray
        self.set_ffweight(ffweightx, ffweighty, ffweightz, ffweightT)
        return True
    
    def get_ffweight(self):
        return self._ffweigth
    
    def set_ffweight(self, ffweightx, ffweighty, ffweightz, ffweightT):
        self._ffweigth = np.array([ffweightx, ffweighty, ffweightz,
                                   ffweightT ** 2, ffweightT ** 2, ffweightT ** 2])
        assert self._ffweigth[3] == self._ffweigth[4] == self._ffweigth[5], \
            ' # IMPORTANT!!!   _ffweigth[3]==_ffweigth[4]==_ffweigth[5]'
        PETSc.Sys.Print('  absolute force free weight %s ' % self._ffweigth)
        return True
    
    def __init__(self, **kwargs):
        # self._ffweigth = ...
        super().__init__(**kwargs)
        self._all_obj_list = uniqueList()  # contain all objects, including subobj within forcefreeComposite.
        self._compst_list = uniqueList()  # forcefreeComposite list.
    
    def add_obj(self, obj):
        if isinstance(obj, ForceFreeComposite):
            self._obj_list.append(obj)
            obj.set_index(self.get_n_obj())
            obj.set_problem(self)
            # obj.set_matrix_method(self.get_matrix_method())
            for sub_obj in obj.get_obj_list():
                self._check_add_obj(sub_obj)
                self._all_obj_list.append(sub_obj)
                self._f_pkg.addDM(sub_obj.get_f_geo().get_dmda())
                self._u_pkg.addDM(sub_obj.get_u_geo().get_dmda())
                self._n_fnode += sub_obj.get_n_f_node()
                self._n_unode += sub_obj.get_n_u_node()
                sub_obj.set_matrix_method(self.get_matrix_method())
            self._f_pkg.addDM(obj.get_f_dmda())
            self._u_pkg.addDM(obj.get_u_dmda())
            self._compst_list.append(obj)
        else:
            self._all_obj_list.append(obj)
            super().add_obj(obj)
        return True
    
    def get_all_obj_list(self, ):
        return self._all_obj_list
    
    def _create_U(self):
        comm = PETSc.COMM_WORLD.tompi4py()
        rank = comm.Get_rank()
        velocity = self._u_pkg.createGlobalVector()
        velocity.zeroEntries()
        for obj0 in self.get_obj_list():
            if isinstance(obj0, ForceFreeComposite):
                for sub_obj, rel_U in zip(obj0.get_obj_list(), obj0.get_rel_U_list()):
                    center = sub_obj.get_u_geo().get_center()
                    sub_nodes = sub_obj.get_u_geo().get_nodes()
                    # sub_obj.show_velocity(length_factor=0.1, show_nodes=True)
                    r = sub_nodes - center
                    t_u = (rel_U[:3] + np.cross(rel_U[3:], r)).flatten()
                    _, u_glbIdx_all = sub_obj.get_u_geo().get_glbIdx()
                    if rank == 0:
                        velocity[u_glbIdx_all] = t_u[:]
            else:
                u0 = obj0.get_velocity()
                _, u_glbIdx_all = obj0.get_u_geo().get_glbIdx()
                if rank == 0:
                    velocity[u_glbIdx_all] = u0[:]
        velocity.assemble()
        self._velocity_petsc = velocity
        return True
    
    def _set_glbIdx(self):
        # global index
        f_isglb = self._f_pkg.getGlobalISs()
        u_isglb = self._u_pkg.getGlobalISs()
        for obj0 in self.get_obj_list():
            if isinstance(obj0, ForceFreeComposite):
                for sub_obj in obj0.get_obj_list():
                    t_f_isglb = f_isglb.pop(0)
                    t_u_isglb = u_isglb.pop(0)
                    sub_obj.get_f_geo().set_glbIdx(t_f_isglb.getIndices())
                    sub_obj.get_u_geo().set_glbIdx(t_u_isglb.getIndices())
                t_f_isglb = f_isglb.pop(0)  # force free additional degree of freedomes
                t_u_isglb = u_isglb.pop(0)  # velocity free additional degree of freedomes
                obj0.set_f_glbIdx(t_f_isglb.getIndices())
                obj0.set_u_glbIdx(t_u_isglb.getIndices())
            else:
                t_f_isglb = f_isglb.pop(0)
                t_u_isglb = u_isglb.pop(0)
                obj0.get_f_geo().set_glbIdx(t_f_isglb.getIndices())
                obj0.get_u_geo().set_glbIdx(t_u_isglb.getIndices())
        return True
    
    def set_force_free(self):
        import numpy.matlib as npm
        ffweight = self._ffweigth
        err_msg = 'self._M_petsc is NOT assembled'
        assert self._M_petsc.isAssembled(), err_msg
        
        for obj1 in self.get_obj_list():
            if isinstance(obj1, ForceFreeComposite):
                center = obj1.get_center()
                _, u_glbIdx_all = obj1.get_u_glbIdx()
                _, f_glbIdx_all = obj1.get_f_glbIdx()
                # self._M_petsc.zeroRows(u_glbIdx_all)
                # self._M_petsc.setValues(u_glbIdx_all, range(f_size), np.zeros(f_size), addv=False)
                # self._M_petsc.setValues(range(u_size), f_glbIdx_all, np.zeros(u_size), addv=False)
                for sub_obj in obj1.get_obj_list():
                    r_u = sub_obj.get_u_geo().get_nodes() - center
                    r_f = sub_obj.get_f_geo().get_nodes() - center
                    t_I = np.array(((-ffweight[0], 0, 0),
                                    (0, -ffweight[1], 0),
                                    (0, 0, -ffweight[2])))
                    tmu1 = npm.repmat(t_I, sub_obj.get_n_u_node(), 1)
                    tmu2 = np.vstack([((0, -ri[2], ri[1]),
                                       (ri[2], 0, -ri[0]),
                                       (-ri[1], ri[0], 0))
                                      for ri in r_u]) * ffweight[3]
                    tmf1 = npm.repmat(t_I, 1, sub_obj.get_n_f_node())
                    tmf2 = np.hstack([((0, -ri[2], ri[1]),
                                       (ri[2], 0, -ri[0]),
                                       (-ri[1], ri[0], 0))
                                      for ri in r_f]) * ffweight[3]
                    tmu = np.hstack((tmu1, tmu2))
                    tmf = np.vstack((tmf1, tmf2))
                    _, sub_u_glbIdx_all = sub_obj.get_u_geo().get_glbIdx()
                    _, sub_f_glbIdx_all = sub_obj.get_f_geo().get_glbIdx()
                    self._M_petsc.setValues(sub_u_glbIdx_all, f_glbIdx_all, tmu, addv=False)
                    self._M_petsc.setValues(u_glbIdx_all, sub_f_glbIdx_all, tmf, addv=False)
                    # # dbg
                    # PETSc.Sys.Print(sub_u_glbIdx_all, f_glbIdx_all)
        self._M_petsc.assemble()
        return True
    
    def create_matrix(self):
        t0 = time()
        self.create_F_U()
        
        # create matrix
        # 1. setup matrix
        if not self._M_petsc.isAssembled():
            self.create_empty_M()
            self._M_destroyed = False
        # 2. set mij part of matrix
        # cmbd_ugeo = geo( )
        # cmbd_ugeo.combine([obj.get_u_geo( ) for obj in self.get_all_obj_list( )])
        # cmbd_ugeo.set_glbIdx_all(np.hstack([obj.get_u_geo( ).get_glbIdx( )[1] for obj in self.get_all_obj_list( )]))
        # cmbd_obj = StokesFlowObj( )
        # cmbd_obj.set_data(cmbd_ugeo, cmbd_ugeo)
        # self._create_matrix_obj(cmbd_obj, self._M_petsc)
        n_obj = len(self.get_all_obj_list())
        for i0, obj1 in enumerate(self.get_all_obj_list()):
            INDEX = ' %d/%d' % (i0 + 1, n_obj)
            self._create_matrix_obj(obj1, self._M_petsc, INDEX)
        # 3. set force and torque free part of matrix
        self.set_force_free()
        # self._M_petsc.view()
        
        t1 = time()
        PETSc.Sys.Print('  %s: create matrix use: %fs' % (str(self), (t1 - t0)))
        
        # # dbg
        # PETSc.Sys.Print('dbg code create_matrix')
        # tM = self.get_M()
        # tU = self.vec_scatter(self._velocity_petsc, destroy=False)
        return True
    
    def _solve_force(self, ksp):
        kwargs = self._kwargs
        getConvergenceHistory = kwargs['getConvergenceHistory']
        ffweight = self._ffweigth
        if getConvergenceHistory:
            ksp.setConvergenceHistory()
            ksp.solve(self._velocity_petsc, self._force_petsc)
            self._convergenceHistory = ksp.getConvergenceHistory()
        else:
            ksp.solve(self._velocity_petsc, self._force_petsc)
        t_force = self.vec_scatter(self._force_petsc, destroy=False)
        
        tmp = []
        for obj0 in self.get_obj_list():
            if isinstance(obj0, ForceFreeComposite):
                center = obj0.get_center()
                for sub_obj in obj0.get_obj_list():
                    _, f_glbIdx_all = sub_obj.get_f_geo().get_glbIdx()
                    sub_obj.set_force(t_force[f_glbIdx_all])
                    tmp.append(t_force[f_glbIdx_all])
                _, f_glbIdx_all = obj0.get_f_glbIdx()
                ref_U = t_force[f_glbIdx_all] * ffweight
                obj0.set_ref_U(ref_U)
                ref_U = obj0.get_ref_U()
                # absolute speed
                for sub_obj, rel_U in zip(obj0.get_obj_list(), obj0.get_rel_U_list()):
                    abs_U = ref_U + rel_U
                    sub_obj.get_u_geo().set_rigid_velocity(abs_U, center=center)
            else:
                _, f_glbIdx_all = obj0.get_f_geo().get_glbIdx()
                obj0.set_force(t_force[f_glbIdx_all])
                tmp.append(t_force[f_glbIdx_all])
        self._force = np.hstack(tmp)
        return True
    
    def _resolve_velocity(self, ksp):
        # self._re_velocity = u_rel + w_rel×ri
        # self._re_velocity + u_ref + w_ref×ri = u_ref + w_ref×ri + u_rel + w_rel×ri
        ffweight = self._ffweigth
        re_velocity_petsc = self._M_petsc.createVecLeft()
        # re_velocity_petsc.set(0)
        self._M_petsc.mult(self._force_petsc, re_velocity_petsc)
        self._re_velocity = self.vec_scatter(re_velocity_petsc)
        
        for obj0 in self.get_obj_list():
            if isinstance(obj0, ForceFreeComposite):
                ref_U = obj0.get_ref_U()
                center = obj0.get_center()
                for sub_obj in obj0.get_obj_list():
                    _, u_glbIdx_all = sub_obj.get_u_geo().get_glbIdx()
                    re_rel_U = self._re_velocity[u_glbIdx_all]
                    sub_nodes = sub_obj.get_u_geo().get_nodes()
                    r = sub_nodes - center
                    t_u = (ref_U[:3] + np.cross(ref_U[3:], r)).flatten()
                    re_abs_U = t_u + re_rel_U
                    sub_obj.set_re_velocity(re_abs_U)
                
                # # dbg
                # t_list = []
                # for sub_obj in obj0.get_obj_list():
                #     _, u_glbIdx_all = sub_obj.get_u_geo().get_glbIdx()
                #     re_rel_U = self._re_velocity[u_glbIdx_all]
                #     sub_nodes = sub_obj.get_u_geo().get_nodes()
                #     r = sub_nodes - center
                #     t_u = (ref_U[:3] + np.cross(ref_U[3:], r)).flatten()
                #     re_abs_U = t_u + re_rel_U
                #     t_geo = sub_obj.get_u_geo().copy()
                #     t_geo.set_velocity(re_abs_U)
                #     t_list.append(t_geo)
                # t_geo2 = geo()
                # t_geo2.combine(t_list)
                # t_geo2.show_velocity()
                
                # _, u_glbIdx_all = obj0.get_u_glbIdx()
                # re_sum = self._re_velocity[u_glbIdx_all] * ([-1] * 3 + [1] * 3) / ffweight
                # obj0.set_total_force(re_sum)  # force free, analytically they are zero.
            else:
                _, u_glbIdx_all = obj0.get_u_geo().get_glbIdx()
                obj0.set_re_velocity(self._re_velocity[u_glbIdx_all])
        self._finish_solve = True
        return ksp.getResidualNorm()
    
    def show_velocity(self, length_factor=1, show_nodes=True):
        geo_list = uniqueList()
        for obj1 in self.get_obj_list():
            if isinstance(obj1, ForceFreeComposite):
                for obj2 in obj1.get_obj_list():
                    geo_list.append(obj2.get_u_geo())
            else:
                geo_list.append(obj1.get_u_geo())
        temp_geo = base_geo()
        temp_geo.combine(geo_list)
        temp_geo.show_velocity(length_factor=length_factor, show_nodes=show_nodes)
        return True
    
    def show_force(self, length_factor=1, show_nodes=True):
        geo_list = uniqueList()
        t_force = []
        for obj1 in self.get_obj_list():
            if isinstance(obj1, ForceFreeComposite):
                for obj2 in obj1.get_obj_list():
                    geo_list.append(obj2.get_u_geo())
                    t_force.append(obj2.get_force())
            else:
                geo_list.append(obj1.get_u_geo())
                t_force.append(obj1.get_force())
        temp_geo = base_geo()
        temp_geo.combine(geo_list)
        temp_geo.set_velocity(np.hstack(t_force))
        temp_geo.show_velocity(length_factor=length_factor, show_nodes=show_nodes)
        return True
    
    def show_re_velocity(self, length_factor=1, show_nodes=True):
        geo_list = uniqueList()
        for obj1 in self.get_obj_list():
            if isinstance(obj1, ForceFreeComposite):
                for obj2 in obj1.get_obj_list():
                    t_geo = obj2.get_u_geo().copy()
                    t_geo.set_velocity(obj2.get_re_velocity())
                    geo_list.append(t_geo)
            else:
                t_geo = obj1.get_u_geo().copy()
                t_geo.set_velocity(obj1.get_re_velocity())
                geo_list.append(t_geo)
        temp_geo = base_geo()
        temp_geo.combine(geo_list)
        temp_geo.show_velocity(length_factor=length_factor, show_nodes=show_nodes)
        return True
    
    # def vtk_self(self, filename, stp_idx=0):
    #     self.check_finish_solve()
    #     obj_list = uniqueList()
    #     for obj0 in self.get_obj_list():
    #         if isinstance(obj0, forcefreeComposite):
    #             for obj1 in obj0.get_obj_list():
    #                 obj_list.append(obj1)
    #         else:
    #             obj_list.append(obj0)
    #     obj0 = StokesFlowObj()
    #     obj0.combine(obj_list, set_re_u=True, set_force=True)
    #     obj0.set_name('Prb')
    #     obj0.vtk(filename, stp_idx)
    #     return True
    
    def vtk_check(self, filename: str, obj: "StokesFlowObj", ref_slt=None):
        obj.print_info()
        obj_tube = list(tube_flatten((obj,)))
        err = []
        for obj in obj_tube:
            if isinstance(obj, StokesFlowObj):
                err.append(self._vtk_check(filename + '_' + str(obj) + '_check', obj, ref_slt))
            elif isinstance(obj, ForceFreeComposite):
                err_msg = 'ref_slt must be None if imput is a forcefreeComposite. '
                assert ref_slt is None, err_msg
                for t_err in self._vtk_composite_check(filename, obj):
                    err.append(t_err)
            else:
                err_msg = 'unknown obj type. '
                raise err_msg
        return tube_flatten((err,))
    
    def _vtk_composite_check(self, filename: str, obj: "ForceFreeComposite"):
        error = []
        ref_U = obj.get_ref_U()
        ref_center = obj.get_center()
        for sub_obj, rel_U in zip(obj.get_obj_list(), obj.get_rel_U_list()):
            rel_center = sub_obj.get_u_geo().get_center()
            sub_nodes = sub_obj.get_u_geo().get_nodes()
            rel_r = sub_nodes - rel_center
            ref_r = sub_nodes - ref_center
            rel_u = (rel_U[:3] + np.cross(rel_U[3:], rel_r)).flatten()
            ref_u = (ref_U[:3] + np.cross(ref_U[3:], ref_r)).flatten()
            tu = ref_u + rel_u
            sub_obj.get_u_geo().set_velocity(tu)
            error.append(self._vtk_check(filename + '_' + str(sub_obj) + '_check',
                                         sub_obj, INDEX=str(sub_obj)))
        return tube_flatten((error,))
    
    def _save_M_mat_dict(self, M_dict, obj):
        if isinstance(obj, ForceFreeComposite):
            for subobj in obj.get_obj_list():
                super()._save_M_mat_dict(M_dict, subobj)
            t_name_all = str(obj) + '_Idx_all'
            t_name = str(obj) + '_Idx'
            u_glbIdx, u_glbIdx_all = obj.get_u_glbIdx()
            M_dict[t_name_all] = u_glbIdx_all
            M_dict[t_name] = u_glbIdx
        else:
            super()._save_M_mat_dict(M_dict, obj)
        return True
    
    def _unpick_addDM(self, obj1):
        if isinstance(obj1, ForceFreeComposite):
            for sub_obj in obj1.get_obj_list():
                super()._unpick_addDM(sub_obj)
            self._f_pkg.addDM(obj1.get_f_dmda())
            self._u_pkg.addDM(obj1.get_u_dmda())
        else:
            super()._unpick_addDM(obj1)
        return True
    
    def unpick_myself(self, check_MPISIZE=True):
        super(ForceFreeProblem, self).unpick_myself(check_MPISIZE=check_MPISIZE)
        # this property has been added in some update.
        if not hasattr(self, '_ffweigth'):
            ffweight = self.get_kwargs()['ffweight']
            self._ffweigth = np.array([ffweight, ffweight, ffweight,
                                       ffweight ** 2, ffweight ** 2, ffweight ** 2])
            PETSc.Sys.Print('  absolute force free weight %s ' % self._ffweigth)
        return True
    
    def update_location(self, eval_dt, print_handle=''):
        super().update_location(eval_dt, print_handle)
        self.set_force_free()
        return True


class ForceFreeIterateProblem(ForceFreeProblem):
    def set_ffweight(self, ffweightx, ffweighty, ffweightz, ffweightT):
        pass
        return True
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._iterComp = ...  # type:  ForceFreeComposite
    
    def add_obj(self, obj):
        if isinstance(obj, ForceFreeComposite):
            self._obj_list.append(obj)
            obj.set_index(self.get_n_obj())
            obj.set_problem(self)
            for sub_obj in obj.get_obj_list():
                self._check_add_obj(sub_obj)
                self._all_obj_list.append(sub_obj)
                self._f_pkg.addDM(sub_obj.get_f_geo().get_dmda())
                self._u_pkg.addDM(sub_obj.get_u_geo().get_dmda())
                self._n_fnode += sub_obj.get_n_f_node()
                self._n_unode += sub_obj.get_n_u_node()
                sub_obj.set_matrix_method(self.get_matrix_method())
        else:
            super().add_obj(obj)
        return True
    
    def _set_glbIdx(self):
        return StokesFlowProblem._set_glbIdx(self)
    
    def _solve_force(self, ksp):
        kwargs = self._kwargs
        getConvergenceHistory = kwargs['getConvergenceHistory']
        if getConvergenceHistory:
            ksp.setConvergenceHistory()
            ksp.solve(self._velocity_petsc, self._force_petsc)
            self._convergenceHistory = ksp.getConvergenceHistory()
        else:
            ksp.solve(self._velocity_petsc, self._force_petsc)
        t_force = self.vec_scatter(self._force_petsc, destroy=False)
        
        tmp = []
        for obj0 in self.get_obj_list():
            if isinstance(obj0, ForceFreeComposite):
                center = obj0.get_center()
                for sub_obj in obj0.get_obj_list():
                    _, f_glbIdx_all = sub_obj.get_f_geo().get_glbIdx()
                    sub_obj.set_force(t_force[f_glbIdx_all])
                    tmp.append(t_force[f_glbIdx_all])
                ref_U = obj0.get_ref_U()
                # absolute speed
                for sub_obj, rel_U in zip(obj0.get_obj_list(), obj0.get_rel_U_list()):
                    abs_U = ref_U + rel_U
                    sub_obj.get_u_geo().set_rigid_velocity(abs_U, center=center)
            else:
                _, f_glbIdx_all = obj0.get_f_geo().get_glbIdx()
                obj0.set_force(t_force[f_glbIdx_all])
                tmp.append(t_force[f_glbIdx_all])
        self._force = np.hstack(tmp)
        return True
    
    def _resolve_velocity(self, ksp):
        # self._re_velocity = u_rel + w_rel×ri
        # self._re_velocity + u_ref + w_ref×ri = u_ref + w_ref×ri + u_rel + w_rel×ri
        ffweight = self._ffweigth
        re_velocity_petsc = self._M_petsc.createVecLeft()
        # re_velocity_petsc.set(0)
        self._M_petsc.mult(self._force_petsc, re_velocity_petsc)
        self._re_velocity = self.vec_scatter(re_velocity_petsc)
        
        for obj0 in self.get_obj_list():
            if isinstance(obj0, ForceFreeComposite):
                ref_U = obj0.get_ref_U()
                center = obj0.get_center()
                for sub_obj in obj0.get_obj_list():
                    _, u_glbIdx_all = sub_obj.get_u_geo().get_glbIdx()
                    re_rel_U = self._re_velocity[u_glbIdx_all]
                    sub_nodes = sub_obj.get_u_geo().get_nodes()
                    r = sub_nodes - center
                    t_u = (ref_U[:3] + np.cross(ref_U[3:], r)).flatten()
                    re_abs_U = t_u + re_rel_U
                    sub_obj.set_re_velocity(re_abs_U)
            else:
                _, u_glbIdx_all = obj0.get_u_geo().get_glbIdx()
                obj0.set_re_velocity(self._re_velocity[u_glbIdx_all])
        self._finish_solve = True
        return ksp.getResidualNorm()
    
    def set_iterate_comp(self, iterComp):
        # set objects that varying their velocity to reach force free condition.
        # other object in the problem have given velocity.
        self._iterComp = iterComp
        return True
    
    def _create_U(self):
        comm = PETSc.COMM_WORLD.tompi4py()
        rank = comm.Get_rank()
        velocity = self._u_pkg.createGlobalVector()
        velocity.zeroEntries()
        for obj0 in self.get_obj_list():
            if isinstance(obj0, ForceFreeComposite):
                center = obj0.get_center()
                for sub_obj, rel_U in zip(obj0.get_obj_list(), obj0.get_rel_U_list()):
                    sub_nodes = sub_obj.get_u_geo().get_nodes()
                    # sub_obj.show_velocity(length_factor=0.1, show_nodes=True)
                    r = sub_nodes - center
                    abs_U = rel_U + obj0.get_ref_U()
                    t_u = (abs_U[:3] + np.cross(abs_U[3:], r)).flatten()
                    _, u_glbIdx_all = sub_obj.get_u_geo().get_glbIdx()
                    if rank == 0:
                        velocity[u_glbIdx_all] = t_u[:]
            else:
                u0 = obj0.get_velocity()
                _, u_glbIdx_all = obj0.get_u_geo().get_glbIdx()
                if rank == 0:
                    velocity[u_glbIdx_all] = u0[:]
        velocity.assemble()
        self._velocity_petsc = velocity
        return True
    
    def create_matrix(self):
        return StokesFlowProblem.create_matrix(self)
    
    def solve_sumForce(self, refU):
        self._iterComp.set_ref_U(refU)
        self.create_F_U()
        self.solve()
        center = self._iterComp.get_center()
        sum_force = np.sum([tobj.get_total_force(center=center)
                            for tobj in self._iterComp.get_obj_list()], axis=0)
        tF = sum_force[:3]
        tT = sum_force[3:]
        return tF, tT
    
    def each_iterate(self, u0, w0, u1, w1):
        F00, T00 = self.solve_sumForce(np.hstack((u0, w0)))
        F01, T01 = self.solve_sumForce(np.hstack((u0, w1)))
        w2 = (w0 - w1) / (T00 - T01) * (0 - T00) + w0
        PETSc.Sys.Print('  w0=%s, w1=%s, T00=%s, T01=%s, w2=%s' % (w0, w1, T00, T01, w2))
        F02, T02 = self.solve_sumForce(np.hstack((u0, w2)))
        F12, T12 = self.solve_sumForce(np.hstack((u1, w2)))
        u2 = (u0 - u1) / (F02 - F12) * (0 - F02) + u0
        PETSc.Sys.Print('  u0=%s, u1=%s, F02=%s, F12=%s, u2=%s' % (u0, u1, F02, F12, u2))
        return u2, w2
    
    def do_iterate(self, ini_refU0=np.zeros(6), ini_refU1=np.ones(6), max_it=1000, tolerate=1e-3):
        u0 = ini_refU0[:3]
        u1 = ini_refU1[:3]
        w0 = ini_refU0[3:]
        w1 = ini_refU1[3:]
        PETSc.Sys.Print('-->>iterate: %d' % 0)
        F_reference, _ = self.solve_sumForce(refU=np.hstack((u1, w0)))
        _, T_reference = self.solve_sumForce(refU=np.hstack((u0, w1)))
        tol = tolerate * 100
        n_it = 0  # # of iterate
        while np.any(tol > tolerate) and n_it < max_it:
            PETSc.Sys.Print('-->>iterate: %d' % (n_it + 1))
            u2, w2 = self.each_iterate(u0, w0, u1, w1)
            F22, T22 = self.solve_sumForce(np.hstack((u2, w2)))
            Ftol = np.abs(F22 / F_reference)
            Ttol = np.abs(T22 / T_reference)
            tol = np.hstack((Ftol, Ttol))
            PETSc.Sys.Print(
                    '  u2=%s, w2=%s, F22=%s, T22=%s, Ftol=%s, Ttol=%s' % (
                        u2, w2, F22, T22, Ftol, Ttol))
            u0, u1 = u1, u2
            w0, w1 = w1, w2
            n_it = n_it + 1
        return np.hstack((u2, w2)), Ftol, Ttol
    
    def each_iterate2(self, u0, w0, u1, w1, F11, T11, relax_fct=1):
        F10, T10 = self.solve_sumForce(np.hstack((u1, w0)))
        F01, T01 = self.solve_sumForce(np.hstack((u0, w1)))
        w2 = (w0 - w1) / (T10 - T11) * (0 - T10) * relax_fct + w0
        u2 = (u0 - u1) / (F01 - F11) * (0 - F01) * relax_fct + u0
        PETSc.Sys.Print('  w0=%s, w1=%s, T10=%s, T11=%s, w2=%s' % (w0, w1, T10, T11, w2))
        PETSc.Sys.Print('  u0=%s, u1=%s, F01=%s, F11=%s, u2=%s' % (u0, u1, F01, F11, u2))
        return u2, w2
    
    def do_iterate2(self, ini_refU0=np.zeros(6), ini_refU1=np.ones(6), max_it=1000, tolerate=1e-3):
        u0 = ini_refU0[:3]
        u1 = ini_refU1[:3]
        w0 = ini_refU0[3:]
        w1 = ini_refU1[3:]
        PETSc.Sys.Print('-->>iterate: %d' % 0)
        F_reference, _ = self.solve_sumForce(refU=np.hstack((u1, w0)))
        _, T_reference = self.solve_sumForce(refU=np.hstack((u0, w1)))
        n_it = 0  # # of iterate
        F11, T11 = self.solve_sumForce(np.hstack((u1, w1)))
        Ftol = np.abs(F11 / F_reference)
        Ttol = np.abs(T11 / T_reference)
        PETSc.Sys.Print(
                '  u1=%s, w1=%s, F11=%s, T11=%s, Ftol=%s, Ttol=%s' % (u1, w1, F11, T11, Ftol, Ttol))
        tol = np.hstack((Ftol, Ttol))
        while np.any(tol > tolerate) and n_it < max_it:
            PETSc.Sys.Print('-->>iterate: %d' % (n_it + 1))
            u2, w2 = self.each_iterate2(u0, w0, u1, w1, F11, T11)
            F22, T22 = self.solve_sumForce(np.hstack((u2, w2)))
            Ftol = np.abs(F22 / F_reference)
            Ttol = np.abs(T22 / T_reference)
            tol = np.hstack((Ftol, Ttol))
            PETSc.Sys.Print(
                    '  u2=%s, w2=%s, F22=%s, T22=%s, Ftol=%s, Ttol=%s' % (
                        u2, w2, F22, T22, Ftol, Ttol))
            u0, u1 = u1, u2
            w0, w1 = w1, w2
            F11, T11 = F22, T22
            n_it = n_it + 1
        return np.hstack((u2, w2)), Ftol, Ttol
    
    def do_iterate3(self, ini_refU0=np.zeros(6), ini_refU1=np.ones(6), max_it=100,
                    rtol=1e-3, atol=1e-10, relax_fct=1):
        u0 = ini_refU0[:3]
        u1 = ini_refU1[:3]
        w0 = ini_refU0[3:]
        w1 = ini_refU1[3:]
        n_it = 0  # # of iterate
        F11, T11 = self.solve_sumForce(np.hstack((u1, w1)))
        # max_rel_err = np.max((2 * np.abs(u0 - u1) / (np.abs(u0) + np.abs(u1)),
        #                   2 * np.abs(w0 - w1) / (np.abs(w0) + np.abs(w1))))
        max_rel_err = np.max((np.abs((u0 - u1) / np.linalg.norm(u1)),
                              np.abs((w0 - w1) / np.linalg.norm(w1))))
        max_abs_err = np.max((np.abs(u0 - u1), np.abs(w0 - w1)))
        PETSc.Sys.Print('  u0=%s, w0=%s, u1=%s, w1=%s' % (u0, w0, u1, w1))
        PETSc.Sys.Print('  max_rel_err=%e, max_abs_err=%e' % (max_rel_err, max_abs_err))
        while max_rel_err > rtol and max_abs_err > atol and n_it < max_it:
            PETSc.Sys.Print('-->>iterate: %d' % (n_it + 1))
            u2, w2 = self.each_iterate2(u0, w0, u1, w1, F11, T11, relax_fct=relax_fct)
            F22, T22 = self.solve_sumForce(np.hstack((u2, w2)))
            u0, u1 = u1, u2
            w0, w1 = w1, w2
            F11, T11 = F22, T22
            n_it = n_it + 1
            # max_rel_err = np.max((2 * np.abs(u0 - u1) / (np.linalg.norm(u0) + np.linalg.norm(u1)),
            #                   2 * np.abs(w0 - w1) / (np.linalg.norm(w0) + np.linalg.norm(w1))))
            max_rel_err = np.max((np.abs((u0 - u1) / np.linalg.norm(u1)),
                                  np.abs((w0 - w1) / np.linalg.norm(w1))))
            max_abs_err = np.max((np.abs(u0 - u1), np.abs(w0 - w1)))
            PETSc.Sys.Print('  u2=%s, w2=%s' % (u2, w2))
            PETSc.Sys.Print('  max_rel_err=%e, max_abs_err=%e' % (max_rel_err, max_abs_err))
            # PETSc.Sys.Print('  DBG', max_rel_err > rtol, max_abs_err > atol, n_it < max_it)
        refU = np.hstack((u1, w1))
        self._iterComp.set_ref_U(refU)
        return np.hstack((u1, w1))
    
    def set_force_free(self):
        pass


class ForceFree1DInfProblem(ForceFreeProblem):
    def _init_kwargs(self, axis='z', **kwargs):
        # axis: symmetrical axis
        if axis == 'x':
            ffweightF = kwargs['ffweightx'] / kwargs['zoom_factor']
        elif axis == 'y':
            ffweightF = kwargs['ffweighty'] / kwargs['zoom_factor']
        elif axis == 'z':
            ffweightF = kwargs['ffweightz'] / kwargs['zoom_factor']
        else:
            err_msg = 'wrong symmetrical axis, it should be one of (x, y, z). '
            raise ValueError(err_msg)
        ffweightT = kwargs['ffweightT'] / kwargs['zoom_factor']
        self._ffweigth = np.array([ffweightF, ffweightT ** 2])
        PETSc.Sys.Print('  absolute force free weight of 1D symmetrical problem is %s ' %
                        self._ffweigth)
        return True
    
    def __init__(self, axis='z', **kwargs):
        super().__init__(**kwargs)
        self._axis = axis  # symmetrical axis
    
    def get_axis(self):
        return self._axis
    
    # def _create_U(self):
    #     comm = PETSc.COMM_WORLD.tompi4py()
    #     rank = comm.Get_rank()
    #     velocity = self._u_pkg.createGlobalVector()
    #     velocity.zeroEntries()
    #     for obj0 in self.get_obj_list():
    #         if isinstance(obj0, forcefreeComposite):
    #             center = obj0.get_center()
    #             for sub_obj, rel_U in zip(obj0.get_obj_list(), obj0.get_rel_U_list()):
    #                 sub_nodes = sub_obj.get_u_geo().get_nodes()
    #                 # sub_obj.show_velocity(length_factor=0.1, show_nodes=True)
    #                 r = sub_nodes - center
    #                 t_u = (rel_U[:3] + np.cross(rel_U[3:], r)).flatten()
    #                 _, u_glbIdx_all = sub_obj.get_u_geo().get_glbIdx()
    #                 if rank == 0:
    #                     velocity[u_glbIdx_all] = t_u[:]
    #         else:
    #             u0 = obj0.get_velocity()
    #             _, u_glbIdx_all = obj0.get_u_geo().get_glbIdx()
    #             if rank == 0:
    #                 velocity[u_glbIdx_all] = u0[:]
    #     velocity.assemble()
    #     self._velocity_petsc = velocity
    #     return True
    
    # def _set_glbIdx(self):
    #     # global index
    #     f_isglb = self._f_pkg.getGlobalISs()
    #     u_isglb = self._u_pkg.getGlobalISs()
    #     for obj0 in self.get_obj_list():
    #         if isinstance(obj0, forcefreeComposite):
    #             for sub_obj in obj0.get_obj_list():
    #                 t_f_isglb = f_isglb.pop(0)
    #                 t_u_isglb = u_isglb.pop(0)
    #                 sub_obj.get_f_geo().set_glbIdx(t_f_isglb.getIndices())
    #                 sub_obj.get_u_geo().set_glbIdx(t_u_isglb.getIndices())
    #             t_f_isglb = f_isglb.pop(0)  # force free additional degree of freedomes
    #             t_u_isglb = u_isglb.pop(0)  # velocity free additional degree of freedomes
    #             obj0.set_f_glbIdx(t_f_isglb.getIndices())
    #             obj0.set_u_glbIdx(t_u_isglb.getIndices())
    #         else:
    #             t_f_isglb = f_isglb.pop(0)
    #             t_u_isglb = u_isglb.pop(0)
    #             obj0.get_f_geo().set_glbIdx(t_f_isglb.getIndices())
    #             obj0.get_u_geo().set_glbIdx(t_u_isglb.getIndices())
    #     return True
    
    def set_force_free(self):
        import numpy.matlib as npm
        ffweight = self._ffweigth
        err_msg = 'self._M_petsc is NOT assembled'
        assert self._M_petsc.isAssembled(), err_msg
        
        for obj1 in self.get_obj_list():
            if isinstance(obj1, ForceFreeComposite):
                center = obj1.get_center()
                _, u_glbIdx_all = obj1.get_u_glbIdx()
                _, f_glbIdx_all = obj1.get_f_glbIdx()
                # self._M_petsc.zeroRows(u_glbIdx_all)
                # self._M_petsc.setValues(u_glbIdx_all, range(f_size), np.zeros(f_size), addv=False)
                # self._M_petsc.setValues(range(u_size), f_glbIdx_all, np.zeros(u_size), addv=False)
                for sub_obj in obj1.get_obj_list():
                    r_u = sub_obj.get_u_geo().get_nodes() - center
                    r_f = sub_obj.get_f_geo().get_nodes() - center
                    axis = self.get_axis()
                    if axis == 'x':
                        t_I = np.array((-ffweight[0], 0, 0))
                        tmu2 = np.vstack([(0, -ri[2], ri[1]) for ri in r_u]) * ffweight[1]
                        tmf2 = np.hstack([(0, -ri[2], ri[1]) for ri in r_f]) * ffweight[1]
                    elif axis == 'y':
                        t_I = np.array((0, -ffweight[0], 0))
                        tmu2 = np.vstack([(ri[2], 0, -ri[0]) for ri in r_u]) * ffweight[1]
                        tmf2 = np.hstack([(ri[2], 0, -ri[0]) for ri in r_f]) * ffweight[1]
                    elif axis == 'z':
                        t_I = np.array((0, 0, -ffweight[0]))
                        tmu2 = np.vstack([(-ri[1], ri[0], 0) for ri in r_u]) * ffweight[1]
                        tmf2 = np.hstack([(-ri[1], ri[0], 0) for ri in r_f]) * ffweight[1]
                    tmu1 = npm.repmat(t_I, sub_obj.get_n_u_node(), 1)
                    tmf1 = npm.repmat(t_I, 1, sub_obj.get_n_f_node())
                    tmu = np.dstack((tmu1.flatten(), tmu2.flatten()))[0]
                    tmf = np.vstack((tmf1, tmf2))
                    _, sub_u_glbIdx_all = sub_obj.get_u_geo().get_glbIdx()
                    _, sub_f_glbIdx_all = sub_obj.get_f_geo().get_glbIdx()
                    self._M_petsc.setValues(sub_u_glbIdx_all, f_glbIdx_all, tmu, addv=False)
                    self._M_petsc.setValues(u_glbIdx_all, sub_f_glbIdx_all, tmf, addv=False)
                    # # dbg
                    # PETSc.Sys.Print(sub_u_glbIdx_all, f_glbIdx_all)
        self._M_petsc.assemble()
        return True
    
    # def create_matrix(self):
    #     t0 = time()
    #     self.create_F_U()
    #
    #     # create matrix
    #     # 1. setup matrix
    #     if not self._M_petsc.isAssembled():
    #         self.create_empty_M()
    #     # 2. set mij part of matrix
    #     # cmbd_ugeo = geo( )
    #     # cmbd_ugeo.combine([obj.get_u_geo( ) for obj in self.get_all_obj_list( )])
    #     # cmbd_ugeo.set_glbIdx_all(np.hstack([obj.get_u_geo( ).get_glbIdx( )[1] for obj in self.get_all_obj_list( )]))
    #     # cmbd_obj = StokesFlowObj( )
    #     # cmbd_obj.set_data(cmbd_ugeo, cmbd_ugeo)
    #     # self._create_matrix_obj(cmbd_obj, self._M_petsc)
    #     n_obj = len(self.get_all_obj_list())
    #     for i0, obj1 in enumerate(self.get_all_obj_list()):
    #         INDEX = ' %d/%d' % (i0 + 1, n_obj)
    #         self._create_matrix_obj(obj1, self._M_petsc, INDEX)
    #     # 3. set force and torque free part of matrix
    #     self.set_force_free()
    #     # self._M_petsc.view()
    #
    #     t1 = time()
    #     PETSc.Sys.Print('  %s: create matrix use: %fs' % (str(self), (t1 - t0)))
    #     return True
    
    def _solve_force(self, ksp):
        kwargs = self._kwargs
        getConvergenceHistory = kwargs['getConvergenceHistory']
        ffweight = self._ffweigth
        axis = self.get_axis()
        if getConvergenceHistory:
            ksp.setConvergenceHistory()
            ksp.solve(self._velocity_petsc, self._force_petsc)
            self._convergenceHistory = ksp.getConvergenceHistory()
        else:
            ksp.solve(self._velocity_petsc, self._force_petsc)
            # # dbg
            # re_velocity_petsc = self._M_petsc.createVecLeft()
            # self._M_petsc.mult(self._force_petsc, re_velocity_petsc)
        t_force = self.vec_scatter(self._force_petsc, destroy=False)
        
        tmp = []
        for obj0 in self.get_obj_list():
            if isinstance(obj0, ForceFreeComposite):
                center = obj0.get_center()
                for sub_obj in obj0.get_obj_list():
                    _, f_glbIdx_all = sub_obj.get_f_geo().get_glbIdx()
                    sub_obj.set_force(t_force[f_glbIdx_all])
                    tmp.append(t_force[f_glbIdx_all])
                _, f_glbIdx_all = obj0.get_f_glbIdx()
                ref_U = t_force[f_glbIdx_all] * ffweight
                if axis == 'x':
                    ref_U = np.array([ref_U[0], 0, 0, ref_U[1], 0, 0])
                elif axis == 'y':
                    ref_U = np.array([0, ref_U[0], 0, 0, ref_U[1], 0])
                elif axis == 'z':
                    ref_U = np.array([0, 0, ref_U[0], 0, 0, ref_U[1]])
                obj0.set_ref_U(ref_U)
                # absolute speed
                for sub_obj, rel_U in zip(obj0.get_obj_list(), obj0.get_rel_U_list()):
                    abs_U = ref_U + rel_U
                    sub_obj.get_u_geo().set_rigid_velocity(abs_U, center=center)
            else:
                _, f_glbIdx_all = obj0.get_f_geo().get_glbIdx()
                obj0.set_force(t_force[f_glbIdx_all])
                tmp.append(t_force[f_glbIdx_all])
        self._force = np.hstack(tmp)
        return True
    
    def _resolve_velocity(self, ksp):
        ffweight = self._ffweigth
        re_velocity_petsc = self._M_petsc.createVecLeft()
        # re_velocity_petsc.set(0)
        self._M_petsc.mult(self._force_petsc, re_velocity_petsc)
        self._re_velocity = self.vec_scatter(re_velocity_petsc)
        axis = self.get_axis()
        
        for obj0 in self.get_obj_list():
            if isinstance(obj0, ForceFreeComposite):
                ref_U = obj0.get_ref_U()
                center = obj0.get_center()
                for sub_obj in obj0.get_obj_list():
                    _, u_glbIdx_all = sub_obj.get_u_geo().get_glbIdx()
                    re_rel_U = self._re_velocity[u_glbIdx_all]
                    sub_nodes = sub_obj.get_u_geo().get_nodes()
                    r = sub_nodes - center
                    t_u = (ref_U[:3] + np.cross(ref_U[3:], r)).flatten()
                    re_abs_U = t_u + re_rel_U
                    sub_obj.set_re_velocity(re_abs_U)
                # _, u_glbIdx_all = obj0.get_u_glbIdx()
                # re_sum = self._re_velocity[u_glbIdx_all] * [-1, 1] / ffweight
                # if axis == 'x':
                #     re_sum = np.array([re_sum[0], 0, 0, re_sum[1], 0, 0])
                # elif axis == 'y':
                #     re_sum = np.array([0, re_sum[0], 0, 0, re_sum[1], 0])
                # elif axis == 'z':
                #     re_sum = np.array([0, 0, re_sum[0], 0, 0, re_sum[1]])
                # obj0.set_total_force(re_sum)  # force free, analytically they are zero.
            else:
                _, u_glbIdx_all = obj0.get_u_geo().get_glbIdx()
                obj0.set_re_velocity(self._re_velocity[u_glbIdx_all])
        self._finish_solve = True
        return ksp.getResidualNorm()


class GivenForceProblem(ForceFreeProblem):
    def _create_U(self):
        comm = PETSc.COMM_WORLD.tompi4py()
        rank = comm.Get_rank()
        ffweight = self._ffweigth
        velocity = self._u_pkg.createGlobalVector()
        velocity.zeroEntries()
        for obj0 in self.get_obj_list():
            if isinstance(obj0, GivenForceComposite):
                center = obj0.get_center()
                for sub_obj, rel_U in zip(obj0.get_obj_list(), obj0.get_rel_U_list()):
                    sub_nodes = sub_obj.get_u_geo().get_nodes()
                    # sub_obj.show_velocity(length_factor=0.1, show_nodes=True)
                    r = sub_nodes - center
                    t_u = (rel_U[:3] + np.cross(rel_U[3:], r)).flatten()
                    _, u_glbIdx_all = sub_obj.get_u_geo().get_glbIdx()
                    if rank == 0:
                        velocity[u_glbIdx_all] = t_u[:]
                _, u_glbIdx_all = obj0.get_u_glbIdx()
                givenF = obj0.get_givenF() * (
                        [-1] * 3 + [1] * 3)  # sum(-1*F)=-F_give, sum(r*F)=T_give
                if rank == 0:
                    velocity[u_glbIdx_all] = givenF * ffweight
            else:
                u0 = obj0.get_velocity()
                _, u_glbIdx_all = obj0.get_u_geo().get_glbIdx()
                if rank == 0:
                    velocity[u_glbIdx_all] = u0[:]
        velocity.assemble()
        self._velocity_petsc = velocity
        return True


class givenForce1DInfPoblem(ForceFree1DInfProblem, GivenForceProblem):
    def _create_U(self):
        comm = PETSc.COMM_WORLD.tompi4py()
        rank = comm.Get_rank()
        ffweight = self._ffweigth
        velocity = self._u_pkg.createGlobalVector()
        velocity.zeroEntries()
        for obj0 in self.get_obj_list():
            if isinstance(obj0, GivenForce1DInfComposite):
                center = obj0.get_center()
                for sub_obj, rel_U in zip(obj0.get_obj_list(), obj0.get_rel_U_list()):
                    sub_nodes = sub_obj.get_u_geo().get_nodes()
                    # sub_obj.show_velocity(length_factor=0.1, show_nodes=True)
                    r = sub_nodes - center
                    t_u = (rel_U[:3] + np.cross(rel_U[3:], r)).flatten()
                    _, u_glbIdx_all = sub_obj.get_u_geo().get_glbIdx()
                    if rank == 0:
                        velocity[u_glbIdx_all] = t_u[:]
                _, u_glbIdx_all = obj0.get_u_glbIdx()
                givenF = obj0.get_givenF() * (
                        [-1] * 3 + [1] * 3)  # sum(-1*F)=-F_give, sum(r*F)=T_give
                axis = self.get_axis()
                if rank == 0:
                    if axis == 'x':
                        velocity[u_glbIdx_all] = givenF[[0, 3]] * ffweight
                    elif axis == 'y':
                        velocity[u_glbIdx_all] = givenF[[1, 4]] * ffweight
                    elif axis == 'z':
                        velocity[u_glbIdx_all] = givenF[[2, 5]] * ffweight
            else:
                u0 = obj0.get_velocity()
                _, u_glbIdx_all = obj0.get_u_geo().get_glbIdx()
                if rank == 0:
                    velocity[u_glbIdx_all] = u0[:]
        velocity.assemble()
        self._velocity_petsc = velocity
        return True


class StokesletsInPipeforcefreeProblem(StokesletsInPipeProblem, ForceFreeProblem):
    def _nothing(self):
        pass


class StokesletsInPipeforcefreeIterateProblem(StokesletsInPipeProblem, ForceFreeIterateProblem):
    def _nothing(self):
        pass


class StokesletsTwoPlaneProblem(StokesFlowProblem):
    # stokes flow between two plane, one is move in a constant velocity.
    # U_all = U_shear + U_twoPlane.
    # two planes are paralleled with XY plane, shear flow is vertical to z axis.
    # see Liron, Nadav, and S. Mochon. "Stokes flow for a stokeslets between two parallel flat plates." Journal of Engineering Mathematics 10.4 (1976): 287-303.
    def _init_kwargs(self, **kwargs):
        self._twoPlateHeight = kwargs['twoPlateHeight']
        return True
    
    def _check_add_obj(self, obj):
        h = self._twoPlateHeight
        nodes = obj.get_u_geo().get_nodes()
        err_msg = 'z coordinate of nodes is out of range (%f, %f)' % (0, h)
        assert np.all(nodes[:, 2] < h) and np.all(nodes[:, 2] > 0), err_msg
        return True


class _GivenFlowProblem(StokesFlowProblem):
    # assuming the problem have a given background flow, subtract, solve and add again.
    def get_given_flow(self, obj):
        given_u = np.zeros(obj.get_n_u_node() * obj.get_n_unknown())
        for obj2 in self.get_all_obj_list():
            if isinstance(obj2, FundSoltObj):
                for location, force, StokesletsHandle in obj2.get_point_force_list():
                    # subtract the velocity due to the force at obj1
                    m_f2 = StokesletsHandle(obj.get_u_nodes(), location)
                    given_u = given_u + np.dot(m_f2, force)
        return given_u
    
    def get_given_flow_at(self, location):
        # from src.geo import geo
        temp_geo1 = base_geo()  # velocity nodes
        temp_geo1.set_nodes(location, deltalength=0)
        temp_obj1 = StokesFlowObj()
        temp_obj1.set_data(temp_geo1, temp_geo1, np.zeros(location.size))
        return self.get_given_flow(temp_obj1)
    
    def subtract_given_flow_obj(self, obj):
        given_u = self.get_given_flow(obj)
        ugeo = obj.get_u_geo()
        ugeo.set_velocity(ugeo.get_velocity() - given_u)
        return True
    
    def add_given_flow_obj(self, obj):
        given_u = self.get_given_flow(obj)
        ugeo = obj.get_u_geo()
        ugeo.set_velocity(ugeo.get_velocity() + given_u)
        return True
    
    def _create_U(self):
        for obj0 in self.get_all_obj_list():
            self.subtract_given_flow_obj(obj0)
        super(_GivenFlowProblem, self)._create_U()
        for obj0 in self.get_all_obj_list():
            self.add_given_flow_obj(obj0)
        return True
    
    def _resolve_velocity(self, ksp):
        ksp_norm = super()._resolve_velocity(ksp)
        for obj0 in self.get_all_obj_list():
            given_u = self.get_given_flow(obj0)
            obj0.set_re_velocity(obj0.get_re_velocity() + given_u)
        return ksp_norm
    
    def solve_obj_u(self, obj: 'StokesFlowObj', INDEX=''):
        obj_u = super().solve_obj_u(obj, INDEX)
        given_u = self.get_given_flow(obj)
        return obj_u + given_u
    
    def update_location(self, eval_dt, print_handle=''):
        super().update_location(eval_dt, print_handle)
        self.create_F_U()
        return True


class StrainRateBaseProblem(_GivenFlowProblem):
    def _init_kwargs(self, **kwargs):
        super()._init_kwargs(**kwargs)
        self._basei = 0
        self.set_basei(kwargs['basei'])
        return True
    
    def base_fun(self):
        # u=(z, 0, 0)
        def base0(unodes):
            n_nodes = unodes.shape[0]
            u = np.vstack((unodes[:, 2], np.zeros(n_nodes), np.zeros(n_nodes))).T.flatten()
            return u
        
        # u=(x, -y, 0)
        def base1(unodes):
            n_nodes = unodes.shape[0]
            u = np.vstack((unodes[:, 0], -1 * unodes[:, 1], np.zeros(n_nodes))).T.flatten()
            return u
        
        # u=(0, -y, z)
        def base2(unodes):
            n_nodes = unodes.shape[0]
            u = np.vstack((np.zeros(n_nodes), -1 * unodes[:, 1], unodes[:, 2])).T.flatten()
            return u
        
        # u=(y, x, 0)
        def base3(unodes):
            n_nodes = unodes.shape[0]
            u = np.vstack((unodes[:, 1], unodes[:, 0], np.zeros(n_nodes))).T.flatten()
            return u
        
        # u=(z, 0, x)
        def base4(unodes):
            n_nodes = unodes.shape[0]
            u = np.vstack((unodes[:, 2], np.zeros(n_nodes), unodes[:, 0])).T.flatten()
            return u
        
        # u=(0, z, y)
        def base5(unodes):
            n_nodes = unodes.shape[0]
            u = np.vstack((np.zeros(n_nodes), unodes[:, 2], unodes[:, 1])).T.flatten()
            return u
        
        # u=(0, -z, y)
        def base6(unodes):
            n_nodes = unodes.shape[0]
            u = np.vstack((np.zeros(n_nodes), -1 * unodes[:, 2], unodes[:, 1])).T.flatten()
            return u
        
        # u=(z, 0, -x)
        def base7(unodes):
            n_nodes = unodes.shape[0]
            u = np.vstack((unodes[:, 2], np.zeros(n_nodes), -1 * unodes[:, 0])).T.flatten()
            return u
        
        # u=(-y, x, 0)
        def base8(unodes):
            n_nodes = unodes.shape[0]
            u = np.vstack((-1 * unodes[:, 1], unodes[:, 0], np.zeros(n_nodes))).T.flatten()
            return u
        
        # u=(0, 0, 0)
        def base9(unodes):
            n_nodes = unodes.shape[0]
            u = np.vstack((np.zeros(n_nodes), np.zeros(n_nodes), np.zeros(n_nodes))).T.flatten()
            return u
        
        # u=(0, 1, 0)
        def base10(unodes):
            n_nodes = unodes.shape[0]
            u = np.vstack((np.zeros(n_nodes), np.ones(n_nodes), np.zeros(n_nodes))).T.flatten()
            return u
        
        # ABC flow case, A = B = C = 1
        def base_ABC(unodes):
            A, B, C = 1, 1, 1
            x, y, z = unodes[:, 0], unodes[:, 1], unodes[:, 2]
            Ub = np.vstack((A * np.sin(z) + C * np.cos(y),
                            B * np.sin(x) + A * np.cos(z),
                            C * np.sin(y) + B * np.cos(x))).T.flatten()
            return Ub
        
        # ABC flow case, A = B = C = 1, D = E = F = changes, H = I = J = 0,
        def base_ABCDEFHIJ(unodes):
            OptDB = PETSc.Options()
            A = OptDB.getReal('ABC_A', 1)
            B = OptDB.getReal('ABC_B', 1)
            C = OptDB.getReal('ABC_C', 1)
            D = OptDB.getReal('ABC_D', 1)
            E = OptDB.getReal('ABC_E', 1)
            F = OptDB.getReal('ABC_F', 1)
            G = OptDB.getReal('ABC_G', 0)
            H = OptDB.getReal('ABC_H', 0)
            I = OptDB.getReal('ABC_I', 0)
            x, y, z = unodes[:, 0], unodes[:, 1], unodes[:, 2]
            Ub = np.vstack((A * np.sin(D * z + G) + C * np.cos(F * y + I),
                            B * np.sin(E * x + H) + A * np.cos(D * z + G),
                            C * np.sin(F * y + I) + B * np.cos(E * x + H))).T.flatten()
            return Ub
        
        # u=(z, 0, 0)
        def base_shear(unodes):
            n_nodes = unodes.shape[0]
            u = np.vstack((unodes[:, 2], np.zeros(n_nodes), np.zeros(n_nodes))).T.flatten()
            return u
        
        _base_fun = {
            0:           base0,
            1:           base1,
            2:           base2,
            3:           base3,
            4:           base4,
            5:           base5,
            6:           base6,
            7:           base7,
            8:           base8,
            9:           base9,
            10:          base10,
            'ABC':       base_ABC,
            'ABCDEFHIJ': base_ABCDEFHIJ,
            'shear':     base_shear,
            }
        return _base_fun
    
    def get_given_flow(self, obj):
        basei = self.get_basei()
        given_u = super().get_given_flow(obj)
        given_u = given_u + self.base_fun()[basei](obj.get_u_nodes())
        return given_u
    
    def set_basei(self, basei):
        base_keys = list(self.base_fun().keys())
        assert basei in base_keys
        self._basei = basei
        return True
    
    def get_basei(self):
        return self._basei


class ShearFlowProblem(_GivenFlowProblem):
    def _init_kwargs(self, **kwargs):
        super()._init_kwargs(**kwargs)
        self._planeShearRate = np.zeros(3)
        self._planeShearNorm = kwargs['planeShearNorm']
        self.set_planeShearRate(kwargs['planeShearRate'])
        return True
    
    def get_given_flow(self, obj):
        given_u = super().get_given_flow(obj)
        # in this case, background flow is a given shear flow
        planeShearRate = self._planeShearRate  # for shear flow
        planeShearNorm = self._planeShearNorm
        ugeo = obj.get_u_geo()
        t1 = np.einsum('ij,j', ugeo.get_nodes(), planeShearNorm)
        given_u = given_u + np.einsum('i, j->ij', t1, planeShearRate.flatten()).ravel()
        return given_u
    
    def set_planeShearRate(self, planeShearRate):
        planeShearNorm = self._planeShearNorm
        self._planeShearRate = np.array(planeShearRate).ravel()
        # err_msg = 'shear flow velocity is must vertical to z axis. '
        # assert self._planeShearRate[0, -1] == 0. and self._planeShearRate.size == 3, err_msg
        err_msg = 'restriction: dot(planeShearRate, planeShearNorm)==0. '
        assert np.isclose(np.dot(planeShearRate, planeShearNorm), 0), err_msg
        return True
    
    def get_planeShearRate(self):
        return self._planeShearRate


class doubleletProblem(_GivenFlowProblem):
    def _init_kwargs(self, **kwargs):
        super()._init_kwargs(**kwargs)
        self._doublelet = np.zeros(3)
        self.set_doublelet(kwargs['doublelet'])
        return True
    
    def get_given_flow(self, obj):
        given_u = super().get_given_flow(obj)
        # in this case, background flow is a given shear flow
        u_weight = self._planeShearRate  # for shear flow
        ugeo = obj.get_u_geo()
        given_u = given_u + np.dot(ugeo.get_nodes()[:, 2].reshape((-1, 1)), u_weight).flatten()
        return given_u
    
    def set_doublelet(self, doublelet):
        self._doublelet = np.array(doublelet).reshape((1, 3))
        return True
    
    def get_doublelet(self):
        return self._doublelet


class FreeVortexProblem(_GivenFlowProblem):
    # assume vortex is in XY plane
    def _init_kwargs(self, **kwargs):
        super()._init_kwargs(**kwargs)
        self._vortexStrength = kwargs['vortexStrength']
        return True
    
    def get_given_flow(self, obj):
        given_u = super().get_given_flow(obj)
        # in this case, background flow is a given Free Vortex flow
        phi, rho, z = obj.get_u_geo().get_polar_coord()
        u_phi = self._vortexStrength / (2 * np.pi * rho)
        given_u = given_u + np.dstack((-u_phi * np.sin(phi),
                                       u_phi * np.cos(phi),
                                       np.zeros_like(phi))).flatten()
        return given_u
    
    def get_vortexStrength(self):
        return self._vortexStrength


class LambOseenVortexProblem(_GivenFlowProblem):
    # assume vortex is in XY plane
    def _init_kwargs(self, **kwargs):
        super()._init_kwargs(**kwargs)
        self._vortexStrength = kwargs['vortexStrength']
        return True
    
    def get_given_flow(self, obj):
        given_u = super().get_given_flow(obj)
        # in this case, background flow is a given Free Vortex flow
        phi, rho, z = obj.get_u_geo().get_polar_coord()
        u_phi = self._vortexStrength / (2 * np.pi * rho) * (1 - np.exp(-rho ** 2 / 4))
        # u_phi = self._vortexStrength / (2 * np.pi * rho) * (1 - np.exp(-rho ** 2))
        given_u = given_u + np.dstack((-u_phi * np.sin(phi),
                                       u_phi * np.cos(phi),
                                       np.zeros_like(phi))).flatten()
        return given_u
    
    def set_vortexStrength(self, vortexStrength):
        self._vortexStrength = vortexStrength
        return True
    
    def get_vortexStrength(self):
        return self._vortexStrength


class StokesletsFlowProblem(_GivenFlowProblem):
    def _init_kwargs(self, **kwargs):
        super()._init_kwargs(**kwargs)
        self._StokesletsStrength = np.array(kwargs['StokesletsStrength']).reshape((1, 3)).flatten()
        return True
    
    def get_given_flow(self, obj):
        given_u = super().get_given_flow(obj)
        # in this case, background flow is a given shear flow
        given_u_fun = lambda x0, x1, x2, f0, f1, f2: np.array(
                [f0 * x0 ** 2 * (x0 ** 2 + x1 ** 2 + x2 ** 2) ** (-1.5) +
                 f0 * (x0 ** 2 + x1 ** 2 + x2 ** 2) ** (-0.5) +
                 f1 * x0 * x1 * (x0 ** 2 + x1 ** 2 + x2 ** 2) ** (-1.5) +
                 f2 * x0 * x2 * (x0 ** 2 + x1 ** 2 + x2 ** 2) ** (-1.5),
                 f0 * x0 * x1 * (x0 ** 2 + x1 ** 2 + x2 ** 2) ** (-1.5) +
                 f1 * x1 ** 2 * (x0 ** 2 + x1 ** 2 + x2 ** 2) ** (-1.5) +
                 f1 * (x0 ** 2 + x1 ** 2 + x2 ** 2) ** (-0.5) +
                 f2 * x1 * x2 * (x0 ** 2 + x1 ** 2 + x2 ** 2) ** (-1.5),
                 f0 * x0 * x2 * (x0 ** 2 + x1 ** 2 + x2 ** 2) ** (-1.5) +
                 f1 * x1 * x2 * (x0 ** 2 + x1 ** 2 + x2 ** 2) ** (-1.5) +
                 f2 * x2 ** 2 * (x0 ** 2 + x1 ** 2 + x2 ** 2) ** (-1.5) +
                 f2 * (x0 ** 2 + x1 ** 2 + x2 ** 2) ** (-0.5)]) / (8 * np.pi)
        unodes = obj.get_u_geo().get_nodes()
        given_u = given_u + given_u_fun(*unodes, *self._StokesletsStrength)
        return given_u


class PoiseuilleFlowProblem(_GivenFlowProblem):
    def _init_kwargs(self, **kwargs):
        super()._init_kwargs(**kwargs)
        self._PoiseuilleStrength = kwargs['PoiseuilleStrength']
        return True
    
    def get_given_flow(self, obj):
        given_u = super().get_given_flow(obj)
        # in this case, background flow is a given Poiseuille flow
        u_weight = self._PoiseuilleStrength  # for Poiseuille flow
        tgeo = obj.get_u_geo()
        _, rho, _ = tgeo.get_polar_coord()
        given_z = (1 - rho ** 2) * u_weight
        given_u = given_u + np.dstack((np.zeros_like(given_z), np.zeros_like(given_z),
                                       given_z,)).flatten()
        return given_u


class FundSoltObj(StokesFlowObj):
    def __init__(self):
        super(FundSoltObj, self).__init__()
        # each element contain two vectors and a type ((x1,2,3), (f1,2,3), StokesletsHandle)
        # self._point_force_list = []
        self._point_force_list = uniqueList()
        # the following properties store the location history of the composite.
        self._force_norm_hist = []  # [[force1 hist], [force2 hist]...]
    
    def set_data(self, *args, **kwargs):
        super().set_data(*args, **kwargs)
        self._type = 'fund sold obj'
        return True
    
    def add_point_force(self, location: np.ndarray, force: np.ndarray,
                        StokesletsHandle=light_stokeslets_matrix_3d):
        err_msg = 'both location and force are vectors with shape (3, 0)'
        assert location.shape == (3,) and force.shape == (3,), err_msg
        
        self._point_force_list.append((location, force, StokesletsHandle))
        self._force_norm_hist.append([])
        return True
    
    def get_point_force_list(self):
        return self._point_force_list
    
    def dbg_set_point_force_list(self, point_force_list):
        self._point_force_list = point_force_list
        return True
    
    def move(self, displacement):
        super(FundSoltObj, self).move(displacement)
        t_list = []
        for location, force, StokesletsHandle in self.get_point_force_list():
            location = location + displacement
            t_list.append((location, force, StokesletsHandle))
        self._point_force_list = t_list
        return True
    
    def node_rotation(self, norm=np.array([0, 0, 1]), theta=0, rotation_origin=None):
        # The rotation is counterclockwise
        if rotation_origin is None:
            rotation_origin = self.get_u_geo().get_origin()
        else:
            rotation_origin = np.array(rotation_origin).reshape((3,))
        
        super(FundSoltObj, self).node_rotation(norm=norm, theta=theta,
                                               rotation_origin=rotation_origin)
        rot_mtx = get_rot_matrix(norm=norm, theta=theta)
        t_list = []
        for location0, force0, StokesletsHandle in self.get_point_force_list():
            location = np.dot(rot_mtx, (location0 - rotation_origin)) + rotation_origin
            force = np.dot(rot_mtx, (force0 + location0 - rotation_origin)) \
                    + rotation_origin - location
            t_list.append((location, force, StokesletsHandle))
        self._point_force_list = t_list
        return True
    
    def update_location(self, eval_dt, print_handle=''):
        super(FundSoltObj, self).update_location(eval_dt, print_handle)
        for (location, _, _), t_hist in zip(self.get_point_force_list(), self._force_norm_hist):
            t_hist.append(location)
        return True
    
    def get_force_norm_hist(self):
        return self._force_norm_hist


class _GivenFlowForceFreeProblem(_GivenFlowProblem, ForceFreeProblem):
    def _create_U(self):
        # u_fi:         velocity due to point force on the boundary, unknown;
        # u_ref, w_ref: reference velocity of the composite, rigid body motion, unknown;
        # u_rel, w_rel: relative velocity of each part, rigid body motion, known;
        # u_bi:         background flow velocity, known;
        # u_ti:         total velocity that keeps rigid body motion on the surfaces of each part.
        # u_fi + u_bi = u_ti = u_ref + w_ref × ri + u_rel + w_rel × ri .
        # u_fi - u_ref - w_ref × ri = u_rel + w_rel × ri - u_bi
        comm = PETSc.COMM_WORLD.tompi4py()
        rank = comm.Get_rank()
        velocity = self._u_pkg.createGlobalVector()
        velocity.zeroEntries()
        for obj0 in self.get_obj_list():
            if isinstance(obj0, ForceFreeComposite):
                center = obj0.get_center()
                for sub_obj, rel_U in zip(obj0.get_obj_list(), obj0.get_rel_U_list()):
                    sub_nodes = sub_obj.get_u_geo().get_nodes()
                    # sub_obj.show_velocity(length_factor=0.1, show_nodes=True)
                    r = sub_nodes - center
                    t_u = (rel_U[:3] + np.cross(rel_U[3:], r)).flatten()
                    _, u_glbIdx_all = sub_obj.get_u_geo().get_glbIdx()
                    givenU = self.get_given_flow(sub_obj)
                    if rank == 0:
                        velocity[u_glbIdx_all] = t_u[:] - givenU
                        # # dbg
                        # t_list = []
                        # for sub_obj, rel_U in zip(obj0.get_obj_list(), obj0.get_rel_U_list()):
                        #     givenU = self.get_given_flow(sub_obj)
                        #     t_geo = sub_obj.get_u_geo().copy()
                        #     t_geo.set_velocity(givenU)
                        #     t_list.append(t_geo)
                        # t_geo2 = geo()
                        # t_geo2.combine(t_list)
                        # t_geo2.show_velocity()
            else:
                u0 = obj0.get_velocity()
                _, u_glbIdx_all = obj0.get_u_geo().get_glbIdx()
                givenU = self.get_given_flow(obj0)
                if rank == 0:
                    velocity[u_glbIdx_all] = u0[:] - givenU
        velocity.assemble()
        self._velocity_petsc = velocity
        return True


class _GivenFlowForceFreeIterateProblem(_GivenFlowProblem, ForceFreeIterateProblem):
    def _create_U(self):
        comm = PETSc.COMM_WORLD.tompi4py()
        rank = comm.Get_rank()
        velocity = self._u_pkg.createGlobalVector()
        velocity.zeroEntries()
        for obj0 in self.get_obj_list():
            if isinstance(obj0, ForceFreeComposite):
                center = obj0.get_center()
                for sub_obj, rel_U in zip(obj0.get_obj_list(), obj0.get_rel_U_list()):
                    sub_nodes = sub_obj.get_u_geo().get_nodes()
                    # sub_obj.show_velocity(length_factor=0.1, show_nodes=True)
                    r = sub_nodes - center
                    abs_U = rel_U + obj0.get_ref_U()
                    t_u = (abs_U[:3] + np.cross(abs_U[3:], r)).flatten()
                    _, u_glbIdx_all = sub_obj.get_u_geo().get_glbIdx()
                    givenU = self.get_given_flow(sub_obj)
                    if rank == 0:
                        velocity[u_glbIdx_all] = t_u[:] - givenU
            else:
                u0 = obj0.get_velocity()
                _, u_glbIdx_all = obj0.get_u_geo().get_glbIdx()
                givenU = self.get_given_flow(obj0)
                if rank == 0:
                    velocity[u_glbIdx_all] = u0[:] - givenU
        velocity.assemble()
        self._velocity_petsc = velocity
        return True


class ShearFlowForceFreeProblem(ShearFlowProblem, _GivenFlowForceFreeProblem):
    def _nothing(self):
        pass


class ShearFlowForceFreeIterateProblem(ShearFlowProblem, _GivenFlowForceFreeIterateProblem):
    def _nothing(self):
        pass


class StrainRateBaseForceFreeProblem(StrainRateBaseProblem, _GivenFlowForceFreeProblem):
    def _nothing(self):
        pass


class StrainRateBaseForceFreeIterateProblem(StrainRateBaseProblem,
                                            _GivenFlowForceFreeIterateProblem):
    def _nothing(self):
        pass


class FreeVortexForceFreeProblem(FreeVortexProblem, _GivenFlowForceFreeProblem):
    def _nothing(self):
        pass


class FreeVortexForceFreeIterateProblem(FreeVortexProblem, _GivenFlowForceFreeIterateProblem):
    def _nothing(self):
        pass


class LambOseenVortexForceFreeProblem(LambOseenVortexProblem, _GivenFlowForceFreeProblem):
    def _nothing(self):
        pass


class LambOseenVortexForceFreeIterateProblem(LambOseenVortexProblem,
                                             _GivenFlowForceFreeIterateProblem):
    def _nothing(self):
        pass


class DualPotentialProblem(StokesFlowProblem):
    def __init__(self, **kwargs):
        super(DualPotentialProblem, self).__init__(**kwargs)
        self._n_unknown = 4
    
    def _create_U(self):
        comm = PETSc.COMM_WORLD.tompi4py()
        rank = comm.Get_rank()
        velocity = self._u_pkg.createGlobalVector()
        velocity.zeroEntries()
        for i0, obj0 in enumerate(self.get_obj_list()):
            u0 = np.vstack((obj0.get_velocity().reshape(-1, 3).T,
                            np.zeros(obj0.get_n_f_node()))).flatten(order='F')
            _, u_glbIdx_all = obj0.get_u_geo().get_glbIdx()
            if rank == 0:
                velocity[u_glbIdx_all] = u0[:]
        velocity.assemble()
        self._velocity_petsc = velocity
        return True


class DualPotentialObj(StokesFlowObj):
    def __init__(self):
        super(DualPotentialObj, self).__init__()
        self._n_unknown = 4
    
    def set_data(self, *args, **kwargs):
        super().set_data(*args, **kwargs)
        self._type = 'dual potential obj'
        return True


class GivenTorqueIterateVelocity1DProblem(StokesFlowProblem):
    def __init__(self, axis='z', givenF=0, **kwargs):
        super().__init__(**kwargs)
        err_msg = 'wrong symmetrical axis, it should be one of (x, y, z). '
        assert axis in ('x', 'y', 'z'), err_msg
        self._axis = axis  # symmetrical axis
        self._iterObj = []
        self._givenF = givenF[2]
    
    def set_iterate_obj(self, obj_list):
        # set objects that varying their velocity to reach force free condition.
        # other object in the problem have given velocity.
        self._iterObj = list(tube_flatten((obj_list,)))
        return True
    
    def solve_sumForce(self, U, W=1, center=np.zeros(3)):
        axis = self._axis  # type: str
        # assert 1 == 2, 'check center of the tobj.set_rigid_velocity()'
        if axis == 'x':
            for tobj in self._iterObj:
                tobj.set_rigid_velocity((U, 0, 0, W, 0, 0))
        elif axis == 'y':
            for tobj in self._iterObj:
                tobj.set_rigid_velocity((0, U, 0, 0, W, 0))
        elif axis == 'z':
            for tobj in self._iterObj:
                tobj.set_rigid_velocity((0, 0, U, 0, 0, W))
        self.create_F_U()
        self.solve()
        sum_force = np.sum([tobj.get_total_force(center=center) for tobj in self._iterObj], axis=0)
        if axis == 'x':
            tf = sum_force[0]
        elif axis == 'y':
            tf = sum_force[1]
        elif axis == 'z':
            tf = sum_force[2]
        return tf
    
    def each_iterate(self, u0, u1):
        f0 = self.solve_sumForce(u0, W=1)
        f1 = self.solve_sumForce(u1, W=1)
        u2 = (u0 - u1) / (f0 - f1) * (self._givenF - f0) + u0
        PETSc.Sys.Print('  u0=%f, u1=%f, f0=%f, f1=%f, u2=%f' % (u0, u1, f0, f1, u2))
        return f0, f1, u2
    
    def do_iterate(self, tolerate=1e-3, max_it=1000):
        f_reference = self.solve_sumForce(U=1, W=0)
        u0 = 0
        u1 = 1
        tol = tolerate * 100
        n_it = 0  # # of iterate
        while tol > tolerate and n_it < max_it:
            f0, f1, u2 = self.each_iterate(u0, u1)
            u0, u1 = u1, u2
            tol = np.abs(f1 / f_reference)
            n_it = n_it + 1
        return u0, tol


class _GivenForceGivenFlowProblem(GivenForceProblem, _GivenFlowProblem):
    def _create_U(self):
        # u_fi: velocity due to point force on the boundary, unknown;
        # u_ref: reference velocity of the composite, rigid body motion, unknown;
        # u_rel: relative velocity of each part, rigid body motion, known;
        # u_bi: background flow velocity, known;
        # u_ti: total velocity that keeps rigid body motion on the surfaces of each part.
        # u_fi + u_bi = u_ti = u_ref + u_rel.
        # u_fi - u_ref = u_rel - u_bi
        comm = PETSc.COMM_WORLD.tompi4py()
        rank = comm.Get_rank()
        ffweight = self._ffweigth
        velocity = self._u_pkg.createGlobalVector()
        velocity.zeroEntries()
        for obj0 in self.get_obj_list():
            if isinstance(obj0, ForceFreeComposite):
                center = obj0.get_center()
                for sub_obj, rel_U in zip(obj0.get_obj_list(), obj0.get_rel_U_list()):
                    sub_nodes = sub_obj.get_u_geo().get_nodes()
                    u_bi = self.get_given_flow(sub_obj)
                    # sub_obj.show_velocity(length_factor=0.1, show_nodes=True)
                    r = sub_nodes - center
                    t_u = (rel_U[:3] + np.cross(rel_U[3:], r)).flatten() - u_bi
                    _, u_glbIdx_all = sub_obj.get_u_geo().get_glbIdx()
                    if rank == 0:
                        velocity[u_glbIdx_all] = t_u[:]
                if isinstance(obj0, GivenForceComposite):
                    _, u_glbIdx_all = obj0.get_u_glbIdx()
                    givenF = obj0.get_givenF() * (
                            [-1] * 3 + [1] * 3)  # sum(-1*F)=-F_give, sum(r*F)=T_give
                    if rank == 0:
                        velocity[u_glbIdx_all] = givenF * ffweight
            else:
                u0 = obj0.get_velocity()
                u_bi = self.get_given_flow(obj0)
                _, u_glbIdx_all = obj0.get_u_geo().get_glbIdx()
                if rank == 0:
                    velocity[u_glbIdx_all] = u0[:] - u_bi
        velocity.assemble()
        self._velocity_petsc = velocity
        return True
    
    def _resolve_velocity(self, ksp):
        # ksp_norm = super(GivenForceProblem, self)._resolve_velocity(ksp)
        ksp_norm = GivenForceProblem._resolve_velocity(self, ksp)
        for obj0 in self.get_all_obj_list():
            given_u = self.get_given_flow(obj0)
            obj0.set_re_velocity(obj0.get_re_velocity() + given_u)
        return ksp_norm


class _GivenTorqueGivenVelocityGivenFlowProblem(GivenForceProblem, _GivenFlowProblem):
    def _init_kwargs(self, **kwargs):
        StokesFlowProblem._init_kwargs(self, **kwargs)
        ffweightT = kwargs['ffweightT'] / kwargs['zoom_factor']
        self._ffweigth = np.array([ffweightT ** 2, ffweightT ** 2, ffweightT ** 2])
        err_msg = ' # IMPORTANT!!!   _ffweigth[0]==_ffweigth[1]==_ffweigth[2]'
        assert self._ffweigth[0] == self._ffweigth[1] == self._ffweigth[2], err_msg
        PETSc.Sys.Print('  absolute force free weight %s ' % self._ffweigth)
        return True
    
    def _create_U(self):
        # u_fi: velocity due to point force on the boundary, unknown;
        # u_ref: reference translation velocity of the composite, rigid body motion, unknown;
        # w_ref: reference rotation velocity of the composite, rigid body motion, unknown;
        # u_rel: relative translation velocity of each part, rigid body motion, known;
        # w_rel: relative rotation velocity of each part, rigid body motion, known;
        # u_bi: velocity due to background flow, known;
        # u_ti: total velocity that keeps rigid body motion on the surfaces of each part.
        # ri: location of each point.
        # u_fi + u_bi = u_ti = u_ref + w_ref×ri + u_rel + w_rel×ri.
        # u_fi + ri×w_ref(=-w_ref×ri) = u_ref + u_rel + w_rel×ri - u_bi
        comm = PETSc.COMM_WORLD.tompi4py()
        rank = comm.Get_rank()
        ffweight = self._ffweigth
        velocity = self._u_pkg.createGlobalVector()
        velocity.zeroEntries()
        for obj0 in self.get_obj_list():
            if isinstance(obj0, GivenTorqueComposite):
                center = obj0.get_center()
                ref_U = obj0.get_givenU()
                for sub_obj, rel_U in zip(obj0.get_obj_list(), obj0.get_rel_U_list()):
                    sub_nodes = sub_obj.get_u_geo().get_nodes()
                    u_bi = self.get_given_flow(sub_obj)
                    # sub_obj.show_velocity(length_factor=0.1, show_nodes=True)
                    r = sub_nodes - center
                    t_u = (rel_U[:3] + np.cross(rel_U[3:], r) + ref_U).flatten() - u_bi
                    _, u_glbIdx_all = sub_obj.get_u_geo().get_glbIdx()
                    if rank == 0:
                        velocity[u_glbIdx_all] = t_u[:]
                _, u_glbIdx_all = obj0.get_u_glbIdx()
                givenT = obj0.get_givenT()  # sum(r*F)=T_give
                if rank == 0:
                    velocity[u_glbIdx_all] = givenT * ffweight
            else:
                u0 = obj0.get_velocity()
                u_bi = self.get_given_flow(obj0)
                _, u_glbIdx_all = obj0.get_u_geo().get_glbIdx()
                if rank == 0:
                    velocity[u_glbIdx_all] = u0[:] - u_bi
        velocity.assemble()
        self._velocity_petsc = velocity
        return True
    
    def _resolve_velocity(self, ksp):
        # self._re_velocity = u_ref + u_rel + w_rel×ri - u_bi
        # self._re_velocity + w_ref×ri + u_bi = u_ref + w_ref×ri + u_rel + w_rel×ri
        ffweight = self._ffweigth
        re_velocity_petsc = self._M_petsc.createVecLeft()
        # re_velocity_petsc.set(0)
        self._M_petsc.mult(self._force_petsc, re_velocity_petsc)
        self._re_velocity = self.vec_scatter(re_velocity_petsc)
        
        for obj0 in self.get_obj_list():
            if isinstance(obj0, GivenTorqueComposite):
                ref_U = obj0.get_ref_U()
                center = obj0.get_center()
                # re_sum = 0
                for sub_obj in obj0.get_obj_list():
                    u_b = self.get_given_flow(sub_obj)
                    _, u_glbIdx_all = sub_obj.get_u_geo().get_glbIdx()
                    re_rel_U = self._re_velocity[u_glbIdx_all]
                    sub_nodes = sub_obj.get_u_geo().get_nodes()
                    r = sub_nodes - center
                    t_u = (ref_U[:3] + np.cross(ref_U[3:], r)).flatten()
                    re_abs_U = t_u + re_rel_U + u_b
                    sub_obj.set_re_velocity(re_abs_U)
                    # re_sum = re_sum + sub_obj.get_total_force(center=center)
                # obj0.set_total_force(re_sum)  # torque free, analytically they are zero.
            else:
                _, u_glbIdx_all = obj0.get_u_geo().get_glbIdx()
                u_b = self.get_given_flow(obj0)
                obj0.set_re_velocity(self._re_velocity[u_glbIdx_all] + u_b)
        self._finish_solve = True
        return ksp.getResidualNorm()
    
    def set_force_free(self):
        ffweight = self._ffweigth
        err_msg = 'self._M_petsc is NOT assembled'
        assert self._M_petsc.isAssembled(), err_msg
        
        for obj1 in self.get_obj_list():
            if isinstance(obj1, GivenTorqueComposite):
                center = obj1.get_center()
                _, u_glbIdx_all = obj1.get_u_glbIdx()
                _, f_glbIdx_all = obj1.get_f_glbIdx()
                # self._M_petsc.zeroRows(u_glbIdx_all)
                # self._M_petsc.setValues(u_glbIdx_all, range(f_size), np.zeros(f_size), addv=False)
                # self._M_petsc.setValues(range(u_size), f_glbIdx_all, np.zeros(u_size), addv=False)
                for sub_obj in obj1.get_obj_list():
                    r_u = sub_obj.get_u_geo().get_nodes() - center
                    r_f = sub_obj.get_f_geo().get_nodes() - center
                    tmu = np.vstack([((0, -ri[2], ri[1]),
                                      (ri[2], 0, -ri[0]),
                                      (-ri[1], ri[0], 0))
                                     for ri in r_u]) * ffweight[0]
                    tmf = np.hstack([((0, -ri[2], ri[1]),
                                      (ri[2], 0, -ri[0]),
                                      (-ri[1], ri[0], 0))
                                     for ri in r_f]) * ffweight[0]
                    _, sub_u_glbIdx_all = sub_obj.get_u_geo().get_glbIdx()
                    _, sub_f_glbIdx_all = sub_obj.get_f_geo().get_glbIdx()
                    self._M_petsc.setValues(sub_u_glbIdx_all, f_glbIdx_all, tmu, addv=False)
                    self._M_petsc.setValues(u_glbIdx_all, sub_f_glbIdx_all, tmf, addv=False)
                    # # dbg
                    # PETSc.Sys.Print(sub_u_glbIdx_all, f_glbIdx_all)
        self._M_petsc.assemble()
        return True


class GivenForceShearFlowProblem(_GivenForceGivenFlowProblem, ShearFlowProblem):
    def _init_kwargs(self, **kwargs):
        GivenForceProblem._init_kwargs(self, **kwargs)
        ShearFlowProblem._init_kwargs(self, **kwargs)
        # super(GivenForceProblem, self)._init_kwargs(**kwargs)
        # super(ShearFlowProblem, self)._init_kwargs(**kwargs)
        return True


class GivenForcePoiseuilleFlowProblem(_GivenForceGivenFlowProblem, PoiseuilleFlowProblem):
    def _init_kwargs(self, **kwargs):
        GivenForceProblem._init_kwargs(self, **kwargs)
        PoiseuilleFlowProblem._init_kwargs(self, **kwargs)
        # super(GivenForceProblem, self)._init_kwargs(**kwargs)
        # super()._init_kwargs(**kwargs)
        return True


class GivenTorqueGivenVelocityShearFlowProblem(_GivenTorqueGivenVelocityGivenFlowProblem,
                                               ShearFlowProblem):
    def _init_kwargs(self, **kwargs):
        _GivenTorqueGivenVelocityGivenFlowProblem._init_kwargs(self, **kwargs)
        ShearFlowProblem._init_kwargs(self, **kwargs)
        return True


class ForceSphere2DProblem(StokesFlowProblem):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._lamb_inter_list = []
        self._ptc_lub_list = []
        self._Minf_petsc = self._M_petsc  # M matrix, infinite part
        self._Rlub_petsc = PETSc.Mat().create(comm=PETSc.COMM_WORLD)  # R matrix, lubrication part
        self._Rtol_petsc = PETSc.Mat().create(comm=PETSc.COMM_WORLD)  # full modified R matrix, Rtol = I + Minf^-1 * Rlub
        self._Fmdf_petsc = PETSc.Vec().create(comm=PETSc.COMM_WORLD)  # modified force vector, Fmdf = Minf * F
        self._velocity = np.zeros([0])  # velocity information
        self._spin = np.zeros([0])  # spin information
        self._re_fmdf = np.zeros([0])  # resolved information of modified force
    
    def get_lamb_inter_list(self):
        return self._lamb_inter_list
    
    def get_ptc_lub_list(self):
        return self._ptc_lub_list
    
    def get_Minf_petsc(self):
        return self._Minf_petsc
    
    def get_Rlub_petsc(self):
        return self._Rlub_petsc
    
    def get_Rtol_petsc(self):
        return self._Rtol_petsc
    
    def get_Fmdf_petsc(self):
        return self._Fmdf_petsc
    
    def get_velocity(self):
        return self._velocity
    
    def get_velocity_x(self):
        return self._velocity[0::self._n_unknown]
    
    def get_velocity_y(self):
        return self._velocity[1::self._n_unknown]
    
    def get_velocity_z(self):
        return self._velocity[2::self._n_unknown]
    
    def get_spin(self):
        return self._spin
    
    def set_velocity_petsc(self, velocity_petsc):
        self._velocity_petsc = velocity_petsc
        return True
    
    def initial_lub(self):
        assert self.get_n_obj() == 1
        rs2 = self.get_kwargs()['rs2']
        tobj = self.get_obj_list()[0]
        
        # self._lamb_inter_list, self._ptc_lub_list = fs2.MMD_lub_wrapper(tobj, tobj, rs2)
        self._lamb_inter_list, self._ptc_lub_list = fs2.MMD_lub_petsc(tobj, tobj, rs2)
        
        # # dbg
        # comm = PETSc.COMM_WORLD.tompi4py()
        # rank = comm.Get_rank()
        # print('dbg initial_lub, current rank =', rank)
        # print(np.vstack(self._lamb_inter_list))
        return True
    
    def _create_U(self):
        velocity = self._u_pkg.createGlobalVector()
        velocity.zeroEntries()
        velocity.assemble()
        self._velocity_petsc = velocity
        self._velocity = self.vec_scatter(velocity, destroy=False)
        return True
    
    def _create_F(self):
        assert self.get_n_obj() == 1
        
        comm = PETSc.COMM_WORLD.tompi4py()
        rank = comm.Get_rank()
        obji = self.get_obj_list()[0]
        NS = obji.get_u_geo().get_n_nodes()
        phi = obji.get_u_geo().get_phi()
        For = self.get_kwargs()['For']
        Tor = self.get_kwargs()['Tor']
        
        self._Fmdf_petsc = self._f_pkg.createGlobalVector()
        self._Fmdf_petsc.zeroEntries()
        self._Fmdf_petsc.assemble()
        
        F1 = fs2.F_fun(NS, For, Tor, phi)
        self._force_petsc = self._f_pkg.createGlobalVector()
        self._force_petsc.zeroEntries()
        _, f_glbIdx_all = obji.get_f_geo().get_glbIdx()
        if rank == 0:
            self._force_petsc[f_glbIdx_all] = F1[:]
        self._force_petsc.assemble()
        self._force = self.vec_scatter(self._force_petsc, destroy=False)
        return True
    
    def create_empty_M(self):
        self._check_create_empty_M()
        
        # create matrix
        for tmat in (self._Minf_petsc, self._Rlub_petsc, self._Rtol_petsc):
            tmat.setSizes((self._velocity_petsc.getSizes(), self._force_petsc.getSizes()))
            tmat.setType('dense')
            tmat.setFromOptions()
            tmat.setUp()
        return self._M_petsc
    
    def _create_matrix_obj(self, obj1, m_petsc, INDEX='', *args):
        # obj1 contain velocity information, obj2 contain force information
        kwargs = self.get_kwargs()
        
        n_obj = len(self.get_all_obj_list())
        for i0, obj2 in enumerate(self.get_all_obj_list()):
            kwargs['INDEX'] = ' %d/%d, ' % (i0 + 1, n_obj) + INDEX
            assert 'forceSphere2d' in obj2.get_matrix_method()
            self._check_args_dict[obj2.get_matrix_method()](**kwargs)
            self._method_dict[obj2.get_matrix_method()](obj1, obj2, m_petsc, **kwargs)
        m_petsc.assemble()
        return True
    
    def create_matrix_light(self):
        i0, n_obj, obj1 = 0, 1, self.get_all_obj_list()[0]
        INDEX = ' %d/%d' % (i0 + 1, n_obj)
        self._create_matrix_obj(obj1, self._M_petsc, INDEX)
        return True
    
    def create_matrix(self, PETScSysPrint=True):
        assert self.get_n_obj() == 1
        # print('dbg1')
        
        t0 = time()
        self.create_F_U()
        if not self._M_petsc.isAssembled():
            self.create_empty_M()
            self._M_destroyed = False
        self.create_matrix_light()
        t1 = time()
        PETSc.Sys.Print('  %s: create matrix use: %fs' % (str(self), (t1 - t0)))
        return True
    
    def solve_resistance(self, ini_guess=None, PETScSysPrint=True):
        t0 = time()
        kwargs = self._kwargs
        solve_method = kwargs['solve_method']
        precondition_method = kwargs['precondition_method']
        
        if ini_guess is not None:
            err_msg = 'size of initial guess for force vector must equal to the number of M matrix rows. '
            assert self._velocity_petsc.getSize() == ini_guess.getSize(), err_msg
            self._velocity_petsc[:] = ini_guess[:]
        
        ksp = PETSc.KSP()
        ksp.create(comm=PETSc.COMM_WORLD)
        ksp.setType(solve_method)
        ksp.getPC().setType(precondition_method)
        ksp.setOperators(self.get_Rtol_petsc())
        # OptDB = PETSc.Options()
        ksp.setFromOptions()
        ksp.setGMRESRestart(ksp.getTolerances()[-1])
        ksp.setInitialGuessNonzero(True)
        ksp.setUp()
        
        self._solve_velocity(ksp)
        self._residualNorm = self._resolve_fmdf(ksp)
        ksp.destroy()
        
        if PETScSysPrint:
            t1 = time()
            PETSc.Sys.Print('  %s: solve matrix equation use: %fs, with residual norm %e' %
                            (str(self) + 'solve_resistance', (t1 - t0), self._residualNorm))
        return self._residualNorm
    
    def _solve_velocity(self, ksp):
        assert self.get_n_obj() == 1
        frac = 1 / (np.pi * self.get_kwargs()['mu'])  # 前置系数
        
        # self.get_Minf_petsc().view()
        self.get_Minf_petsc().mult(self.get_force_petsc(), self.get_Fmdf_petsc())
        self.get_Fmdf_petsc().scale(frac)
        # ksp.getOperators()[0].view()
        print(self._velocity_petsc.getSizes())
        print(self.get_Fmdf_petsc().getSizes())
        print()
        # assert 1 == 2
        if self._kwargs['getConvergenceHistory']:
            ksp.setConvergenceHistory()
            ksp.solve(self.get_Fmdf_petsc(), self._velocity_petsc)
            self._convergenceHistory = ksp.getConvergenceHistory()
        else:
            ksp.solve(self.get_Fmdf_petsc(), self._velocity_petsc)
        
        # the following codes are commanded for speedup.
        # # reorder force from petsc index to normal index, and separate to each object.
        # t_velocity = self.vec_scatter(self._velocity_petsc, destroy=False)
        # tmp_velocity = []
        # tmp_spin = []
        # for obj0 in self.get_obj_list():  # type: geo.sphere_particle_2d
        #     ugeo = obj0.get_u_geo()
        #     _, u_glbIdx_all = ugeo.get_glbIdx()
        #     obj0.set_velocity(t_velocity[u_glbIdx_all[:ugeo.get_dof() * ugeo.get_n_nodes()]])
        #     obj0.set_spin(t_velocity[u_glbIdx_all[ugeo.get_dof() * ugeo.get_n_nodes():]])
        #     tmp_velocity.append(obj0.get_velocity())
        #     tmp_spin.append(obj0.get_spin())
        # self._velocity = np.hstack(tmp_velocity)
        # self._spin = np.hstack(tmp_spin)
        return True
    
    def _resolve_fmdf(self, ksp):
        assert self.get_n_obj() == 1
        
        resolve_fmdf_petsc = self.get_Rtol_petsc().createVecRight()
        self.get_Rtol_petsc().mult(self._velocity_petsc, resolve_fmdf_petsc)
        self._re_fmdf = self.vec_scatter(resolve_fmdf_petsc)
        for obj0 in self.get_all_obj_list():
            _, f_glbIdx_all = obj0.get_f_geo().get_glbIdx()
            obj0.set_re_fmdf(self._re_fmdf[f_glbIdx_all])
        self._finish_solve = True
        return ksp.getResidualNorm()


class ForceSphereObj(StokesFlowObj):
    def __init__(self):
        super().__init__()
        self._re_fmdf = np.zeros([0])  # resolved information of modified force
    
    def get_re_fmdf(self):
        return self._re_fmdf
    
    def set_re_fmdf(self, re_fmdf):
        self._re_fmdf = re_fmdf
    
    def set_data(self, f_geo: base_geo, u_geo: base_geo, name='...', **kwargs):
        err_msg = 'current version, the geometry contains force and velocity must same. '
        assert f_geo == u_geo, err_msg
        super().set_data(f_geo, u_geo, name=name, **kwargs)
    
    def get_spin(self):
        return self.get_u_geo().get_spin()
    
    def set_spin(self, w: np.array):
        return self.get_u_geo().set_spin(w)


problem_dic = {
    'rs':                           StokesFlowProblem,
    'rs_plane':                     StokesFlowProblem,
    'lg_rs':                        StokesFlowProblem,
    'tp_rs':                        StokesFlowProblem,
    'pf':                           StokesFlowProblem,
    'pf_dualPotential':             DualPotentialProblem,
    'rs_stokesletsInPipe':          StokesletsInPipeProblem,
    'pf_stokesletsInPipe':          StokesletsInPipeProblem,
    'pf_stokesletsInPipeforcefree': StokesletsInPipeforcefreeProblem,
    'pf_stokesletsTwoPlane':        StokesletsTwoPlaneProblem,
    'pf_infhelix':                  StokesFlowProblem,
    'pf_selfRepeat':                SelfRepeatHlxProblem,
    'pf_selfRotate':                SelfRotateProblem,
    'rs_selfRotate':                SelfRotateProblem,
    'lg_rs_selfRotate':             SelfRotateProblem,
    'pf_sphere':                    StokesFlowProblem,
    }

obj_dic = {
    'rs':                           StokesFlowObj,
    'rs_plane':                     StokesFlowObj,
    'lg_rs':                        StokesFlowObj,
    'tp_rs':                        StokesFlowObj,
    'pf':                           StokesFlowObj,
    'pf_dualPotential':             DualPotentialObj,
    'rs_stokesletsInPipe':          StokesFlowObj,
    'pf_stokesletsInPipe':          StokesFlowObj,
    'pf_stokesletsInPipeforcefree': StokesFlowObj,
    'pf_stokesletsTwoPlane':        StokesFlowObj,
    'pf_infhelix':                  StokesFlowObj,
    'pf_stokesletsRingInPipe':      StokesFlowRingObj,
    'pf_selfRepeat':                SelfRepeatObj,
    'pf_selfRotate':                SelfRotateObj,
    'rs_selfRotate':                SelfRotateObj,
    'lg_rs_selfRotate':             SelfRotateObj,
    'KRJ_slb':                      StokesFlowObj,
    'lighthill_slb':                StokesFlowObj,
    'pf_sphere':                    StokesFlowObj,
    }

# names of models that need two geometries.
two_geo_method_list = ('pf', 'ps', 'ps_ds', 'pf_ds', 'pf_sphere'
                                                     'pf_stokesletsInPipe',
                       'pf_stokesletsInPipeforcefree',
                       'pf_stokesletsTwoPlane', 'pf_infhelix',
                       'pf_ShearFlow',
                       'pf_dualPotential',
                       'pf_stokesletsRingInPipe',)

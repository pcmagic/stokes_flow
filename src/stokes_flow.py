# coding=utf-8
"""
functions for solving stokes flow using regularised stokeslets (and its improved) method.
Zhang Ji, 20160409
"""
import sys

# sys.path = ['..'] + sys.path
from pyvtk import *
import copy
import numpy as np
import scipy.io as sio
from evtk.hl import pointsToVTK, gridToVTK
from memory_profiler import profile
from petsc4py import PETSc
# import warnings
import pickle
from src.geo import *
from src.support_class import *
from time import time
from src.ref_solution import *
from pyvtk import *
from multiprocessing import Pool, cpu_count, Queue, Process
from tqdm import tqdm, tqdm_notebook, tnrange


class stokesFlowProblem:
    def __init__(self, **kwargs):
        self._obj_list = uniqueList( )  # contain objects
        self._kwargs = kwargs  # kwargs associate with solving method,
        self._force = np.zeros([0])  # force information
        self._force_petsc = PETSc.Vec( ).create(comm=PETSc.COMM_WORLD)
        self._velocity_petsc = PETSc.Vec( ).create(comm=PETSc.COMM_WORLD)  # velocity information
        self._re_velocity = np.zeros([0])  # resolved velocity information
        self._n_fnode = 0
        self._n_unode = 0
        self._f_pkg = PETSc.DMComposite( ).create(comm=PETSc.COMM_WORLD)
        self._u_pkg = PETSc.DMComposite( ).create(comm=PETSc.COMM_WORLD)
        self._M_petsc = PETSc.Mat( ).create(comm=PETSc.COMM_WORLD)  # M matrix
        self._M_destroyed = False  # weather Mat().destroy() have been called.
        self._dimension = -1  # 2 dimension or 3 dimension
        self._finish_solve = False
        self._pick_M = False  # weather save M matrix and F vector in binary files.
        self._n_unknown = 3  # number of unknowns at each  node.
        self._pick_filename = '..'  # prefix filename of pick files. i.e. filename_F.bin, filename_M.bin, filename_pick.bin.
        self._residualNorm = 0.
        self._convergenceHistory = np.zeros([0])

        from src import StokesFlowMethod
        self._method_dict = {
            'rs':                            StokesFlowMethod.regularized_stokeslets_matrix_3d_petsc,
            'tp_rs':                         StokesFlowMethod.two_para_regularized_stokeslets_matrix_3d,
            'lg_rs':                         StokesFlowMethod.legendre_regularized_stokeslets_matrix_3d,
            'rs_precondition':               StokesFlowMethod.regularized_stokeslets_matrix_3d_petsc,
            'tp_rs_precondition':            StokesFlowMethod.two_para_regularized_stokeslets_matrix_3d,
            'lg_rs_precondition':            StokesFlowMethod.legendre_regularized_stokeslets_matrix_3d,
            'pf':                            StokesFlowMethod.point_force_matrix_3d_petsc,
            'rs_stokeslets':                 StokesFlowMethod.regularized_stokeslets_matrix_3d_petsc,
            'tp_rs_stokeslets':              StokesFlowMethod.two_para_regularized_stokeslets_matrix_3d,
            'lg_rs_stokeslets':              StokesFlowMethod.legendre_regularized_stokeslets_matrix_3d,
            'rs_stokesletsInPipe':           StokesFlowMethod.regularized_stokeslets_matrix_3d_petsc,
            'pf_stokesletsInPipe':           StokesFlowMethod.point_force_matrix_3d_petsc,
            'rs_stokeslets_precondition':    StokesFlowMethod.regularized_stokeslets_matrix_3d_petsc,
            'tp_rs_stokeslets_precondition': StokesFlowMethod.two_para_regularized_stokeslets_matrix_3d,
            'lg_rs_stokeslets_precondition': StokesFlowMethod.legendre_regularized_stokeslets_matrix_3d,
            'pf_stokeslets':                 StokesFlowMethod.point_force_matrix_3d_petsc,
            'pf_stokesletsTwoPlate':         StokesFlowMethod.two_plate_matrix_3d_petsc,
        }
        self._check_args_dict = {
            'rs':                            StokesFlowMethod.check_regularized_stokeslets_matrix_3d,
            'tp_rs':                         StokesFlowMethod.check_two_para_regularized_stokeslets_matrix_3d,
            'lg_rs':                         StokesFlowMethod.check_legendre_regularized_stokeslets_matrix_3d,
            'rs_precondition':               StokesFlowMethod.check_regularized_stokeslets_matrix_3d,
            'tp_rs_precondition':            StokesFlowMethod.check_two_para_regularized_stokeslets_matrix_3d,
            'lg_rs_precondition':            StokesFlowMethod.check_legendre_regularized_stokeslets_matrix_3d,
            'pf':                            StokesFlowMethod.check_point_force_matrix_3d_petsc,
            'rs_stokeslets':                 StokesFlowMethod.check_regularized_stokeslets_matrix_3d,
            'tp_rs_stokeslets':              StokesFlowMethod.check_two_para_regularized_stokeslets_matrix_3d,
            'lg_rs_stokeslets':              StokesFlowMethod.check_legendre_regularized_stokeslets_matrix_3d,
            'rs_stokesletsInPipe':           StokesFlowMethod.check_regularized_stokeslets_matrix_3d,
            'pf_stokesletsInPipe':           StokesFlowMethod.check_point_force_matrix_3d_petsc,
            'rs_stokeslets_precondition':    StokesFlowMethod.check_regularized_stokeslets_matrix_3d,
            'tp_rs_stokeslets_precondition': StokesFlowMethod.check_two_para_regularized_stokeslets_matrix_3d,
            'lg_rs_stokeslets_precondition': StokesFlowMethod.check_legendre_regularized_stokeslets_matrix_3d,
            'pf_stokeslets':                 StokesFlowMethod.check_point_force_matrix_3d_petsc,
            'pf_stokesletsTwoPlate':         StokesFlowMethod.check_two_plate_matrix_3d_petsc,
        }
        self._check_args_dict[kwargs['matrix_method']](**kwargs)

    def add_obj(self, obj):
        """
        Add a new object to the problem.

        :type obj: stokesFlowObj
        :param obj: added object
        :return: none.
        """
        self._obj_list.append(obj)
        obj.set_index(self.get_n_obj( ))
        obj.set_problem(self)
        self._f_pkg.addDM(obj.get_f_geo( ).get_dmda( ))
        self._u_pkg.addDM(obj.get_u_geo( ).get_dmda( ))
        self._n_fnode += obj.get_n_f_node( )
        self._n_unode += obj.get_n_u_node( )
        return True

    def __repr__(self):
        return 'StokesFlowProblem'

    def __str__(self):
        return self._kwargs['name']

    def create_matrix_obj(self, obj1, m_petsc, *args):
        kwargs = self._kwargs
        for obj2 in self.get_all_obj_list( ):
            self._method_dict[self.get_matrix_method( )](obj1, obj2, m_petsc, **kwargs)
        m_petsc.assemble( )
        return True

    def create_F_U(self):
        comm = PETSc.COMM_WORLD.tompi4py( )
        size = comm.Get_size( )
        rank = comm.Get_rank( )
        # create f and u DMComposite
        self._f_pkg.setFromOptions( )
        self._f_pkg.setUp( )
        self._u_pkg.setFromOptions( )
        self._u_pkg.setUp( )

        # velocity of problem and global index of force and velocity geometries.
        velocity = self._u_pkg.createGlobalVector( )
        f_isglb = self._f_pkg.getGlobalISs( )
        u_isglb = self._u_pkg.getGlobalISs( )
        for i0, obj0 in enumerate(self.get_obj_list( )):
            obj0.get_f_geo( ).set_glbIdx(f_isglb[i0].getIndices( ))
            obj0.get_u_geo( ).set_glbIdx(u_isglb[i0].getIndices( ))
            u0 = obj0.get_velocity( )
            _, u_glbIdx_all = obj0.get_u_geo( ).get_glbIdx( )
            if rank == 0:
                velocity[u_glbIdx_all] = u0[:]
        velocity.assemble( )
        self._velocity_petsc = velocity

        # force
        self._force_petsc = self._f_pkg.createGlobalVector( )
        return True

    def create_matrix(self):
        t0 = time( )
        comm = PETSc.COMM_WORLD.tompi4py( )
        rank = comm.Get_rank( )
        err_msg = 'at least one object is necessary. '
        assert len(self._obj_list) != 0, err_msg
        kwargs = self._kwargs
        solve_method = kwargs['solve_method']

        err_msg = 'unequal force and velocity degrees of freedom, only lsqr method is accept. '
        for obj1 in self.get_obj_list( ):
            assert obj1.get_n_force( ) == obj1.get_n_velocity( ) or solve_method == 'lsqr', err_msg

        self.create_F_U( )
        # create matrix
        if not self._M_petsc.isAssembled( ):
            self._M_petsc.setSizes((self._velocity_petsc.getSizes( ), self._force_petsc.getSizes( )))
            self._M_petsc.setType('dense')
            self._M_petsc.setFromOptions( )
            self._M_petsc.setUp( )
        for obj1 in self.get_obj_list( ):
            self.create_matrix_obj(obj1, self._M_petsc)
        # self._M_petsc.view()

        t1 = time( )
        PETSc.Sys.Print('%s: create matrix use: %fs' % (str(self), (t1 - t0)))
        return True

    def set_matrix(self, m_petsc):
        self.create_F_U( )
        self._M_petsc = m_petsc
        return True

    def create_obj_matrix(self, objf: 'stokesFlowObj',  # force object
                          obju: 'stokesFlowObj',  # velocity object
                          copy_obj=True,
                          **kwargs):
        if copy_obj:
            obj1 = objf.copy( )
            obj2 = obju.copy( )
        else:
            obj1 = objf
            obj2 = obju
        t_f_pkg = PETSc.DMComposite( ).create(comm=PETSc.COMM_WORLD)
        t_u_pkg = PETSc.DMComposite( ).create(comm=PETSc.COMM_WORLD)
        t_f_pkg.addDM(obj1.get_f_geo( ).get_dmda( ))
        t_u_pkg.addDM(obj2.get_u_geo( ).get_dmda( ))
        t_f_pkg.setFromOptions( )
        t_f_pkg.setUp( )
        t_u_pkg.setFromOptions( )
        t_u_pkg.setUp( )
        f_isglb = t_f_pkg.getGlobalISs( )
        u_isglb = t_u_pkg.getGlobalISs( )
        obj1.get_f_geo( ).set_glbIdx(f_isglb[0].getIndices( ))
        obj2.get_u_geo( ).set_glbIdx(u_isglb[0].getIndices( ))
        t_velocity = t_u_pkg.createGlobalVector( )
        t_force = t_f_pkg.createGlobalVector( )
        m_petsc = PETSc.Mat( ).create(comm=PETSc.COMM_WORLD)
        m_petsc.setSizes((t_velocity.getSizes( ), t_force.getSizes( )))
        m_petsc.setType('dense')
        m_petsc.setFromOptions( )
        m_petsc.setUp( )
        self._method_dict[kwargs['matrix_method']](obj2, obj1, m_petsc, **kwargs)
        m_petsc.assemble( )
        t_velocity.destroy( )
        t_force.destroy( )
        return m_petsc

    def solve(self, ini_guess=None):
        t0 = time( )
        kwargs = self._kwargs
        solve_method = kwargs['solve_method']
        precondition_method = kwargs['precondition_method']
        comm = PETSc.COMM_WORLD.tompi4py( )
        rank = comm.Get_rank( )

        if ini_guess is None:
            self._force_petsc.set(0)
        else:
            err_msg = 'size of initial guess for force vector must equal to the number of M matrix rows. '
            assert self._force_petsc.getSize( ) == ini_guess.getSize( ), err_msg
            self._force_petsc[:] = ini_guess[:]

        ksp = PETSc.KSP( )
        ksp.create(comm=PETSc.COMM_WORLD)
        ksp.setType(solve_method)
        ksp.getPC( ).setType(precondition_method)
        ksp.setOperators(self._M_petsc)
        OptDB = PETSc.Options( )
        ksp.setFromOptions( )
        if not OptDB.getBool('debug', False):
            tolerance = ksp.getTolerances( )
            ksp.setGMRESRestart(tolerance[-1])
        ksp.setInitialGuessNonzero(True)
        ksp.setUp( )

        self._solve_force(ksp)
        self._residualNorm = self._resolve_velocity(ksp)
        ksp.destroy( )

        t1 = time( )
        PETSc.Sys.Print('%s: solve matrix equation use: %fs, with residual norm %e' %
                        (str(self), (t1 - t0), self._residualNorm))
        return self._residualNorm

    def _solve_force(self, ksp):
        if self._kwargs['getConvergenceHistory']:
            ksp.setConvergenceHistory( )
            ksp.solve(self._velocity_petsc, self._force_petsc)
            self._convergenceHistory = ksp.getConvergenceHistory( )
        else:
            ksp.solve(self._velocity_petsc, self._force_petsc)
        self._force = self.vec_scatter(self._force_petsc, destroy=False)
        for obj0 in self.get_obj_list( ):
            _, f_glbIdx_all = obj0.get_f_geo( ).get_glbIdx( )
            obj0.set_force(self._force[f_glbIdx_all])
        return True

    def _resolve_velocity(self, ksp):
        re_velocity_petsc = self._M_petsc.createVecLeft( )
        self._M_petsc.mult(self._force_petsc, re_velocity_petsc)
        self._re_velocity = self.vec_scatter(re_velocity_petsc)
        for obj0 in self.get_obj_list( ):
            _, u_glbIdx_all = obj0.get_u_geo( ).get_glbIdx( )
            obj0.set_re_velocity(self._re_velocity[u_glbIdx_all])
        self._finish_solve = True
        return ksp.getResidualNorm( )

    def vtk_self(self, filename):
        self.check_finish_solve( )
        comm = PETSc.COMM_WORLD.tompi4py( )
        rank = comm.Get_rank( )
        obj0 = stokesFlowObj( )
        obj0.combine(self.get_obj_list( ), set_re_u=True, set_force=True)
        obj0.set_name('Prb')
        if rank == 0:
            obj0.vtk(filename)
        return True

    def solve_obj_u(self, obj: "stokesFlowObj", ):
        """
        solve velocity for given object.
        """
        self.check_finish_solve( )
        comm = PETSc.COMM_WORLD.tompi4py( )
        rank = comm.Get_rank( )
        kwargs = self._kwargs
        n_node_threshold = kwargs['n_node_threshold']

        # partition object into several parts if it contain too many nodes; and then solve in a loop.
        sub_obj_list = uniqueList( )
        n_obj_nodes = obj.get_n_u_node( )
        n_sub_obj = int(obj.get_n_u_node( ) / n_node_threshold) + 1
        obj_nodes = obj.get_u_nodes( )
        for i0 in range(n_sub_obj):
            sub_geo1 = geo( )
            id0 = i0 * n_obj_nodes // n_sub_obj
            id1 = (i0 + 1) * n_obj_nodes // n_sub_obj
            sub_geo1.set_nodes(obj_nodes[id0:id1], resetVelocity=True, deltalength=obj.get_u_geo( ).get_deltaLength( ))
            sub_obj1 = stokesFlowObj( )
            sub_obj_kwargs = {'name': '%s_sub_%d' % (str(obj), i0)}
            sub_obj1.set_data(sub_geo1, sub_geo1, **sub_obj_kwargs)
            sub_obj_list.append(sub_obj1)

        obj_u = obj.get_velocity( ).copy( )
        for i1, sub_obj1 in enumerate(sub_obj_list):
            sub_u_dmda = sub_obj1.get_u_geo( ).get_dmda( )
            sub_u_pkg = PETSc.DMComposite( ).create(comm=PETSc.COMM_WORLD)
            sub_u_pkg.addDM(sub_u_dmda)
            sub_u_pkg.setFromOptions( )
            sub_u_pkg.setUp( )
            sub_u_isglb = sub_u_pkg.getGlobalISs( )
            sub_obj_u_petsc = sub_u_dmda.createGlobalVector( )
            sub_obj1.get_u_geo( ).set_glbIdx(sub_u_isglb[0].getIndices( ))
            m_petsc = PETSc.Mat( ).create(comm=PETSc.COMM_WORLD)
            m_petsc.setSizes((sub_obj_u_petsc.getSizes( ), self._force_petsc.getSizes( )))
            m_petsc.setType('dense')
            m_petsc.setFromOptions( )
            m_petsc.setUp( )
            self.create_matrix_obj(sub_obj1, m_petsc)
            # sub_obj_u_petsc.set(0)
            m_petsc.mult(self._force_petsc, sub_obj_u_petsc)
            sub_obj_u = self.vec_scatter(sub_obj_u_petsc)
            id0 = i1 * n_obj_nodes // n_sub_obj * 3
            id1 = (i1 + 1) * n_obj_nodes // n_sub_obj * 3
            obj_u[id0:id1] = sub_obj_u[:]
            m_petsc.destroy( )
            sub_u_pkg.destroy( )
        return obj_u

    def vtk_check(self, filename: str,
                  obj: "stokesFlowObj",
                  ref_slt: "slt" = None):
        # Todo: special case: forcefreeComposite.
        return self._vtk_check(filename, obj, ref_slt)

    def _vtk_check(self, filename: str,
                   obj: "stokesFlowObj",
                   ref_slt: "slt" = None):
        """
        check velocity at the surface of objects.

        :type filename: str
        :param filename: output file name
        :type obj: stokesFlowObj
        :param obj: check object (those known exact velocity information. )
        :param ref_slt: reference solution function headle
        :return: none.
        """
        self.check_finish_solve( )
        comm = PETSc.COMM_WORLD.tompi4py( )
        rank = comm.Get_rank( )
        kwargs = self._kwargs
        n_node_threshold = kwargs['n_node_threshold']

        obj_u = self.solve_obj_u(obj)
        obj.set_re_velocity(obj_u)

        if ref_slt is None:
            u_exact = obj.get_velocity( )
        else:
            u_exact = ref_slt.get_solution(obj.get_u_geo( ))
        if rank == 0:
            velocity_x = obj_u[0::3].copy( )
            velocity_y = obj_u[1::3].copy( )
            velocity_z = obj_u[2::3].copy( )
            velocity_err_x = u_exact[0::3].copy( ) - velocity_x
            velocity_err_y = u_exact[1::3].copy( ) - velocity_y
            velocity_err_z = u_exact[2::3].copy( ) - velocity_z
            rel_err_x = np.abs(velocity_err_x / velocity_x)
            rel_err_y = np.abs(velocity_err_y / velocity_y)
            rel_err_z = np.abs(velocity_err_z / velocity_z)
            nodes = obj.get_u_nodes( )
            pointsToVTK(filename, nodes[:, 0], nodes[:, 1], nodes[:, 2],
                        data={"velocity_err": (velocity_err_x, velocity_err_y, velocity_err_z),
                              "velocity":     (velocity_x, velocity_y, velocity_z),
                              "rel_err":      (rel_err_x, rel_err_y, rel_err_z)})

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
            field_range = field_range.transpose( )
            n_range = field_range.shape
        if n_range != (2, 3):
            err_msg = 'maximum and minimum coordinates for the rectangular velocity field are necessary, ' + \
                      'i.e. range = [[0,0,0],[10,10,10]]. '
            raise ValueError(err_msg)
        self.check_finish_solve( )
        n_grid = n_grid.ravel( )
        if n_grid.shape != (3,):
            err_msg = 'mesh number of each axis for the rectangular velocity field is necessary, ' + \
                      'i.e. n_grid = [100, 100, 100]. '
            raise ValueError(err_msg)

        return field_range, n_grid

    def vtk_velocity(self, filename: str):
        comm = PETSc.COMM_WORLD.tompi4py( )
        rank = comm.Get_rank( )
        self.check_finish_solve( )
        field_range, n_grid = self.check_vtk_velocity( )
        region_type = self._kwargs['region_type']

        if not self._M_destroyed:
            self._M_petsc.destroy( )
            self._M_destroyed = True
        myregion = region( )
        full_region_x, full_region_y, full_region_z = myregion.type[region_type](field_range, n_grid)

        # solve velocity at cell center.
        n_para = 3 * n_grid[1] * n_grid[2]

        # to handle big problem, solve velocity field at every splice along x axis.
        if rank == 0:
            u_x = np.zeros((n_grid[0], n_grid[1], n_grid[2]))
            u_y = np.zeros((n_grid[0], n_grid[1], n_grid[2]))
            u_z = np.zeros((n_grid[0], n_grid[1], n_grid[2]))
        else:
            u_x = None
            u_y = None
            u_z = None
        obj0 = stokesFlowObj( )
        for i0 in range(full_region_x.shape[0]):
            temp_x = full_region_x[i0]
            temp_y = full_region_y[i0]
            temp_z = full_region_z[i0]
            temp_nodes = np.c_[temp_x.ravel( ), temp_y.ravel( ), temp_z.ravel( )]
            temp_geo = geo( )
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
                      pointData={"velocity": (u_x, u_y, u_z)})
        return True

    def vtk_obj(self, filename):
        self.check_finish_solve( )
        for obj1 in self._obj_list:
            obj1.vtk(filename)

    def vtk_tetra(self,
                  filename: str,
                  bgeo: geo):
        comm = PETSc.COMM_WORLD.tompi4py( )
        rank = comm.Get_rank( )
        self.check_finish_solve( )

        bnodes = bgeo.get_nodes( )
        belems, elemtype = bgeo.get_mesh( )
        err_msg = 'mesh type is NOT tetrahedron. '
        assert elemtype == 'tetra', err_msg

        PETSc.Sys.Print('export to %s.vtk, element type is %s, contain %d nodes and %d elements. '
                        % (filename, elemtype, bnodes.shape[0], belems.shape[0]))

        obj1 = stokesFlowObj( )
        obj1.set_data(bgeo, bgeo)
        u = self.solve_obj_u(obj1)
        if rank == 0:
            u = np.array(u).reshape(bnodes.shape)
            vtk = VtkData(
                    UnstructuredGrid(bnodes,
                                     tetra=belems,
                                     ),
                    PointData(Vectors(u, name='u')),
                    str(self)
            )
            vtk.tofile(filename)
        return True

    def saveM_ASCII(self, filename: str = '..', ):
        if filename[-4:] != '.txt':
            filename = filename + '.txt'
        viewer = PETSc.Viewer( ).createASCII(filename, 'w', comm=PETSc.COMM_WORLD)
        viewer(self._M_petsc)
        viewer.destroy( )
        PETSc.Sys.Print('%s: save M matrix to %s' % (str(self), filename))
        return True

    def saveF_ASCII(self, filename: str = '..', ):
        if filename[-4:] != '.txt':
            filename = filename + '.txt'
        viewer = PETSc.Viewer( ).createASCII(filename, 'w', comm=PETSc.COMM_WORLD)
        viewer(self._force_petsc)
        viewer.destroy( )
        PETSc.Sys.Print('%s: save force to %s' % (str(self), filename))
        return True

    def saveV_ASCII(self, filename: str = '..', ):
        if filename[-4:] != '.txt':
            filename = filename + '.txt'
        viewer = PETSc.Viewer( ).createASCII(filename, 'w', comm=PETSc.COMM_WORLD)
        viewer(self._velocity_petsc)
        viewer.destroy( )
        PETSc.Sys.Print('%s: save velocity to %s' % (str(self), filename))
        return True

    def saveF_Binary(self, filename: str = '..', ):
        if filename[-4:] != '.bin':
            filename = filename + '.bin'
        viewer = PETSc.Viewer( ).createBinary(filename, 'w', comm=PETSc.COMM_WORLD)
        viewer(self._force_petsc)
        viewer.destroy( )
        PETSc.Sys.Print('%s: save force to %s' % (str(self), filename))
        return True

    def saveV_Binary(self, filename: str = '..', ):
        if filename[-4:] != '.bin':
            filename = filename + '.bin'
        viewer = PETSc.Viewer( ).createBinary(filename, 'w', comm=PETSc.COMM_WORLD)
        viewer(self._velocity_petsc)
        viewer.destroy( )
        PETSc.Sys.Print('%s: save velocity to %s' % (str(self), filename))
        return True

    def saveM_Binary(self, filename: str = '..', ):
        if filename[-4:] != '.bin':
            filename = filename + '.bin'
        viewer = PETSc.Viewer( ).createBinary(filename, 'w', comm=PETSc.COMM_WORLD)
        viewer.pushFormat(viewer.Format.NATIVE)
        viewer(self._M_petsc)
        viewer.destroy( )
        PETSc.Sys.Print('%s: save M matrix to %s' % (str(self), filename))
        return True

    def loadM_Binary(self, filename: str):
        if filename[-4:] != '.bin':
            filename = filename + '.bin'
        viewer = PETSc.Viewer( ).createBinary(filename, 'r')
        self._M_petsc = PETSc.Mat( ).create(comm=PETSc.COMM_WORLD)
        self._M_petsc.setSizes((self._velocity_petsc.getSizes( ), self._force_petsc.getSizes( )))
        self._M_petsc.setType('dense')
        self._M_petsc.setFromOptions( )
        self._M_petsc = self._M_petsc.load(viewer)
        return True

    def destroy(self):
        self._force_petsc.destroy( )
        self._velocity_petsc.destroy( )
        self._f_pkg.destroy( )
        self._u_pkg.destroy( )
        if not self._M_destroyed:
            self._M_petsc.destroy( )
            self._M_destroyed = True
        return True

    def pickmyself(self, filename: str, check=False, pick_M=False):
        t0 = time( )
        comm = PETSc.COMM_WORLD.tompi4py( )
        rank = comm.Get_rank( )
        self._pick_filename = filename
        self._pick_M = pick_M

        if not check and self._finish_solve and pick_M:
            self.saveM_Binary(filename + '_M')
        if not check:
            # self._force_petsc.view()
            self.destroy( )

        if rank == 0:
            with open(filename + '_pick.bin', 'wb') as output:
                pickler = pickle.Pickler(output, -1)
                pickler.dump(self)

        if not check:
            self.unpickmyself( )
        t1 = time( )
        PETSc.Sys( ).Print('%s: pick the problem use: %fs' % (str(self), (t1 - t0)))
        return True

    def unpickmyself(self):
        filename = self._pick_filename
        err_msg = 'call pickmyself() before unpickmyself(). i.e. store date first and reload them at restart mode. '
        assert filename != '..', err_msg

        self._f_pkg = PETSc.DMComposite( ).create(comm=PETSc.COMM_WORLD)
        self._u_pkg = PETSc.DMComposite( ).create(comm=PETSc.COMM_WORLD)
        self._M_petsc = PETSc.Mat( ).create(comm=PETSc.COMM_WORLD)  # M matrix
        for obj1 in self._obj_list:
            obj1.unpickmyself( )
            self._f_pkg.addDM(obj1.get_f_geo( ).get_dmda( ))
            self._u_pkg.addDM(obj1.get_u_geo( ).get_dmda( ))
        self._f_pkg.setFromOptions( )
        self._u_pkg.setFromOptions( )
        # Todo: setUp f_pkg and u_pkg at a appropriate time
        # self._f_pkg.setUp()
        # self._u_pkg.setUp()

        if self._finish_solve:
            self.create_F_U( )
            self._force_petsc[:] = self._force[:]
            self._force_petsc.assemble( )
            # self._force_petsc.view()
        if self._finish_solve and self._pick_M:
            self.loadM_Binary(filename + '_M')
        return True

    def view_log_M(self, **kwargs):
        m = self._M_petsc.getDenseArray( )
        view_args = {
            'vmin':  -10,
            'vmax':  0,
            'title': 'log10_abs_' + kwargs['matrix_method'],
            'cmap':  'gray'
        }
        self._view_matrix(np.log10(np.abs(m) + 1e-100), **view_args)

    def view_M(self, **kwargs):
        m = self._M_petsc.getDenseArray( )
        view_args = {
            'vmin':        None,
            'vmax':        None,
            'title': kwargs['matrix_method'],
            'cmap':        'gray'
        }
        self._view_matrix(m, **view_args)

    def _view_matrix(self, m, **kwargs):
        args = {
            'vmin':  None,
            'vmax':  None,
            'title': ' ',
            'cmap':  None
        }
        for key, value in args.items( ):
            if key in kwargs:
                args[key] = kwargs[key]

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots( )
        cax = ax.matshow(m,
                         origin='lower',
                         vmin=args['vmin'],
                         vmax=args['vmax'],
                         cmap=plt.get_cmap(args['cmap']))
        fig.colorbar(cax)
        plt.title(args['title'])
        plt.show( )

    def check_finish_solve(self):
        err_msg = 'call solve() method first.'
        assert self._finish_solve, err_msg
        return True

    def print_info(self):
        for obj in self.get_obj_list( ):
            obj.print_info( )
        PETSc.Sys.Print('%s: force nodes: %d, velocity nodes: %d'
                        % (str(self), self.get_n_f_node( ), self.get_n_u_node( )))

    def get_M(self, **kwargs):
        err_msg = 'this method must be called befor method vtk_velocity(), the latter one would destroy the M matrix. '
        assert not self._M_destroyed, err_msg

        M = self._M_petsc.getDenseArray( ).copy( )
        return M

    def get_n_f_node(self):
        return self._n_fnode

    def get_n_u_node(self):
        return self._n_unode

    def get_n_force(self):
        return self._force_petsc.getSizes( )[1]

    def get_n_velocity(self):
        return self._velocity_petsc.getSizes( )[1]

    def get_obj_list(self):
        return self._obj_list

    def get_all_obj_list(self):
        return self.get_obj_list( )

    def get_n_obj(self):
        return len(self._obj_list)

    def get_force(self):
        return self._force

    def set_kwargs(self, **kwargs):
        self._kwargs = kwargs  # kwargs associate with solving method,
        return True

    def get_force_petsc(self):
        return self._force_petsc

    def get_n_unknown(self):
        return self._n_unknown

    def get_kwargs(self):
        return self._kwargs

    def get_matrix_method(self):
        return self._kwargs['matrix_method']

    def get_residualNorm(self):
        self.check_finish_solve( )
        return self._residualNorm

    def get_convergenceHistory(self):
        self.check_finish_solve( )
        return self._convergenceHistory

    def get_name(self):
        return self._kwargs['name']

    def vec_scatter(self, vec_petsc, destroy=True):
        scatter, temp = PETSc.Scatter( ).toAll(vec_petsc)
        scatter.scatterBegin(vec_petsc, temp, False, PETSc.Scatter.Mode.FORWARD)
        scatter.scatterEnd(vec_petsc, temp, False, PETSc.Scatter.Mode.FORWARD)
        vec = temp.getArray( )
        if destroy:
            vec_petsc.destroy( )
        return vec

    def show_velocity(self, length_factor=1, show_nodes=True):
        geo_list = uniqueList( )
        for obj1 in self.get_obj_list( ):
            geo_list.append(obj1.get_u_geo( ))
        temp_geo = geo( )
        temp_geo.combine(geo_list)
        temp_geo.show_velocity(length_factor=length_factor, show_nodes=show_nodes)
        return True

    def show_f_nodes(self):
        geo_list = uniqueList( )
        for obj1 in self.get_obj_list( ):
            geo_list.append(obj1.get_f_geo( ))
        temp_geo = geo( )
        temp_geo.combine(geo_list)
        temp_geo.show_nodes( )
        return True

    def show_u_nodes(self):
        geo_list = uniqueList( )
        for obj1 in self.get_obj_list( ):
            geo_list.append(obj1.get_u_geo( ))
        temp_geo = geo( )
        temp_geo.combine(geo_list)
        temp_geo.show_nodes( )
        return True

    def show_f_u_nodes(self):
        geo_list = uniqueList( )
        for obj1 in self.get_obj_list( ):
            geo_list.append(obj1.get_f_geo( ))
        for obj1 in self.get_obj_list( ):
            geo_list.append(obj1.get_u_geo( ))
        temp_geo = geo( )
        temp_geo.combine(geo_list)
        temp_geo.show_nodes( )
        return True


class stokesFlowObj:
    # general class of object, contain general properties of objcet.
    def __init__(self):
        self._index = -1  # index of object
        self._f_geo = geo( )  # global coordinates of force nodes
        self._u_geo = geo( )  # global coordinates of velocity nodes
        self._re_velocity = np.zeros([0])  # resolved information
        self._force = np.zeros([0])  # force information
        self._type = 'uninitialized'  # object type
        self._name = '...'  # object name
        self._n_unknown = 3
        self._problem = None
        # self._copy_lock = threading.Lock()

    def __repr__(self):
        return self.get_obj_name( )

    def __str__(self):
        return self.get_name( )

    def print_info(self):
        PETSc.Sys.Print('%s: father %s, type %s, index %d, force nodes %d, velocity nodes %d'
                        % (self.get_name( ), self._problem.get_name( ), self._type, self.get_index( ),
                           self.get_n_f_node( ), self.get_n_u_node( )))
        return True

    def set_data(self,
                 f_geo: geo,
                 u_geo: geo,
                 name='...', **kwargs):
        err_msg = 'f_geo and u_geo need geo objects contain force and velocity nodes, respectively. '
        assert isinstance(f_geo, geo) and isinstance(u_geo, geo), err_msg

        self._f_geo = f_geo
        self._u_geo = u_geo
        self._force = np.zeros(self.get_f_nodes( ).size)
        self._name = name
        self._type = 'general obj'
        return True

    def set_velocity(self, velocity: np.array, **kwargs):
        return self.get_u_geo( ).set_velocity(velocity)

    def set_rigid_velocity(self, U):
        return self.get_u_geo( ).set_rigid_velocity(U)

    def get_problem(self):
        return self._problem

    def set_problem(self, problem: 'stokesFlowProblem'):
        self._problem = problem
        return True

    def copy(self) -> object:
        """
        copy a new object.
        """
        # with self._copy_lock:
        problem = self._problem
        self._problem = None
        self.get_f_geo( ).destroy_dmda( )
        if self.get_f_geo( ) is not self.get_u_geo( ):
            self.get_u_geo( ).destroy_dmda( )

        obj2 = copy.deepcopy(self)
        self.set_problem(problem)
        obj2.set_problem(problem)
        obj2.set_index(-1)
        self.get_f_geo( ).set_dmda( )
        if self.get_f_geo( ) is not self.get_u_geo( ):
            self.get_u_geo( ).set_dmda( )
        obj2.get_f_geo( ).set_dmda( )
        if obj2.get_f_geo( ) is not obj2.get_u_geo( ):
            obj2.get_u_geo( ).set_dmda( )
        return obj2

    def combine(self, obj_list: uniqueList, set_re_u=False, set_force=False):
        fgeo_list = uniqueList( )
        ugeo_list = uniqueList( )
        for obj0 in obj_list:
            err_msg = 'one or more objects in obj_list are not stokesFlowObj object. '
            assert isinstance(obj0, stokesFlowObj), err_msg
            fgeo_list.append(obj0.get_f_geo( ))
            ugeo_list.append(obj0.get_u_geo( ))

        fgeo = geo( )
        ugeo = geo( )
        fgeo.combine(fgeo_list)
        ugeo.combine(ugeo_list)
        self.set_data(fgeo, ugeo)

        if set_re_u:
            self.set_re_velocity(np.zeros([0]))
            for obj0 in obj_list:
                self.set_re_velocity(np.hstack((self.get_re_velocity( ), obj0.get_re_velocity( ))))
        if set_force:
            self.set_force(np.zeros([0]))
            for obj0 in obj_list:
                self.set_force(np.hstack((self.get_force( ), obj0.get_force( ))))
        return True

    def move(self, displacement):
        self.get_f_geo( ).move(displacement)
        if self.get_f_geo( ) is not self.get_u_geo( ):
            self.get_u_geo( ).move(displacement)
        return True

    def node_rotation(self, norm=np.array([0, 0, 1]), theta=0, rotation_origin=None):
        self.get_f_geo( ).node_rotation(norm, theta, rotation_origin)
        if self.get_f_geo( ) is not self.get_u_geo( ):
            self.get_u_geo( ).node_rotation(norm, theta, rotation_origin)
        return True

    def zoom(self, factor):
        self.get_f_geo( ).node_zoom(factor)
        if self.get_f_geo( ) is not self.get_u_geo( ):
            self.get_u_geo( ).node_zoom(factor)
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
        return self._f_geo.get_nodes( )

    def get_u_nodes(self):
        return self._u_geo.get_nodes( )

    def get_force(self):
        return self._force

    def get_force_x(self):
        return self._force[0::self._n_unknown]

    def get_force_y(self):
        return self._force[1::self._n_unknown]

    def get_force_z(self):
        return self._force[2::self._n_unknown]

    def set_force(self, force):
        self._force = force

    def get_re_velocity(self):
        return self._re_velocity

    def get_velocity(self):
        return self.get_u_geo( ).get_velocity( )

    def set_re_velocity(self, re_velocity):
        self._re_velocity = re_velocity

    def get_n_f_node(self):
        return self.get_f_nodes( ).shape[0]

    def get_n_u_node(self):
        return self.get_u_nodes( ).shape[0]

    def get_n_velocity(self):
        return self.get_velocity( ).size

    def get_n_force(self):
        return self.get_f_nodes( ).shape[0] * self._n_unknown

    def get_f_geo(self):
        return self._f_geo

    def get_u_geo(self):
        return self._u_geo

    def vec_scatter(self, vec_petsc, destroy=True):
        scatter, temp = PETSc.Scatter( ).toAll(vec_petsc)
        scatter.scatter(vec_petsc, temp, False, PETSc.Scatter.Mode.FORWARD)
        vec = temp.getArray( )
        if destroy:
            vec_petsc.destroy( )
        return vec

    def vtk(self, filename):
        if str(self) == '...':
            return

        comm = PETSc.COMM_WORLD.tompi4py( )
        rank = comm.Get_rank( )
        if rank == 0:
            force_x = self._force[0::self._n_unknown].copy( )
            force_y = self._force[1::self._n_unknown].copy( )
            force_z = self._force[2::self._n_unknown].copy( )
            velocity_x = self._re_velocity[0::3].copy( )
            velocity_y = self._re_velocity[1::3].copy( )
            velocity_z = self._re_velocity[2::3].copy( )
            velocity_err_x = np.abs(self._re_velocity[0::3] - self.get_velocity( )[0::3])
            velocity_err_y = np.abs(self._re_velocity[1::3] - self.get_velocity( )[1::3])
            velocity_err_z = np.abs(self._re_velocity[2::3] - self.get_velocity( )[2::3])

            f_filename = filename + '_' + str(self) + '_force'
            pointsToVTK(f_filename, self.get_f_nodes( )[:, 0], self.get_f_nodes( )[:, 1], self.get_f_nodes( )[:, 2],
                        data={"force": (force_x, force_y, force_z), })
            u_filename = filename + '_' + str(self) + '_velocity'
            pointsToVTK(u_filename, self.get_u_nodes( )[:, 0], self.get_u_nodes( )[:, 1], self.get_u_nodes( )[:, 2],
                        data={"velocity":     (velocity_x, velocity_y, velocity_z),
                              "velocity_err": (velocity_err_x, velocity_err_y, velocity_err_z), })

        return True

    def get_n_unknown(self):
        return self._n_unknown

    def show_velocity(self, length_factor=1, show_nodes=True):
        self.get_u_geo( ).show_velocity(length_factor, show_nodes)
        return True

    def show_f_nodes(self, linestyle='-'):
        self.get_f_geo( ).show_nodes(linestyle)
        return True

    def show_u_nodes(self, linestyle='-'):
        self.get_u_geo( ).show_nodes(linestyle)
        return True

    def show_f_u_nodes(self, linestyle='-'):
        geo_list = uniqueList( )
        geo_list.append(self.get_u_geo( ))
        geo_list.append(self.get_f_geo( ))
        temp_geo = geo( )
        temp_geo.combine(geo_list)
        temp_geo.show_nodes(linestyle)
        return True

    def unpickmyself(self):
        self.get_u_geo( ).set_dmda( )
        self.get_f_geo( ).set_dmda( )
        return True


class surf_forceObj(stokesFlowObj):
    def __init__(self, filename: str = '..'):
        super( ).__init__(filename)
        self._norm = np.zeros([0])  # information about normal vector at each point.

    def import_mat(self, filename: str):
        """
        Import geometries from Matlab file.

        :type filename: str
        :param filename: name of mat file containing object information
        """
        mat_contents = sio.loadmat(filename)
        velocity = mat_contents['U'].astype(np.float)
        origin = mat_contents['origin'].astype(np.float)
        f_nodes = mat_contents['f_nodes'].astype(np.float)
        u_nodes = mat_contents['u_nodes'].astype(np.float)
        norm = mat_contents['norm'].astype(np.float)
        para = {
            'norm':   norm,
            'origin': origin
        }
        self.set_data(f_nodes, u_nodes, velocity, **para)

    def import_nodes(self, filename: str):
        """
        Import geometries from Matlab file.

        :type filename: str
        :param filename: name of mat file containing object information
        """
        mat_contents = sio.loadmat(filename)
        origin = mat_contents['origin'].astype(np.float)
        f_nodes = mat_contents['f_nodes'].astype(np.float)
        u_nodes = mat_contents['u_nodes'].astype(np.float)
        norm = mat_contents['norm'].astype(np.float)
        para = {
            'norm':   norm,
            'origin': origin
        }
        self.set_nodes(f_nodes, u_nodes, **para)

    def set_nodes(self,
                  f_geo: geo,
                  u_geo: geo,
                  **kwargs):
        need_args = []
        for key in need_args:
            if not key in kwargs:
                err_msg = 'information about ' + key + \
                          ' is nesscery for surface force method. '
                raise ValueError(err_msg)

        args = {'origin': np.array([0, 0, 0])}
        for key, value in args.items( ):
            if not key in kwargs:
                kwargs[key] = args[key]

        super( ).set_nodes(f_geo, u_geo, **kwargs)
        self._norm = kwargs['norm']
        self._type = 'surface force obj'

    def get_norm(self):
        return self._norm


class pointSourceObj(stokesFlowObj):
    def __init__(self, filename: str = '..'):
        super( ).__init__(filename)
        self._n_unknown = 4

    def set_nodes(self,
                  f_geo: geo,
                  u_geo: geo,
                  **kwargs):
        super( ).set_nodes(f_geo, u_geo, **kwargs)
        self._type = 'point source obj'

    def vtk(self, filename):
        if str(self) == '...':
            return

        comm = PETSc.COMM_WORLD.tompi4py( )
        rank = comm.Get_rank( )
        if rank == 0:
            force_x = self._force[0::self._n_unknown].copy( )
            force_y = self._force[1::self._n_unknown].copy( )
            force_z = self._force[2::self._n_unknown].copy( )
            pointSource = self._force[3::self._n_unknown].copy( )
            velocity_x = self._re_velocity[0::3].copy( )
            velocity_y = self._re_velocity[1::3].copy( )
            velocity_z = self._re_velocity[2::3].copy( )
            velocity_err_x = np.abs(self._re_velocity[0::3] - self._velocity[0::3])
            velocity_err_y = np.abs(self._re_velocity[1::3] - self._velocity[1::3])
            velocity_err_z = np.abs(self._re_velocity[2::3] - self._velocity[2::3])

            f_filename = filename + '_' + str(self) + '_force'
            pointsToVTK(f_filename, self.get_f_nodes( )[:, 0], self.get_f_nodes( )[:, 1], self.get_f_nodes( )[:, 2],
                        data={"force":        (force_x, force_y, force_z),
                              "point_source": pointSource})
            u_filename = filename + '_' + str(self) + '_velocity'
            pointsToVTK(u_filename, self.get_u_nodes( )[:, 0], self.get_u_nodes( )[:, 1], self.get_u_nodes( )[:, 2],
                        data={"velocity":     (velocity_x, velocity_y, velocity_z),
                              "velocity_err": (velocity_err_x, velocity_err_y, velocity_err_z), })

            del force_x, force_y, force_z, \
                velocity_x, velocity_y, velocity_z, \
                velocity_err_x, velocity_err_y, velocity_err_z


class pointSourceProblem(stokesFlowProblem):
    def __init__(self, **kwargs):
        super( ).__init__(**kwargs)
        self._n_unknown = 4

    def vtk_force(self, filename):
        self.check_finish_solve( )
        comm = PETSc.COMM_WORLD.tompi4py( )
        rank = comm.Get_rank( )

        if rank == 0:
            force_x = self._force[0::self._n_unknown].copy( )
            force_y = self._force[1::self._n_unknown].copy( )
            force_z = self._force[2::self._n_unknown].copy( )
            pointSource = self._force[3::self._n_unknown].copy( )
            velocity_x = self._re_velocity[0::3].copy( )
            velocity_y = self._re_velocity[1::3].copy( )
            velocity_z = self._re_velocity[2::3].copy( )
            velocity_err_x = np.abs(self._re_velocity[0::3] - self._velocity[0::3])
            velocity_err_y = np.abs(self._re_velocity[1::3] - self._velocity[1::3])
            velocity_err_z = np.abs(self._re_velocity[2::3] - self._velocity[2::3])
            nodes = np.ones([self._f_node_index_list[-1], 3], order='F')
            for i, obj in enumerate(self._obj_list):
                nodes[self._f_node_index_list[i]:self._f_node_index_list[i + 1], :] = obj.get_f_nodes( )
            pointsToVTK(filename, nodes[:, 0], nodes[:, 1], nodes[:, 2],
                        data={"force":        (force_x, force_y, force_z),
                              "velocity":     (velocity_x, velocity_y, velocity_z),
                              "velocity_err": (velocity_err_x, velocity_err_y, velocity_err_z),
                              "point_source": pointSource})
            del force_x, force_y, force_z, \
                velocity_x, velocity_y, velocity_z, \
                velocity_err_x, velocity_err_y, velocity_err_z, \
                nodes, pointSource


class point_source_dipoleObj(stokesFlowObj):
    def __init__(self, filename: str = '..'):
        super( ).__init__(filename)
        self._n_unknown = 6
        self._pf_geo = geo( )
        self._pf_velocity = np.array(0)

    def set_nodes(self,
                  f_geo: geo,
                  u_geo: geo,
                  **kwargs):
        super( ).set_nodes(f_geo, u_geo, **kwargs)
        self._pf_geo = kwargs['pf_geo']
        self._type = 'dipole obj'

    def set_velocity(self,
                     velocity: np.array,
                     **kwargs):
        need_args = []
        for key in need_args:
            if not key in kwargs:
                err_msg = 'information about ' + key + \
                          ' is nesscery for surface force method. '
                raise ValueError(err_msg)

        args = {'pf_velocity': np.zeros(velocity.shape)}
        for key, value in args.items( ):
            if not key in kwargs:
                kwargs[key] = args[key]

        self._velocity = velocity.reshape(velocity.size)
        self._pf_velocity = kwargs['pf_velocity']

    def vtk(self, filename):
        if str(self) == '...':
            return

        comm = PETSc.COMM_WORLD.tompi4py( )
        rank = comm.Get_rank( )
        if rank == 0:
            num_unknown = self._n_unknown
            force_x = self._force[0::num_unknown].copy( )
            force_y = self._force[1::num_unknown].copy( )
            force_z = self._force[2::num_unknown].copy( )
            dipole_x = self._force[3::num_unknown].copy( )
            dipole_y = self._force[4::num_unknown].copy( )
            dipole_z = self._force[5::num_unknown].copy( )
            velocity_x = self._re_velocity[0::3].copy( )
            velocity_y = self._re_velocity[1::3].copy( )
            velocity_z = self._re_velocity[2::3].copy( )
            velocity_err_x = np.abs(self._re_velocity[0::3] - self.get_velocity( )[0::3])
            velocity_err_y = np.abs(self._re_velocity[1::3] - self.get_velocity( )[1::3])
            velocity_err_z = np.abs(self._re_velocity[2::3] - self.get_velocity( )[2::3])

            f_filename = filename + '_' + str(self) + '_force'
            pointsToVTK(f_filename, self.get_f_nodes( )[:, 0], self.get_f_nodes( )[:, 1], self.get_f_nodes( )[:, 2],
                        data={"force":  (force_x, force_y, force_z),
                              "dipole": (dipole_x, dipole_y, dipole_z), })
            u_filename = filename + '_' + str(self) + '_velocity'
            pointsToVTK(u_filename, self.get_u_nodes( )[:, 0], self.get_u_nodes( )[:, 1], self.get_u_nodes( )[:, 2],
                        data={"velocity":     (velocity_x, velocity_y, velocity_z),
                              "velocity_err": (velocity_err_x, velocity_err_y, velocity_err_z), })
        return True

    def get_pf_geo(self):
        return self._pf_geo

    def get_pf_velocity(self):
        return self._pf_velocity


class point_source_dipoleProblem(stokesFlowProblem):
    def __init__(self, **kwargs):
        super( ).__init__(**kwargs)
        self._n_unknown = 6
        self._ini_problem = []

    def vtk_force(self, filename):
        self.check_finish_solve( )
        comm = PETSc.COMM_WORLD.tompi4py( )
        rank = comm.Get_rank( )

        if rank == 0:
            num_unknown = self._n_unknown
            force_x = self._force[0::num_unknown].copy( )
            force_y = self._force[1::num_unknown].copy( )
            force_z = self._force[2::num_unknown].copy( )
            dipole_x = self._force[3::num_unknown].copy( )
            dipole_y = self._force[4::num_unknown].copy( )
            dipole_z = self._force[5::num_unknown].copy( )
            velocity_x = self._re_velocity[0::3].copy( )
            velocity_y = self._re_velocity[1::3].copy( )
            velocity_z = self._re_velocity[2::3].copy( )
            velocity_err_x = np.abs(self._re_velocity[0::3] - self._velocity[0::3])
            velocity_err_y = np.abs(self._re_velocity[1::3] - self._velocity[1::3])
            velocity_err_z = np.abs(self._re_velocity[2::3] - self._velocity[2::3])
            nodes = np.ones([self._f_node_index_list[-1], 3], order='F')
            for i, obj in enumerate(self._obj_list):
                nodes[self._f_node_index_list[i]:self._f_node_index_list[i + 1], :] = obj.get_f_nodes( )
            pointsToVTK(filename, nodes[:, 0], nodes[:, 1], nodes[:, 2],
                        data={"force":        (force_x, force_y, force_z),
                              "velocity":     (velocity_x, velocity_y, velocity_z),
                              "velocity_err": (velocity_err_x, velocity_err_y, velocity_err_z),
                              "dipole":       (dipole_x, dipole_y, dipole_z), })

    def ini_guess(self):
        comm = PETSc.COMM_WORLD.tompi4py( )
        rank = comm.Get_rank( )
        if rank == 1:
            print('%s: get ini guess for ps_ds method.' % str(self))

        ini_problem = self.ini_problem( )
        ini_problem.create_matrix( )
        residualNorm = ini_problem.solve( )
        pf_force = ini_problem.get_force( )
        temp0 = np.array(pf_force).reshape((-1, 3))
        ini_guess = np.hstack((temp0, np.zeros(temp0.shape))).flatten( )

        return ini_guess, residualNorm, ini_problem

    def ini_problem(self):
        comm = PETSc.COMM_WORLD.tompi4py( )
        rank = comm.Get_rank( )
        if rank == 1:
            print('%s: generate ini guess problem.' % str(self))

        problem_arguments = self._kwargs.copy( )
        problem_arguments['matrix_method'] = 'pf'
        problem_arguments['name'] = 'ini_' + problem_arguments['name']
        problem_arguments['fileHeadle'] += '_ini'
        ini_problem = stokesFlowProblem(**problem_arguments)
        for obj0 in self.get_obj_list( ):
            obj1 = stokesFlowObj( )
            para = {'name': obj0.get_name( )}
            obj1.set_data(obj0.get_f_geo( ), obj0.get_pf_geo( ), obj0.get_pf_velocity( ), **para)
            ini_problem.add_obj(obj1)

        self._ini_problem = ini_problem
        return ini_problem

    # def solve(self, ini_guess=None, Tolerances={}):
    #     if ini_guess is None:
    #         ini_guess = np.zeros(self.get_n_f_node() * self.get_n_unknown())
    #
    #     return super().solve(ini_guess=ini_guess, Tolerances=Tolerances)

    def get_ini_problem(self):
        return self._ini_problem


class point_force_dipoleObj(stokesFlowObj):
    def __init__(self, filename: str = '..'):
        super( ).__init__(filename)
        self._n_unknown = 12

    def set_nodes(self,
                  f_geo: geo,
                  u_geo: geo,
                  **kwargs):
        super( ).set_nodes(f_geo, u_geo, **kwargs)
        self._type = 'dipole obj'

    def vtk(self, filename):
        if str(self) == '...':
            return

        comm = PETSc.COMM_WORLD.tompi4py( )
        rank = comm.Get_rank( )
        if rank == 0:
            num_unknown = self._n_unknown
            force_x = self._force[0::num_unknown].copy( )
            force_y = self._force[1::num_unknown].copy( )
            force_z = self._force[2::num_unknown].copy( )
            dipole_1 = self._force[3::num_unknown].copy( )
            dipole_2 = self._force[4::num_unknown].copy( )
            dipole_3 = self._force[5::num_unknown].copy( )
            dipole_4 = self._force[6::num_unknown].copy( )
            dipole_5 = self._force[7::num_unknown].copy( )
            dipole_6 = self._force[8::num_unknown].copy( )
            dipole_7 = self._force[9::num_unknown].copy( )
            dipole_8 = self._force[10::num_unknown].copy( )
            dipole_9 = self._force[11::num_unknown].copy( )
            velocity_x = self._re_velocity[0::3].copy( )
            velocity_y = self._re_velocity[1::3].copy( )
            velocity_z = self._re_velocity[2::3].copy( )
            velocity_err_x = np.abs(self._re_velocity[0::3] - self.get_velocity( )[0::3])
            velocity_err_y = np.abs(self._re_velocity[1::3] - self.get_velocity( )[1::3])
            velocity_err_z = np.abs(self._re_velocity[2::3] - self.get_velocity( )[2::3])

            f_filename = filename + '_' + str(self) + '_force'
            pointsToVTK(f_filename, self.get_f_nodes( )[:, 0], self.get_f_nodes( )[:, 1], self.get_f_nodes( )[:, 2],
                        data={"force":   (force_x, force_y, force_z),
                              "dipole1": (dipole_1, dipole_2, dipole_3),
                              "dipole2": (dipole_4, dipole_5, dipole_6),
                              "dipole3": (dipole_7, dipole_8, dipole_9)})
            u_filename = filename + '_' + str(self) + '_velocity'
            pointsToVTK(u_filename, self.get_u_nodes( )[:, 0], self.get_u_nodes( )[:, 1], self.get_u_nodes( )[:, 2],
                        data={"velocity":     (velocity_x, velocity_y, velocity_z),
                              "velocity_err": (velocity_err_x, velocity_err_y, velocity_err_z), })
        return True


class point_force_dipoleProblem(stokesFlowProblem):
    def __init__(self, **kwargs):
        super( ).__init__(**kwargs)
        self._n_unknown = 12

    def vtk_force(self, filename):
        self.check_finish_solve( )
        comm = PETSc.COMM_WORLD.tompi4py( )
        rank = comm.Get_rank( )

        if rank == 0:
            num_unknown = self._n_unknown
            force_x = self._force[0::num_unknown].copy( )
            force_y = self._force[1::num_unknown].copy( )
            force_z = self._force[2::num_unknown].copy( )
            dipole_1 = self._force[3::num_unknown].copy( )
            dipole_2 = self._force[4::num_unknown].copy( )
            dipole_3 = self._force[5::num_unknown].copy( )
            dipole_4 = self._force[6::num_unknown].copy( )
            dipole_5 = self._force[7::num_unknown].copy( )
            dipole_6 = self._force[8::num_unknown].copy( )
            dipole_7 = self._force[9::num_unknown].copy( )
            dipole_8 = self._force[10::num_unknown].copy( )
            dipole_9 = self._force[11::num_unknown].copy( )
            velocity_x = self._re_velocity[0::3].copy( )
            velocity_y = self._re_velocity[1::3].copy( )
            velocity_z = self._re_velocity[2::3].copy( )
            velocity_err_x = np.abs(self._re_velocity[0::3] - self._velocity[0::3])
            velocity_err_y = np.abs(self._re_velocity[1::3] - self._velocity[1::3])
            velocity_err_z = np.abs(self._re_velocity[2::3] - self._velocity[2::3])
            nodes = np.ones([self._f_node_index_list[-1], 3], order='F')
            for i, obj in enumerate(self._obj_list):
                nodes[self._f_node_index_list[i]:self._f_node_index_list[i + 1], :] = obj.get_f_nodes( )
            pointsToVTK(filename, nodes[:, 0], nodes[:, 1], nodes[:, 2],
                        data={"force":        (force_x, force_y, force_z),
                              "velocity":     (velocity_x, velocity_y, velocity_z),
                              "velocity_err": (velocity_err_x, velocity_err_y, velocity_err_z),
                              "dipole":       (
                                  dipole_1, dipole_2, dipole_3, dipole_4, dipole_5, dipole_6, dipole_7, dipole_8,
                                  dipole_9), })


class preconditionObj(stokesFlowObj):
    def __init__(self, filename: str = '..'):
        super( ).__init__(filename)
        self._inv_petsc = PETSc.Mat( ).create(comm=PETSc.COMM_WORLD)

    def create_inv_diag_matrix(self):
        t0 = time( )
        comm = PETSc.COMM_WORLD.tompi4py( )
        rank = comm.Get_rank( )
        problem = self.get_problem( )
        # assert isinstance(problem, stokesFlowProblem)
        matrix_method = problem.get_matrix_method( )
        kwargs = problem.get_kwargs( )
        temp_m_petsc = matrix_method(self, self, **kwargs)

        I_petsc = PETSc.Mat( ).create(comm=PETSc.COMM_WORLD)
        temp_m_petsc.convert(mat_type='dense', out=I_petsc)
        I_petsc.zeroEntries( )
        I_petsc.shift(alpha=1)
        temp_m_petsc.convert(mat_type='dense', out=self._inv_petsc)
        rperm, cperm = temp_m_petsc.getOrdering(ord_type='MATORDERINGNATURAL')
        temp_m_petsc.factorLU(rperm, cperm)
        # temp_m_petsc.factorCholesky(rperm)
        temp_m_petsc.matSolve(I_petsc, self._inv_petsc)
        temp_m_petsc.destroy( )
        I_petsc.destroy( )

        t1 = time( )
        PETSc.Sys.Print('%s: solve inverse matrix: %fs.' % (str(self), (t1 - t0)))
        return True

    def get_inv_petsc(self):
        return self._inv_petsc

    def set_problem(self, problem: 'stokesFlowProblem'):
        super( ).set_problem(problem)

        if not self.get_inv_petsc( ).isAssembled( ):
            self.create_inv_diag_matrix( )
        return True


class preconditionProblem(stokesFlowProblem):
    def __init__(self, **kwargs):
        super( ).__init__(**kwargs)
        self._inv_diag_petsc = PETSc.Mat( ).create(comm=PETSc.COMM_WORLD)  # diagonal elements of M matrix

    def create_inv_diag_matrix(self):
        """
        calculate self-to-self elements for the M matrix.
        :return:
        """
        if not self._inv_diag_petsc.isAssembled( ):
            self._inv_diag_petsc.setSizes(((None, self._u_index_list[-1]), (None, self._f_index_list[-1])))
            self._inv_diag_petsc.setType('dense')
            self._inv_diag_petsc.setFromOptions( )
            self._inv_diag_petsc.setUp( )

        for i, obj1 in enumerate(self._obj_list):
            inv_petsc = obj1.get_inv_petsc( )
            err_msg = 'create inv matrix of %s first. ' % str(obj1)
            assert inv_petsc.isAssembled( ), err_msg

            velocity_index_begin = self._f_index_list[i]
            force_index_begin = self._f_index_list[i]
            force_index_end = self._f_index_list[i + 1]
            temp_m_start, temp_m_end = inv_petsc.getOwnershipRange( )

            temp_inv_petsc = inv_petsc.getDenseArray( )
            for k in range(temp_m_start, temp_m_end):
                self._inv_diag_petsc.setValues(velocity_index_begin + k,
                                               np.arange(force_index_begin, force_index_end, dtype='int32'),
                                               temp_inv_petsc[k - temp_m_start, :])
        self._inv_diag_petsc.assemble( )

        return True

    def create_matrix(self):
        self.create_inv_diag_matrix( )
        super( ).create_matrix( )
        return True

    def solve(self, ini_guess=None, Tolerances={}):
        err_msg = 'create the inverse of main-obj diagonal matrix first. '
        assert self._inv_diag_petsc.isAssembled( ), err_msg
        assert 1 == 2, 'DO NOT WORK, self._velocity property is changing. '

        old_M_petsc = PETSc.Mat( ).create(comm=PETSc.COMM_WORLD)
        self._M_petsc.convert(mat_type='dense', out=old_M_petsc)
        self._inv_diag_petsc.matMult(old_M_petsc, self._M_petsc)

        old_velocity = self._velocity.copy( )
        inv_diag = self._inv_diag_petsc.getDenseArray( )
        self._velocity = inv_diag.dot(old_velocity)

        super( ).solve(ini_guess=ini_guess, Tolerances=Tolerances)
        self._M_petsc = old_M_petsc
        self._velocity = old_velocity

        return True


class stokesletsProblem(stokesFlowProblem):
    def __init__(self, **kwargs):
        super( ).__init__(**kwargs)
        self._stokeslets_post = np.array(kwargs['stokeslets_post']).reshape(1, 3)
        stokeslets_f = kwargs['stokeslets_f']
        self.set_stokeslets_f(stokeslets_f)

    def get_stokeslets_post(self):
        return self._stokeslets_post

    def set_stokeslets_post(self, stokeslets_post):
        errmsg = 'stokeslets_post (stokeslets postion) is a np.ndarray obj in shape (1, 3)'
        assert isinstance(stokeslets_post, np.ndarray), errmsg
        assert stokeslets_post.shape == (1, 3), errmsg
        self._stokeslets_post = stokeslets_post
        return True

    def get_stokeslets_f(self):
        return self._stokeslets_f

    def set_stokeslets_f(self, stokeslets_f):
        errmsg = 'stokeslets_f (stokeslets force) is a np.ndarray obj in shape (3, )'
        assert isinstance(stokeslets_f, np.ndarray), errmsg
        assert stokeslets_f.shape == (3,), errmsg
        self._stokeslets_f = stokeslets_f
        self._stokeslets_f_petsc = PETSc.Vec( ).create(comm=PETSc.COMM_WORLD)
        self._stokeslets_f_petsc.setSizes(stokeslets_f.size)
        self._stokeslets_f_petsc.setFromOptions( )
        self._stokeslets_f_petsc.setUp( )
        self._stokeslets_f_petsc[:] = stokeslets_f[:]
        self._stokeslets_f_petsc.assemble( )
        return True

    def unpickmyself(self):
        super( ).unpickmyself( )
        stokeslets_f = self.get_stokeslets_f( )
        self.set_stokeslets_f(stokeslets_f)
        return True

    def vtk_velocity(self, filename: str):
        self.check_finish_solve( )
        field_range, n_grid = self.check_vtk_velocity( )
        region_type = self._kwargs['region_type']

        if not self._M_destroyed:
            self._M_petsc.destroy( )
            self._M_destroyed = True
        myregion = region( )
        full_region_x, full_region_y, full_region_z = myregion.type[region_type](field_range, n_grid)

        # solve velocity at cell center.
        n_para = 3 * n_grid[1] * n_grid[2]
        comm = PETSc.COMM_WORLD.tompi4py( )
        rank = comm.Get_rank( )

        # to handle big problem, solve velocity field at every splice along x axis.
        # u_total = u1 + u2. satisfying boundary conditions.
        # u1: cancelled background flow at boundary.
        # u2: background flow due to stokeslets.
        if rank == 0:
            u_x = np.zeros((n_grid[0], n_grid[1], n_grid[2]))
            u_y = np.zeros((n_grid[0], n_grid[1], n_grid[2]))
            u_z = np.zeros((n_grid[0], n_grid[1], n_grid[2]))
            u1_x = np.zeros((n_grid[0], n_grid[1], n_grid[2]))
            u1_y = np.zeros((n_grid[0], n_grid[1], n_grid[2]))
            u1_z = np.zeros((n_grid[0], n_grid[1], n_grid[2]))
            u2_x = np.zeros((n_grid[0], n_grid[1], n_grid[2]))
            u2_y = np.zeros((n_grid[0], n_grid[1], n_grid[2]))
            u2_z = np.zeros((n_grid[0], n_grid[1], n_grid[2]))
        else:
            u_x = None
            u_y = None
            u_z = None
        m1_petsc = PETSc.Mat( ).create(comm=PETSc.COMM_WORLD)
        m1_petsc.setSizes(((None, n_para), (None, self._f_index_list[-1])))
        m1_petsc.setType('dense')
        m1_petsc.setFromOptions( )
        m1_petsc.setUp( )
        obj1 = stokesFlowObj( )
        geo_stokeslets = geo( )
        geo_stokeslets.set_nodes(self.get_stokeslets_post( ))
        obj_stokeslets = stokesFlowObj( )
        obj_stokeslets.set_data(geo_stokeslets, geo_stokeslets, np.zeros(self.get_stokeslets_post( ).size))
        for i0 in range(full_region_x.shape[0]):
            # u1: cancelled background flow to satisfy boundary conditions.
            temp_x = full_region_x[i0]
            temp_y = full_region_y[i0]
            temp_z = full_region_z[i0]
            temp_nodes = np.c_[temp_x.ravel( ), temp_y.ravel( ), temp_z.ravel( )]
            temp_geo = geo( )
            temp_geo.set_nodes(temp_nodes)
            obj1.set_data(temp_geo, temp_geo, np.zeros(temp_nodes.size))
            self.create_matrix_obj(obj1, m1_petsc)
            m1_petsc.assemble( )
            u1_petsc = m1_petsc.createVecLeft( )
            # u1_petsc.set(0)
            m1_petsc.mult(self._force_petsc, u1_petsc)
            u1 = self.vec_scatter(u1_petsc)
            # u2: add background flow due to stokeslets.
            from src.StokesFlowMethod import stokeslets_matrix_3d_petsc
            m2_petsc = stokeslets_matrix_3d_petsc(obj1, obj_stokeslets)
            u2_petsc = m2_petsc.createVecLeft( )
            # u2_petsc.set(0)
            m2_petsc.mult(self._stokeslets_f_petsc, u2_petsc)
            u2 = self.vec_scatter(u2_petsc)
            m2_petsc.destroy( )
            if rank == 0:
                u1_x[i0, :, :] = u1[0::3].reshape((n_grid[1], n_grid[2])).copy( )
                u1_y[i0, :, :] = u1[1::3].reshape((n_grid[1], n_grid[2])).copy( )
                u1_z[i0, :, :] = u1[2::3].reshape((n_grid[1], n_grid[2])).copy( )
                u2_x[i0, :, :] = u2[0::3].reshape((n_grid[1], n_grid[2])).copy( )
                u2_y[i0, :, :] = u2[1::3].reshape((n_grid[1], n_grid[2])).copy( )
                u2_z[i0, :, :] = u2[2::3].reshape((n_grid[1], n_grid[2])).copy( )
                u_x[i0, :, :] = u1_x[i0, :, :] + u2_x[i0, :, :]
                u_y[i0, :, :] = u1_y[i0, :, :] + u2_y[i0, :, :]
                u_z[i0, :, :] = u1_z[i0, :, :] + u2_z[i0, :, :]
            else:
                u_x = None
                u_y = None
                u_z = None

        m1_petsc.destroy( )
        if rank == 0:
            # output data
            gridToVTK(filename, full_region_x, full_region_y, full_region_z,
                      pointData={"u1": (u1_x, u1_y, u1_z),
                                 "u2": (u2_x, u2_y, u2_z),
                                 "u":  (u_x, u_y, u_z)})
        return True

    def vtk_tetra(self,
                  filename: str,
                  bgeo: geo):
        comm = PETSc.COMM_WORLD.tompi4py( )
        rank = comm.Get_rank( )
        self.check_finish_solve( )

        bnodes = bgeo.get_nodes( )
        belems, elemtype = bgeo.get_mesh( )
        err_msg = 'mesh type is NOT tetrahedron. '
        assert elemtype == 'tetra', err_msg

        PETSc.Sys.Print('export to %s.vtk, element type is %s, contain %d nodes and %d elements. '
                        % (filename, elemtype, bnodes.shape[0], belems.shape[0]))

        n_para = bgeo.get_n_nodes( ) * 3
        # u1: cancelled background flow to satisfy boundary conditions.
        m1_petsc = PETSc.Mat( ).create(comm=PETSc.COMM_WORLD)
        m1_petsc.setSizes(((None, n_para), (None, self._f_index_list[-1])))
        m1_petsc.setType('dense')
        m1_petsc.setFromOptions( )
        m1_petsc.setUp( )
        obj1 = stokesFlowObj( )
        obj1.set_data(bgeo, bgeo, np.zeros(bnodes.size))
        self.create_matrix_obj(obj1, m1_petsc)
        m1_petsc.assemble( )
        u1_petsc = m1_petsc.createVecLeft( )
        # u1_petsc.set(0)
        m1_petsc.mult(self._force_petsc, u1_petsc)
        u1 = self.vec_scatter(u1_petsc)
        # u2: add background flow due to stokeslets.
        from src.StokesFlowMethod import stokeslets_matrix_3d_petsc
        geo_stokeslets = geo( )
        geo_stokeslets.set_nodes(self.get_stokeslets_post( ))
        obj_stokeslets = stokesFlowObj( )
        obj_stokeslets.set_data(geo_stokeslets, geo_stokeslets, np.zeros(self.get_stokeslets_post( ).size))
        m2_petsc = stokeslets_matrix_3d_petsc(obj1, obj_stokeslets)
        u2_petsc = m2_petsc.createVecLeft( )
        # u2_petsc.set(0)
        m2_petsc.mult(self._stokeslets_f_petsc, u2_petsc)
        u2 = self.vec_scatter(u2_petsc)
        m1_petsc.destroy( )
        m2_petsc.destroy( )

        if rank == 0:
            u1 = np.array(u1).reshape(bnodes.shape)
            u2 = np.array(u2).reshape(bnodes.shape)
            u = u1 + u2
            vtk = VtkData(
                    UnstructuredGrid(bnodes,
                                     tetra=belems,
                                     ),
                    PointData(Vectors(u, name='u'),
                              Vectors(u1, name='u1'),
                              Vectors(u2, name='u2')),
                    str(self)
            )
            vtk.tofile(filename)
        return True


class stokesletsObj(stokesFlowObj):
    def __init__(self):
        super( ).__init__( )
        self._stokeslets_post = np.array((0, 0, 0)).reshape(1, 3)
        self._stokeslets_f = np.array((0, 0, 0)).reshape(1, 3)
        self._stokeslets_f_petsc = PETSc.Vec( ).create(comm=PETSc.COMM_WORLD)

    def set_data(self,
                 f_geo: geo,
                 u_geo: geo,
                 name='...', **kwargs):
        super( ).set_data(f_geo, u_geo, name, **kwargs)
        self._stokeslets_post = np.array(kwargs['stokeslets_post']).reshape(1, 3)
        stokeslets_f = kwargs['stokeslets_f']
        self.set_stokeslets_f(stokeslets_f)

    def get_stokeslets_post(self):
        return self._stokeslets_post

    def set_stokeslets_post(self, stokeslets_post):
        errmsg = 'stokeslets_post (stokeslets postion) is a np.ndarray obj in shape (1, 3)'
        assert isinstance(stokeslets_post, np.ndarray), errmsg
        assert stokeslets_post.shape == (1, 3), errmsg
        self._stokeslets_post = stokeslets_post
        return True

    def get_stokeslets_f(self):
        return self._stokeslets_f

    def set_stokeslets_f(self, stokeslets_f):
        errmsg = 'stokeslets_f (stokeslets force) is a np.ndarray obj in shape (3, )'
        assert isinstance(stokeslets_f, np.ndarray), errmsg
        assert stokeslets_f.shape == (3,), errmsg
        self._stokeslets_f = stokeslets_f
        self._stokeslets_f_petsc = PETSc.Vec( ).create(comm=PETSc.COMM_WORLD)
        self._stokeslets_f_petsc.setSizes(stokeslets_f.size)
        self._stokeslets_f_petsc.setFromOptions( )
        self._stokeslets_f_petsc.setUp( )
        self._stokeslets_f_petsc[:] = stokeslets_f[:]
        self._stokeslets_f_petsc.assemble( )
        return True

    def unpickmyself(self):
        super( ).unpickmyself( )
        stokeslets_f = self.get_stokeslets_f( )
        self.set_stokeslets_f(stokeslets_f)
        return True

    def vtk(self, filename):
        if str(self) == '...':
            return
        comm = PETSc.COMM_WORLD.tompi4py( )
        rank = comm.Get_rank( )

        # u2: add background flow due to stokeslets.
        from src.StokesFlowMethod import stokeslets_matrix_3d_petsc
        geo_stokeslets = geo( )
        geo_stokeslets.set_nodes(self.get_stokeslets_post( ))
        obj_stokeslets = stokesFlowObj( )
        obj_stokeslets.set_data(geo_stokeslets, geo_stokeslets, np.zeros(self.get_stokeslets_post( ).size))
        m2_petsc = stokeslets_matrix_3d_petsc(self, obj_stokeslets)
        u2_petsc = m2_petsc.createVecLeft( )
        # u2_petsc.set(0)
        m2_petsc.mult(self._stokeslets_f_petsc, u2_petsc)
        u2 = self.vec_scatter(u2_petsc)

        if rank == 0:
            force_x = self._force[0::self._n_unknown].copy( )
            force_y = self._force[1::self._n_unknown].copy( )
            force_z = self._force[2::self._n_unknown].copy( )
            # u1: cancelled background flow to satisfy boundary conditions.
            re_velocity_x = self._re_velocity[0::3].copy( )
            re_velocity_y = self._re_velocity[1::3].copy( )
            re_velocity_z = self._re_velocity[2::3].copy( )
            velocity_err_x = np.abs(self.get_velocity( )[0::3].copy( ) - re_velocity_x)
            velocity_err_y = np.abs(self.get_velocity( )[1::3].copy( ) - re_velocity_y)
            velocity_err_z = np.abs(self.get_velocity( )[2::3].copy( ) - re_velocity_z)
            velocity_x = self.get_velocity( )[0::3].copy( )
            velocity_y = self.get_velocity( )[1::3].copy( )
            velocity_z = self.get_velocity( )[2::3].copy( )
            # u2: add background flow due to stokeslets.
            stokeslets_x = u2[0::3].copy( )
            stokeslets_y = u2[1::3].copy( )
            stokeslets_z = u2[2::3].copy( )
            # u = u1 + u2
            u_x = velocity_x + stokeslets_x
            u_y = velocity_y + stokeslets_y
            u_z = velocity_z + stokeslets_z
            re_u_x = re_velocity_x + stokeslets_x
            re_u_y = re_velocity_y + stokeslets_y
            re_u_z = re_velocity_z + stokeslets_z

            f_filename = filename + '_' + str(self) + '_force'
            pointsToVTK(f_filename, self.get_f_nodes( )[:, 0], self.get_f_nodes( )[:, 1], self.get_f_nodes( )[:, 2],
                        data={"force": (force_x, force_y, force_z), })
            u_filename = filename + '_' + str(self) + '_velocity'
            pointsToVTK(u_filename, self.get_u_nodes( )[:, 0], self.get_u_nodes( )[:, 1], self.get_u_nodes( )[:, 2],
                        data={"re_u":  (re_u_x, re_u_y, re_u_z),
                              "re_u1": (re_velocity_x, re_velocity_y, re_velocity_z),
                              "re_u2": (stokeslets_x, stokeslets_y, stokeslets_z),
                              "u":     (u_x, u_y, u_z),
                              "u1":    (velocity_x, velocity_y, velocity_z),
                              "u2":    (stokeslets_x, stokeslets_y, stokeslets_z),
                              "u_err": (velocity_err_x, velocity_err_y, velocity_err_z), })

        return True


class stokesletsPreconditionProblem(stokesletsProblem, preconditionProblem):
    def __init__(self, **kwargs):
        super( ).__init__(**kwargs)
        # for precondition problem
        self._inv_diag_petsc = PETSc.Mat( ).create(comm=PETSc.COMM_WORLD)  # diagonal elements of M matrix
        # for stokeslets problem
        self._stokeslets_post = np.array(kwargs['stokeslets_post']).reshape(1, 3)
        stokeslets_f = kwargs['stokeslets_f']
        self.set_stokeslets_f(stokeslets_f)


class stokesletsPreconditionObj(stokesletsObj, preconditionObj):
    def nothing(self):
        pass


class stokesletsInPipeProblem(stokesFlowProblem):
    # pipe center line along z asix
    def __init__(self, **kwargs):
        super( ).__init__(**kwargs)
        self._fpgeo = geo( )  # force geo of pipe
        self._vpgeo = geo( )  # velocity geo of pipe
        self._cpgeo = geo( )  # check geo of pipe
        self._m_pipe = PETSc.Mat( ).create(comm=PETSc.COMM_WORLD)
        self._m_pipe_check = PETSc.Mat( ).create(comm=PETSc.COMM_WORLD)

        # create a empty matrix, and a empty velocity vecters, to avoid use too much time to allocate memory.
        self._t_m = PETSc.Mat( ).create(
                comm=PETSc.COMM_WORLD)  # M matrix associated with u1 part ,velocity due to pipe boundary.
        self._t_u11 = []  # a list contain three u1 component, for interpolation
        self._t_u12 = []
        self._t_u13 = []
        self._set_f123( )
        self._stokeslet_m = PETSc.Mat( ).create(comm=PETSc.COMM_WORLD)  # M matrix associated with stokeslet singularity
        self._t_u2 = PETSc.Vec( ).create(comm=PETSc.COMM_WORLD)

        self._f1_list = []  # list of forces lists for each object at or outside pipe associated with force-nodes at x axis
        self._f2_list = []  # list of forces lists for each object at or outside pipe associated with force-nodes at y axis
        self._f3_list = []  # list of forces lists for each object at or outside pipe associated with force-nodes at z axis
        self._residualNorm_list = []  # residualNorm of f1, f2, and f3 of different b
        self._err_list = []  # relative velocity error solved using check geo.

    def _set_f123(self):
        # set point source vector f1, f2, f3.
        fgeo = geo( )
        fgeo.set_nodes((0, 0, 0), deltalength=0)
        t_f_pkg = PETSc.DMComposite( ).create(comm=PETSc.COMM_WORLD)
        t_f_pkg.addDM(fgeo.get_dmda( ))
        t_f_pkg.setFromOptions( )
        t_f_pkg.setUp( )
        f_isglb = t_f_pkg.getGlobalISs( )
        fgeo.set_glbIdx(f_isglb[0].getIndices( ))
        f1_petsc = t_f_pkg.createGlobalVector( )
        f2_petsc = t_f_pkg.createGlobalVector( )
        f3_petsc = t_f_pkg.createGlobalVector( )
        f1_petsc[:] = (1, 0, 0)
        f2_petsc[:] = (0, 1, 0)
        f3_petsc[:] = (0, 0, 1)
        f1_petsc.assemble( )
        f2_petsc.assemble( )
        f3_petsc.assemble( )
        t_f_pkg.destroy( )

        self._f123_petsc = (f1_petsc, f2_petsc, f3_petsc)
        self._stokeslet_geo = fgeo
        return True

    def add_obj(self, obj):
        _, b, _ = obj.get_f_geo( ).get_polar_coord( )
        b_list = self.get_b_list( )
        b0 = np.min(b_list)
        b1 = np.max(b_list)
        err_msg = 'b is out of range (%f, %f)' % (b0, b1)
        assert all(b >= b0) and all(b <= b1), err_msg

        return super( ).add_obj(obj)

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

    def debug_solve_stokeslets_b(self, b, node):
        t_geo = geo( )
        t_geo.set_nodes(node, deltalength=0)
        phi, rho, z = t_geo.get_polar_coord( )
        node_rpz = np.vstack((rho, phi, z)).T
        return self._solve_stokeslets_b(b, node, node_rpz)

    def debug_solve_u_pipe(self, pgeo, outputHandle, greenFun):
        return self._solve_u_pipe(pgeo, outputHandle, greenFun)

    def debug_solve_stokeslets_fnode(self, fnode, unodes):
        self._solve_stokeslets_fnode(fnode, unodes)
        return True

    def _solve_u1_b_list(self, k, ugeo, use_cart=False):
        # solve velocity component due to boundary as a function of b and u_node location. Here, b is in self._b_list.
        # total u = u1 + u2, u1: force at (or outside) pipe boundary
        kwargs = self.get_kwargs( )
        temp_m = self._t_m
        temp_obj1 = stokesFlowObj( )
        temp_obj1.set_data(self._fpgeo, ugeo)
        self._method_dict[kwargs['matrix_method']](temp_obj1, temp_obj1, temp_m, **kwargs)
        temp_m.assemble( )
        temp_m = self._t_m

        for i0, ID in enumerate(k):
            f1 = self._f1_list[ID]
            f2 = self._f2_list[ID]
            f3 = self._f3_list[ID]
            temp_m.mult(f1, self._t_u11[i0])
            temp_m.mult(f2, self._t_u12[i0])
            temp_m.mult(f3, self._t_u13[i0])

            if use_cart:
                pass
            else:
                uphi, _, _ = ugeo.get_polar_coord( )
                # Transform to polar coord
                ux1 = self._t_u11[i0][0::3].copy( )
                uy1 = self._t_u11[i0][1::3].copy( )
                uz1 = self._t_u11[i0][2::3].copy( )
                uR1 = np.cos(uphi) * ux1 + np.sin(uphi) * uy1
                uPhi1 = -np.sin(uphi) * ux1 + np.cos(uphi) * uy1
                self._t_u11[i0][:] = np.dstack((uR1, uPhi1, uz1)).flatten( )
                ux2 = self._t_u12[i0][0::3].copy( )
                uy2 = self._t_u12[i0][1::3].copy( )
                uz2 = self._t_u12[i0][2::3].copy( )
                uR2 = np.cos(uphi) * ux2 + np.sin(uphi) * uy2
                uPhi2 = -np.sin(uphi) * ux2 + np.cos(uphi) * uy2
                self._t_u12[i0][:] = np.dstack((uR2, uPhi2, uz2)).flatten( )
                ux3 = self._t_u13[i0][0::3].copy( )
                uy3 = self._t_u13[i0][1::3].copy( )
                uz3 = self._t_u13[i0][2::3].copy( )
                uR3 = np.cos(uphi) * ux3 + np.sin(uphi) * uy3
                uPhi3 = -np.sin(uphi) * ux3 + np.cos(uphi) * uy3
                self._t_u13[i0][:] = np.dstack((uR3, uPhi3, uz3)).flatten( )
            self._t_u11[i0].assemble( )
            self._t_u12[i0].assemble( )
            self._t_u13[i0].assemble( )
        return True

    def _solve_stokeslets_b(self, b, unode_xyz, unode_rpz, use_cart=False, u_glbIdx=[]):
        from src.stokesletsInPipe import detail
        from src.StokesFlowMethod import point_force_matrix_3d_petsc
        unode = unode_xyz
        # INDEX = np.abs(unode_xyz[:, 2]) > self._lp / 4
        # unode_num = unode_xyz[~INDEX]  # numerical part
        # unode_the = unode_rpz[INDEX]  # theoretical result part
        #
        # # theoretical part
        # greenFun = detail(threshold=self._th, b=b)
        # greenFun.solve_prepare()
        # u1_list = []
        # u2_list = []
        # u3_list = []
        # for R, phi, z in unode_the:
        #     t_u1 = greenFun.solve_u1(R, phi, z)
        #     u1_list.append(t_u1)
        #     t_u2 = greenFun.solve_u2(R, phi, z)
        #     u2_list.append(t_u2)
        #     t_u3 = greenFun.solve_u3(R, phi, z)
        #     u3_list.append(t_u3)
        # u1 = np.vstack(u1_list)
        # u2 = np.vstack(u2_list)
        # u3 = np.vstack(u3_list)
        # if use_cart:
        #     phi = unode_the[:,1]
        #     t_ux1 = np.cos(phi) * u1[:, 0] - np.sin(phi) * u1[:, 1]
        #     t_ux2 = np.cos(phi) * u2[:, 0] - np.sin(phi) * u2[:, 1]
        #     t_ux3 = np.cos(phi) * u3[:, 0] - np.sin(phi) * u3[:, 1]
        #     t_uy1 = np.sin(phi) * u1[:, 0] + np.cos(phi) * u1[:, 1]
        #     t_uy2 = np.sin(phi) * u2[:, 0] + np.cos(phi) * u2[:, 1]
        #     t_uy3 = np.sin(phi) * u3[:, 0] + np.cos(phi) * u3[:, 1]
        #     u1 = np.dstack((t_ux1, t_uy1, u1[:, 2])).flatten()
        #     u2 = np.dstack((t_ux2, t_uy2, u2[:, 2])).flatten()
        #     u3 = np.dstack((t_ux3, t_uy3, u3[:, 2])).flatten()
        #     u_the = (u1, u2, u3)
        # else:
        #     u_the = (u1.flatten(), u2.flatten(), u3.flatten())

        # velocity due to stokesles.
        kwargs = self.get_kwargs( )
        stokeslet_geo = self._stokeslet_geo
        stokeslet_node = np.hstack((b, 0, 0))
        stokeslet_geo.set_nodes(stokeslet_node, deltalength=0)
        ugeo = geo( )
        ugeo.set_nodes(unode, deltalength=0)
        ugeo.set_glbIdx(u_glbIdx)
        obj1 = stokesFlowObj( )
        obj1.set_data(stokeslet_geo, ugeo)
        stokeslet_m = self._stokeslet_m
        point_force_matrix_3d_petsc(obj1, obj1, stokeslet_m, **kwargs)
        stokeslet_m.assemble( )
        # velocity due to boundary, lagrange interploation
        b_list = self.get_b_list( )
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
        l1 = ((b - b_list[k[1]]) * (b - b_list[k[2]])) / ((b_list[k[0]] - b_list[k[1]]) * (b_list[k[0]] - b_list[k[2]]))
        l2 = ((b - b_list[k[0]]) * (b - b_list[k[2]])) / ((b_list[k[1]] - b_list[k[0]]) * (b_list[k[1]] - b_list[k[2]]))
        l3 = ((b - b_list[k[0]]) * (b - b_list[k[1]])) / ((b_list[k[2]] - b_list[k[0]]) * (b_list[k[2]] - b_list[k[1]]))
        # ux
        t_u1 = self._t_u11[0] * l1 + self._t_u11[1] * l2 + self._t_u11[2] * l3  # velocity due to boundary.
        stokeslet_m.mult(self._f123_petsc[0], self._t_u2)
        u_petsc.append(t_u1 + self._t_u2)
        # uy
        t_u1 = self._t_u12[0] * l1 + self._t_u12[1] * l2 + self._t_u12[2] * l3  # velocity due to boundary.
        stokeslet_m.mult(self._f123_petsc[1], self._t_u2)
        u_petsc.append(t_u1 + self._t_u2)
        # uz
        t_u1 = self._t_u13[0] * l1 + self._t_u13[1] * l2 + self._t_u13[2] * l3  # velocity due to boundary.
        stokeslet_m.mult(self._f123_petsc[2], self._t_u2)
        u_petsc.append(t_u1 + self._t_u2)
        return u_petsc

    def _solve_stokeslets_fnode(self, fnode, unodes, u_glbIdx=[]):
        fnode = fnode.reshape((1, 3))
        unodes = unodes.reshape((-1, 3))
        ugeo = geo( )
        ugeo.set_nodes(unodes, resetVelocity=True, deltalength=0)
        uphi, urho, uz = ugeo.get_polar_coord( )
        fgeo = geo( )
        fgeo.set_nodes(fnode, resetVelocity=True, deltalength=0)
        fphi, frho, fz = fgeo.get_polar_coord( )

        # calculate ux, uy, uz in local coordinate, definding by fnode.
        b = frho
        R = urho
        phi = uphi - fphi
        x = R * np.cos(phi)
        y = R * np.sin(phi)
        z = uz - fz
        t_node_xyz = np.vstack((x, y, z)).T
        t_node_rpz = np.vstack((R, phi, z)).T
        u_fx_petsc, u_fy_petsc, u_fz_petsc = \
            self._solve_stokeslets_b(b, t_node_xyz, t_node_rpz, use_cart=True, u_glbIdx=u_glbIdx)
        u_fx_loc = u_fx_petsc.getArray( )
        u_fy_loc = u_fy_petsc.getArray( )
        u_fz_loc = u_fz_petsc.getArray( )

        # shift to global coordinate
        theta = np.arctan2(fnode[0, 1], fnode[0, 0])
        T = np.array(((np.cos(theta), np.sin(theta), 0),
                      (-np.sin(theta), np.cos(theta), 0),
                      (0, 0, 1)))
        Tinv = np.array(((np.cos(theta), -np.sin(theta), 0),
                         (np.sin(theta), np.cos(theta), 0),
                         (0, 0, 1)))
        temp_loc = np.dstack((u_fx_loc.reshape((-1, 3)), u_fy_loc.reshape((-1, 3)), u_fz_loc.reshape((-1, 3))))
        temp_glb = np.tensordot(Tinv, np.tensordot(temp_loc, T, axes=(2, 0)), axes=(1, 1))
        u_fx_glb = np.dstack((temp_glb[0, :, 0], temp_glb[1, :, 0], temp_glb[2, :, 0])).flatten( )
        u_fy_glb = np.dstack((temp_glb[0, :, 1], temp_glb[1, :, 1], temp_glb[2, :, 1])).flatten( )
        u_fz_glb = np.dstack((temp_glb[0, :, 2], temp_glb[1, :, 2], temp_glb[2, :, 2])).flatten( )
        u_fx_petsc.setValues(range(u_fx_petsc.getOwnershipRange( )[0], u_fx_petsc.getOwnershipRange( )[1]), u_fx_glb)
        u_fy_petsc.setValues(range(u_fy_petsc.getOwnershipRange( )[0], u_fy_petsc.getOwnershipRange( )[1]), u_fy_glb)
        u_fz_petsc.setValues(range(u_fz_petsc.getOwnershipRange( )[0], u_fz_petsc.getOwnershipRange( )[1]), u_fz_glb)

        # # dbg
        # from src.StokesFlowMethod import light_stokeslets_matrix_3d
        # dbg_m = light_stokeslets_matrix_3d(unodes, fnode)
        # dbg_ux = np.dot(dbg_m, np.array((1, 0, 0)))
        # dbg_uy = np.dot(dbg_m, np.array((0, 1, 0)))
        # dbg_uz = np.dot(dbg_m, np.array((0, 0, 1)))
        # print((np.max(np.abs(u_fx_glb - dbg_ux)), np.max(np.abs(u_fy_glb - dbg_uy)), np.max(np.abs(u_fz_glb - dbg_uz))))
        return u_fx_petsc, u_fy_petsc, u_fz_petsc

    def _check_f_accuracy(self, b, greenFun, waitBar=np.array((1, 1)), **kwargs):
        cpgeo = self._cpgeo
        outputHandle = 'check'
        a_u11, a_u21, a_u31 = self._solve_u_pipe(cpgeo, outputHandle, greenFun, waitBar)
        m_petsc = self._m_pipe_check

        f1_petsc, c_u11_petsc = m_petsc.createVecs( )
        f1_petsc[:] = self._f1_list[-1][:]
        f1_petsc.assemble( )
        # c_u11_petsc.set(0)
        m_petsc.mult(f1_petsc, c_u11_petsc)
        c_u11 = self.vec_scatter(c_u11_petsc)
        f2_petsc, c_u21_petsc = m_petsc.createVecs( )
        f2_petsc[:] = self._f2_list[-1][:]
        f2_petsc.assemble( )
        # c_u21_petsc.set(0)
        m_petsc.mult(f2_petsc, c_u21_petsc)
        c_u21 = self.vec_scatter(c_u21_petsc)
        f3_petsc, c_u31_petsc = m_petsc.createVecs( )
        f3_petsc[:] = self._f3_list[-1][:]
        f3_petsc.assemble( )
        # c_u31_petsc.set(0)
        m_petsc.mult(f3_petsc, c_u31_petsc)
        c_u31 = self.vec_scatter(c_u31_petsc)

        err1 = np.sqrt(np.sum((a_u11 - c_u11) ** 2) / np.sum(a_u11 ** 2))
        err2 = np.sqrt(np.sum((a_u21 - c_u21) ** 2) / np.sum(a_u21 ** 2))
        err3 = np.sqrt(np.sum((a_u31 - c_u31) ** 2) / np.sum(a_u31 ** 2))
        PETSc.Sys( ).Print('      relative err: %f, %f, %f' % (err1, err2, err3))
        self._err_list.append((err1, err2, err3))
        return True

    def set_prepare(self, fileHeadle):
        t_headle = '_force_pipe.mat'
        if fileHeadle[-len(t_headle):] != t_headle:
            fileHeadle = fileHeadle + t_headle
        mat_contents = sio.loadmat(fileHeadle)

        self.set_b_list(mat_contents['b'].flatten( ))
        self._f1_list = [f1 for f1 in mat_contents['f1_list']]
        self._f2_list = [f2 for f2 in mat_contents['f2_list']]
        self._f3_list = [f3 for f3 in mat_contents['f3_list']]
        self._residualNorm_list = mat_contents['residualNorm'].tolist( )
        self._err_list = mat_contents['err'].tolist( )
        self._dp = mat_contents['dp'][0, 0]
        self._rp = mat_contents['rp'][0, 0]
        self._lp = mat_contents['lp'][0, 0]
        self._ep = mat_contents['ep'][0, 0]
        self._th = mat_contents['th'][0, 0]
        self._factor = mat_contents['factor'][0, 0]

        kwargs = self.get_kwargs( )
        kwargs['dp'] = self._dp
        kwargs['rp'] = self._rp
        kwargs['lp'] = self._lp
        kwargs['ep'] = self._ep
        kwargs['th'] = self._th
        kwargs['factor'] = self._factor
        kwargs['check_acc'] = False
        self._kwargs['unpickedPrb'] = True
        self._pipe_geo(**kwargs)
        self._kwargs = kwargs

        # PETSC version
        self._f_list_numpy2PETSC( )
        return True

    def solve_prepare(self):
        kwargs = self.get_kwargs( )
        self._dp = kwargs['dp']
        self._rp = kwargs['rp']
        self._lp = kwargs['lp']
        self._ep = kwargs['ep']
        self._th = kwargs['th']
        self._factor = kwargs['factor']
        self._b_list = np.linspace(kwargs['b0'], kwargs['b1'], kwargs['nb'])  # list of b (force location).

        PETSc.Sys.Print('                b_list: ')
        PETSc.Sys.Print(self.get_b_list( ))
        self._f1_list.clear( )
        self._f2_list.clear( )
        self._f3_list.clear( )
        self._pipe_geo(**kwargs)
        self._solve_m_pipe(**kwargs)

        ini_guess = (None, None, None,)
        from src.stokesletsInPipe import detail
        for i0, b in enumerate(tqdm(self.get_b_list( ))):
            greenFun = detail(threshold=self._th, b=b)
            greenFun.solve_prepare( )
            waitBar = np.array((i0 + 1, self.get_n_b( )))
            problem_u1, problem_u2, problem_u3 = self._solve_f_pipe(b, ini_guess, greenFun, waitBar, **kwargs)
            # # numpy based version
            # self._f1_list.append(self.vec_scatter(problem_u1.get_force_petsc()))
            # self._f2_list.append(self.vec_scatter(problem_u2.get_force_petsc()))
            # self._f3_list.append(self.vec_scatter(problem_u3.get_force_petsc()))
            # PETSC based version
            self._f1_list.append(problem_u1.get_force_petsc( ))
            self._f2_list.append(problem_u2.get_force_petsc( ))
            self._f3_list.append(problem_u3.get_force_petsc( ))
            self._residualNorm_list.append(
                    (problem_u1.get_residualNorm( ), problem_u2.get_residualNorm( ), problem_u3.get_residualNorm( )))
            if kwargs['check_acc']:
                self._check_f_accuracy(b, greenFun, waitBar, **kwargs)
        self._m_pipe.destroy( )
        self._m_pipe_check.destroy( )
        return True

    def get_f_list(self):
        # PETSC version
        self._f_list_PETSC2numpy( )
        return self._f1_list, self._f2_list, self._f3_list

    def _pipe_geo(self, **kwargs):
        dp = self._dp
        rp = self._rp
        lp = self._lp
        ep = self._ep
        factor = self._factor

        vpgeo = tunnel_geo( )  # velocity node geo of pipe
        dth = 2 * np.arcsin(dp / 2 / rp)

        # debug
        # OptDB = PETSc.Options()
        # factor = OptDB.getReal('dbg_factor', 2.5)
        # PETSc.Sys.Print('--------------------> DBG: factor=%f' % factor)
        fpgeo = vpgeo.create_deltatheta(dth=dth, radius=rp, length=lp, epsilon=ep, with_cover=True, factor=factor)
        t_pkg = PETSc.DMComposite( ).create(comm=PETSc.COMM_WORLD)
        t_pkg.addDM(fpgeo.get_dmda( ))
        t_pkg.setFromOptions( )
        t_pkg.setUp( )
        t_isglb = t_pkg.getGlobalISs( )
        fpgeo.set_glbIdx(t_isglb[0].getIndices( ))
        # cbd_geo = geo()
        # cbd_geo.combine(geo_list=[vpgeo, fpgeo, ])
        # cbd_geo.show_nodes(linestyle='-')
        self._fpgeo = fpgeo
        self._vpgeo = vpgeo

        if kwargs['check_acc']:
            cpgeo = tunnel_geo( )
            cpgeo.create_deltatheta(dth=dth * 0.79, radius=rp, length=lp, epsilon=0, with_cover=True, factor=1)
            self._cpgeo = cpgeo

        # if kwargs['plot_geo']:
        #     cbd_geo = geo()
        #     cbd_geo.combine(geo_list=[vpgeo, fpgeo, ])
        #     cbd_geo.show_nodes(linestyle='-')
        return True

    def _solve_m_pipe(self, **kwargs):
        # generate geo and associated nodes: a finite length pipe with covers at both side.
        t0 = time( )
        obj1 = stokesFlowObj( )
        obj1.set_data(self._fpgeo, self._vpgeo)
        PETSc.Sys( ).Print('Stokeslets in pipe prepare, contain %d nodes' % self._vpgeo.get_n_nodes( ))
        self._m_pipe = self.create_obj_matrix(obj1, obj1, copy_obj=False, **kwargs)

        if kwargs['check_acc']:
            obj2 = stokesFlowObj( )
            obj2.set_data(self._fpgeo, self._cpgeo)
            PETSc.Sys( ).Print('Stokeslets in pipe check, contain %d nodes' % self._cpgeo.get_n_nodes( ))
            self._m_pipe_check = self.create_obj_matrix(obj2, obj2, copy_obj=False, **kwargs)
            t1 = time( )
            PETSc.Sys( ).Print('  create matrix use %fs:' % (t1 - t0))
        return True

    def _solve_f_pipe(self, b, ini_guess, greenFun, waitBar=np.array((1, 1)), **kwargs):
        # calculate force at each nodes at (or outside) the pipe boundary.
        vpgeo = self._vpgeo
        outputHandle = 'vpgeo'
        u11, u21, u31 = self._solve_u_pipe(vpgeo, outputHandle, greenFun, waitBar)

        # for each direction, solve force at (or outside) nodes.
        f1 = np.array((1, 0, 0))
        f2 = np.array((0, 1, 0))
        f3 = np.array((0, 0, 1))
        stokeslets_post = np.hstack((b, 0, 0)).reshape(1, 3)
        fpgeo = self._fpgeo
        kwargs_u1 = kwargs.copy( )
        kwargs_u1['deltaLength'] = self._dp
        kwargs_u1['epsilon'] = self._ep
        kwargs_u1['delta'] = self._dp * self._ep
        kwargs_u1['d_radia'] = self._dp / 2
        kwargs_u1['name'] = '  _%05d/%05d_u1' % (waitBar[0], waitBar[1])
        kwargs_u1['plot'] = False
        kwargs_u1['fileHeadle'] = 'stokesletsInPipeProblem_u1'
        kwargs_u1['restart'] = False
        kwargs_u1['getConvergenceHistory'] = False
        kwargs_u1['pickProblem'] = False
        kwargs_u1['stokeslets_post'] = stokeslets_post

        kwargs_u1['stokeslets_f'] = f1
        problem_u1 = stokesletsProblem(**kwargs_u1)
        obj_u1 = stokesletsObj( )
        obj_u1_kwargs = {'name':            'stokesletsInPipeObj_u1',
                         'stokeslets_f':    f1,
                         'stokeslets_post': stokeslets_post}
        vpgeo.set_velocity(u11)
        obj_u1.set_data(fpgeo, vpgeo, **obj_u1_kwargs)
        problem_u1.add_obj(obj_u1)
        problem_u1.set_matrix(self._m_pipe)
        problem_u1.solve(ini_guess=ini_guess[0])

        kwargs_u2 = kwargs_u1.copy( )
        kwargs_u2['stokeslets_f'] = f2
        kwargs_u2['name'] = '  _%05d/%05d_u2' % (waitBar[0], waitBar[1])
        kwargs_u2['fileHeadle'] = 'stokesletsInPipeProblem_u2'
        problem_u2 = stokesletsProblem(**kwargs_u2)
        obj_u2 = stokesletsObj( )
        obj_u2_kwargs = {'name':            'stokesletsInPipeObj_u2',
                         'stokeslets_f':    f2,
                         'stokeslets_post': stokeslets_post}
        vpgeo.set_velocity(u21)
        obj_u2.set_data(fpgeo, vpgeo, **obj_u2_kwargs)
        problem_u2.add_obj(obj_u2)
        problem_u2.set_matrix(self._m_pipe)
        problem_u2.solve(ini_guess=ini_guess[1])

        kwargs_u3 = kwargs_u1.copy( )
        kwargs_u3['stokeslets_f'] = f3
        kwargs_u3['name'] = '  _%05d/%05d_u3' % (waitBar[0], waitBar[1])
        kwargs_u3['fileHeadle'] = 'stokesletsInPipeProblem_u3'
        problem_u3 = stokesletsProblem(**kwargs_u3)
        obj_u3 = stokesletsObj( )
        obj_u3_kwargs = {'name':            'stokesletsInPipeObj_u3',
                         'stokeslets_f':    f3,
                         'stokeslets_post': stokeslets_post}
        vpgeo.set_velocity(u31)
        obj_u3.set_data(fpgeo, vpgeo, **obj_u3_kwargs)
        problem_u3.add_obj(obj_u3)
        problem_u3.set_matrix(self._m_pipe)
        problem_u3.solve(ini_guess=ini_guess[2])

        return problem_u1, problem_u2, problem_u3

    def _solve_u_pipe(self, pgeo, outputHandle, greenFun, waitBar=np.array((1, 1))):
        t0 = time( )
        from src.StokesFlowMethod import stokeslets_matrix_3d
        # 1 velocity at pipe
        iscover = pgeo.get_iscover( )
        uR1 = np.zeros(np.sum(~iscover))
        uPhi1 = np.zeros_like(uR1)
        uz1 = np.zeros_like(uR1)
        uR2 = np.zeros_like(uR1)
        uPhi2 = np.zeros_like(uR1)
        uz2 = np.zeros_like(uR1)
        uR3 = np.zeros_like(uR1)
        uPhi3 = np.zeros_like(uR1)
        uz3 = np.zeros_like(uR1)
        # 2 velocity at cover
        #  see Liron, N., and R. Shahar. "Stokes flow due to a Stokeslet in a pipe." Journal of Fluid Mechanics 86.04 (1978): 727-744.
        tuR1_list = []
        tuPhi1_list = []
        tuz1_list = []
        tuR2_list = []
        tuPhi2_list = []
        tuz2_list = []
        tuR3_list = []
        tuPhi3_list = []
        tuz3_list = []
        cover_start_list = pgeo.get_cover_start_list( )
        n_cover_node = 0
        for t_nodes in cover_start_list:
            tR = t_nodes[0]
            tphi = t_nodes[1]
            tz = np.abs(t_nodes[2])
            sign_z = np.sign(t_nodes[2])
            n_cover_node = n_cover_node + tphi.size
            tuR1, tuPhi1, tuz1, tuR2, tuPhi2, tuz2, tuR3, tuPhi3, tuz3 = greenFun.solve_u(tR, tphi, tz)
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
        cover_end_list = pgeo.get_cover_end_list( )
        for t_nodes in cover_end_list:
            tR = t_nodes[0]
            tphi = t_nodes[1]
            tz = np.abs(t_nodes[2])
            sign_z = np.sign(t_nodes[2])
            n_cover_node = n_cover_node + tphi.size
            tuR1, tuPhi1, tuz1, tuR2, tuPhi2, tuz2, tuR3, tuPhi3, tuz3 = greenFun.solve_u(tR, tphi, tz)
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
        assert n_cover_node == np.sum(iscover), 'something is wrong'

        pphi, _, _ = pgeo.get_polar_coord( )
        ux1 = np.cos(pphi) * uR1 - np.sin(pphi) * uPhi1
        ux2 = np.cos(pphi) * uR2 - np.sin(pphi) * uPhi2
        ux3 = np.cos(pphi) * uR3 - np.sin(pphi) * uPhi3
        uy1 = np.sin(pphi) * uR1 + np.cos(pphi) * uPhi1
        uy2 = np.sin(pphi) * uR2 + np.cos(pphi) * uPhi2
        uy3 = np.sin(pphi) * uR3 + np.cos(pphi) * uPhi3
        u1 = np.vstack((ux1, uy1, uz1)).T
        u2 = np.vstack((ux2, uy2, uz2)).T
        u3 = np.vstack((ux3, uy3, uz3)).T

        # u2, stokeslets, sigulatrity.
        b = greenFun.get_b( )
        stokeslets_post = np.hstack((b, 0, 0)).reshape(1, 3)
        deltalength = 0
        geo_stokeslets = geo( )
        geo_stokeslets.set_nodes(stokeslets_post, deltalength, resetVelocity=True)
        obj_stokeslets = stokesFlowObj( )
        obj_stokeslets.set_data(geo_stokeslets, geo_stokeslets)
        obj_p = stokesFlowObj( )
        obj_p.set_data(pgeo, pgeo)
        m2 = stokeslets_matrix_3d(obj_p, obj_stokeslets)
        f12 = np.array((1, 0, 0))
        f22 = np.array((0, 1, 0))
        f32 = np.array((0, 0, 1))
        u12 = np.dot(m2, f12)
        u22 = np.dot(m2, f22)
        u32 = np.dot(m2, f32)
        u11 = u1.flatten( ) - u12
        u21 = u2.flatten( ) - u22
        u31 = u3.flatten( ) - u32

        # # 3 subtract singularity.
        # geo_stokeslets = geo()
        # b = greenFun.get_b()
        # stokeslets_post = np.hstack((b, 0, 0)).reshape(1, 3)
        # deltalength = 0
        # geo_stokeslets.set_nodes(stokeslets_post, deltalength, resetVelocity=True)
        # obj_stokeslets = stokesFlowObj()
        # obj_stokeslets.set_data(geo_stokeslets, geo_stokeslets)
        # obj_p = stokesFlowObj()
        # obj_p.set_data(pgeo, pgeo)
        # m2_petsc = stokeslets_matrix_3d_petsc(obj_p, obj_stokeslets)
        # f1 = np.array((1, 0, 0))
        # f2 = np.array((0, 1, 0))
        # f3 = np.array((0, 0, 1))
        # # 3.1 x axis
        # f1_petsc, u12_petsc = m2_petsc.createVecs()
        # u12_petsc.set(0)
        # f1_petsc[:] = f1[:]
        # m2_petsc.mult(f1_petsc, u12_petsc)
        # u12 = self.vec_scatter(u12_petsc)
        # u11 = u1.flatten() - u12
        # # 3.2 y axis
        # f2_petsc, u22_petsc = m2_petsc.createVecs()
        # u22_petsc.set(0)
        # f2_petsc[:] = f2[:]
        # m2_petsc.mult(f2_petsc, u22_petsc)
        # u22 = self.vec_scatter(u22_petsc)
        # u21 = u2.flatten() - u22
        # # 3.3 z axis
        # f3_petsc, u32_petsc = m2_petsc.createVecs()
        # u32_petsc.set(0)
        # f3_petsc[:] = f3[:]
        # m2_petsc.mult(f3_petsc, u32_petsc)
        # u32 = self.vec_scatter(u32_petsc)
        # u31 = u3.flatten() - u32
        t1 = time( )
        PETSc.Sys( ).Print('  _%05d/%05d_b=%f:    calculate %s boundary condation use: %fs' % (
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

    def create_matrix_obj(self, obj1, m, INDEX='', *args):
        # create a empty matrix, and a empty velocity vecters, to avoid use too much time to allocate memory.
        ugeo = obj1.get_u_geo( ).copy( )
        kwargs = self.get_kwargs( )
        t_u_pkg = PETSc.DMComposite( ).create(comm=PETSc.COMM_WORLD)
        t_u_pkg.addDM(ugeo.get_dmda( ))
        t_u_pkg.setFromOptions( )
        t_u_pkg.setUp( )
        self._t_u11 = [t_u_pkg.createGlobalVector( ), t_u_pkg.createGlobalVector( ), t_u_pkg.createGlobalVector( )]
        self._t_u12 = [t_u_pkg.createGlobalVector( ), t_u_pkg.createGlobalVector( ), t_u_pkg.createGlobalVector( )]
        self._t_u13 = [t_u_pkg.createGlobalVector( ), t_u_pkg.createGlobalVector( ), t_u_pkg.createGlobalVector( )]
        self._t_u2 = t_u_pkg.createGlobalVector( )

        stokeslet_m = PETSc.Mat( ).create(comm=PETSc.COMM_WORLD)
        stokeslet_m.setSizes((self._t_u11[0].getSizes( ), self._f123_petsc[0].getSizes( )))
        stokeslet_m.setType('dense')
        stokeslet_m.setFromOptions( )
        stokeslet_m.setUp( )
        self._stokeslet_m = stokeslet_m

        u_isglb = t_u_pkg.getGlobalISs( )
        u_glbIdx_pipe = u_isglb[0].getIndices( )
        temp_m = PETSc.Mat( ).create(comm=PETSc.COMM_WORLD)
        temp_m.setSizes((self._t_u11[0].getSizes( ), self._f1_list[0].getSizes( )))
        temp_m.setType('dense')
        temp_m.setFromOptions( )
        temp_m.setUp( )
        self._t_m = temp_m

        _, u_glbIdx_all = obj1.get_u_geo( ).get_glbIdx( )
        unodes = ugeo.get_nodes( )
        desc = 'object level' + INDEX
        for obj2 in tqdm(self.get_all_obj_list( ), desc=desc):
            f_nodes = obj2.get_f_nodes( )
            _, f_glbIdx_all = obj2.get_f_geo( ).get_glbIdx( )
            f_dmda = obj2.get_f_geo( ).get_dmda( )
            for i0 in tqdm(range(obj2.get_n_f_node( )), desc='f_node level', leave=False):
                t_f_node = f_nodes[i0]
                f_glb = f_glbIdx_all[i0 * 3]
                u1, u2, u3 = self._solve_stokeslets_fnode(t_f_node, unodes, u_glbIdx_pipe)
                u_range = u1.getOwnershipRange( )
                u_glbIdx = u_glbIdx_all[u_range[0]:u_range[1]]
                m.setValues(u_glbIdx, f_glb + 0, u1, addv=False)
                m.setValues(u_glbIdx, f_glb + 1, u2, addv=False)
                m.setValues(u_glbIdx, f_glb + 2, u3, addv=False)
        t_u_pkg.destroy( )
        m.assemble( )
        return True

    def _f_list_numpy2PETSC(self):
        t_f1_list = uniqueList( )  # list of forces lists for each object at or outside pipe associated with force-nodes at x axis
        t_f2_list = uniqueList( )  # list of forces lists for each object at or outside pipe associated with force-nodes at y axis
        t_f3_list = uniqueList( )  # list of forces lists for each object at or outside pipe associated with force-nodes at z axis

        f_pkg = PETSc.DMComposite( ).create(comm=PETSc.COMM_WORLD)
        f_pkg.addDM(self._fpgeo.get_dmda( ))
        f_pkg.setFromOptions( )
        f_pkg.setUp( )
        for f1 in self._f1_list:
            f1_petsc = f_pkg.createGlobalVector( )
            f1_petsc.setFromOptions( )
            f1_petsc.setUp( )
            f1_petsc[:] = f1[:]
            f1_petsc.assemble( )
            t_f1_list.append(f1_petsc)
        for f2 in self._f2_list:
            f2_petsc = f_pkg.createGlobalVector( )
            f2_petsc.setFromOptions( )
            f2_petsc.setUp( )
            f2_petsc[:] = f2[:]
            f2_petsc.assemble( )
            t_f2_list.append(f2_petsc)
        for f3 in self._f3_list:
            f3_petsc = f_pkg.createGlobalVector( )
            f3_petsc.setFromOptions( )
            f3_petsc.setUp( )
            f3_petsc[:] = f3[:]
            f3_petsc.assemble( )
            t_f3_list.append(f3_petsc)
        self._f1_list = t_f1_list
        self._f2_list = t_f2_list
        self._f3_list = t_f3_list
        f_pkg.destroy( )
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

    def pickmyself(self, filename: str, check=False, pick_M=False):
        t0 = time( )
        comm = PETSc.COMM_WORLD.tompi4py( )
        rank = comm.Get_rank( )
        self._pick_filename = filename
        self._pick_M = pick_M

        # PETSC based version
        if not check:
            self._f_list_PETSC2numpy( )

        if rank == 0:
            with open(filename + '_pick.bin', 'wb') as output:
                pickler = pickle.Pickler(output, -1)
                pickler.dump(self)

        if not check and self._finish_solve:
            self.saveF_Binary(filename + '_F')
        if not check and self._finish_solve and pick_M:
            self.saveM_Binary(filename + '_M')
        if not check:
            self.unpickmyself( )
        t1 = time( )
        PETSc.Sys( ).Print('%s: pick the problem use: %fs' % (str(self), (t1 - t0)))
        return True

    def unpickmyself(self):
        super( ).unpickmyself( )
        self._pipe_geo(**self.get_kwargs( ))
        self._kwargs['unpickedPrb'] = True
        # create a empty matrix, and a empty velocity vecters, to avoid use too much time to allocate memory.
        self._set_f123( )
        # PETSC based version
        self._f_list_numpy2PETSC( )
        return True


class forceFreeComposite:
    def __init__(self, center=np.zeros(3), name='...'):
        self._obj_list = uniqueList( )
        self._rel_U_list = []
        self._index = -1  # index of object
        self._problem = None
        self._center = center
        self._n_fnode = 0
        self._n_unode = 0
        self._f_glbIdx = np.array([])  # global indices
        self._f_glbIdx_all = np.array([])  # global indices for all process.
        self._u_glbIdx = np.array([])  # global indices
        self._u_glbIdx_all = np.array([])  # global indices for all process.
        self._type = 'forceFreeComposite'  # object type
        self._name = name  # object name
        self.set_dmda( )
        self._ref_U = np.zeros(6)  # ux, uy, uz, omega_x, omega_y, omega_z
        self._re_sum = np.ones(
                6)  # analitically, re_sum={sum(fi), sum(cross(ri, fi))}==np.zeros(6) to satisfy the force free equations.
        self._min_ds = np.inf  # min deltalength of objects in the composite

    def __repr__(self):
        return self.get_obj_name( )

    def __str__(self):
        return self.get_name( )

    def add_obj(self, obj, rel_U):
        self._obj_list.append(obj)
        obj.set_index(self.get_n_obj( ))
        obj.set_problem(self)
        self._rel_U_list.append(rel_U)
        self._n_fnode += obj.get_n_f_node( )
        self._n_unode += obj.get_n_u_node( )
        self._min_ds = np.min((self._min_ds, obj.get_u_geo( ).get_deltaLength( )))
        return True

    def get_f_dmda(self):
        return self._f_dmda

    def get_u_dmda(self):
        return self._u_dmda

    def set_dmda(self):
        self._f_dmda = PETSc.DMDA( ).create(sizes=(6,), dof=1, stencil_width=0,
                                            comm=PETSc.COMM_WORLD)  # addtional degrees of freedom for force free.
        self._f_dmda.setFromOptions( )
        self._f_dmda.setUp( )
        self._u_dmda = PETSc.DMDA( ).create(sizes=(6,), dof=1, stencil_width=0,
                                            comm=PETSc.COMM_WORLD)  # addtional degrees of freedom for force free.
        self._u_dmda.setFromOptions( )
        self._u_dmda.setUp( )
        return True

    def destroy_dmda(self):
        self._f_dmda.destroy( )
        self._u_dmda.destroy( )
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
        comm = PETSc.COMM_WORLD.tompi4py( )
        self._f_glbIdx = indices
        self._f_glbIdx_all = np.hstack(comm.allgather(indices))
        return True

    def get_f_glbIdx(self):
        return self._f_glbIdx, self._f_glbIdx_all

    def set_u_glbIdx(self, indices):
        comm = PETSc.COMM_WORLD.tompi4py( )
        self._u_glbIdx = indices
        self._u_glbIdx_all = np.hstack(comm.allgather(indices))
        return True

    def get_u_glbIdx(self):
        return self._u_glbIdx, self._u_glbIdx_all

    def get_combined_obj(self):
        obj0 = stokesFlowObj( )
        obj0.combine(self.get_obj_list( ), set_re_u=True, set_force=True)
        return obj0

    def set_problem(self, problem: 'stokesFlowProblem'):
        self._problem = problem
        return True

    def clear_obj_list(self):
        self._obj_list = uniqueList( )
        self._rel_U_list = []
        return True

    def print_info(self):
        PETSc.Sys.Print('%s: father %s, type %s, index %d, force nodes %d, velocity nodes %d'
                        % (self.get_name( ), self._problem.get_name( ), self._type, self.get_index( ),
                           self.get_n_f_node( ), self.get_n_u_node( )))
        for obj in self._obj_list:
            obj.print_info( )
        return True

    def copy(self):
        composite2 = copy.copy(self)
        composite2.set_problem(self._problem)
        composite2.set_index(-1)
        composite2.set_dmda( )
        composite2.clear_obj_list( )
        for sub_obj, rel_U in zip(self.get_obj_list( ), self.get_rel_U_list( )):
            obj1 = sub_obj.copy( )
            composite2.add_obj(obj1, rel_U)
        return composite2

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

    def set_re_sum(self, re_sum):
        self._re_sum = re_sum
        return True

    def get_re_sum(self):
        return self._re_sum

    def show_velocity(self, length_factor=1, show_nodes=True):
        geo_list = uniqueList( )
        for obj1 in self.get_obj_list( ):
            geo_list.append(obj1.get_u_geo( ))
        temp_geo = geo( )
        temp_geo.combine(geo_list)
        temp_geo.show_velocity(length_factor=length_factor, show_nodes=show_nodes)
        return True

    def show_f_nodes(self, linestyle='-'):
        geo_list = uniqueList( )
        for obj1 in self.get_obj_list( ):
            geo_list.append(obj1.get_f_geo( ))
        temp_geo = geo( )
        temp_geo.combine(geo_list)
        temp_geo.show_nodes(linestyle)
        return True

    def show_u_nodes(self, linestyle='-'):
        geo_list = uniqueList( )
        for obj1 in self.get_obj_list( ):
            geo_list.append(obj1.get_u_geo( ))
        temp_geo = geo( )
        temp_geo.combine(geo_list)
        temp_geo.show_nodes(linestyle)
        return True

    def show_f_u_nodes(self, linestyle='-'):
        geo_list = uniqueList( )
        for obj1 in self.get_obj_list( ):
            geo_list.append(obj1.get_f_geo( ))
        for obj1 in self.get_obj_list( ):
            geo_list.append(obj1.get_u_geo( ))
        temp_geo = geo( )
        temp_geo.combine(geo_list)
        temp_geo.show_nodes(linestyle)
        return True

    def vtk(self, filename):
        for obj0 in self._obj_list:
            obj0.vtk(filename)

    def unpickmyself(self):
        self.set_dmda( )
        for sub_obj in self.get_obj_list( ):
            sub_obj.unpickmyself( )
        return True


class forceFreeProblem(stokesFlowProblem):
    def __init__(self, **kwargs):
        super( ).__init__(**kwargs)
        self._all_obj_list = uniqueList( )  # contain all objects, including subobj within forceFreeComposite.
        self._compst_list = uniqueList( )  # forceFreeComposite list.

    def add_obj(self, obj):
        if isinstance(obj, forceFreeComposite):
            self._obj_list.append(obj)
            obj.set_index(self.get_n_obj( ))
            obj.set_problem(self)
            for sub_obj in obj.get_obj_list( ):
                self._all_obj_list.append(sub_obj)
                self._f_pkg.addDM(sub_obj.get_f_geo( ).get_dmda( ))
                self._u_pkg.addDM(sub_obj.get_u_geo( ).get_dmda( ))
                self._n_fnode += sub_obj.get_n_f_node( )
                self._n_unode += sub_obj.get_n_u_node( )
            self._f_pkg.addDM(obj.get_f_dmda( ))
            self._u_pkg.addDM(obj.get_u_dmda( ))
            self._compst_list.append(obj)
        else:
            self._all_obj_list.append(obj)
            super( ).add_obj(obj)
        return True

    def get_all_obj_list(self, ):
        return self._all_obj_list

    def create_F_U(self):
        comm = PETSc.COMM_WORLD.tompi4py( )
        size = comm.Get_size( )
        rank = comm.Get_rank( )
        kwargs = self._kwargs
        # create f and u DMComposite
        self._f_pkg.setFromOptions( )
        self._f_pkg.setUp( )
        self._u_pkg.setFromOptions( )
        self._u_pkg.setUp( )

        # global index
        f_isglb = self._f_pkg.getGlobalISs( )
        u_isglb = self._u_pkg.getGlobalISs( )
        for obj0 in self.get_obj_list( ):
            if isinstance(obj0, forceFreeComposite):
                for sub_obj in obj0.get_obj_list( ):
                    t_f_isglb = f_isglb.pop(0)
                    t_u_isglb = u_isglb.pop(0)
                    sub_obj.get_f_geo( ).set_glbIdx(t_f_isglb.getIndices( ))
                    sub_obj.get_u_geo( ).set_glbIdx(t_u_isglb.getIndices( ))
                t_f_isglb = f_isglb.pop(0)  # force free additional degree of freedomes
                t_u_isglb = u_isglb.pop(0)  # velocity free additional degree of freedomes
                obj0.set_f_glbIdx(t_f_isglb.getIndices( ))
                obj0.set_u_glbIdx(t_u_isglb.getIndices( ))
            else:
                t_f_isglb = f_isglb.pop(0)
                t_u_isglb = u_isglb.pop(0)
                obj0.get_f_geo( ).set_glbIdx(t_f_isglb.getIndices( ))
                obj0.get_u_geo( ).set_glbIdx(t_u_isglb.getIndices( ))

        # velocity
        velocity = self._u_pkg.createGlobalVector( )
        velocity.zeroEntries( )
        for obj0 in self.get_obj_list( ):
            if isinstance(obj0, forceFreeComposite):
                center = obj0.get_center( )
                for sub_obj, rel_U in zip(obj0.get_obj_list( ), obj0.get_rel_U_list( )):
                    sub_nodes = sub_obj.get_u_geo( ).get_nodes( )
                    r = sub_nodes - center
                    t_u = (rel_U[:3] + np.cross(rel_U[3:], r)).flatten( )
                    _, u_glbIdx_all = sub_obj.get_u_geo( ).get_glbIdx( )
                    if rank == 0:
                        velocity[u_glbIdx_all] = t_u[:]
            else:
                u0 = obj0.get_velocity( )
                _, u_glbIdx_all = obj0.get_u_geo( ).get_glbIdx( )
                if rank == 0:
                    velocity[u_glbIdx_all] = u0[:]
        velocity.assemble( )
        self._velocity_petsc = velocity

        # force
        self._force_petsc = self._f_pkg.createGlobalVector( )
        return True

    def set_force_free(self):
        import numpy.matlib as npm
        kwargs = self._kwargs
        zoom_factor = kwargs['zoom_factor']
        ffweight = kwargs['ffweight'] / zoom_factor

        for obj1 in self.get_obj_list( ):
            if isinstance(obj1, forceFreeComposite):
                center = obj1.get_center( )
                _, u_glbIdx_all = obj1.get_u_glbIdx( )
                _, f_glbIdx_all = obj1.get_f_glbIdx( )
                # self._M_petsc.zeroRows(u_glbIdx_all)
                # self._M_petsc.setValues(u_glbIdx_all, range(f_size), np.zeros(f_size), addv=False)
                # self._M_petsc.setValues(range(u_size), f_glbIdx_all, np.zeros(u_size), addv=False)
                for sub_obj in obj1.get_obj_list( ):
                    r_u = sub_obj.get_u_geo( ).get_nodes( ) - center
                    r_f = sub_obj.get_f_geo( ).get_nodes( ) - center
                    t_I = np.array(((-ffweight, 0, 0),
                                    (0, -ffweight, 0),
                                    (0, 0, -ffweight ** 3)))
                    tmu1 = npm.repmat(t_I, sub_obj.get_n_u_node( ), 1)
                    tmu2 = np.vstack([((0, -ri[2], ri[1]),
                                       (ri[2], 0, -ri[0]),
                                       (-ri[1], ri[0], 0))
                                      for ri in r_u]) * ffweight ** 2
                    tmf1 = npm.repmat(t_I, 1, sub_obj.get_n_f_node( ))
                    tmf2 = np.hstack([((0, -ri[2], ri[1]),
                                       (ri[2], 0, -ri[0]),
                                       (-ri[1], ri[0], 0))
                                      for ri in r_f]) * ffweight ** 2
                    tmu = np.hstack((tmu1, tmu2))
                    tmf = np.vstack((tmf1, tmf2))
                    _, sub_u_glbIdx_all = sub_obj.get_u_geo( ).get_glbIdx( )
                    _, sub_f_glbIdx_all = sub_obj.get_f_geo( ).get_glbIdx( )
                    self._M_petsc.setValues(sub_u_glbIdx_all, f_glbIdx_all, tmu, addv=False)
                    self._M_petsc.setValues(u_glbIdx_all, sub_f_glbIdx_all, tmf, addv=False)
                    # # dbg
                    # PETSc.Sys.Print(sub_u_glbIdx_all, f_glbIdx_all)
        self._M_petsc.assemble( )
        return True

    def create_matrix(self):
        t0 = time( )
        err_msg = 'at least one object is necessary. '
        assert len(self._obj_list) != 0, err_msg
        kwargs = self._kwargs
        solve_method = kwargs['solve_method']

        err_msg = 'unequal force and velocity degrees of freedom, only lsqr method is accept. '
        for obj1 in self.get_all_obj_list( ):
            assert obj1.get_n_force( ) == obj1.get_n_velocity( ) or solve_method == 'lsqr', err_msg

        self.create_F_U( )

        # create matrix
        # 1. setup matrix
        if not self._M_petsc.isAssembled( ):
            self._M_petsc.setSizes((self._velocity_petsc.getSizes( ), self._force_petsc.getSizes( )))
            self._M_petsc.setType('dense')
            self._M_petsc.setFromOptions( )
            self._M_petsc.setUp( )
            # Todo: error occure if matrix is huge, set zero one part aftre another.
            self._M_petsc.zeroEntries( )
        f_size = self._force_petsc.getSize( )  # including additional degree of freedome.
        u_size = self._velocity_petsc.getSize( )  # including additional degree of freedome.
        # 2. set mij part of matrix
        cmbd_ugeo = geo( )
        cmbd_ugeo.combine([obj.get_u_geo( ) for obj in self.get_all_obj_list( )])
        cmbd_ugeo.set_glbIdx_all(np.hstack([obj.get_u_geo( ).get_glbIdx( )[1] for obj in self.get_all_obj_list( )]))
        cmbd_obj = stokesFlowObj( )
        cmbd_obj.set_data(cmbd_ugeo, cmbd_ugeo)
        self.create_matrix_obj(cmbd_obj, self._M_petsc)
        # n_obj = len(self.get_all_obj_list())
        # for i0, obj1 in enumerate(self.get_all_obj_list()):
        #     INDEX = '  %d/%d' % (i0+1, n_obj)
        #     self.create_matrix_obj(obj1, self._M_petsc, INDEX)
        # 3. set force and torque free part of matrix
        self.set_force_free( )
        # self._M_petsc.view()

        t1 = time( )
        PETSc.Sys.Print('%s: create matrix use: %fs' % (str(self), (t1 - t0)))
        return True

    def _solve_force(self, ksp):
        kwargs = self._kwargs
        zoom_factor = kwargs['zoom_factor']
        ffweight = kwargs['ffweight'] / zoom_factor
        zoom_factor = kwargs['zoom_factor']
        getConvergenceHistory = kwargs['getConvergenceHistory']
        if getConvergenceHistory:
            ksp.setConvergenceHistory( )
            ksp.solve(self._velocity_petsc, self._force_petsc)
            self._convergenceHistory = ksp.getConvergenceHistory( )
        else:
            ksp.solve(self._velocity_petsc, self._force_petsc)
        self._force = self.vec_scatter(self._force_petsc, destroy=False)

        for obj0 in self.get_obj_list( ):
            if isinstance(obj0, forceFreeComposite):
                for sub_obj in obj0.get_obj_list( ):
                    _, f_glbIdx_all = sub_obj.get_f_geo( ).get_glbIdx( )
                    sub_obj.set_force(self._force[f_glbIdx_all])
                _, f_glbIdx_all = obj0.get_f_glbIdx( )
                ref_U = self._force[f_glbIdx_all] * [ffweight, ffweight, ffweight ** 3, ffweight ** 2, ffweight ** 2,
                                                     ffweight ** 2]
                obj0.set_ref_U(ref_U)
                # absolute speed
                for sub_obj, rel_U in zip(obj0.get_obj_list( ), obj0.get_rel_U_list( )):
                    abs_U = ref_U + rel_U
                    sub_obj.get_u_geo( ).set_rigid_velocity(abs_U)
            else:
                _, f_glbIdx_all = obj0.get_f_geo( ).get_glbIdx( )
                obj0.set_force(self._force[f_glbIdx_all])
        return True

    def _resolve_velocity(self, ksp):
        re_velocity_petsc = self._M_petsc.createVecLeft( )
        # re_velocity_petsc.set(0)
        self._M_petsc.mult(self._force_petsc, re_velocity_petsc)
        self._re_velocity = self.vec_scatter(re_velocity_petsc)
        for obj0 in self.get_obj_list( ):
            if isinstance(obj0, forceFreeComposite):
                ref_U = obj0.get_ref_U( )
                center = obj0.get_center( )
                for sub_obj in obj0.get_obj_list( ):
                    _, u_glbIdx_all = sub_obj.get_u_geo( ).get_glbIdx( )
                    re_rel_U = self._re_velocity[u_glbIdx_all]
                    sub_nodes = sub_obj.get_u_geo( ).get_nodes( )
                    r = sub_nodes - center
                    t_u = (ref_U[:3] + np.cross(ref_U[3:], r)).flatten( )
                    re_abs_U = t_u + re_rel_U
                    sub_obj.set_re_velocity(re_abs_U)
                _, u_glbIdx_all = obj0.get_u_glbIdx( )
                obj0.set_re_sum(self._re_velocity[u_glbIdx_all])  # force free, analytically they are zero.
            else:
                _, u_glbIdx_all = obj0.get_u_geo( ).get_glbIdx( )
                obj0.set_re_velocity(self._re_velocity[u_glbIdx_all])
        self._finish_solve = True
        return ksp.getResidualNorm( )

    def vtk_self(self, filename):
        self.check_finish_solve( )
        obj_list = uniqueList( )
        for obj0 in self.get_obj_list( ):
            if isinstance(obj0, forceFreeComposite):
                for obj1 in obj0.get_obj_list( ):
                    obj_list.append(obj1)
            else:
                obj_list.append(obj0)
        obj0 = stokesFlowObj( )
        obj0.combine(obj_list, set_re_u=True, set_force=True)
        obj0.set_name('Prb')
        obj0.vtk(filename)
        return True

    def unpickmyself(self):
        filename = self._pick_filename
        err_msg = 'call pickmyself() before unpickmyself(). i.e. store date first and reload them at restart mode. '
        assert filename != '..', err_msg

        self._f_pkg = PETSc.DMComposite( ).create(comm=PETSc.COMM_WORLD)
        self._u_pkg = PETSc.DMComposite( ).create(comm=PETSc.COMM_WORLD)
        self._M_petsc = PETSc.Mat( ).create(comm=PETSc.COMM_WORLD)  # M matrix
        for obj1 in self._obj_list:
            obj1.unpickmyself( )
            if isinstance(obj1, forceFreeComposite):
                for sub_obj in obj1.get_obj_list( ):
                    self._f_pkg.addDM(sub_obj.get_f_geo( ).get_dmda( ))
                    self._u_pkg.addDM(sub_obj.get_u_geo( ).get_dmda( ))
                self._f_pkg.addDM(obj1.get_f_dmda( ))
                self._u_pkg.addDM(obj1.get_u_dmda( ))
            else:
                self._f_pkg.addDM(obj1.get_f_geo( ).get_dmda( ))
                self._u_pkg.addDM(obj1.get_u_geo( ).get_dmda( ))
        self._f_pkg.setFromOptions( )
        self._u_pkg.setFromOptions( )
        # Todo: setUp f_pkg and u_pkg at a appropriate time
        # self._f_pkg.setUp()
        # self._u_pkg.setUp()

        if self._finish_solve:
            self.create_F_U( )
            self._force_petsc[:] = self._force[:]
            self._force_petsc.assemble( )
            # self._force_petsc.view()
        if self._finish_solve and self._pick_M:
            self.loadM_Binary(filename + '_M')
        return True


class stokesletsInPipeForceFreeProblem(stokesletsInPipeProblem, forceFreeProblem):
    def add_obj(self, obj):
        b_list = self.get_b_list( )
        b0 = np.min(b_list)
        b1 = np.max(b_list)
        err_msg = 'b is out of range (%f, %f)' % (b0, b1)

        if isinstance(obj, forceFreeComposite):
            self._obj_list.append(obj)
            obj.set_index(self.get_n_obj( ))
            obj.set_problem(self)
            for sub_obj in obj.get_obj_list( ):
                _, b, _ = sub_obj.get_f_geo( ).get_polar_coord( )
                assert all(b <= b1), err_msg
                self._all_obj_list.append(sub_obj)
                self._f_pkg.addDM(sub_obj.get_f_geo( ).get_dmda( ))
                self._u_pkg.addDM(sub_obj.get_u_geo( ).get_dmda( ))
                self._n_fnode += sub_obj.get_n_f_node( )
                self._n_unode += sub_obj.get_n_u_node( )
            self._f_pkg.addDM(obj.get_f_dmda( ))
            self._u_pkg.addDM(obj.get_u_dmda( ))
            self._compst_list.append(obj)
        else:
            self._all_obj_list.append(obj)
            super(stokesletsInPipeProblem).add_obj(obj)
        return True


problem_dic = {
    'rs':                            stokesFlowProblem,
    'lg_rs':                         stokesFlowProblem,
    'tp_rs':                         stokesFlowProblem,
    'pf':                            stokesFlowProblem,
    'tp_rs_precondition':            preconditionProblem,
    'lg_rs_precondition':            preconditionProblem,
    'rs_precondition':               preconditionProblem,
    'sf':                            stokesFlowProblem,
    'sf_debug':                      stokesFlowProblem,
    'ps':                            pointSourceProblem,
    'ps_ds':                         point_source_dipoleProblem,
    'pf_ds':                         point_force_dipoleProblem,
    'rs_stokeslets':                 stokesletsProblem,
    'lg_rs_stokeslets':              stokesletsProblem,
    'tp_rs_stokeslets':              stokesletsProblem,
    'pf_stokeslets':                 stokesletsProblem,
    'rs_stokesletsInPipe':           stokesletsInPipeProblem,
    'pf_stokesletsInPipe':           stokesletsInPipeProblem,
    'rs_stokeslets_precondition':    stokesletsPreconditionProblem,
    'lg_rs_stokeslets_precondition': stokesletsPreconditionProblem,
    'tp_rs_stokeslets_precondition': stokesletsPreconditionProblem,
    'pf_stokesletsTwoPlate':         stokesFlowProblem,
}
obj_dic = {
    'tp_rs':                         stokesFlowObj,
    'lg_rs':                         stokesFlowObj,
    'rs':                            stokesFlowObj,
    'pf':                            stokesFlowObj,
    'tp_rs_precondition':            preconditionObj,
    'lg_rs_precondition':            preconditionObj,
    'rs_precondition':               preconditionObj,
    'sf':                            surf_forceObj,
    'sf_debug':                      surf_forceObj,
    'ps':                            pointSourceObj,
    'ps_ds':                         point_source_dipoleObj,
    'pf_ds':                         point_force_dipoleObj,
    'rs_stokeslets':                 stokesletsObj,
    'lg_rs_stokeslets':              stokesletsObj,
    'tp_rs_stokeslets':              stokesletsObj,
    'pf_stokeslets':                 stokesletsObj,
    'rs_stokesletsInPipe':           stokesFlowObj,
    'pf_stokesletsInPipe':           stokesFlowObj,
    'rs_stokeslets_precondition':    stokesletsPreconditionObj,
    'lg_rs_stokeslets_precondition': stokesletsPreconditionObj,
    'tp_rs_stokeslets_precondition': stokesletsPreconditionObj,
    'pf_stokesletsTwoPlate':         stokesFlowObj,
}

# names of models that need two geometries.
two_geo_method_list = ('pf', 'ps', 'ps_ds', 'pf_ds',
                       'pf_stokeslets')

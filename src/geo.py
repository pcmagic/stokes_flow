# coding=utf-8

import copy
import numpy as np
from numpy import sin, cos
from scipy.io import savemat, loadmat
from petsc4py import PETSc
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from src.support_class import *
import abc

__all__ = ['geo', 'sphere_geo', 'ellipse_geo', 'geoComposit',
           'tunnel_geo', 'pipe_cover_geo', 'supHelix',
           'infgeo_1d', 'infHelix', 'infPipe',
           'region', ]


class geo():
    def __init__(self):
        self._nodes = np.array([])
        self._elems = np.array([])
        self._elemtype = ' '
        self._normal = np.array([])  # norm of surface at each point.
        self._geo_norm = np.array((0, 0, 1))  # describing the aspect of the geo.
        self._origin = np.array((0, 0, 0))
        self._u = np.array([])
        self._deltaLength = 0
        self._dmda = None  # dof management
        self._stencil_width = 0  # --->>>if change in further version, deal with combine method.
        self._glbIdx = np.array([])  # global indices
        self._glbIdx_all = np.array([])  # global indices for all process.
        self._selfIdx = np.array([])  # indices of _glbIdx in _glbIdx_all
        self._dof = 3  # degrees of freedom pre node.

    def mat_nodes(self, filename: str = '..',
                  mat_handle: str = 'nodes'):
        err_msg = 'wrong mat file name. '
        assert filename != '..', err_msg
        filename = check_file_extension(filename, '.mat')

        mat_contents = loadmat(filename)
        nodes = mat_contents[mat_handle].astype(np.float, order='F')
        err_msg = 'nodes is a n*3 numpy array containing x, y and z coordinates. '
        assert nodes.shape[1] == 3, err_msg
        self._nodes = nodes
        self._u = np.zeros(self._nodes.size)
        self.set_dmda()
        return True

    def mat_elmes(self, filename: str = '..',
                  mat_handle: str = 'elmes',
                  elemtype: str = ' '):
        err_msg = 'wrong mat file name. '
        assert filename != '..', err_msg

        mat_contents = loadmat(filename)
        elems = mat_contents[mat_handle].astype(np.int, order='F')
        elems = elems - elems.min()
        self._elems = elems
        self._elemtype = elemtype
        return True

    def text_nodes(self, filename: str = '..'):
        err_msg = 'wrong mat file name. '
        assert filename != '..', err_msg
        nodes = np.loadtxt(filename)
        err_msg = 'nodes is a n*3 numpy array containing x, y and z coordinates. '
        assert nodes.shape[1] == 3, err_msg
        self._nodes = np.asfortranarray(nodes)
        self._u = np.zeros(self._nodes.size)
        self.set_dmda()
        return True

    def mat_origin(self, filename: str = '..',
                   mat_handle: str = 'origin'):
        err_msg = 'wrong mat file name. '
        assert filename != '..', err_msg

        mat_contents = loadmat(filename)
        self._origin = mat_contents[mat_handle].astype(np.float)
        return True

    def mat_velocity(self, filename: str = '..',
                     mat_handle: str = 'U'):
        err_msg = 'wrong mat file name. '
        assert filename != '..', err_msg

        mat_contents = loadmat(filename)
        self._u = mat_contents[mat_handle].flatten()
        return True

    def node_rotation(self, norm=np.array([0, 0, 1]), theta=0, rotation_origin=None):
        # The rotation is counterclockwise
        if rotation_origin is None:
            rotation_origin = self.get_origin()
        else:
            rotation_origin = np.array(rotation_origin).reshape((3,))

        rotation = get_rot_matrix(norm, theta)
        self._nodes = np.dot(rotation, (self._nodes - rotation_origin).T).T + \
                      rotation_origin  # The rotation is counterclockwise
        t_origin = self._origin
        self._origin = np.dot(rotation, (self._origin - rotation_origin)) + rotation_origin
        self._geo_norm = np.dot(rotation, (self._geo_norm + t_origin - rotation_origin)) \
                         + rotation_origin - self._origin
        return True

    def coord_rotation(self, norm=np.array([0, 0, 1]), theta=0):
        # TODO: check the direction.
        assert 1 == 2
        # theta = -theta # The rotation is counterclockwise
        rotation = get_rot_matrix(norm, theta)

        temp_u = self._u.reshape((3, -1), order='F')
        self._u = rotation.dot(temp_u).T.flatten()
        self._nodes = np.dot(rotation, self._nodes.T).T
        self._origin = 000
        self._geo_norm = 000
        return True

    def node_zoom(self, factor, zoom_origin=None):
        if zoom_origin is None:
            zoom_origin = self.get_origin()
        self._nodes = (self._nodes - zoom_origin) * factor + zoom_origin
        return True

    def node_zoom_x(self, factor, zoom_origin=None):
        if zoom_origin is None:
            zoom_origin = self.get_origin()
        self._nodes[:, 0] = (self._nodes[:, 0] - zoom_origin[0]) * factor + zoom_origin[0]
        return True

    def node_zoom_y(self, factor, zoom_origin=None):
        if zoom_origin is None:
            zoom_origin = self.get_origin()
        self._nodes[:, 1] = (self._nodes[:, 1] - zoom_origin[1]) * factor + zoom_origin[1]
        return True

    def node_zoom_z(self, factor, zoom_origin=None):
        if zoom_origin is None:
            zoom_origin = self.get_origin()
        self._nodes[:, 2] = (self._nodes[:, 2] - zoom_origin[2]) * factor + zoom_origin[2]
        return True

    def get_nodes(self):
        return self._nodes

    def get_nodes_petsc(self):
        nodes_petsc = self.get_dmda().createGlobalVector()
        nodes_petsc[:] = self._nodes.reshape((3, -1))[:]
        nodes_petsc.assemble()
        return nodes_petsc

    def set_nodes(self, nodes, deltalength, resetVelocity=False):
        nodes = np.array(nodes).reshape((-1, 3))
        self._nodes = nodes
        self._deltaLength = deltalength
        self.set_dmda()

        if resetVelocity:
            self._u = np.zeros(self._nodes.size)
        return True

    def get_nodes_x(self):
        return self._nodes[:, 0]

    def get_nodes_y(self):
        return self._nodes[:, 1]

    def get_nodes_z(self):
        return self._nodes[:, 2]

    def get_nodes_x_petsc(self):
        x_petsc = self.get_dmda().createGlobalVector()
        t_x = np.matlib.repmat(self._nodes[:, 0].reshape((-1, 1)), 1, 3).flatten()
        x_petsc[:] = t_x[:]
        x_petsc.assemble()
        return x_petsc

    def get_nodes_y_petsc(self):
        y_petsc = self.get_dmda().createGlobalVector()
        t_y = np.matlib.repmat(self._nodes[:, 1].reshape((-1, 1)), 1, 3).flatten()
        y_petsc[:] = t_y[:]
        y_petsc.assemble()
        return y_petsc

    def get_nodes_z_petsc(self):
        z_petsc = self.get_dmda().createGlobalVector()
        t_z = np.matlib.repmat(self._nodes[:, 2].reshape((-1, 1)), 1, 3).flatten()
        z_petsc[:] = t_z[:]
        z_petsc.assemble()
        return z_petsc

    def get_n_nodes(self):
        return self._nodes.shape[0]

    def get_n_velocity(self):
        return self._u.size

    def get_velocity(self):
        return self._u.flatten()

    def set_velocity(self, velocity):
        err_msg = 'set nodes first. '
        assert self._nodes.size != 0, err_msg

        err_msg = 'velocity is a numpy array having a similar size of nodes. '
        assert velocity.size == self._nodes.size, err_msg
        self._u = velocity.flatten()
        return True

    def set_rigid_velocity(self, U=np.array((0, 0, 0, 0, 0, 0)), center=None):
        """
        :type U: np.array
        :param U: [u1, u2, u3, w1, w2, w3], velocity and angular velocity.
        :type center: np.array
        :param center: rotation center.
        """
        if center is None:
            center = self._origin
        center = np.array(center)
        err_msg = 'center is a np.array containing 3 scales. '
        assert center.size == 3, err_msg

        r = self._nodes - center
        self._u = np.zeros(self._nodes.size)
        self._u[0::3] = U[0] + U[4] * r[:, 2] - U[5] * r[:, 1]
        self._u[1::3] = U[1] + U[5] * r[:, 0] - U[3] * r[:, 2]
        self._u[2::3] = U[2] + U[3] * r[:, 1] - U[4] * r[:, 0]
        return True

    def get_velocity_x(self):
        return self._u[0::3].flatten()

    def get_velocity_y(self):
        return self._u[1::3].flatten()

    def get_velocity_z(self):
        return self._u[2::3].flatten()

    def get_polar_coord(self):
        phi = np.arctan2(self.get_nodes_y(), self.get_nodes_x())
        rho = np.sqrt(self.get_nodes_x() ** 2 + self.get_nodes_y() ** 2)
        z = self.get_nodes_z()
        return phi, rho, z

    def get_normal(self):
        return self._normal

    def set_normal(self, normal):
        self._normal = normal
        return True

    def get_geo_norm(self):
        return self._geo_norm

    def get_origin(self):
        return self._origin

    def get_center(self):
        return self.get_origin()

    def set_origin(self, origin):
        self._origin = origin
        return True

    def set_center(self, origin):
        return self.set_origin(origin=origin)

    def get_deltaLength(self):
        return self._deltaLength

    def set_deltaLength(self, deltaLength):
        self._deltaLength = deltaLength
        return True

    def copy(self) -> 'geo':
        self.destroy_dmda()
        geo2 = copy.deepcopy(self)
        self.set_dmda()
        geo2.set_dmda()
        return geo2

    def move(self, displacement: np.array):
        displacement = np.array(displacement).reshape((3,))

        self.set_nodes(self.get_nodes() + displacement, self.get_deltaLength())
        self.set_origin(self.get_origin() + displacement)
        return True

    def combine(self, geo_list, deltaLength=None):
        if len(geo_list) == 0:
            return False
        for geo1 in geo_list:
            err_msg = 'some objects in geo_list are not geo object. '
            assert isinstance(geo1, geo), err_msg
            err_msg = 'one or more objects not finished create yet. '
            assert geo1.get_n_nodes() != 0, err_msg
        if deltaLength is None:
            deltaLength = geo_list[0].get_deltaLength()

        geo1 = geo_list.pop(0)
        self.set_nodes(geo1.get_nodes(), deltalength=deltaLength)
        self.set_velocity(geo1.get_velocity())
        for geo1 in geo_list:
            self.set_nodes(np.vstack((self.get_nodes(), geo1.get_nodes())), deltalength=deltaLength)
            self.set_velocity(np.hstack((self.get_velocity(), geo1.get_velocity())))
        self.set_dmda()
        return True

    def save_nodes(self, filename):
        comm = PETSc.COMM_WORLD.tompi4py()
        rank = comm.Get_rank()
        filename = check_file_extension(filename, extension='.mat')
        if rank == 0:
            savemat(filename,
                    {'nodes': self.get_nodes()},
                    oned_as='column')
        return True

    def _show_velocity(self, length_factor=1, show_nodes=True):
        comm = PETSc.COMM_WORLD.tompi4py()
        rank = comm.Get_rank()
        if rank == 0:
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.set_aspect('equal')
            # Be carefull. the axis using in matplotlib is a left-handed coordinate system
            if show_nodes:
                ax.plot(self.get_nodes_x(), self.get_nodes_y(), self.get_nodes_z(), linestyle='None', c='b',
                        marker='o')
            INDEX = np.zeros_like(self.get_nodes_z(), dtype=bool)
            INDEX[:] = True
            length = 1 / np.mean(self._deltaLength) * length_factor
            ax.quiver(self.get_nodes_x()[INDEX], self.get_nodes_y()[INDEX], self.get_nodes_z()[INDEX],
                      self.get_velocity_x()[INDEX], self.get_velocity_y()[INDEX], self.get_velocity_z()[INDEX],
                      color='r', length=length)
            # ax.quiver(self.get_nodes_x(), self.get_nodes_y(), self.get_nodes_z(),
            #           0, 0, self.get_nodes_z(), length=self._deltaLength * 2)

            X = np.hstack((self.get_nodes_x()))
            Y = np.hstack((self.get_nodes_y()))
            Z = np.hstack((self.get_nodes_z()))
            max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max() / 2.0
            mid_x = (X.max() + X.min()) * 0.5
            mid_y = (Y.max() + Y.min()) * 0.5
            mid_z = (Z.max() + Z.min()) * 0.5
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
            ax.set_xlabel('x', size='xx-large')
            ax.set_ylabel('y', size='xx-large')
            ax.set_zlabel('z', size='xx-large')
        else:
            fig = None
        return fig

    def show_velocity(self, length_factor=1, show_nodes=True):
        comm = PETSc.COMM_WORLD.tompi4py()
        rank = comm.Get_rank()
        self._show_velocity(length_factor=length_factor, show_nodes=show_nodes)
        if rank == 0:
            plt.grid()
            plt.get_current_fig_manager().window.showMaximized()
            plt.show()
        return True

    def core_show_nodes(self, linestyle='-'):
        comm = PETSc.COMM_WORLD.tompi4py()
        rank = comm.Get_rank()
        if rank == 0:
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.set_aspect('equal')
            ax.plot(self.get_nodes_x(), self.get_nodes_y(), self.get_nodes_z(),
                    linestyle=linestyle,
                    color='b',
                    marker='.')

            X = np.hstack((self.get_nodes_x()))
            Y = np.hstack((self.get_nodes_y()))
            Z = np.hstack((self.get_nodes_z()))
            max_range = np.array([X.max() - X.min(),
                                  Y.max() - Y.min(),
                                  Z.max() - Z.min()]).max() / 2.0
            mid_x = (X.max() + X.min()) * 0.5
            mid_y = (Y.max() + Y.min()) * 0.5
            mid_z = (Z.max() + Z.min()) * 0.5
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
            ax.set_xlabel('x', size='xx-large')
            ax.set_ylabel('y', size='xx-large')
            ax.set_zlabel('z', size='xx-large')
        else:
            fig = None
        return fig

    def show_nodes(self, linestyle='-'):
        comm = PETSc.COMM_WORLD.tompi4py()
        rank = comm.Get_rank()
        self.core_show_nodes(linestyle=linestyle)
        if rank == 0:
            plt.grid()
            plt.get_current_fig_manager().window.showMaximized()
            plt.show()
        return True

    def png_nodes(self, finename, linestyle='-'):
        comm = PETSc.COMM_WORLD.tompi4py()
        rank = comm.Get_rank()
        finename = check_file_extension(finename, '.png')

        fig = self.core_show_nodes(linestyle=linestyle)
        if rank == 0:
            fig.set_size_inches(18.5, 10.5)
            fig.savefig(finename, dpi=100)
            plt.close()
        return True

    def get_mesh(self):
        return self._elems, self._elemtype

    def get_dmda(self):
        return self._dmda

    def set_dmda(self):
        if self.get_dmda() is not None:
            self._dmda.destroy()
        if not hasattr(self, '_dof'):
            self._dof = 3
        self._dmda = PETSc.DMDA().create(sizes=(self.get_n_nodes(),), dof=self._dof,
                                         stencil_width=self._stencil_width, comm=PETSc.COMM_WORLD)
        self._dmda.setFromOptions()
        self._dmda.setUp()
        # self._dmda.createGlobalVector()
        return True

    def destroy_dmda(self):
        self._dmda.destroy()
        self._dmda = None
        return True

    def get_dof(self):
        return self._dof

    def set_dof(self, dof):
        self._dof = dof
        return True

    def set_glbIdx(self, indices):
        comm = PETSc.COMM_WORLD.tompi4py()
        self._glbIdx = indices
        self._glbIdx_all = np.hstack(comm.allgather(indices))
        self._selfIdx = np.searchsorted(self._glbIdx_all, self._glbIdx)
        return True

    def set_glbIdx_all(self, indices):
        self._glbIdx = []
        self._selfIdx = []
        self._glbIdx_all = indices
        return True

    def get_glbIdx(self):
        return self._glbIdx, self._glbIdx_all

    def get_selfIdx(self):
        return self._selfIdx

        # def _heaviside(self, n, factor):
        #     f = lambda x: 1 / (1 + np.exp(-factor * x))
        #     x = np.linspace(-0.5, 0.5, n)
        #     return (f(x) - f(-0.5)) / (f(0.5) - f(-0.5))


class geoComposit(uniqueList):
    def __init__(self, geo_list=[]):
        acceptType = geo
        super().__init__(acceptType)
        geo_list = tube_flatten((geo_list,))
        for geoi in geo_list:
            self.append(geoi)

    def core_show_nodes(self, linestyle='-'):
        color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k', ]
        comm = PETSc.COMM_WORLD.tompi4py()
        rank = comm.Get_rank()
        if len(self) == 0:
            return False
        if rank == 0:
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.set_aspect('equal')
            xlim_list = np.zeros((len(self), 2))
            ylim_list = np.zeros((len(self), 2))
            zlim_list = np.zeros((len(self), 2))
            for i0, geo0 in enumerate(self):
                if geo0.get_n_nodes() > 0:
                    ax.plot(geo0.get_nodes_x(), geo0.get_nodes_y(), geo0.get_nodes_z(),
                            linestyle=linestyle,
                            color=color_list[i0 % len(color_list)],
                            marker='.')

                    X = np.hstack((geo0.get_nodes_x()))
                    Y = np.hstack((geo0.get_nodes_y()))
                    Z = np.hstack((geo0.get_nodes_z()))
                    max_range = np.array([X.max() - X.min(),
                                          Y.max() - Y.min(),
                                          Z.max() - Z.min()]).max() / 2.0
                    mid_x = (X.max() + X.min()) * 0.5
                    mid_y = (Y.max() + Y.min()) * 0.5
                    mid_z = (Z.max() + Z.min()) * 0.5
                    xlim_list[i0] = (mid_x - max_range, mid_x + max_range)
                    ylim_list[i0] = (mid_y - max_range, mid_y + max_range)
                    zlim_list[i0] = (mid_z - max_range, mid_z + max_range)
                else:
                    xlim_list[i0] = (np.nan, np.nan)
                    ylim_list[i0] = (np.nan, np.nan)
                    zlim_list[i0] = (np.nan, np.nan)
            ax.set_xlim(np.nanmin(xlim_list), np.nanmax(xlim_list))
            ax.set_ylim(np.nanmin(ylim_list), np.nanmax(ylim_list))
            ax.set_zlim(np.nanmin(zlim_list), np.nanmax(zlim_list))
            ax.set_xlabel('x', size='xx-large')
            ax.set_ylabel('y', size='xx-large')
            ax.set_zlabel('z', size='xx-large')
        else:
            fig = None
        return fig

    def show_nodes(self, linestyle='-'):
        if len(self) == 0:
            return False

        comm = PETSc.COMM_WORLD.tompi4py()
        rank = comm.Get_rank()
        self.core_show_nodes(linestyle=linestyle)
        if rank == 0:
            plt.grid()
            plt.get_current_fig_manager().window.showMaximized()
            plt.show()
        return True

    def move(self, displacement: np.array):
        if len(self) == 0:
            return False
        else:
            for sub_geo in self:
                sub_geo.move(displacement=displacement)
        return True


class _ThickLine_geo(geo):
    def __init__(self):
        super().__init__()
        self._r = 0  # radius of thick line itself, thick is a cycle.
        self._dth = 0  # anglar between nodes in a cycle.
        self._angleCycle = np.array([])
        self._frenetFrame = (
            np.array([]).reshape((-1, 3)), np.array([]).reshape((-1, 3)), np.array([]).reshape((-1, 3)))
        self._iscover = np.array([]).reshape((-1, 3))  # start: -1, body: 0, end: 1
        self._factor = 1e-5
        self._left_hand = False
        self._check_epsilon = True

    def set_check_epsilon(self, check_epsilon):
        self._check_epsilon = check_epsilon
        return True

    def get_check_epsilon(self):
        return self._check_epsilon

    def _get_theta(self):
        def eqr(dth, ds, r):
            return (ds / (2 * r)) ^ 2 + np.sin(dth / 4) ** 2 - np.sin(dth / 2) ** 2

        from scipy import optimize as sop
        self._dth = sop.brentq(eqr, -1e-3 * np.pi, np.pi, args=(self.get_deltaLength(), self._r))
        return self._dth

    def _get_deltalength(self):
        # dl = 2 * self._r * np.sqrt(np.sin(self._dth / 2) ** 2 - np.sin(self._dth / 4) ** 2)
        dl = 2 * self._r * np.sin(self._dth / 2)
        self.set_deltaLength(dl)
        return dl

    @abc.abstractmethod
    def _get_axis(self):
        return

    @abc.abstractmethod
    def _get_fgeo_axis(self, epsilon):
        return

    def _body_pretreatment(self, nodes, **kwargs):
        return True

    def _strat_pretreatment(self, nodes, **kwargs):
        return True

    def _end_pretreatment(self, nodes, **kwargs):
        return True

    def _create_deltatheta(self, dth: float,  # delta theta of the cycle for the mesh
                           radius: float,  # radius of the cycle
                           epsilon=0,
                           with_cover=0):
        # the tunnel is along z axis
        err_msg = 'dth must less than pi'
        assert dth < np.pi, err_msg
        self._dth = dth
        self._r = radius
        deltalength = self._get_deltalength()
        nc = np.ceil(2 * np.pi / dth).astype(int)
        angleCycle = np.linspace(0, 2 * np.pi, nc, endpoint=False)
        axisNodes, T_frame, N_frame, B_frame = self._get_axis()
        fgeo_axisNodes, fgeo_T_frame, fgeo_N_frame, fgeo_B_frame = self._get_fgeo_axis(epsilon)
        iscover = []
        vgeo_nodes = []
        fgeo_nodes = []
        epsilon = (radius + epsilon * deltalength) / radius
        if self.get_check_epsilon():
            err_msg = 'epsilon > %f. ' % (-radius / deltalength)
            assert epsilon > 0, err_msg
        ai_para = 0

        # cover at start
        if with_cover == 1:
            # old version, cover is a plate.
            nc = np.ceil((radius - deltalength) / deltalength).astype(int)
            ri = np.linspace(deltalength / 2, radius, nc, endpoint=False)
            # self
            for i0 in np.arange(0, nc):
                ai_para = ai_para + 1
                ni = np.ceil(2 * np.pi * ri[i0] / deltalength).astype(int)
                ai = np.linspace(0, 2 * np.pi, ni, endpoint=False) + (-1) ** ai_para * dth / 4
                t_cover = np.ones_like(ai, dtype=bool)
                t_cover[:] = -1
                iscover.append(t_cover)
                nodes_cycle = np.vstack((np.cos(ai) * ri[i0], np.sin(ai) * ri[i0], np.zeros_like(ai))).T
                t_nodes = axisNodes[0] + np.dot(nodes_cycle,
                                                np.vstack((N_frame[0], B_frame[0], np.zeros_like(T_frame[0]))))
                vgeo_nodes.append(t_nodes)
                tf_nodes = fgeo_axisNodes[0] + np.dot(nodes_cycle * epsilon,
                                                      np.vstack((N_frame[0], B_frame[0], np.zeros_like(T_frame[0]))))
                fgeo_nodes.append(tf_nodes)
                self._strat_pretreatment(t_nodes)
        elif with_cover == 2:
            # 20170929, new version, cover is a hemisphere
            vhsgeo = sphere_geo()
            vhsgeo.create_half_delta(deltalength, radius)
            vhsgeo.node_rotation((1, 0, 0), np.pi / 2 + ai_para)
            t_nodes = axisNodes[0] + np.dot(vhsgeo.get_nodes(),
                                            np.vstack((-T_frame[0], N_frame[0], B_frame[0])))
            vgeo_nodes.append(t_nodes)
            fhsgeo = vhsgeo.copy()
            # fhsgeo.show_nodes()
            fhsgeo.node_zoom(epsilon)
            # fhsgeo.show_nodes()
            tf_nodes = fgeo_axisNodes[0] + np.dot(fhsgeo.get_nodes(),
                                                  np.vstack((-T_frame[0], N_frame[0], B_frame[0])))
            fgeo_nodes.append(tf_nodes)
            self._strat_pretreatment(t_nodes)
            iscover.append(np.ones(vhsgeo.get_n_nodes(), dtype=bool))

        # body
        for i0, nodei_line in enumerate(axisNodes):
            ai_para = ai_para + 1
            ai = angleCycle + (-1) ** ai_para * dth / 4
            nodes_cycle = np.vstack((np.cos(ai) * radius, np.sin(ai) * radius, np.zeros_like(ai))).T
            t_nodes = nodei_line + np.dot(nodes_cycle,
                                          np.vstack((N_frame[i0], B_frame[i0], np.zeros_like(T_frame[i0]))))
            vgeo_nodes.append(t_nodes)
            t_cover = np.ones_like(ai, dtype=bool)
            t_cover[:] = 0
            iscover.append(t_cover)
            nodes_cycle = np.vstack((np.cos(ai) * radius, np.sin(ai) * radius, np.zeros_like(ai))).T * epsilon
            tf_nodes = fgeo_axisNodes[i0] + np.dot(nodes_cycle, np.vstack(
                    (fgeo_N_frame[i0], fgeo_B_frame[i0], np.zeros_like(fgeo_T_frame[i0]))))
            fgeo_nodes.append(tf_nodes)
            self._body_pretreatment(t_nodes)

        # cover at end
        if with_cover == 1:
            # old version, cover is a plate.
            nc = np.ceil((radius - deltalength) / deltalength).astype(int)
            ri = np.linspace(deltalength / 2, radius, nc, endpoint=False)[-1::-1]
            for i0 in np.arange(0, nc):
                ai_para = ai_para + 1
                ni = np.ceil(2 * np.pi * ri[i0] / deltalength).astype(int)
                ai = np.linspace(0, 2 * np.pi, ni, endpoint=False) + (-1) ** ai_para * dth / 4
                t_cover = np.ones_like(ai, dtype=bool)
                t_cover[:] = 1
                iscover.append(t_cover)
                nodes_cycle = np.vstack((np.cos(ai) * ri[i0], np.sin(ai) * ri[i0], np.zeros_like(ai))).T
                t_nodes = axisNodes[-1] + np.dot(nodes_cycle,
                                                 np.vstack((N_frame[-1], B_frame[-1], np.zeros_like(T_frame[-1]))))
                vgeo_nodes.append(t_nodes)
                tf_nodes = fgeo_axisNodes[-1] + np.dot(nodes_cycle * epsilon, np.vstack(
                        (fgeo_N_frame[-1], fgeo_B_frame[-1], np.zeros_like(fgeo_T_frame[-1]))))
                fgeo_nodes.append(tf_nodes)
                self._end_pretreatment(t_nodes)
        elif with_cover == 2:
            # 20170929, new version, cover is a hemisphere
            vhsgeo = sphere_geo()
            vhsgeo.create_half_delta(deltalength, radius)
            vhsgeo.node_rotation((1, 0, 0), -np.pi / 2 - ai_para)
            t_nodes = axisNodes[-1] + np.dot(vhsgeo.get_nodes(),
                                             np.vstack((T_frame[-1], N_frame[-1], B_frame[-1])))
            vgeo_nodes.append(np.flipud(t_nodes))
            fhsgeo = vhsgeo.copy()
            fhsgeo.node_zoom(epsilon)
            tf_nodes = fgeo_axisNodes[-1] + np.dot(fhsgeo.get_nodes(),
                                                   np.vstack((T_frame[-1], N_frame[-1], B_frame[-1])))
            fgeo_nodes.append(np.flipud(tf_nodes))
            self._end_pretreatment(t_nodes)
            iscover.append(np.ones(vhsgeo.get_n_nodes(), dtype=bool))

        self._iscover = np.hstack(iscover)
        self._nodes = np.asfortranarray(np.vstack(vgeo_nodes))
        self.set_dmda()
        self._u = np.zeros(self._nodes.size)
        self._normal = np.zeros((self._nodes.shape[0], 2), order='F')
        fgeo = geo()
        fgeo.set_dof(self.get_dof())
        fgeo.set_nodes(np.asfortranarray(np.vstack(fgeo_nodes)), deltalength=deltalength * epsilon, resetVelocity=True)
        return fgeo

    def get_iscover(self):
        return self._iscover

    def _factor_fun(self, n, factor):
        err_msg = 'factor must positive'
        assert factor > 0, err_msg

        if np.abs(factor - 1) < 0.01:
            y = np.linspace(0, 1, n)
        else:
            f1 = lambda x: (np.exp(x * factor) - 1) / (2 * (np.exp(0.5 * factor) - 1))
            f2 = lambda x: np.log(2 * (np.exp(0.5 / factor) - 1) * x + 1) * factor
            x = np.linspace(-0.5, 0.5, n)
            y1 = np.sign(x) * f1(np.abs(x)) + 0.5
            y2 = np.sign(x) * f2(np.abs(x)) + 0.5
            y = (y1 * factor + y2 / factor) / (y1[-1] * factor + y2[-1] / factor)
        return y


class ellipse_geo(geo):
    def create_n(self, n: int,  # number of nodes.
                 headA: float,  # major axis = 2*headA
                 headC: float):  # minor axis = 2*headC
        err_msg = 'both major and minor axises should positive. '
        assert headA > 0 and headC > 0, err_msg

        jj = np.arange(n)
        xlocH = -1 + 2 * jj / (n - 1)
        numf = 0.5

        prefac = 3.6 * np.sqrt(headC / headA)
        spherePhi = np.ones(n)
        for i0 in range(0, n):
            if i0 == 0 or i0 == n - 1:
                spherePhi[i0] = 0
            else:
                tr = np.sqrt(1 - xlocH[i0] ** 2)
                wgt = prefac * (1 - numf * (1 - tr)) / tr
                spherePhi[i0] = (spherePhi[i0 - 1] + wgt / np.sqrt(n)) % (2 * np.pi)

        tsin = np.sqrt(1 - xlocH ** 2)
        self._nodes = np.zeros((n, 3), order='F')
        self._nodes[:, 0] = headC * xlocH
        self._nodes[:, 1] = headA * tsin * np.cos(spherePhi)
        self._nodes[:, 2] = headA * tsin * np.sin(spherePhi)
        self.set_dmda()

        self._u = np.zeros(self._nodes.size)
        self._normal = np.zeros((n, 2), order='F')
        return True

    def create_delta(self, ds: float,  # length of the mesh
                     a: float,  # axis1 = 2*a
                     b: float):  # axis2 = 2*b
        err_msg = 'both major and minor axises should positive. '
        assert a > 0 and b > 0, err_msg
        self._deltaLength = ds

        # fit arc length as function F of theta using 2-degree pylonomial
        from scipy.special import ellipeinc
        from scipy.optimize import curve_fit
        func = lambda theta, a, b: a * theta ** 2 + b * theta

        theta = np.linspace(0, np.pi / 2, 100)
        arcl = b * ellipeinc(theta, 1 - (a / b) ** 2)
        popt, _ = curve_fit(func, theta, arcl)
        # # dbg
        # plt.plot(theta, arcl, '.')
        # plt.plot(theta, func(theta, popt[0], popt[1]))
        # plt.show()
        # assert 1 == 2

        # divided arc length equally, and get theta using F^-1.
        n = np.ceil(arcl[-1] / ds).astype(int)
        t_arcl = np.linspace(0, arcl[-1], n, endpoint=False) + ds / 2
        # do something to correct the fitting error.
        while t_arcl[-1] > arcl[-1]:
            t_arcl = t_arcl[:-1]
        t_theta1 = (-popt[1] + np.sqrt(popt[1] ** 2 + 4 * popt[0] * t_arcl)) / (2 * popt[0])
        t_theta2 = (-popt[1] - np.sqrt(popt[1] ** 2 + 4 * popt[0] * t_arcl)) / (2 * popt[0])
        b_theta1 = [a and b for a, b in zip(t_theta1 > 0, t_theta1 < np.pi / 2)]
        b_theta2 = [a and b for a, b in zip(t_theta2 > 0, t_theta2 < np.pi / 2)]
        err_msg = 'something is wrong, theta of ellipse is uncertain. '
        assert all([a != b for a, b in zip(b_theta1, b_theta2)]), err_msg
        t_theta0 = t_theta1 * b_theta1 + t_theta2 * b_theta2
        t_theta = np.hstack((t_theta0, np.pi / 2, np.pi - t_theta0[::-1]))
        t_x = a * np.cos(t_theta)
        t_y = b * np.sin(t_theta)

        # generate nodes.
        x = []
        y = []
        z = []
        ai_para = 0
        for xi, yi in zip(t_x, t_y):
            ai_para = ai_para + 1
            ni = np.ceil(2 * np.pi * yi / ds).astype(int)
            ai, da = np.linspace(0, 2 * np.pi, ni, endpoint=False, retstep=True)
            ai = ai + (-1) ** ai_para * da / 4 + np.sign(xi) * np.pi / 2
            x.append(xi * np.ones_like(ai))
            y.append(np.sign(xi) * yi * np.cos(ai))
            z.append(np.sign(xi) * yi * np.sin(ai))
        self._nodes = np.vstack((np.hstack(x), np.hstack(y), np.hstack(z))).T
        self.set_dmda()
        self._u = np.zeros(self._nodes.size)
        self._normal = np.zeros((self._nodes.shape[0], 2), order='F')
        return True

    def create_half_delta(self, ds: float,  # length of the mesh
                          a: float,  # axis1 = 2*a
                          b: float):  # axis2 = 2*b
        err_msg = 'both major and minor axises should positive. '
        assert a > 0 and b > 0, err_msg
        self._deltaLength = ds

        # fit arc length as function F of theta using 2-degree pylonomial
        from scipy.special import ellipeinc
        from scipy.optimize import curve_fit
        func = lambda theta, a, b: a * theta ** 2 + b * theta

        theta = np.linspace(0, np.pi / 2, 100)
        arcl = b * ellipeinc(theta, 1 - (a / b) ** 2)
        popt, _ = curve_fit(func, theta, arcl)
        # # dbg
        # plt.plot(theta, arcl, '.')
        # plt.plot(theta, func(theta, popt[0], popt[1]))
        # plt.show()
        # assert 1 == 2

        # divided arc length equally, and get theta using F^-1.
        n = np.ceil(arcl[-1] / ds).astype(int)
        t_arcl = np.linspace(0, arcl[-1], n, endpoint=False) + ds / 2
        # do something to correct the fitting error.
        while t_arcl[-1] > arcl[-1]:
            t_arcl = t_arcl[:-1]
        t_theta1 = (-popt[1] + np.sqrt(popt[1] ** 2 + 4 * popt[0] * t_arcl)) / (2 * popt[0])
        t_theta2 = (-popt[1] - np.sqrt(popt[1] ** 2 + 4 * popt[0] * t_arcl)) / (2 * popt[0])
        b_theta1 = [a and b for a, b in zip(t_theta1 > 0, t_theta1 < np.pi / 2)]
        b_theta2 = [a and b for a, b in zip(t_theta2 > 0, t_theta2 < np.pi / 2)]
        err_msg = 'something is wrong, theta of ellipse is uncertain. '
        assert all([a != b for a, b in zip(b_theta1, b_theta2)]), err_msg
        t_theta0 = t_theta1 * b_theta1 + t_theta2 * b_theta2
        t_x = a * np.cos(t_theta0)
        t_y = b * np.sin(t_theta0)

        # generate nodes.
        x = []
        y = []
        z = []
        ai_para = 0
        for xi, yi in zip(t_x, t_y):
            ai_para = ai_para + 1
            ni = np.ceil(2 * np.pi * yi / ds).astype(int)
            ai, da = np.linspace(0, 2 * np.pi, ni, endpoint=False, retstep=True)
            ai = ai + (-1) ** ai_para * da / 4 + np.sign(xi) * np.pi / 2
            x.append(xi * np.ones_like(ai))
            y.append(np.sign(xi) * yi * np.cos(ai))
            z.append(np.sign(xi) * yi * np.sin(ai))
        self._nodes = np.vstack((np.hstack(x), np.hstack(y), np.hstack(z))).T
        self.set_dmda()
        self._u = np.zeros(self._nodes.size)
        self._normal = np.zeros((self._nodes.shape[0], 2), order='F')
        return True


class sphere_geo(ellipse_geo):
    def create_n(self, n: int,  # number of nodes.
                 radius: float, *args):  # radius
        err_msg = 'additional parameters are useless.  '
        assert not args, err_msg
        self._deltaLength = np.sqrt(4 * np.pi * radius * radius / n)
        return super().create_n(n, radius, radius)

    def create_delta(self, deltaLength: float,  # length of the mesh
                     radius: float, *args):  # radius
        err_msg = 'additional parameters are useless.  '
        assert not args, err_msg
        return super().create_delta(deltaLength, radius, radius)

    def create_half_delta(self, ds: float,  # length of the mesh
                          a: float, *args):
        err_msg = 'additional parameters are useless.  '
        assert not args, err_msg
        return super().create_half_delta(ds, a, a)

    def normal(self):
        self._normal = np.zeros((self._nodes.shape[0],
                                 2))  # {Sin[a] Sin[b], -Cos[a] Sin[b], Cos[b]} = {n1, n2, n3} is the normal vector
        normal_vector = self._nodes / np.sqrt(
                self._nodes[:, 0] ** 2 + self._nodes[:, 1] ** 2 + self._nodes[:, 2] ** 2).reshape(self._nodes.shape[0],
                                                                                                  1)
        self._normal[:, 1] = np.arccos(normal_vector[:, 2])  # b
        self._normal[:, 0] = np.arcsin(normal_vector[:, 0] / np.sin(self._normal[:, 1]))  # a
        return True


# noinspection PyUnresolvedReferences
class tunnel_geo(_ThickLine_geo):
    def __init__(self):
        super().__init__()
        self._length = 0
        self._cover_strat_list = []
        self._cover_end_list = []

    def create_n(self, n: int,  # number of nodes.
                 length: float,  # length of the tunnel
                 radius: float):  # radius of the tunnel
        deltaLength = np.sqrt(2 * np.pi * radius * length / n)
        self._deltaLength = deltaLength
        deltaTheta = deltaLength / radius

        # the geo is symmetrical
        if n % 2:  # if n is odd
            n_half = int((n - 1) / 2)
            theta = np.arange(-n_half, n_half + 1) * deltaTheta
        else:  # if n is even
            n_half = int(n / 2)
            theta = np.arange(-n_half, n_half) * deltaTheta + deltaTheta / 2
        self._nodes = np.zeros((n, 3), order='F')
        self._nodes[:, 0] = deltaLength * theta / 2 / np.pi
        self._nodes[:, 1] = radius * np.sin(theta)
        self._nodes[:, 2] = radius * np.cos(theta)
        self.set_dmda()

        self._u = np.zeros(self._nodes.size)
        self._normal = np.zeros((n, 2), order='F')
        self._geo_norm = np.array((1, 0, 0))
        return True

    def create_deltalength(self, deltaLength: float,  # length of the mesh
                           length: float,  # length of the tunnel
                           radius: float):  # radius of the tunnel
        # the tunnel is along z axis
        self._deltaLength = deltaLength
        a = np.arange(0, 2 * np.pi - deltaLength / radius / 2, deltaLength / radius)
        x, y = np.cos(a) * radius, np.sin(a) * radius
        z = np.linspace(-length / 2, length / 2, num=np.ceil((length / deltaLength)).astype(int))
        n_a, n_z = a.size, z.size

        self._nodes = np.zeros((n_a * n_z, 3), order='F')
        self._nodes[:, 0] = np.tile(z, n_a).reshape(n_a, -1).flatten(order='F')
        self._nodes[:, 1] = np.tile(x, (n_z, 1)).reshape(-1, 1).flatten(order='F')
        self._nodes[:, 2] = np.tile(y, (n_z, 1)).reshape(-1, 1).flatten(order='F')
        self.set_dmda()

        self._u = np.zeros(self._nodes.size)
        self._normal = np.zeros((self._nodes.shape[0], 2), order='F')
        self._geo_norm = np.array((0, 0, 1))
        return True

    def create_deltatheta(self, dth: float,  # delta theta of the cycle for the mesh
                          radius: float,
                          length: float,
                          epsilon=0,
                          with_cover=0,
                          factor=1,
                          left_hand=False):
        self._length = length
        self._factor = factor
        self._left_hand = left_hand
        self._geo_norm = np.array((0, 0, 1))
        return self._create_deltatheta(dth, radius, epsilon, with_cover)

    def _get_axis(self):
        length = self._length
        factor = self._factor
        left_hand = self._left_hand
        ds = self.get_deltaLength()
        nl = np.ceil(length / ds).astype(int)
        z = self._factor_fun(nl, factor) * length - length / 2
        self._axisNodes = np.vstack((np.zeros_like(z), np.zeros_like(z), z)).T
        if left_hand:
            T_frame = np.vstack((np.zeros(nl), np.zeros(nl), np.ones(nl))).T  # (0, 0, 1)
            N_frame = np.vstack((np.ones(nl), np.zeros(nl), np.zeros(nl))).T  # (1, 0, 0)
            B_frame = np.vstack((np.zeros(nl), np.ones(nl), np.zeros(nl))).T  # (0, 1, 0)
        else:
            T_frame = np.vstack((np.zeros(nl), np.zeros(nl), np.ones(nl))).T  # (0, 0, 1)
            N_frame = np.vstack((np.zeros(nl), np.ones(nl), np.zeros(nl))).T  # (0, 1, 0)
            B_frame = np.vstack((np.ones(nl), np.zeros(nl), np.zeros(nl))).T  # (1, 0, 0)
        self._frenetFrame = (T_frame, N_frame, B_frame)
        return self._axisNodes, self._frenetFrame[0], self._frenetFrame[1], self._frenetFrame[2]

    def _get_fgeo_axis(self, epsilon):
        length = self._length
        factor = self._factor
        nl = self._axisNodes.shape[0]
        ds = -self.get_deltaLength() * epsilon / 4
        z = self._factor_fun(nl, factor) * (length - ds * 2) - length / 2 + ds
        axisNodes = np.vstack((np.zeros_like(z), np.zeros_like(z), z)).T
        return axisNodes, self._frenetFrame[0], self._frenetFrame[1], self._frenetFrame[2]

    def _strat_pretreatment(self, nodes, **kwargs):
        def cart2pol(x, y):
            rho = np.sqrt(x ** 2 + y ** 2)
            phi = np.arctan2(y, x)
            return rho, phi

        r, ai = cart2pol(nodes[:, 0], nodes[:, 1])
        self._cover_strat_list.append((np.mean(r), ai, np.mean(nodes[:, 2])))
        return True

    def _end_pretreatment(self, nodes, **kwargs):
        def cart2pol(x, y):
            rho = np.sqrt(x ** 2 + y ** 2)
            phi = np.arctan2(y, x)
            return (rho, phi)

        r, ai = cart2pol(nodes[:, 0], nodes[:, 1])
        self._cover_end_list.append((np.mean(r), ai, np.mean(nodes[:, 2])))
        return True

    def get_cover_start_list(self):
        return self._cover_strat_list

    def get_cover_end_list(self):
        return self._cover_end_list

    def normal(self):
        self._normal = np.zeros((self._nodes.shape[0],
                                 2))  # {Sin[a] Sin[b], -Cos[a] Sin[b], Cos[b]} = {n1, n2, n3} is the normal vector
        normal_vector = -1 * self._nodes / np.sqrt(self._nodes[:, 1] ** 2 + self._nodes[:, 2] ** 2).reshape(
                self._nodes.shape[0], 1)  # -1 means swap direction
        self._normal[:, 1] = np.arccos(normal_vector[:, 2])  # b
        self._normal[:, 0] = 0  # a
        return True

    def node_zoom_radius(self, factor):
        def cart2pol(x, y):
            rho = np.sqrt(x ** 2 + y ** 2)
            phi = np.arctan2(y, x)
            return rho, phi

        def pol2cart(rho, phi):
            x = rho * np.cos(phi)
            y = rho * np.sin(phi)
            return x, y

        # zooming geo along radius of tunnel, keep longitude axis.
        # 1. copy
        temp_geo = geo()
        temp_nodes = self.get_nodes() - self.get_origin()
        temp_geo.set_nodes(temp_nodes, self.get_deltaLength())
        # temp_geo.show_nodes()
        # 2. rotation, tunnel center line along x axis.
        temp_norm = self._geo_norm
        rotation_norm = np.cross(temp_norm, [1, 0, 0])
        temp_theta = -np.arccos(temp_norm[0] / np.linalg.norm(temp_norm))
        doRotation = (not np.array_equal(rotation_norm, np.array((0, 0, 0)))) and temp_theta != 0.
        if doRotation:
            temp_geo.node_rotation(rotation_norm, temp_theta)
        # 3. zooming
        temp_nodes = temp_geo.get_nodes()
        temp_R, temp_phi = cart2pol(temp_nodes[:, 1], temp_nodes[:, 2])
        temp_R = temp_R * factor
        X1 = np.min(temp_nodes[:, 0])
        X2 = np.max(temp_nodes[:, 0])
        factor = (factor - 1) / 2 + 1
        temp_nodes[:, 0] = (temp_nodes[:, 0] - (X1 + X2) / 2) * factor + (X1 + X2) / 2
        temp_nodes[:, 1], temp_nodes[:, 2] = pol2cart(temp_R, temp_phi)
        temp_geo.set_nodes(temp_nodes, self.get_deltaLength())

        # 4. rotation back
        if doRotation:
            temp_geo.node_rotation(rotation_norm, -temp_theta)
        # 5. set
        # temp_geo.show_nodes()
        self.set_nodes(temp_geo.get_nodes() + self.get_origin(), self.get_deltaLength())
        return True


class pipe_cover_geo(tunnel_geo):
    def __init__(self):
        super().__init__()
        self._cover_node_list = uniqueList()

    def create_with_cover(self, deltaLength: float,  # length of the mesh
                          length: float,  # length of the tunnel
                          radius: float,  # radius of the tunnel
                          a_factor=1e-6,
                          z_factor=1e-6):
        # the tunnel is along z axis.
        self._deltaLength = deltaLength
        # pipe
        na = np.ceil(2 * np.pi * radius / deltaLength).astype(int)
        a = np.linspace(-1, 1, na, endpoint=False)
        a = (1 / (1 + np.exp(-a_factor * a)) - 1 / (1 + np.exp(a_factor))) / (
                1 / (1 + np.exp(-a_factor)) - 1 / (1 + np.exp(a_factor))) * 2 * np.pi
        nz = np.ceil(length / deltaLength).astype(int)
        nodes_z = np.linspace(1, -1, nz)
        nodes_z = np.sign(nodes_z) * (np.exp(np.abs(nodes_z) * z_factor) - 1) / (np.exp(z_factor) - 1) * length / 2
        a, nodes_z = np.meshgrid(a, nodes_z)
        a = a.flatten()
        nodes_z = nodes_z.flatten()
        nodes_x = np.cos(a) * radius
        nodes_y = np.sin(a) * radius

        iscover = np.ones_like(nodes_z, dtype=bool)
        iscover[:] = False

        # cover
        nc = np.ceil((radius - deltaLength) / deltaLength).astype(int) + 1
        ri = np.linspace(radius, deltaLength / 2, nc)[1:]
        cover_node_list = uniqueList()
        for i0 in range(0, int(nc - 2)):
            ni = np.ceil(2 * np.pi * ri[i0] / deltalength).astype(int)
            ai = np.linspace(0, 2 * np.pi, ni, endpoint=False)
            t_cover = np.ones_like(ai, dtype=bool)
            t_cover[:] = True

            # cover z>0
            nodes_z = np.hstack((length / 2 * np.ones(ai.shape), nodes_z))
            nodes_x = np.hstack((np.cos(ai) * ri[i0], nodes_x))
            nodes_y = np.hstack((np.sin(ai) * ri[i0], nodes_y))
            t_cover_nodes = (ri[i0], ai, length / 2)
            cover_node_list.append(t_cover_nodes)
            iscover = np.hstack((t_cover, iscover))
            # cover z<  0
            nodes_z = np.hstack((nodes_z, -length / 2 * np.ones(ai.shape)))
            nodes_x = np.hstack((nodes_x, np.cos(ai) * ri[i0]))
            nodes_y = np.hstack((nodes_y, np.sin(ai) * ri[i0]))
            iscover = np.hstack((iscover, t_cover))

        self._nodes = np.zeros((nodes_z.size, 3), order='F')
        self._nodes[:, 0] = nodes_x
        self._nodes[:, 1] = nodes_y
        self._nodes[:, 2] = nodes_z
        self.set_dmda()
        self._iscover = iscover
        self._u = np.zeros(self._nodes.size)
        self._normal = np.zeros((self._nodes.shape[0], 2), order='F')
        self._geo_norm = np.array((1, 0, 0))
        self._cover_node_list = cover_node_list
        return True

    def get_cover_node_list(self):
        return self._cover_node_list


class supHelix(_ThickLine_geo):
    _helix = lambda self, R, B, s: np.vstack((R * np.cos((B ** 2 + R ** 2) ** (-1 / 2) * s),
                                              R * np.sin((B ** 2 + R ** 2) ** (-1 / 2) * s),
                                              B * (B ** 2 + R ** 2) ** (-1 / 2) * s)).T
    _helix_left_hand = lambda self, R, B, s: np.vstack((R * np.sin((B ** 2 + R ** 2) ** (-1 / 2) * s),
                                                        R * np.cos((B ** 2 + R ** 2) ** (-1 / 2) * s),
                                                        B * (B ** 2 + R ** 2) ** (-1 / 2) * s)).T
    _T_frame = lambda self, R, B, s: np.vstack(
            ((-1) * R * (B ** 2 + R ** 2) ** (-1 / 2) * np.sin((B ** 2 + R ** 2) ** (-1 / 2) * s),
             R * (B ** 2 + R ** 2) ** (-1 / 2) * np.cos((B ** 2 + R ** 2) ** (-1 / 2) * s),
             B * (B ** 2 + R ** 2) ** (-1 / 2) * np.ones_like(s))).T
    _N_frame = lambda self, R, B, s: np.vstack(
            ((-1) * np.cos((B ** 2 + R ** 2) ** (-1 / 2) * s),
             (-1) * np.sin((B ** 2 + R ** 2) ** (-1 / 2) * s),
             np.zeros_like(s))).T
    _B_frame = lambda self, R, B, s: np.vstack(
            (B * (B ** 2 + R ** 2) ** (-1 / 2) * np.sin((B ** 2 + R ** 2) ** (-1 / 2) * s),
             (-1) * B * (B ** 2 + R ** 2) ** (-1 / 2) * np.cos((B ** 2 + R ** 2) ** (-1 / 2) * s),
             R * (B ** 2 + R ** 2) ** (-1 / 2) * np.ones_like(s))).T

    _T_frame_left_hand = lambda self, R, B, s: np.vstack(
            (R * (B ** 2 + R ** 2) ** (-1 / 2) * np.cos((B ** 2 + R ** 2) ** (-1 / 2) * s),
             (-1) * R * (B ** 2 + R ** 2) ** (-1 / 2) * np.sin((B ** 2 + R ** 2) ** (-1 / 2) * s),
             B * (B ** 2 + R ** 2) ** (-1 / 2) * np.ones_like(s))).T
    _N_frame_left_hand = lambda self, R, B, s: np.vstack(
            ((-1) * np.sin((B ** 2 + R ** 2) ** (-1 / 2) * s),
             (-1) * np.cos((B ** 2 + R ** 2) ** (-1 / 2) * s),
             np.zeros_like(s))).T
    _B_frame_left_hand = lambda self, R, B, s: np.vstack(
            (B * (B ** 2 + R ** 2) ** (-1 / 2) * np.cos((B ** 2 + R ** 2) ** (-1 / 2) * s),
             (-1) * B * (B ** 2 + R ** 2) ** (-1 / 2) * np.sin((B ** 2 + R ** 2) ** (-1 / 2) * s),
             (-1) * R * (B ** 2 + R ** 2) ** (-1 / 2) * np.ones_like(s))).T

    def __init__(self):
        super().__init__()
        self._R = 0  # major radius of helix
        self._rho = 0  # minor radius of helix
        self._B = 0  # B = pitch / (2 * np.pi)
        self._n_c = 0  # number of period

    def supHelixLength(self, R, B, r, b, s):
        import scipy.integrate as integrate

        A = R
        a = r
        dr = lambda s: 2 ** (-1 / 2) * ((a ** 2 + b ** 2) ** (-1) * (A ** 2 + B ** 2) ** (-2) * (
                2 * a ** 2 * A ** 4 + a ** 2 * A ** 2 * b ** 2 + 2 * A ** 4 * b ** 2 + 4 * a ** 2 * A ** 2 * b * B + 4 * a ** 2 * A ** 2 * B ** 2 + 2 * a ** 2 * b ** 2 * B ** 2 + 4 * A ** 2 * b ** 2 * B ** 2 + 4 * a ** 2 * b * B ** 3 + 2 * a ** 2 * B ** 4 + 2 * b ** 2 * B ** 4 + (
            -4) * a * A * b ** 2 * (A ** 2 + B ** 2) * cos(
                (a ** 2 + b ** 2) ** (-1 / 2) * s) + a ** 2 * A ** 2 * b ** 2 * cos(
                2 * (a ** 2 + b ** 2) ** (-1 / 2) * s))) ** (1 / 2);
        t_ans = integrate.quad(dr, 0, s, limit=100)
        PETSc.Sys.Print(t_ans[1])
        return t_ans[0]

    def create_n(self, R, B, r, n_node, n_c=1, eh=1):
        assert 1 == 2, 'The method DO NOT finished!!!'
        sH1 = lambda s: (R ** 2 + B ** 2) ** (-1 / 2) * (
                (R ** 2 + B ** 2) ** (1 / 2) * (R + (-1) * r * cos((r ** 2 + b ** 2) ** (-1 / 2) * s)) * cos(
                b * ((r ** 2 + b ** 2) * (R ** 2 + B ** 2)) ** (-1 / 2) * s) + r * B * sin(
                (r ** 2 + b ** 2) ** (-1 / 2) * s) * sin(
                b * ((r ** 2 + b ** 2) * (R ** 2 + B ** 2)) ** (-1 / 2) * s))
        sH2 = lambda s: (R ** 2 + B ** 2) ** (-1 / 2) * (
                (-1) * r * B * cos(b * ((r ** 2 + b ** 2) * (R ** 2 + B ** 2)) ** (-1 / 2) * s) * sin(
                (r ** 2 + b ** 2) ** (-1 / 2) * s) + (R ** 2 + B ** 2) ** (1 / 2) * (
                        R + (-1) * r * cos((r ** 2 + b ** 2) ** (-1 / 2) * s)) * sin(
                b * ((r ** 2 + b ** 2) * (R ** 2 + B ** 2)) ** (-1 / 2) * s))
        sH3 = lambda s: ((r ** 2 + b ** 2) * (R ** 2 + B ** 2)) ** (-1 / 2) * (
                b * B * s + r * R * (r ** 2 + b ** 2) ** (1 / 2) * sin((r ** 2 + b ** 2) ** (-1 / 2) * s))
        sHf1 = lambda s: (R ** 2 + B ** 2) ** (-1 / 2) * (
                (R ** 2 + B ** 2) ** (1 / 2) * (R + (-1) * af * cos((r ** 2 + b ** 2) ** (-1 / 2) * s)) * cos(
                b * ((r ** 2 + b ** 2) * (R ** 2 + B ** 2)) ** (-1 / 2) * s) + af * B * sin(
                (r ** 2 + b ** 2) ** (-1 / 2) * s) * sin(
                b * ((r ** 2 + b ** 2) * (R ** 2 + B ** 2)) ** (-1 / 2) * s))
        sHf2 = lambda s: (R ** 2 + B ** 2) ** (-1 / 2) * (
                (-1) * af * B * cos(b * ((r ** 2 + b ** 2) * (R ** 2 + B ** 2)) ** (-1 / 2) * s) * sin(
                (r ** 2 + b ** 2) ** (-1 / 2) * s) + (R ** 2 + B ** 2) ** (1 / 2) * (
                        R + (-1) * af * cos((r ** 2 + b ** 2) ** (-1 / 2) * s)) * sin(
                b * ((r ** 2 + b ** 2) * (R ** 2 + B ** 2)) ** (-1 / 2) * s))
        sHf3 = lambda s: ((r ** 2 + b ** 2) * (R ** 2 + B ** 2)) ** (-1 / 2) * (
                b * B * s + R * af * (r ** 2 + b ** 2) ** (1 / 2) * sin((r ** 2 + b ** 2) ** (-1 / 2) * s))

        from scipy import optimize as sop
        b = 2 ** (-1 / 2) * (n_node ** (-2) * n_c * (B ** 2 * n_c + n_c * R ** 2 + (B ** 2 + R ** 2) ** (1 / 2) * (
                B ** 2 * n_c ** 2 + 4 * n_node ** 2 * r ** 2 + n_c ** 2 * R ** 2) ** (1 / 2))) ** (
                    1 / 2)
        si = np.arange(n_node) * b * 2 * np.pi
        nodes = np.vstack((sH1(si), sH2(si), sH3(si) - B * 2 * np.pi * n_c / 2)).T
        self._nodes = nodes
        self.set_dmda()
        self._u = np.zeros(self._nodes.size)
        self._normal = np.zeros((n_node, 2), order='F')
        self.set_deltaLength(2 * np.pi * b)

        af = r - 2 * np.pi * b * eh
        err_msg = 'epsilon of helix eh is too big, cause minor radius of force geo < 0. '
        assert r > 0, err_msg
        fgeo = geo()
        fgeo.set_nodes(np.vstack((sHf1(si), sHf2(si), sHf3(si) - B * 2 * np.pi * n_c / 2,)).T,
                       deltalength=2 * np.pi * b, resetVelocity=True)
        return fgeo

    def create_deltatheta(self, dth: float,  # delta theta of the cycle for the mesh
                          radius: float,
                          R, B, n_c,
                          epsilon=0,
                          with_cover=False,
                          factor=1,
                          left_hand=False):
        # definition of parameters see self.__init__()
        self._R = R
        self._rho = radius
        self._B = B
        self._n_c = n_c
        self._factor = factor
        self._left_hand = left_hand
        return self._create_deltatheta(dth, radius, epsilon, with_cover)

    def _get_axis(self):
        R = self._R
        B = self._B
        n_c = self._n_c
        factor = self._factor
        left_hand = self._left_hand
        ds = self.get_deltaLength()
        length = np.sqrt(R ** 2 + B ** 2) * 2 * np.pi * n_c
        nl = np.ceil(length / ds).astype(int)
        s = self._factor_fun(nl, factor) * length - length / 2
        if left_hand:
            self._frenetFrame = (
                self._T_frame_left_hand(R, B, s), self._N_frame_left_hand(R, B, s), self._B_frame_left_hand(R, B, s))
            self._axisNodes = self._helix_left_hand(R, B, s)
        else:
            self._frenetFrame = (self._T_frame(R, B, s), self._N_frame(R, B, s), self._B_frame(R, B, s))
            self._axisNodes = self._helix(R, B, s)
        return self._axisNodes, self._frenetFrame[0], self._frenetFrame[1], self._frenetFrame[2]

    def _get_fgeo_axis(self, epsilon):
        R = self._R
        B = self._B
        n_c = self._n_c
        factor = self._factor
        left_hand = self._left_hand
        length = np.sqrt(R ** 2 + B ** 2) * 2 * np.pi * n_c
        nl = self._axisNodes.shape[0]
        ds = -self.get_deltaLength() * epsilon / 1
        s = self._factor_fun(nl, factor) * (length - ds / 2) + ds / 4 - length / 2
        if left_hand:
            frenetFrame = (
                self._T_frame_left_hand(R, B, s), self._N_frame_left_hand(R, B, s), self._B_frame_left_hand(R, B, s))
            axisNodes = self._helix_left_hand(R, B, s)
        else:
            frenetFrame = (self._T_frame(R, B, s), self._N_frame(R, B, s), self._B_frame(R, B, s))
            axisNodes = self._helix(R, B, s)
        return axisNodes, frenetFrame[0], frenetFrame[1], frenetFrame[2]


# symmetric geo with infinity length, i.e. infinite long helix, infinite long tube.
class infgeo_1d(geo):  # periodism along z direction.
    def __init__(self, max_length):
        super().__init__()
        self._max_length = max_length  # cut of of infinite long geo
        self._nSegment = 0  # number of subsections of geo

    def get_nSegment(self):
        return self._nSegment

    def get_maxlength(self):
        return self._max_length

    @abc.abstractmethod
    def coord_x123(self, percentage):
        return

    def rot_matrix(self, percentage):
        return np.identity(3)

    def show_segment(self, linestyle='-'):
        return super().show_nodes(linestyle)

    def show_nodes(self, linestyle='-'):
        t_nodes = []
        for percentage in np.linspace(-1, 1, self.get_nSegment()):
            t_nodes.append(self.coord_x123(percentage))
        t_nodes = np.vstack(t_nodes)
        t_geo = geo()
        t_geo.set_nodes(t_nodes, deltalength=0)
        return t_geo.show_nodes(linestyle)


# a infinite long helix along z axis
class infHelix(infgeo_1d):
    def __init__(self, max_length):
        super().__init__(max_length)  # here maxlength means the cut off max theta of helix
        self._R = 0  # major radius of helix
        self._rho = 0  # minor radius of helix
        self._ph = 0  # pitch of helix
        self._phi = 0  # define the coordinates of nodes at the reference cross section.
        self._theta0 = 0  # define the reference location (original rotation) of the helix

    def coord_x123(self, percentage):
        R = self._R
        rho = self._rho
        ph = self._ph
        phi = self._phi
        theta0 = self._theta0
        th = (percentage * self._max_length) % (2 * np.pi) + theta0
        # th = percentage * self._maxlength + theta0

        # definition of parameters see __init__()
        # x1, x2, x3, coordinates of helix nodes
        x1 = lambda theta: np.cos(theta) * (R - rho * np.sin(phi)) + (ph * rho * np.cos(phi) * np.sin(
                theta)) / np.sqrt(ph ** 2 + 4 * np.pi ** 2 * R ** 2)
        x2 = lambda theta: - (ph * rho * np.cos(phi) * np.cos(theta)) / np.sqrt(
                ph ** 2 + 4 * np.pi ** 2 * R ** 2) + (R - rho * np.sin(phi)) * np.sin(theta)
        x3 = lambda theta: (ph * theta) / (2. * np.pi) + (2 * np.pi * R * rho * np.cos(phi)) / np.sqrt(
                ph ** 2 + 4 * np.pi ** 2 * R ** 2)
        return np.vstack((x1(th), x2(th), x3(percentage * self._max_length))).T

    def rot_matrix(self, percentage):
        th = percentage * self._max_length
        Rmxt = np.identity(3)
        Rmxt[0][0] = np.cos(th)
        Rmxt[0][1] = -np.sin(th)
        Rmxt[1][0] = np.sin(th)
        Rmxt[1][1] = np.cos(th)
        return Rmxt

    def Frenetframe(self, percentage):
        th = percentage * self._max_length + self._theta0
        ph = self._ph
        lh = 2 * np.pi * self._R
        s = np.sqrt(lh ** 2 + ph ** 2)
        T = np.array((-lh * np.sin(th) / s, lh * np.cos(th) / s, ph / s))
        N = np.array((ph * np.sin(th) / s, -ph * np.cos(th) / s, lh / s))
        B = np.array((-np.cos(th), -np.sin(th), 0))
        return T, N, B

    def create_n(self, R, rho, ph, n, theta0=0):
        ch = self.get_maxlength() / (2 * np.pi) * 2  # it ranges from -1 to 1, so times two.
        ntheta = (ch * np.sqrt(ph ** 2 + (2 * np.pi * R) ** 2) * n) / (2 * np.pi * rho)
        self._nSegment = ntheta
        self._R = R
        self._rho = rho
        self._ph = ph
        self._theta0 = theta0
        self._phi = np.linspace(0, 2 * np.pi, n, endpoint=False)
        self._nodes = self.coord_x123(0)
        self.set_deltaLength(2 * np.pi * rho / n)
        self.set_origin((0, 0, 0))
        self._u = np.zeros(self._nodes.size)
        self.set_dmda()
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # fig.patch.set_facecolor('white')
        # ax.plot(*self._nodes.T, '*')
        # plt.show()
        # print(self.get_n_nodes())
        return True

    def create_fgeo(self, epsilon):
        fgeo = infHelix(self.get_maxlength())
        deltalength = self.get_deltaLength()
        f_rho = (self._rho + epsilon * deltalength)
        err_msg = 'epsilon > %f. ' % (-self._rho / deltalength)
        assert f_rho > 0, err_msg
        fgeo.create_n(self._R, f_rho, self._ph, self.get_n_nodes(), self._theta0)
        return fgeo

    def get_phi(self):
        return self._phi


# a infinite long pipe along z axis
class infPipe(infgeo_1d):
    def __init__(self, max_length):
        super().__init__(max_length)
        self._R = 0  # radius of pipe
        self._phi = 0  # define the coordinates of nodes at the reference cross section.
        self._theta = 0  # the angle between the cut plane and the z axis

    def coord_x123(self, percentage):
        # return coordinates of helix nodes
        xz = percentage * self._max_length
        R = self._R
        phi = self._phi
        theta = self._theta
        return np.vstack((np.cos(phi) * R, np.sin(phi) * R, np.cos(phi) * R * np.sin(theta) + np.ones_like(phi) * xz)).T

    def create_n(self, R, n, theta=0):
        deltaLength = 2 * np.pi * R / n
        self._nSegment = np.ceil(self.get_maxlength() / deltaLength)
        self._R = R
        self._phi = np.linspace(0, 2 * np.pi, n, endpoint=False)
        self._theta = theta
        self._nodes = self.coord_x123(0)
        self.set_deltaLength(deltaLength)
        self.set_origin((0, 0, 0))
        self._u = np.zeros(self._nodes.size)
        self.set_dmda()
        return True

    def create_fgeo(self, epsilon):
        fgeo = infPipe(self.get_maxlength())
        deltalength = self.get_deltaLength()
        f_R = self._R + epsilon * deltalength
        fgeo.create_n(f_R, self.get_n_nodes(), self._theta)
        return fgeo

    def get_phi(self):
        return self._phi


class region:
    def __init__(self):
        self.type = {'rectangle': self.rectangle,
                     'sector':    self.sector}

    def rectangle(self,
                  field_range: np.array,
                  n_grid: np.array):
        """

        :type self: StokesFlowProblem
        :param self: self
        :type: field_range: np.array
        :param field_range: range of output velocity field.
        :type: n_grid: np.array
        :param n_grid: number of cells at each direction.
        """

        min_range = np.amin(field_range, axis=0)
        max_range = np.amax(field_range, axis=0)
        # noinspection PyUnresolvedReferences
        full_region_x = np.linspace(min_range[0], max_range[0], n_grid[0])
        # noinspection PyUnresolvedReferences
        full_region_y = np.linspace(min_range[1], max_range[1], n_grid[1])
        # noinspection PyUnresolvedReferences
        full_region_z = np.linspace(min_range[2], max_range[2], n_grid[2])
        [full_region_x, full_region_y, full_region_z] = np.meshgrid(full_region_x, full_region_y, full_region_z,
                                                                    indexing='ij')

        return full_region_x, full_region_y, full_region_z

    def sector(self,
               field_range: np.array,
               n_grid: np.array):
        """

        :type self: StokesFlowProblem
        :param self: self
        :type: field_range: np.array
        :param field_range: range of output velocity field.
        :type: n_grid: np.array
        :param n_grid: number of cells at each direction.
        """

        min_range = np.amin(field_range, axis=0)
        max_range = np.amax(field_range, axis=0)
        # noinspection PyUnresolvedReferences
        full_region_x = np.linspace(min_range[0], max_range[0], n_grid[0])
        # noinspection PyUnresolvedReferences
        full_region_r = np.linspace(min_range[1], max_range[1], n_grid[1])
        # noinspection PyUnresolvedReferences
        full_region_theta = np.linspace(min_range[2], max_range[2], n_grid[2])
        [full_region_x, temp_r, temp_theta] = np.meshgrid(full_region_x, full_region_r, full_region_theta,
                                                          indexing='ij')
        full_region_y = temp_r * np.cos(temp_theta)
        full_region_z = temp_r * np.sin(temp_theta)

        return full_region_x, full_region_y, full_region_z

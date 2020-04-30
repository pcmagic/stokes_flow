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
from scipy.special import hyp2f1
from scipy import interpolate, integrate, optimize, sparse
from itertools import compress

__all__ = ['base_geo', 'sphere_geo', 'ellipse_base_geo', 'geoComposit',
           'tunnel_geo', 'pipe_cover_geo', 'supHelix', 'FatHelix',
           'lineOnFatHelix', 'ThickLine_base_geo',
           'SelfRepeat_body_geo', 'SelfRepeat_FatHelix',
           'infgeo_1d', 'infHelix', 'infPipe',
           'slb_helix', 'Johnson_helix', 'expJohnson_helix',
           'regularizeDisk', 'helicoid',
           '_revolve_geo', 'revolve_pipe', 'revolve_ellipse',
           'region', 'set_axes_equal']


class base_geo():
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
        self._type = 'general_geo'  # geo type

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
        self._geo_norm = self._geo_norm / np.linalg.norm(self._geo_norm)
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

    def move(self, displacement: np.array):
        displacement = np.array(displacement).reshape((3,))

        self.set_nodes(self.get_nodes() + displacement, self.get_deltaLength())
        self.set_origin(self.get_origin() + displacement)
        return True

    def mirrorImage(self, norm=np.array([0, 0, 1]), rotation_origin=None):
        if rotation_origin is None:
            rotation_origin = self.get_origin()
        else:
            rotation_origin = np.array(rotation_origin).reshape((3,))
        norm = norm / np.linalg.norm(norm)

        nodes = self.get_nodes()
        dist = nodes - rotation_origin
        parallel = np.einsum('i,j', np.einsum('ij,j', dist, norm), norm)
        perpendicular = dist - parallel
        dist2 = perpendicular + (-1 * parallel)
        nodes2 = dist2 + rotation_origin
        self.set_nodes(nodes2, self.get_deltaLength())
        return True

    def combine(self, geo_list, deltaLength=None, origin=None, geo_norm=None):
        if len(geo_list) == 0:
            return False
        for geo1 in geo_list:
            err_msg = 'some objects in geo_list are not geo object. %s' % str(type(geo1))
            assert isinstance(geo1, base_geo), err_msg
            err_msg = 'one or more objects not finished create yet. '
            assert geo1.get_n_nodes() != 0, err_msg
        if deltaLength is None:
            deltaLength = geo_list[0].get_deltaLength()
        if origin is None:
            origin = geo_list[0].get_origin()
        if geo_norm is None:
            geo_norm = geo_list[0].get_geo_norm()

        geo1 = geo_list.pop(0)
        self.set_nodes(geo1.get_nodes(), deltalength=deltaLength)
        self.set_velocity(geo1.get_velocity())
        for geo1 in geo_list:
            self.set_nodes(np.vstack((self.get_nodes(), geo1.get_nodes())), deltalength=deltaLength)
            self.set_velocity(np.hstack((self.get_velocity(), geo1.get_velocity())))
        self.set_dmda()
        self._geo_norm = geo_norm
        self.set_origin(origin)
        return True

    def get_nodes(self):
        return self._nodes

    def get_nodes_petsc(self):
        nodes_petsc = self.get_dmda().createGlobalVector()
        nodes_petsc[:] = self._nodes.reshape((3, -1))[:]
        nodes_petsc.assemble()
        return nodes_petsc

    def set_nodes(self, nodes, deltalength, resetVelocity=False):
        nodes = np.array(nodes).reshape((-1, 3), order='F')
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

    def set_geo_norm(self, geo_norm):
        geo_norm = np.array(geo_norm).flatten()
        assert geo_norm.size == 3
        self._geo_norm = geo_norm
        return True

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

    def copy(self) -> 'base_geo':
        self.destroy_dmda()
        geo2 = copy.deepcopy(self)
        self.set_dmda()
        geo2.set_dmda()
        return geo2

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
            # Be careful. the axis using in matplotlib is a left-handed coordinate system
            if show_nodes:
                ax.plot(self.get_nodes_x(), self.get_nodes_y(), self.get_nodes_z(),
                        linestyle='None', c='b',
                        marker='o')
            INDEX = np.zeros_like(self.get_nodes_z(), dtype=bool)
            INDEX[:] = True
            length = 1 / np.mean(self._deltaLength) * length_factor
            ax.quiver(self.get_nodes_x()[INDEX], self.get_nodes_y()[INDEX],
                      self.get_nodes_z()[INDEX],
                      self.get_velocity_x()[INDEX], self.get_velocity_y()[INDEX],
                      self.get_velocity_z()[INDEX],
                      color='r', length=length)
            # ax.quiver(self.get_nodes_x(), self.get_nodes_y(), self.get_nodes_z(),
            #           0, 0, self.get_nodes_z(), length=self._deltaLength * 2)

            X = np.hstack((self.get_nodes_x()))
            Y = np.hstack((self.get_nodes_y()))
            Z = np.hstack((self.get_nodes_z()))
            max_range = np.array(
                    [X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max() / 2.0
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
            # plt.get_current_fig_manager().window.showMaximized()
            plt.show()
        return True

    def core_show_nodes(self, linestyle='-', marker='.'):
        comm = PETSc.COMM_WORLD.tompi4py()
        rank = comm.Get_rank()
        if rank == 0:
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.set_aspect('equal')
            ax.plot(self.get_nodes_x(), self.get_nodes_y(), self.get_nodes_z(),
                    linestyle=linestyle,
                    color='b',
                    marker=marker)

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

    def show_nodes(self, linestyle='-', marker='.'):
        comm = PETSc.COMM_WORLD.tompi4py()
        rank = comm.Get_rank()
        self.core_show_nodes(linestyle=linestyle, marker=marker)
        if rank == 0:
            plt.grid()
            # plt.get_current_fig_manager().window.showMaximized()
            plt.show()
        return True

    def png_nodes(self, finename, linestyle='-', marker='.'):
        comm = PETSc.COMM_WORLD.tompi4py()
        rank = comm.Get_rank()
        finename = check_file_extension(finename, '.png')

        fig = self.core_show_nodes(linestyle=linestyle, marker=marker)
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

    def get_type(self):
        return self._type

    def print_info(self):
        pass

    def pickmyself_prepare(self):
        if not self._dmda is None:
            self.destroy_dmda()
        return True


class geoComposit(uniqueList):
    def __init__(self, geo_list=[]):
        acceptType = base_geo
        super().__init__(acceptType)
        geo_list = list(tube_flatten((geo_list,)))
        for geoi in geo_list:
            self.append(geoi)

    def core_show_nodes(self, linestyle='-', marker='.'):
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
                            marker=marker)

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

    def show_nodes(self, linestyle='-', marker='.'):
        if len(self) == 0:
            return False

        comm = PETSc.COMM_WORLD.tompi4py()
        rank = comm.Get_rank()
        self.core_show_nodes(linestyle=linestyle, marker=marker)
        if rank == 0:
            plt.grid()
            # plt.get_current_fig_manager().window.showMaximized()
            plt.show()
        return True

    def move(self, displacement: np.array):
        if len(self) == 0:
            return False
        else:
            for sub_geo in self:
                sub_geo.move(displacement=displacement)
        return True


class ThickLine_base_geo(base_geo):
    def __init__(self):
        super().__init__()
        self._r = 0  # radius of thick line itself, thick is a cycle.
        self._dth = 0  # anglar between nodes in a cycle.
        self._axisNodes = np.array([]).reshape((-1, 3))
        self._frenetFrame = (np.array([]).reshape((-1, 3)),
                             np.array([]).reshape((-1, 3)),
                             np.array([]).reshape((-1, 3)))
        self._iscover = []  # start: -1, body: 0, end: 1
        self._with_cover = 0
        self._factor = 1e-5
        self._left_hand = False
        self._check_epsilon = True
        self._type = '_ThickLine_geo'  # geo type
        self._cover_strat_idx = np.array([])
        self._body_idx_list = []
        self._cover_end_idx = np.array([])
        self._local_rot = True  # special parameter for selfrepeat_geo
        self._node_axisNode_idx = []

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

    @abc.abstractmethod
    def _body_pretreatment(self, nodes, **kwargs):
        return

    @abc.abstractmethod
    def _strat_pretreatment(self, nodes, **kwargs):
        return

    @abc.abstractmethod
    def _end_pretreatment(self, nodes, **kwargs):
        return

    def _create_deltatheta(self, dth: float,  # delta theta of the cycle for the mesh
                           radius: float,  # radius of the cycle
                           epsilon=0, with_cover=0, local_rot=True):
        # the tunnel is along z axis
        err_msg = 'dth must less than pi'
        assert dth < np.pi, err_msg
        self._dth = dth
        self._r = radius
        self._with_cover = with_cover
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
        t_node_idx = 0
        local_rot = self._local_rot

        # cover at start
        if with_cover == 1:
            # old version, cover is a plate.
            nc = np.ceil((radius - deltalength) / deltalength).astype(int)
            ri = np.linspace(deltalength / 2, radius, nc, endpoint=False)
            # self
            tidx = 0
            for i0 in np.arange(0, nc):
                ai_para = ai_para + 1 if local_rot else 0
                ni = np.ceil(2 * np.pi * ri[i0] / deltalength).astype(int)
                ai = np.linspace(0, 2 * np.pi, ni, endpoint=False) + (-1) ** ai_para * dth / 4
                iscover.append(np.ones_like(ai) * -1)
                nodes_cycle = np.vstack(
                        (np.cos(ai) * ri[i0], np.sin(ai) * ri[i0], np.zeros_like(ai))).T
                t_nodes = axisNodes[0] + np.dot(nodes_cycle,
                                                np.vstack((N_frame[0], B_frame[0],
                                                           np.zeros_like(T_frame[0]))))
                vgeo_nodes.append(t_nodes)
                tidx = tidx + t_nodes.shape[0]
                tf_nodes = fgeo_axisNodes[0] + np.dot(nodes_cycle * epsilon,
                                                      np.vstack((N_frame[0], B_frame[0],
                                                                 np.zeros_like(T_frame[0]))))
                fgeo_nodes.append(tf_nodes)
                self._strat_pretreatment(t_nodes)
            self._cover_strat_idx = np.arange(len(vgeo_nodes))
            t_node_idx = self._cover_strat_idx[-1] + 1 if self._cover_strat_idx.size > 0 else 0
            self._node_axisNode_idx.append(np.zeros(tidx))
        elif with_cover == 2:
            # 20170929, new version, cover is a hemisphere
            vhsgeo = sphere_geo()
            vhsgeo.create_half_delta(deltalength, radius)
            vhsgeo.node_rotation((1, 0, 0), np.pi / 2 + ai_para)
            t_nodes = axisNodes[0] + np.dot(vhsgeo.get_nodes(),
                                            np.vstack((-T_frame[0], N_frame[0], B_frame[0])))
            vgeo_nodes.append(t_nodes)
            self._cover_strat_idx = np.arange(t_nodes.shape[0]) + t_node_idx
            t_node_idx = self._cover_strat_idx[-1] + 1
            fhsgeo = vhsgeo.copy()
            # fhsgeo.show_nodes()
            fhsgeo.node_zoom(epsilon)
            # fhsgeo.show_nodes()
            tf_nodes = fgeo_axisNodes[0] + np.dot(fhsgeo.get_nodes(),
                                                  np.vstack((-T_frame[0], N_frame[0], B_frame[0])))
            fgeo_nodes.append(tf_nodes)
            self._strat_pretreatment(t_nodes)
            iscover.append(np.ones(vhsgeo.get_n_nodes()) * -1)
            self._node_axisNode_idx.append(np.zeros(vhsgeo.get_n_nodes()))

        # body
        for i0, nodei_line in enumerate(axisNodes):
            ai_para = ai_para + 1 if local_rot else 0
            ai = angleCycle + (-1) ** ai_para * dth / 4
            nodes_cycle = np.vstack((np.cos(ai) * radius, np.sin(ai) * radius, np.zeros_like(ai))).T
            t_nodes = nodei_line + np.dot(nodes_cycle,
                                          np.vstack((N_frame[i0], B_frame[i0],
                                                     np.zeros_like(T_frame[i0]))))
            vgeo_nodes.append(t_nodes)
            self._body_idx_list.append(np.arange(t_nodes.shape[0]) + t_node_idx)
            t_node_idx = self._body_idx_list[-1][-1] + 1
            iscover.append(np.zeros_like(ai))
            nodes_cycle = np.vstack(
                    (np.cos(ai) * radius, np.sin(ai) * radius, np.zeros_like(ai))).T * epsilon
            tf_nodes = fgeo_axisNodes[i0] + np.dot(nodes_cycle, np.vstack(
                    (fgeo_N_frame[i0], fgeo_B_frame[i0], np.zeros_like(fgeo_T_frame[i0]))))
            fgeo_nodes.append(tf_nodes)
            self._body_pretreatment(t_nodes)
            self._node_axisNode_idx.append(np.ones(ai.size) * i0)

        # cover at end
        if with_cover == 1:
            # old version, cover is a plate.
            nc = np.ceil((radius - deltalength) / deltalength).astype(int)
            ri = np.linspace(deltalength / 2, radius, nc, endpoint=False)[-1::-1]
            tidx = 0
            for i0 in np.arange(0, nc):
                ai_para = ai_para + 1 if local_rot else 0
                ni = np.ceil(2 * np.pi * ri[i0] / deltalength).astype(int)
                ai = np.linspace(0, 2 * np.pi, ni, endpoint=False) + (-1) ** ai_para * dth / 4
                iscover.append(np.ones_like(ai))
                nodes_cycle = np.vstack(
                        (np.cos(ai) * ri[i0], np.sin(ai) * ri[i0], np.zeros_like(ai))).T
                t_nodes = axisNodes[-1] + np.dot(nodes_cycle,
                                                 np.vstack((N_frame[-1], B_frame[-1],
                                                            np.zeros_like(T_frame[-1]))))
                vgeo_nodes.append(t_nodes)
                tidx = tidx + t_nodes.shape[0]
                tf_nodes = fgeo_axisNodes[-1] + np.dot(nodes_cycle * epsilon, np.vstack(
                        (fgeo_N_frame[-1], fgeo_B_frame[-1], np.zeros_like(fgeo_T_frame[-1]))))
                fgeo_nodes.append(tf_nodes)
                self._end_pretreatment(t_nodes)
            self._cover_end_idx = np.arange(len(vgeo_nodes) - t_node_idx) + t_node_idx
            self._node_axisNode_idx.append(np.ones(tidx) * (axisNodes.shape[0] - 1))
        elif with_cover == 2:
            # 20170929, new version, cover is a hemisphere
            vhsgeo = sphere_geo()
            vhsgeo.create_half_delta(deltalength, radius)
            vhsgeo.node_rotation((1, 0, 0), -np.pi / 2 - ai_para)
            t_nodes = axisNodes[-1] + np.dot(vhsgeo.get_nodes(),
                                             np.vstack((T_frame[-1], N_frame[-1], B_frame[-1])))
            vgeo_nodes.append(np.flipud(t_nodes))
            self._cover_end_idx = np.arange(t_nodes.shape[0]) + t_node_idx
            fhsgeo = vhsgeo.copy()
            fhsgeo.node_zoom(epsilon)
            tf_nodes = fgeo_axisNodes[-1] + np.dot(fhsgeo.get_nodes(),
                                                   np.vstack(
                                                           (T_frame[-1], N_frame[-1], B_frame[-1])))
            fgeo_nodes.append(np.flipud(tf_nodes))
            self._end_pretreatment(t_nodes)
            iscover.append(np.ones(vhsgeo.get_n_nodes()))
            self._node_axisNode_idx.append(np.ones(vhsgeo.get_n_nodes()) * (axisNodes.shape[0] - 1))

        self._iscover = np.hstack(iscover)
        self._nodes = np.asfortranarray(np.vstack(vgeo_nodes))
        self.set_dmda()
        self._u = np.zeros(self._nodes.size)
        self._normal = np.zeros((self._nodes.shape[0], 2), order='F')
        self._node_axisNode_idx = np.hstack(self._node_axisNode_idx).astype('int')
        fgeo = self.copy()
        # fgeo.set_dof(self.get_dof())
        fgeo.set_nodes(np.asfortranarray(np.vstack(fgeo_nodes)), deltalength=deltalength * epsilon,
                       resetVelocity=True)
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

    @property
    def axisNodes(self):
        return self._axisNodes

    @property
    def frenetFrame(self):
        return self._frenetFrame

    @property
    def cover_strat_idx(self):
        return self._cover_strat_idx

    @property
    def body_idx_list(self):
        return self._body_idx_list

    @property
    def cover_end_idx(self):
        return self._cover_end_idx

    @property
    def with_cover(self):
        return self._with_cover

    @property
    def cover_start_nodes(self):
        return self.get_nodes()[self.cover_strat_idx]

    @property
    def body_nodes_list(self):
        return [self.get_nodes()[tidx] for tidx in self.body_idx_list]

    @property
    def cover_end_nodes(self):
        return self.get_nodes()[self.cover_end_idx]

    @property
    def node_axisNode_idx(self):
        return self._node_axisNode_idx

    def node_rotation(self, norm=np.array([0, 0, 1]), theta=0, rotation_origin=None):
        # The rotation is counterclockwise
        super().node_rotation(norm, theta, rotation_origin)

        if rotation_origin is None:
            rotation_origin = self.get_origin()
        else:
            rotation_origin = np.array(rotation_origin).reshape((3,))

        rotation = get_rot_matrix(norm, theta)
        t_axisNodes = self._axisNodes
        self._axisNodes = np.dot(rotation, (self._axisNodes - rotation_origin).T).T + \
                          rotation_origin  # The rotation is counterclockwise
        t0 = []
        for i0 in range(3):
            t1 = []
            for t2, taxis0, taxis in zip(self._frenetFrame[i0], t_axisNodes, self._axisNodes):
                t2 = np.dot(rotation, (t2 + taxis0 - rotation_origin)) \
                     + rotation_origin - taxis
                t2 = t2 / np.linalg.norm(t2)
                t1.append(t2)
            t0.append(np.vstack(t1))
        self._frenetFrame = t0
        return True

    def move(self, displacement: np.array):
        super().move(displacement)
        displacement = np.array(displacement).reshape((3,))
        self._axisNodes = self._axisNodes + displacement
        return True

    def nodes_local_coord(self, nodes, axis_idx):
        tnode_line = self.axisNodes[axis_idx]
        tT = self.frenetFrame[0][axis_idx]
        tN = self.frenetFrame[1][axis_idx]
        tB = self.frenetFrame[2][axis_idx]
        tfnodes_local = np.dot((nodes - tnode_line), np.vstack((tN, tB, tT)).T)
        return tfnodes_local

    def selfnodes_local_coord(self, axis_idx):
        nodes = self.get_nodes()[self.body_idx_list[axis_idx]]
        return self.nodes_local_coord(nodes, axis_idx)

    def force_local_coord(self, force, axis_idx):
        tT = self.frenetFrame[0][axis_idx]
        tN = self.frenetFrame[1][axis_idx]
        tB = self.frenetFrame[2][axis_idx]
        tfi_local = np.dot(force, np.vstack((tN, tB, tT)).T)
        return tfi_local

    def frenetFrame_local(self, axis_idx):
        tT = self.frenetFrame[0][axis_idx]
        tN = self.frenetFrame[1][axis_idx]
        tB = self.frenetFrame[2][axis_idx]
        return tT, tN, tB


class ellipse_base_geo(base_geo):
    def __init__(self):
        super().__init__()
        self._type = 'ellipse_geo'  # geo type

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
        self._geo_norm = np.array((1, 0, 0))
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


class sphere_geo(ellipse_base_geo):
    def __init__(self):
        super().__init__()
        self._type = 'sphere_geo'  # geo type

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
                self._nodes[:, 0] ** 2 + self._nodes[:, 1] ** 2 + self._nodes[:, 2] ** 2).reshape(
                self._nodes.shape[0],
                1)
        self._normal[:, 1] = np.arccos(normal_vector[:, 2])  # b
        self._normal[:, 0] = np.arcsin(normal_vector[:, 0] / np.sin(self._normal[:, 1]))  # a
        return True


# noinspection PyUnresolvedReferences
class tunnel_geo(ThickLine_base_geo):
    def __init__(self):
        super().__init__()
        self._length = 0
        self._cover_strat_list = []
        self._cover_end_list = []
        self._type = 'tunnel_geo'  # geo type

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
        normal_vector = -1 * self._nodes / np.sqrt(
                self._nodes[:, 1] ** 2 + self._nodes[:, 2] ** 2).reshape(
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
        temp_geo = base_geo()
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


class _revolve_geo(base_geo):
    def __init__(self):
        super().__init__()

    def create_full_geo(self, n_c):
        # rotate alone z axis
        def rot_nodes(nodes):
            r = nodes[:, 0]
            z = nodes[:, 2]
            theta = np.linspace(0, 2 * np.pi, n_c, endpoint=False)
            x = np.outer(r, np.cos(theta)).flatten()
            y = np.outer(r, np.sin(theta)).flatten()
            z = np.outer(z, np.ones_like(theta)).flatten()
            nodes = np.vstack((x, y, z)).T
            return nodes

        self.set_nodes(rot_nodes(self.get_nodes()), self.get_deltaLength(), resetVelocity=True)
        return True


class revolve_ellipse(_revolve_geo):
    def __init__(self):
        super().__init__()
        self._length = 0
        self._radius = 0
        self._type = 'revolve_ellipse'

    def create_deltaz(self, ds: float,  # length of the mesh
                      a: float,  # axis1 = 2*a
                      b: float):  # axis2 = 2*b
        epsilon1 = 1 / 3
        epsilon2 = 0.3
        err_msg = 'both major and minor axises should positive. '
        assert a > 0 and b > 0, err_msg
        self._deltaLength = ds

        n_2 = np.ceil(a / 2 / ds).astype(int)
        dz = a / n_2
        z0 = np.linspace(a - dz / 2, dz / 2, n_2)
        z1 = np.hstack([z0, np.flipud(z0) * -1])
        x1 = np.sqrt(b ** 2 * (1 - (z1 / a) ** 2))

        # generate nodes.
        self._nodes = np.zeros((x1.size, 3), order='F')
        self._nodes[:, 0] = x1.flatten(order='F')
        self._nodes[:, 2] = z1.flatten(order='F')
        self.set_dmda()
        self._u = np.zeros(self._nodes.size)
        self._normal = np.zeros((self._nodes.shape[0], 2), order='F')
        self._geo_norm = np.array((0, 0, 1))

        # associated force geo
        move_delta = dz * epsilon1
        z0 = z0 - move_delta
        z1 = np.hstack([z0, np.flipud(z0) * -1])
        dx = x1 / b * dz * epsilon2
        x1 = x1 - dx
        f_geo = self.copy()
        fnodes = np.vstack((x1, np.zeros_like(x1), z1)).T
        f_geo.set_nodes(fnodes, 1)
        return f_geo

    def create_half_deltaz(self, ds: float,  # length of the mesh
                           a: float,  # axis1 = 2*a
                           b: float):  # axis2 = 2*b
        epsilon1 = 1 / 3
        epsilon2 = 0.3
        err_msg = 'both major and minor axises should positive. '
        assert a > 0 and b > 0, err_msg
        self._deltaLength = ds

        n_2 = np.ceil(a / 2 / ds).astype(int)
        dz = a / n_2
        z1 = np.linspace(a - dz / 2, dz / 2, n_2)
        x1 = np.sqrt(b ** 2 * (1 - (z1 / a) ** 2))

        # generate nodes.
        self._nodes = np.zeros((x1.size, 3), order='F')
        self._nodes[:, 0] = x1.flatten(order='F')
        self._nodes[:, 2] = z1.flatten(order='F')
        self.set_dmda()
        self._u = np.zeros(self._nodes.size)
        self._normal = np.zeros((self._nodes.shape[0], 2), order='F')
        self._geo_norm = np.array((0, 0, 1))

        # associated force geo
        move_delta = dz * epsilon1
        z1 = z1 - move_delta
        dx = x1 / b * dz * epsilon2
        x1 = x1 - dx
        f_geo = self.copy()
        fnodes = np.vstack((x1, np.zeros_like(x1), z1)).T
        f_geo.set_nodes(fnodes, 1)
        return f_geo

    def create_delta(self, ds: float,  # length of the mesh
                     a: float,  # axis1 = 2*a
                     b: float,  # axis2 = 2*b
                     epsilon):
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

        self._nodes = np.vstack((t_y, np.zeros_like(t_y), np.hstack(t_x))).T
        self.set_dmda()
        self._u = np.zeros(self._nodes.size)
        self._normal = np.zeros((self._nodes.shape[0], 2), order='F')
        self._geo_norm = np.array((0, 0, 1))

        # force geo
        tfct = (a + epsilon * ds) / a
        t_x = a * tfct * np.cos(t_theta)
        t_y = b * tfct * np.sin(t_theta)
        fnodes = np.vstack((t_y, np.zeros_like(t_y), np.hstack(t_x))).T
        f_geo = self.copy()
        f_geo.set_nodes(fnodes, 1)
        return f_geo

    def create_half_delta(self, ds: float,  # length of the mesh
                          a: float,  # axis1 = 2*a
                          b: float,  # axis2 = 2*b
                          epsilon):
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

        self._nodes = np.vstack((t_y, np.zeros_like(t_y), np.hstack(t_x))).T
        self.set_dmda()
        self._u = np.zeros(self._nodes.size)
        self._normal = np.zeros((self._nodes.shape[0], 2), order='F')
        self._geo_norm = np.array((0, 0, 1))

        # force geo
        tfct = (a + epsilon * ds) / a
        t_x = a * tfct * np.cos(t_theta0)
        t_y = b * tfct * np.sin(t_theta0)
        fnodes = np.vstack((t_y, np.zeros_like(t_y), np.hstack(t_x))).T
        f_geo = self.copy()
        f_geo.set_nodes(fnodes, 1)
        return f_geo


class revolve_pipe(_revolve_geo):
    def __init__(self):
        super().__init__()
        self._length = 0
        self._radius = 0
        self._type = 'revolve_pipe'

    def create_deltaz(self, ds: float,  # length of the mesh
                      length: float,  # length of the tunnel
                      radius: float):  # radius of the tunnel
        epsilon_x = 1 / 2
        epsilon_z = 1 / 3
        cover_fct = 2
        self._deltaLength = ds
        self._length = length
        self._radius = radius
        # the tunnel is along z axis
        # due to the symmetry of pipe, generate the first part and get the image as the other part.
        z0 = np.linspace((length - ds) / 2, 0,
                         num=np.ceil((length / ds / 2)).astype(int))[1:]
        z0 = z0 + ds / 2
        x0 = np.ones_like(z0) * radius
        # cover 1
        x1 = np.linspace(0, radius, num=cover_fct * np.ceil((radius / ds)).astype(int))
        z1 = np.ones_like(x1) * length / 2
        # half pard
        xi = np.hstack((x1, x0))
        zi = np.hstack((z1, z0))
        # all
        x = np.hstack((xi, np.flipud(xi)))
        z = np.hstack((zi, np.flipud(zi) * -1))
        self._nodes = np.zeros((x.size, 3), order='F')
        self._nodes[:, 0] = x.flatten(order='F')
        self._nodes[:, 1] = np.zeros_like(x).flatten(order='F')
        self._nodes[:, 2] = z.flatten(order='F')
        self.set_dmda()

        self._u = np.zeros(self._nodes.size)
        self._normal = np.zeros((self._nodes.shape[0], 2), order='F')
        self._geo_norm = np.array((0, 0, 1))

        # associated force geo
        f_geo = self.copy()
        epsilon_x = epsilon_x / cover_fct
        a = (radius - ds * epsilon_x * 2) / radius
        b = ds * epsilon_x
        z0 = z0 - ds * epsilon_z
        x0 = a * x0 + b
        x1 = a * x1 + b
        z1 = np.ones_like(x1) * length / 2 - ds * epsilon_z
        # half pard
        xi = np.hstack((x1, x0))
        zi = np.hstack((z1, z0))
        # all
        x = np.hstack((xi, np.flipud(xi)))
        z = np.hstack((zi, np.flipud(zi) * -1))
        fnodes = np.vstack((x, np.zeros_like(x), z)).T
        f_geo.set_nodes(fnodes, 1)
        return f_geo

    def create_half_deltaz(self, ds: float,  # length of the mesh
                           length: float,  # length of the tunnel
                           radius: float):  # radius of the tunnel
        epsilon_x = 1 / 2
        epsilon_z = 1 / 2
        cover_fct = 1.5
        self._deltaLength = ds
        self._length = length
        self._radius = radius
        # the tunnel is along z axis
        z0 = np.linspace(length / 2, ds / 2, num=np.ceil(length / ds / 2).astype(int))[1:]
        x0 = np.ones_like(z0) * radius
        # cover
        x1 = np.linspace(0, radius, num=np.ceil(cover_fct * radius / ds).astype(int))
        z1 = np.ones_like(x1) * length / 2
        # half part
        xi = np.hstack((x1, x0))
        zi = np.hstack((z1, z0))
        self._nodes = np.zeros((xi.size, 3), order='F')
        self._nodes[:, 0] = xi.flatten(order='F')
        self._nodes[:, 1] = np.zeros_like(xi).flatten(order='F')
        self._nodes[:, 2] = zi.flatten(order='F')
        self.set_dmda()

        self._u = np.zeros(self._nodes.size)
        self._normal = np.zeros((self._nodes.shape[0], 2), order='F')
        self._geo_norm = np.array((0, 0, 1))

        # associated force geo
        f_geo = self.copy()
        epsilon_x = epsilon_x / cover_fct
        a = (radius - ds * epsilon_x * 2) / radius
        b = ds * epsilon_x
        z0 = z0 - ds * epsilon_z
        x0 = a * x0 + b
        x1 = a * x1 + b
        z1 = np.ones_like(x1) * length / 2 - ds * epsilon_z
        # half part
        xi = np.hstack((x1, x0))
        zi = np.hstack((z1, z0))
        fnodes = np.vstack((xi, np.zeros_like(xi), zi)).T
        f_geo.set_nodes(fnodes, 1)
        return f_geo

    def create_half_deltaz_v2(self, ds: float,  # length of the mesh
                              length: float,  # length of the tunnel
                              radius: float):  # radius of the tunnel
        epsilon_x = 1 / 2
        epsilon_z = 1 / 2
        epsilon_3 = 1 / 5  # radio between radii of tangent curve and pipe.
        cover_fct = 1
        tc_fct = 5
        self._deltaLength = ds
        self._length = length
        self._radius = radius
        # the tunnel is along z axis
        z0 = np.linspace(length / 2, ds / 2, num=np.ceil(length / ds / 2).astype(int))[1:]
        x0 = np.ones_like(z0) * radius
        # Tangent curve
        tnz = np.ceil(epsilon_3 * radius / ds).astype(int)
        r_cv = ds * tnz
        z1 = np.flipud(np.arange(tnz) * ds + length / 2)
        x1 = (r_cv ** tc_fct - (z1 - length / 2) ** tc_fct) ** (1 / tc_fct) + radius - r_cv
        # cover
        num = np.ceil(cover_fct * x1[0] / ds).astype(int)
        x2 = np.linspace(0, x1[0], num=num)[:np.ceil(-2 * cover_fct).astype(int)]
        z2 = np.ones_like(x2) * z1[0]
        # half part
        xi = np.hstack((x2, x1, x0))
        zi = np.hstack((z2, z1, z0))
        self._nodes = np.zeros((xi.size, 3), order='F')
        self._nodes[:, 0] = xi.flatten(order='F')
        self._nodes[:, 1] = np.zeros_like(xi).flatten(order='F')
        self._nodes[:, 2] = zi.flatten(order='F')
        self.set_dmda()

        self._u = np.zeros(self._nodes.size)
        self._normal = np.zeros((self._nodes.shape[0], 2), order='F')
        self._geo_norm = np.array((0, 0, 1))

        # associated force geo
        f_geo = self.copy()
        epsilon_x = epsilon_x / cover_fct
        a = (radius - ds * epsilon_x * 2) / radius
        b = ds * epsilon_x
        x0 = a * x0 + b
        z0 = z0 - ds * epsilon_z
        x1 = a * x1 + b
        z1 = z1 - ds * epsilon_z
        x2 = a * x2 + b
        z2 = np.ones_like(x2) * length / 2 - ds * epsilon_z + r_cv
        # half part
        xi = np.hstack((x2, x1, x0))
        zi = np.hstack((z2, z1, z0))
        fnodes = np.vstack((xi, np.zeros_like(xi), zi)).T
        f_geo.set_nodes(fnodes, 1)
        return f_geo

    def create_half_deltaz_v3(self, ds: float,  # length of the mesh
                              length: float,  # length of the tunnel
                              radius: float):  # radius of the tunnel
        epsilon_x = 1 / 2
        epsilon_z = 1 / 2
        epsilon_3 = 1 / 1  # radio between radii of tangent curve and pipe.
        cover_fct = 1.5
        tc_fct = 2
        self._deltaLength = ds
        self._length = length
        self._radius = radius
        # the tunnel is along z axis
        z0 = np.linspace(length / 2, ds / 2, num=np.ceil(length / ds / 2).astype(int))[1:]
        x0 = np.ones_like(z0) * radius
        # Tangent curve
        tnz = np.ceil(epsilon_3 * radius / ds).astype(int)
        r_cv = ds * tnz
        z1 = np.flipud(np.arange(tnz) * ds + length / 2)
        x1 = (r_cv ** tc_fct - (z1 - length / 2) ** tc_fct) ** (1 / tc_fct) + radius - r_cv
        # cover
        num = np.ceil(cover_fct * x1[0] / ds).astype(int)
        x2 = np.linspace(0, x1[0], num=num)[:np.ceil(-2 * cover_fct).astype(int)]
        z2 = np.ones_like(x2) * length / 2 + r_cv
        # half part
        xi = np.hstack((x2, x1, x0))
        zi = np.hstack((z2, z1, z0))
        self._nodes = np.zeros((xi.size, 3), order='F')
        self._nodes[:, 0] = xi.flatten(order='F')
        self._nodes[:, 1] = np.zeros_like(xi).flatten(order='F')
        self._nodes[:, 2] = zi.flatten(order='F')
        self.set_dmda()

        self._u = np.zeros(self._nodes.size)
        self._normal = np.zeros((self._nodes.shape[0], 2), order='F')
        self._geo_norm = np.array((0, 0, 1))

        # associated force geo
        f_geo = self.copy()
        epsilon_x = epsilon_x / cover_fct
        a = (radius - ds * epsilon_x * 2) / radius
        b = ds * epsilon_x
        x0 = a * x0 + b
        z0 = z0 - ds * epsilon_z
        x1 = a * x1 + b
        z1 = z1 - ds * epsilon_z
        x2 = a * x2 + b
        z2 = np.ones_like(x2) * length / 2 - ds * epsilon_z + r_cv
        # half part
        xi = np.hstack((x2, x1, x0))
        zi = np.hstack((z2, z1, z0))
        fnodes = np.vstack((xi, np.zeros_like(xi), zi)).T
        f_geo.set_nodes(fnodes, 1)
        return f_geo


class pipe_cover_geo(tunnel_geo):
    def __init__(self):
        super().__init__()
        self._cover_node_list = uniqueList()
        self._type = 'pipe_cover_geo'  # geo type

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
        nodes_z = np.sign(nodes_z) * (np.exp(np.abs(nodes_z) * z_factor) - 1) / (
                np.exp(z_factor) - 1) * length / 2
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
            ni = np.ceil(2 * np.pi * ri[i0] / deltaLength).astype(int)
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


class supHelix(ThickLine_base_geo):
    _helix_right_hand = lambda self, R, B, s: np.vstack(
            (R * np.cos((B ** 2 + R ** 2) ** (-1 / 2) * s),
             R * np.sin((B ** 2 + R ** 2) ** (-1 / 2) * s),
             B * (B ** 2 + R ** 2) ** (-1 / 2) * s)).T
    _helix_left_hand = lambda self, R, B, s: np.vstack(
            (R * np.sin((B ** 2 + R ** 2) ** (-1 / 2) * s),
             R * np.cos((B ** 2 + R ** 2) ** (-1 / 2) * s),
             B * (B ** 2 + R ** 2) ** (-1 / 2) * s)).T
    _T_frame_right_hand = lambda self, R, B, s: np.vstack(
            ((-1) * R * (B ** 2 + R ** 2) ** (-1 / 2) * np.sin((B ** 2 + R ** 2) ** (-1 / 2) * s),
             R * (B ** 2 + R ** 2) ** (-1 / 2) * np.cos((B ** 2 + R ** 2) ** (-1 / 2) * s),
             B * (B ** 2 + R ** 2) ** (-1 / 2) * np.ones_like(s))).T
    _N_frame_right_hand = lambda self, R, B, s: np.vstack(
            ((-1) * np.cos((B ** 2 + R ** 2) ** (-1 / 2) * s),
             (-1) * np.sin((B ** 2 + R ** 2) ** (-1 / 2) * s),
             np.zeros_like(s))).T
    _B_frame_right_hand = lambda self, R, B, s: np.vstack(
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

    def dbg_frame(self, R, B, s):
        print(self._helix_right_hand(R, B, s))
        print(self._T_frame_right_hand(R, B, s))
        print(self._N_frame_right_hand(R, B, s))
        print(self._B_frame_right_hand(R, B, s))
        # print(self._helix(R, B, s).shape)
        # print(self._T_frame(R, B, s).shape)
        # print(self._N_frame(R, B, s).shape)
        # print(self._B_frame(R, B, s).shape)
        print('N[r[%f,%f,%f]]' % (R, B, s))
        print('N[T0[%f,%f,%f]]' % (R, B, s))
        print('N[N0[%f,%f,%f]]' % (R, B, s))
        print('N[B0[%f,%f,%f]]' % (R, B, s))

    def __init__(self):
        super().__init__()
        self._R = 0  # major radius of helix
        self._rho = 0  # minor radius of helix
        self._B = 0  # B = pitch / (2 * np.pi)
        self._n_c = 0  # number of period
        self._type = 'supHelix'  # geo type

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
                (R ** 2 + B ** 2) ** (1 / 2) * (
                R + (-1) * r * cos((r ** 2 + b ** 2) ** (-1 / 2) * s)) * cos(
                b * ((r ** 2 + b ** 2) * (R ** 2 + B ** 2)) ** (-1 / 2) * s) + r * B * sin(
                (r ** 2 + b ** 2) ** (-1 / 2) * s) * sin(
                b * ((r ** 2 + b ** 2) * (R ** 2 + B ** 2)) ** (-1 / 2) * s))
        sH2 = lambda s: (R ** 2 + B ** 2) ** (-1 / 2) * (
                (-1) * r * B * cos(
                b * ((r ** 2 + b ** 2) * (R ** 2 + B ** 2)) ** (-1 / 2) * s) * sin(
                (r ** 2 + b ** 2) ** (-1 / 2) * s) + (R ** 2 + B ** 2) ** (1 / 2) * (
                        R + (-1) * r * cos((r ** 2 + b ** 2) ** (-1 / 2) * s)) * sin(
                b * ((r ** 2 + b ** 2) * (R ** 2 + B ** 2)) ** (-1 / 2) * s))
        sH3 = lambda s: ((r ** 2 + b ** 2) * (R ** 2 + B ** 2)) ** (-1 / 2) * (
                b * B * s + r * R * (r ** 2 + b ** 2) ** (1 / 2) * sin(
                (r ** 2 + b ** 2) ** (-1 / 2) * s))
        sHf1 = lambda s: (R ** 2 + B ** 2) ** (-1 / 2) * (
                (R ** 2 + B ** 2) ** (1 / 2) * (
                R + (-1) * af * cos((r ** 2 + b ** 2) ** (-1 / 2) * s)) * cos(
                b * ((r ** 2 + b ** 2) * (R ** 2 + B ** 2)) ** (-1 / 2) * s) + af * B * sin(
                (r ** 2 + b ** 2) ** (-1 / 2) * s) * sin(
                b * ((r ** 2 + b ** 2) * (R ** 2 + B ** 2)) ** (-1 / 2) * s))
        sHf2 = lambda s: (R ** 2 + B ** 2) ** (-1 / 2) * (
                (-1) * af * B * cos(
                b * ((r ** 2 + b ** 2) * (R ** 2 + B ** 2)) ** (-1 / 2) * s) * sin(
                (r ** 2 + b ** 2) ** (-1 / 2) * s) + (R ** 2 + B ** 2) ** (1 / 2) * (
                        R + (-1) * af * cos((r ** 2 + b ** 2) ** (-1 / 2) * s)) * sin(
                b * ((r ** 2 + b ** 2) * (R ** 2 + B ** 2)) ** (-1 / 2) * s))
        sHf3 = lambda s: ((r ** 2 + b ** 2) * (R ** 2 + B ** 2)) ** (-1 / 2) * (
                b * B * s + R * af * (r ** 2 + b ** 2) ** (1 / 2) * sin(
                (r ** 2 + b ** 2) ** (-1 / 2) * s))

        from scipy import optimize as sop
        b = 2 ** (-1 / 2) * (n_node ** (-2) * n_c * (
                B ** 2 * n_c + n_c * R ** 2 + (B ** 2 + R ** 2) ** (1 / 2) * (
                B ** 2 * n_c ** 2 + 4 * n_node ** 2 * r ** 2 + n_c ** 2 * R ** 2) ** (
                        1 / 2))) ** (
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
        fgeo = base_geo()
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
            self._frenetFrame = (self._T_frame_left_hand(R, B, s),
                                 self._N_frame_left_hand(R, B, s),
                                 self._B_frame_left_hand(R, B, s))
            self._axisNodes = self._helix_left_hand(R, B, s)
        else:
            self._frenetFrame = (self._T_frame_right_hand(R, B, s),
                                 self._N_frame_right_hand(R, B, s),
                                 self._B_frame_right_hand(R, B, s))
            self._axisNodes = self._helix_right_hand(R, B, s)
        return self._axisNodes, self._frenetFrame[0], self._frenetFrame[1], self._frenetFrame[2]

    def _get_fgeo_axis(self, epsilon):
        R = self._R
        B = self._B
        n_c = self._n_c
        factor = self._factor
        left_hand = self._left_hand
        length = np.sqrt(R ** 2 + B ** 2) * 2 * np.pi * n_c
        nl = self._axisNodes.shape[0]
        ds = -self.get_deltaLength() * epsilon
        s = self._factor_fun(nl, factor) * (length - ds / 2) + ds / 4 - length / 2
        if left_hand:
            frenetFrame = (self._T_frame_left_hand(R, B, s),
                           self._N_frame_left_hand(R, B, s),
                           self._B_frame_left_hand(R, B, s))
            axisNodes = self._helix_left_hand(R, B, s)
        else:
            frenetFrame = (self._T_frame_right_hand(R, B, s),
                           self._N_frame_right_hand(R, B, s),
                           self._B_frame_right_hand(R, B, s))
            axisNodes = self._helix_right_hand(R, B, s)
        return axisNodes, frenetFrame[0], frenetFrame[1], frenetFrame[2]


class FatHelix(ThickLine_base_geo):
    # here, s is a angle.
    _helix_right_hand = lambda self, R1, R2, B, s: np.vstack((R1 * np.cos(s),
                                                              R2 * np.sin(s),
                                                              B * s)).T
    _T_frame_right_hand = lambda self, R1, R2, B, s: np.vstack(
            (-((R1 * np.sin(s)) / np.sqrt(
                    B ** 2 + np.abs(R2 * np.cos(s)) ** 2 + np.abs(R1 * np.sin(s)) ** 2)),
             (R2 * np.cos(s)) / np.sqrt(
                     B ** 2 + np.abs(R2 * np.cos(s)) ** 2 + np.abs(R1 * np.sin(s)) ** 2),
             B / np.sqrt(B ** 2 + np.abs(R2 * np.cos(s)) ** 2 + np.abs(R1 * np.sin(s)) ** 2))).T
    _N_frame_right_hand = lambda self, R1, R2, B, s: np.vstack(
            (-((np.sqrt(2) * R1 * (B ** 2 + R2 ** 2) * np.cos(s)) / np.sqrt(
                    (2 * R1 ** 2 * R2 ** 2 + B ** 2 * (
                            R1 ** 2 + R2 ** 2) + B ** 2 * (R1 - R2) * (R1 + R2) * np.cos(2 * s)) * (
                            B ** 2 + R2 ** 2 * np.cos(
                            s) ** 2 + R1 ** 2 * np.sin(s) ** 2))),
             -((np.sqrt(2) * (B ** 2 + R1 ** 2) * R2 * np.sin(s)) / np.sqrt(
                     (2 * R1 ** 2 * R2 ** 2 + B ** 2 * (
                             R1 ** 2 + R2 ** 2) + B ** 2 * (R1 - R2) * (R1 + R2) * np.cos(
                             2 * s)) * (
                             B ** 2 + R2 ** 2 * np.cos(
                             s) ** 2 + R1 ** 2 * np.sin(s) ** 2))),
             (B * (-R1 ** 2 + R2 ** 2) * np.sin(2 * s)) / (np.sqrt(2) * np.sqrt(
                     (2 * R1 ** 2 * R2 ** 2 + B ** 2 * (R1 ** 2 + R2 ** 2) + B ** 2 * (R1 - R2) * (
                             R1 + R2) * np.cos(
                             2 * s)) * (
                             B ** 2 + R2 ** 2 * np.cos(s) ** 2 + R1 ** 2 * np.sin(s) ** 2))))).T
    _B_frame_right_hand = lambda self, R1, R2, B, s: np.vstack(
            ((np.sqrt(2) * B * R2 * np.sin(s)) / np.sqrt(
                    2 * R1 ** 2 * R2 ** 2 + B ** 2 * (R1 ** 2 + R2 ** 2) + B ** 2 * (R1 - R2) * (
                            R1 + R2) * np.cos(
                            2 * s)),
             -((np.sqrt(2) * B * R1 * np.cos(s)) / np.sqrt(
                     2 * R1 ** 2 * R2 ** 2 + B ** 2 * (R1 ** 2 + R2 ** 2) + B ** 2 * (R1 - R2) * (
                             R1 + R2) * np.cos(2 * s))),
             (np.sqrt(2) * R1 * R2) / np.sqrt(
                     2 * R1 ** 2 * R2 ** 2 + B ** 2 * (R1 ** 2 + R2 ** 2) + B ** 2 * (R1 - R2) * (
                             R1 + R2) * np.cos(
                             2 * s)))).T
    _helix_left_hand = lambda self, R1, R2, B, s: np.vstack((R1 * np.cos(s),
                                                             R2 * -np.sin(s),
                                                             B * s)).T
    _T_frame_left_hand = lambda self, R1, R2, B, s: np.vstack(
            (-((R1 * np.sin(s)) / np.sqrt(
                    B ** 2 + np.abs(R2 * np.cos(s)) ** 2 + np.abs(R1 * np.sin(s)) ** 2)),
             -((R2 * np.cos(s)) / np.sqrt(
                     B ** 2 + np.abs(R2 * np.cos(s)) ** 2 + np.abs(R1 * np.sin(s)) ** 2)),
             B / np.sqrt(B ** 2 + np.abs(R2 * np.cos(s)) ** 2 + np.abs(R1 * np.sin(s)) ** 2))).T
    _N_frame_left_hand = lambda self, R1, R2, B, s: np.vstack(
            (-((np.sqrt(2) * R1 * (B ** 2 + R2 ** 2) * np.cos(s)) /
               np.sqrt((2 * R1 ** 2 * R2 ** 2 + B ** 2 * (R1 ** 2 + R2 ** 2) + B ** 2 * (
                       R1 - R2) * (R1 + R2) * np.cos(2 * s)) *
                       (B ** 2 + R2 ** 2 * np.cos(s) ** 2 + R1 ** 2 * np.sin(s) ** 2))),
             (np.sqrt(2) * (B ** 2 + R1 ** 2) * R2 * np.sin(s)) /
             np.sqrt((2 * R1 ** 2 * R2 ** 2 + B ** 2 * (R1 ** 2 + R2 ** 2) + B ** 2 * (R1 - R2) * (
                     R1 + R2) * np.cos(2 * s)) *
                     (B ** 2 + R2 ** 2 * np.cos(s) ** 2 + R1 ** 2 * np.sin(s) ** 2)),
             (B * (-R1 ** 2 + R2 ** 2) * np.sin(2 * s)) /
             (np.sqrt(2) * np.sqrt((2 * R1 ** 2 * R2 ** 2 + B ** 2 * (R1 ** 2 + R2 ** 2) +
                                    B ** 2 * (R1 - R2) * (R1 + R2) * np.cos(2 * s)) * (
                                           B ** 2 + R2 ** 2 * np.cos(s) ** 2 + R1 ** 2 * np.sin(
                                           s) ** 2))))).T
    _B_frame_left_hand = lambda self, R1, R2, B, s: np.vstack(
            (-((np.sqrt(2) * B * R2 * np.sin(s)) /
               np.sqrt(2 * R1 ** 2 * R2 ** 2 + B ** 2 * (R1 ** 2 + R2 ** 2) + B ** 2 * (R1 - R2) * (
                       R1 + R2) * np.cos(2 * s))),
             -((np.sqrt(2) * B * R1 * np.cos(s)) /
               np.sqrt(2 * R1 ** 2 * R2 ** 2 + B ** 2 * (R1 ** 2 + R2 ** 2) + B ** 2 * (R1 - R2) * (
                       R1 + R2) * np.cos(2 * s))),
             -((np.sqrt(2) * R1 * R2) /
               np.sqrt(2 * R1 ** 2 * R2 ** 2 + B ** 2 * (R1 ** 2 + R2 ** 2) + B ** 2 * (R1 - R2) * (
                       R1 + R2) * np.cos(2 * s))))).T

    def dbg_frame_right_hand(self, R1, R2, B, s):
        print(self._helix_right_hand(R1, R2, B, s))
        print(self._T_frame_right_hand(R1, R2, B, s))
        print(self._N_frame_right_hand(R1, R2, B, s))
        print(self._B_frame_right_hand(R1, R2, B, s))
        # print(self._helix(R1, R2, B, s).shape)
        # print(self._T_frame(R1, R2, B, s).shape)
        # print(self._N_frame(R1, R2, B, s).shape)
        # print(self._B_frame(R1, R2, B, s).shape)
        print('N[r[%f,%f,%f,%f]]' % (R1, R2, B, s))
        print('N[T0[%f,%f,%f,%f]]' % (R1, R2, B, s))
        print('N[N0[%f,%f,%f,%f]]' % (R1, R2, B, s))
        print('N[B0[%f,%f,%f,%f]]' % (R1, R2, B, s))

    def dbg_frame_left_hand(self, R1, R2, B, s):
        print(self._helix_left_hand(R1, R2, B, s))
        print(self._T_frame_left_hand(R1, R2, B, s))
        print(self._N_frame_left_hand(R1, R2, B, s))
        print(self._B_frame_left_hand(R1, R2, B, s))
        print('N[r[%f,%f,%f,%f]]' % (R1, R2, B, s))
        print('N[T0[%f,%f,%f,%f]]' % (R1, R2, B, s))
        print('N[N0[%f,%f,%f,%f]]' % (R1, R2, B, s))
        print('N[B0[%f,%f,%f,%f]]' % (R1, R2, B, s))

    def __init__(self):
        super().__init__()
        self._R1 = 0  # 1th major radius of helix
        self._R2 = 0  # 2th major radius of helix
        self._rho = 0  # minor radius of helix
        self._ph = 0  # pitch of helix
        self._B = 0  # B = pitch / (2 * np.pi)
        self._n_c = 0  # number of period
        self._type = 'FatHelix'  # geo type

    @property
    def rt11(self):
        return self._R1

    @property
    def rt12(self):
        return self._R2

    @property
    def rt2(self):
        return self._rho

    @property
    def R1(self):
        return self._R1

    @property
    def R2(self):
        return self._R2

    @property
    def rho(self):
        return self._rho

    @property
    def ph(self):
        return self._ph

    @property
    def B(self):
        return self._B

    @property
    def ch(self):
        return self._n_c

    def create_deltatheta(self, dth: float, radius: float, R1, R2, B, n_c,
                          epsilon=0, with_cover=False, factor=1, left_hand=False):
        # definition of parameters see self.__init__()
        # err_msg = 'current version only have right hand helix. '
        # assert not left_hand, err_msg
        err_msg = 'R1 must >= R2'
        assert R1 >= R2, err_msg

        self._R1 = R1
        self._R2 = R2
        self._rho = radius
        self._B = B
        self._ph = B * 2 * np.pi
        self._n_c = n_c
        self._factor = factor
        self._left_hand = left_hand
        return self._create_deltatheta(dth, radius, epsilon, with_cover)

    def _get_axis(self):
        R1 = self._R1
        R2 = self._R2
        B = self._B
        n_c = self._n_c
        factor = self._factor
        ds = self.get_deltaLength()
        left_hand = self._left_hand
        # length of center axis
        t = ((R1 - R2) / (R1 + R2)) ** 2
        ellipse_length = np.pi * (R1 + R2) * (1 + 3 * t / (10 + np.sqrt(4 - 3 * t)))
        length = np.sqrt(ellipse_length ** 2 + (B * 2 * np.pi) ** 2) * n_c
        nl = np.ceil(length / ds).astype(int)
        max_theta = 2 * np.pi * n_c
        s = self._factor_fun(nl, factor) * max_theta - max_theta / 2
        if left_hand:
            self._frenetFrame = (self._T_frame_left_hand(R1, R2, B, s),
                                 self._N_frame_left_hand(R1, R2, B, s),
                                 self._B_frame_left_hand(R1, R2, B, s))
            self._axisNodes = self._helix_left_hand(R1, R2, B, s)
        else:
            self._frenetFrame = (self._T_frame_right_hand(R1, R2, B, s),
                                 self._N_frame_right_hand(R1, R2, B, s),
                                 self._B_frame_right_hand(R1, R2, B, s))
            self._axisNodes = self._helix_right_hand(R1, R2, B, s)
        return self._axisNodes, self._frenetFrame[0], self._frenetFrame[1], self._frenetFrame[2]

    def _get_fgeo_axis(self, epsilon):
        R1 = self._R1
        R2 = self._R2
        B = self._B
        n_c = self._n_c
        factor = self._factor
        left_hand = self._left_hand
        # length of center axis
        t = ((R1 - R2) / (R1 + R2)) ** 2
        ellipse_length = np.pi * (R1 + R2) * (1 + 3 * t / (10 + np.sqrt(4 - 3 * t)))
        length = np.sqrt(ellipse_length ** 2 + (B * 2 * np.pi) ** 2) * n_c
        nl = self._axisNodes.shape[0]
        ds = -self.get_deltaLength() * epsilon
        max_theta = 2 * np.pi * n_c * (length - ds / 2) / length
        s = self._factor_fun(nl, factor) * max_theta - max_theta / 2
        if left_hand:
            frenetFrame = (self._T_frame_left_hand(R1, R2, B, s),
                           self._N_frame_left_hand(R1, R2, B, s),
                           self._B_frame_left_hand(R1, R2, B, s))
            axisNodes = self._helix_left_hand(R1, R2, B, s)
        else:
            frenetFrame = (self._T_frame_right_hand(R1, R2, B, s),
                           self._N_frame_right_hand(R1, R2, B, s),
                           self._B_frame_right_hand(R1, R2, B, s))
            axisNodes = self._helix_right_hand(R1, R2, B, s)
        return axisNodes, frenetFrame[0], frenetFrame[1], frenetFrame[2]


class SelfRepeat_body_geo(base_geo):
    def __init__(self, repeat_n, ph, ):
        super().__init__()
        self._repeat_n = repeat_n
        self._ph = ph

    @property
    def ph(self):
        return self._ph

    @property
    def repeat_n(self):
        return self._repeat_n

    def get_bodyi_nodes(self, repeat_i):
        repeat_n = self._repeat_n
        ph = self._ph
        geo_norm = self.get_geo_norm()
        nodes = self.get_nodes()
        tmove = (repeat_i - (repeat_n - 1) / 2) * ph * geo_norm
        nodes_bodyi = nodes + tmove
        return nodes_bodyi


class SelfRepeat_FatHelix(FatHelix):
    def __init__(self, repeat_n):
        super().__init__()
        self._repeat_n = repeat_n  # repeat the period in the middle of geo _repeat_n times
        self._local_rot = np.isclose(repeat_n, 1)
        # self._start_dmda = None  # the start part
        # self._body0_dmda = None  # the middle part
        # self._end_dmda = None  # the end part

    # def set_dmda(self):
    #     def _process_dmda(t_dmda, sizes):
    #         if t_dmda is not None:
    #             t_dmda.destroy()
    #         t_dmda = PETSc.DMDA().create(sizes=(sizes,), dof=self._dof, comm=comm,
    #                                      stencil_width=self._stencil_width, )
    #         t_dmda.setFromOptions()
    #         t_dmda.setUp()
    #         # t_dmda.createGlobalVector()
    #         return t_dmda
    #
    #     super().set_dmda()
    #     comm = PETSc.COMM_WORLD
    #
    #     self._n_start_node = self.get_start_nodes().shape[0]
    #     self._start_dmda = _process_dmda(self.get_start_dmda(), self._n_start_node)
    #
    #     self._n_body0_node = self.get_body0_nodes().shape[0]
    #     self._body0_dmda = _process_dmda(self.get_body0_dmda(), self._n_body0_node)
    #
    #     self._n_end_node = self.get_end_nodes().shape[0]
    #     self._end_dmda = _process_dmda(self.get_end_dmda(), self._n_end_node)
    #     return True
    #
    # def destroy_dmda(self):
    #     self._dmda.destroy()
    #     self._dmda = None
    #     self._start_dmda.destroy()
    #     self._start_dmda = None
    #     self._body0_dmda.destroy()
    #     self._body0_dmda = None
    #     self._end_dmda.destroy()
    #     self._end_dmda = None
    #     return True
    #
    # def pickmyself_prepare(self):
    #     if (self._dmda is not None) or (self._start_dmda is not None) or \
    #             (self._body0_dmda is not None) or (self._end_dmda is not None):
    #         self.destroy_dmda()
    #     return True
    #
    # def get_start_dmda(self):
    #     return self._start_dmda
    #
    # def get_body0_dmda(self):
    #     return self._body0_dmda
    #
    # def get_end_dmda(self):
    #     return self._end_dmda

    def create_deltatheta(self, dth: float, radius: float, R1, R2, B, n_c,
                          epsilon=0, with_cover=False, factor=1, left_hand=False):
        err_msg = 'the helix needs at least 1 period for the repeat case. '
        assert n_c > 1, err_msg
        err_msg = 'assert hfct == 1 for the repeat case. '
        assert np.isclose(factor, 1), err_msg
        return super().create_deltatheta(dth, radius, R1, R2, B, n_c,
                                         epsilon, with_cover, factor,
                                         left_hand)

    def get_n_start_node(self):
        return self.get_start_nodes().shape[0]

    def get_n_body0_node(self):
        return self.get_body0_nodes().shape[0]

    def get_n_end_node(self):
        return self.get_end_nodes().shape[0]

    def get_start_idx(self):
        ph = self._ph
        center = self.get_center()
        geo_norm = self.get_geo_norm()

        idx_start = []
        idx_start.append(self.cover_strat_idx)
        tdst = np.einsum('i,...i', geo_norm, self.axisNodes - center) / np.linalg.norm(geo_norm)
        t1 = tdst < -0.5 * ph  # one end part
        for i0 in compress(self.body_idx_list, t1):
            idx_start.append(i0)
        idx_start = np.hstack(idx_start).astype('int')
        return idx_start

    def get_start_nodes(self):
        repeat_n = self._repeat_n
        ph = self._ph
        geo_norm = self.get_geo_norm()
        idx_start = self.get_start_idx()
        nodes_start = self.get_nodes()[idx_start]
        tmove = -1 * (repeat_n - 1) * ph * geo_norm / 2
        nodes_start = nodes_start + tmove
        return nodes_start

    def get_start_velocity(self):
        idx_start = self.get_start_idx()
        t1 = self.get_velocity().reshape((-1, self.get_dof()))[idx_start]
        return t1

    def get_end_idx(self):
        ph = self._ph
        center = self.get_center()
        geo_norm = self.get_geo_norm()

        idx_end = []
        tdst = np.einsum('i,...i', geo_norm, self.axisNodes - center) / np.linalg.norm(geo_norm)
        t2 = tdst > 0.5 * ph  # another end part
        for i0 in compress(self.body_idx_list, t2):
            idx_end.append(i0)
        idx_end.append(self.cover_end_idx)
        idx_end = np.hstack(idx_end).astype('int')
        return idx_end

    def get_end_nodes(self):
        repeat_n = self._repeat_n
        ph = self._ph
        geo_norm = self.get_geo_norm()
        idx_end = self.get_end_idx()
        nodes_end = self.get_nodes()[idx_end]
        tmove = (repeat_n - 1) * ph * geo_norm / 2
        nodes_end = nodes_end + tmove
        return nodes_end

    def get_end_velocity(self):
        idx_end = self.get_end_idx()
        t1 = self.get_velocity().reshape((-1, self.get_dof()))[idx_end]
        return t1

    def get_body0_idx(self):
        ph = self._ph
        center = self.get_center()
        geo_norm = self.get_geo_norm()

        idx_body0 = []
        tdst = np.einsum('i,...i', geo_norm, self.axisNodes - center) / np.linalg.norm(geo_norm)
        t1 = tdst < -0.5 * ph  # one end part
        t2 = tdst > 0.5 * ph  # another end part
        t3 = np.logical_not(np.logical_or(t1, t2))
        for i0 in compress(self.body_idx_list, t3):
            idx_body0.append(i0)
        idx_body0 = np.hstack(idx_body0).astype('int')
        return idx_body0

    def get_bodyi_nodes(self, repeat_i):
        repeat_n = self._repeat_n
        ph = self._ph
        geo_norm = self.get_geo_norm()
        idx_body0 = self.get_body0_idx()
        nodes_body = self.get_nodes()[idx_body0]
        tmove = (repeat_i - (repeat_n - 1) / 2) * ph * geo_norm
        nodes_bodyi = nodes_body + tmove
        return nodes_bodyi

    def get_body0_nodes(self):
        return self.get_bodyi_nodes(0)

    def get_body0_velocity(self):
        idx_body0 = self.get_body0_idx()
        t1 = self.get_velocity().reshape((-1, self.get_dof()))[idx_body0]
        return t1

    def get_all_nodes_list(self):
        repeat_n = self._repeat_n
        nodes_start = self.get_start_nodes()
        nodes_end = self.get_end_nodes()

        # idx_body
        nodes_body_repeat_list = []
        for i0 in range(repeat_n):
            nodes_bodyi = self.get_bodyi_nodes(i0)
            nodes_body_repeat_list.append(nodes_bodyi)

        return nodes_start, np.vstack(nodes_body_repeat_list), nodes_end

    def get_all_nodes(self):
        return np.vstack(self.get_all_nodes_list())

    @property
    def repeat_n(self):
        return self._repeat_n

    def show_all_nodes(self, linestyle='-'):
        nodes_start, nodes_body_repeat, nodes_end = self.get_all_nodes_list()
        tnodes = np.vstack((nodes_start, nodes_body_repeat, nodes_end))
        cut_slice1 = np.vstack([nodes_start[-1], nodes_body_repeat[0]])
        cut_slice2 = np.vstack([nodes_body_repeat[-1], nodes_end[0]])
        tgeo = base_geo()
        tgeo.set_nodes(tnodes, self.get_deltaLength(), resetVelocity=True)
        tgeo.show_nodes(linestyle)
        axi = plt.gca()
        axi.plot(*cut_slice1.T, linestyle=linestyle, color='r')
        axi.plot(*cut_slice2.T, linestyle=linestyle, color='r')
        return True

    def get_start_geo(self):
        # ph = self._ph
        # center = self.get_center()
        # geo_norm = self.get_geo_norm()
        # tdst = np.einsum('i,...i', geo_norm, self.axisNodes - center) / np.linalg.norm(geo_norm)
        # start_axis_idx = tdst < -0.5 * ph  # one end part
        # end_axis_idx = tdst > 0.5 * ph  # another end part
        # body0_axis_idx = np.logical_not(np.logical_or(start_axis_idx, end_axis_idx))
        # tidx = self.get_start_idx()
        # tnodes = self.get_start_nodes()
        # tu = self.get_velocity()[tidx]
        # taxisNodes = self.axisNodes[start_axis_idx]
        # tfrenetFrame = [i0[start_axis_idx] for i0 in self.frenetFrame]
        # tbody_idx_list = compress(self.body_idx_list, start_axis_idx)
        # tcover_strat_idx = self.cover_strat_idx
        # tcover_end_idx = []
        # tiscover = self.get_iscover()[tidx]
        # tnormal = self.get_normal()[tidx]

        tgeo = base_geo()
        tnodes = np.asfortranarray(self.get_start_nodes())
        tu = self.get_start_velocity()
        tgeo.set_nodes(tnodes, self.get_deltaLength(), resetVelocity=False)
        tgeo.set_velocity(tu)
        tidx = self.get_start_idx()
        tgeo.set_geo_norm(self.get_geo_norm())
        tgeo.set_center(self.get_center())
        tgeo.set_dof(self.get_dof())
        tgeo.set_normal(self.get_normal()[tidx])
        return tgeo

    def get_body0_geo(self):
        tgeo = SelfRepeat_body_geo(self.repeat_n, self.ph)
        tnodes = np.asfortranarray(self.get_body0_nodes())
        tu = self.get_body0_velocity()
        tgeo.set_nodes(tnodes, self.get_deltaLength(), resetVelocity=False)
        tgeo.set_velocity(tu)
        tidx = self.get_start_idx()
        tgeo.set_geo_norm(self.get_geo_norm())
        tgeo.set_center(self.get_center())
        tgeo.set_dof(self.get_dof())
        tgeo.set_normal(self.get_normal()[tidx])
        return tgeo

    def get_body_mid_geo(self):
        tgeo = SelfRepeat_body_geo(self.repeat_n, self.ph)
        t1 = (self.repeat_n - 1) / 2
        tnodes = np.asfortranarray(self.get_bodyi_nodes(t1))
        tu = self.get_body0_velocity()
        tgeo.set_nodes(tnodes, self.get_deltaLength(), resetVelocity=False)
        tgeo.set_velocity(tu)
        tidx = self.get_body0_idx()
        tgeo.set_geo_norm(self.get_geo_norm())
        tgeo.set_center(self.get_center())
        tgeo.set_dof(self.get_dof())
        tgeo.set_normal(self.get_normal()[tidx])
        return tgeo

    def get_end_geo(self):
        tgeo = base_geo()
        tnodes = np.asfortranarray(self.get_end_nodes())
        tu = self.get_end_velocity()
        tgeo.set_nodes(tnodes, self.get_deltaLength(), resetVelocity=False)
        tgeo.set_velocity(tu)
        tidx = self.get_end_idx()
        tgeo.set_geo_norm(self.get_geo_norm())
        tgeo.set_center(self.get_center())
        tgeo.set_dof(self.get_dof())
        tgeo.set_normal(self.get_normal()[tidx])
        return tgeo


class MirrorSym_FatHelix(FatHelix):
    def get_mirror_geo(self):
        center = self.get_center()
        geo_norm = self.get_geo_norm()
        tnodes = self.get_nodes()
        tdst = np.einsum('i,...i', geo_norm, tnodes - center) / np.linalg.norm(geo_norm)
        t2 = tdst > 0  # upper half part
        idx_half = [i0 for i0 in compress(self.body_idx_list, t2)]

        tgeo = base_geo()
        tgeo.set_nodes(tnodes[idx_half], self.get_deltaLength(), resetVelocity=False)
        tgeo.set_velocity(self.get_velocity()[idx_half])
        tgeo.set_geo_norm(self.get_geo_norm())
        tgeo.set_center(self.get_center())
        tgeo.set_dof(self.get_dof())
        tgeo.set_normal(self.get_normal()[idx_half])
        return tgeo


class lineOnFatHelix(FatHelix):
    def __init__(self):
        super().__init__()
        self._type = 'lineOnFatHelix'  # geo type

    def _create_deltatheta(self, dth: float,  # delta theta of the cycle for the mesh
                           radius: float,  # radius of the cycle
                           epsilon=0, with_cover=0, theta0=0):
        # the tunnel is along z axis
        err_msg = 'dth must less than pi'
        assert dth < np.pi, err_msg
        self._dth = dth
        self._r = radius
        deltalength = self._get_deltalength()
        angleCycle = np.zeros(1)
        axisNodes, T_frame, N_frame, B_frame = self._get_axis()
        fgeo_axisNodes, fgeo_T_frame, fgeo_N_frame, fgeo_B_frame = self._get_fgeo_axis(epsilon)
        iscover = []
        vgeo_nodes = []
        fgeo_nodes = []
        epsilon = (radius + epsilon * deltalength) / radius
        if self.get_check_epsilon():
            err_msg = 'epsilon > %f. ' % (-radius / deltalength)
            assert epsilon > 0, err_msg

        # cover at start
        if with_cover == 1:
            # old version, cover is a plate.
            assert 1 == 2
            nc = np.ceil((radius - deltalength) / deltalength).astype(int)
            ri = np.linspace(deltalength / 2, radius, nc, endpoint=False)
            # self
            i0 = 0
            ai_para = ai_para + 1
            ni = np.ceil(2 * np.pi * ri[i0] / deltalength).astype(int)
            ai = np.linspace(0, 2 * np.pi, ni, endpoint=False) + (-1) ** ai_para * dth / 4
            t_cover = np.ones_like(ai, dtype=bool)
            t_cover[:] = -1
            iscover.append(t_cover)
            nodes_cycle = np.vstack(
                    (np.cos(ai) * ri[i0], np.sin(ai) * ri[i0], np.zeros_like(ai))).T
            t_nodes = axisNodes[0] + np.dot(nodes_cycle,
                                            np.vstack((N_frame[0], B_frame[0],
                                                       np.zeros_like(T_frame[0]))))
            vgeo_nodes.append(t_nodes)
            tf_nodes = fgeo_axisNodes[0] + np.dot(nodes_cycle * epsilon,
                                                  np.vstack((N_frame[0], B_frame[0],
                                                             np.zeros_like(T_frame[0]))))
            fgeo_nodes.append(tf_nodes)
            self._strat_pretreatment(t_nodes)
        elif with_cover == 2:
            # 20170929, new version, cover is a hemisphere
            vhsgeo = revolve_ellipse()
            fhsgeo = vhsgeo.create_half_delta(deltalength, radius, radius, epsilon)
            vhsgeo.node_rotation((1, 0, 0), theta0 - np.pi / 2)
            fhsgeo.node_rotation((1, 0, 0), theta0 - np.pi / 2)
            t_nodes = axisNodes[0] + np.dot(vhsgeo.get_nodes(), np.vstack(
                    (-T_frame[0], N_frame[0], B_frame[0])))
            vgeo_nodes.append(np.flipud(t_nodes))
            tf_nodes = fgeo_axisNodes[0] + np.dot(fhsgeo.get_nodes(), np.vstack(
                    (-T_frame[0], N_frame[0], B_frame[0])))
            fgeo_nodes.append(np.flipud(tf_nodes))
            self._strat_pretreatment(t_nodes)
            iscover.append(np.ones(vhsgeo.get_n_nodes(), dtype=bool))

        # body
        for i0, nodei_line in enumerate(axisNodes):
            ai = angleCycle + theta0
            nodes_cycle = np.vstack((np.cos(ai) * radius, np.sin(ai) * radius, np.zeros_like(ai))).T
            t_nodes = nodei_line + np.dot(nodes_cycle,
                                          np.vstack((N_frame[i0], B_frame[i0],
                                                     np.zeros_like(T_frame[i0]))))
            vgeo_nodes.append(t_nodes)
            t_cover = np.ones(ai.shape, dtype=bool)
            t_cover[:] = 0
            iscover.append(t_cover)
            nodes_cycle = np.vstack((np.cos(ai) * radius,
                                     np.sin(ai) * radius,
                                     np.zeros_like(ai))).T * epsilon
            tf_nodes = fgeo_axisNodes[i0] + np.dot(nodes_cycle, np.vstack(
                    (fgeo_N_frame[i0], fgeo_B_frame[i0], np.zeros_like(fgeo_T_frame[i0]))))
            fgeo_nodes.append(tf_nodes)
            self._body_pretreatment(t_nodes)

        # cover at end
        if with_cover == 1:
            assert 1 == 2
            # old version, cover is a plate.
            nc = np.ceil((radius - deltalength) / deltalength).astype(int)
            ri = np.linspace(deltalength / 2, radius, nc, endpoint=False)[-1::-1]
            i0 = 0
            ai_para = ai_para + 1
            ni = np.ceil(2 * np.pi * ri[i0] / deltalength).astype(int)
            ai = np.linspace(0, 2 * np.pi, ni, endpoint=False) + (-1) ** ai_para * dth / 4
            t_cover = np.ones_like(ai, dtype=bool)
            t_cover[:] = 1
            iscover.append(t_cover)
            nodes_cycle = np.vstack(
                    (np.cos(ai) * ri[i0], np.sin(ai) * ri[i0], np.zeros_like(ai))).T
            t_nodes = axisNodes[-1] + np.dot(nodes_cycle,
                                             np.vstack((N_frame[-1], B_frame[-1],
                                                        np.zeros_like(T_frame[-1]))))
            vgeo_nodes.append(t_nodes)
            tf_nodes = fgeo_axisNodes[-1] + np.dot(nodes_cycle * epsilon, np.vstack(
                    (fgeo_N_frame[-1], fgeo_B_frame[-1], np.zeros_like(fgeo_T_frame[-1]))))
            fgeo_nodes.append(tf_nodes)
            self._end_pretreatment(t_nodes)
        elif with_cover == 2:
            # 20170929, new version, cover is a hemisphere
            vhsgeo = revolve_ellipse()
            fhsgeo = vhsgeo.create_half_delta(deltalength, radius, radius, epsilon)
            vhsgeo.node_rotation((1, 0, 0), theta0 - np.pi / 2)
            fhsgeo.node_rotation((1, 0, 0), theta0 - np.pi / 2)
            t_nodes = axisNodes[-1] + np.dot(vhsgeo.get_nodes(), np.vstack(
                    (T_frame[-1], N_frame[-1], B_frame[-1])))
            vgeo_nodes.append(t_nodes)
            tf_nodes = fgeo_axisNodes[-1] + np.dot(fhsgeo.get_nodes(), np.vstack(
                    (T_frame[-1], N_frame[-1], B_frame[-1])))
            fgeo_nodes.append(tf_nodes)
            self._end_pretreatment(t_nodes)
            iscover.append(np.ones(vhsgeo.get_n_nodes(), dtype=bool))

        self._iscover = np.hstack(iscover)
        self._nodes = np.asfortranarray(np.vstack(vgeo_nodes))
        self.set_dmda()
        self._u = np.zeros(self._nodes.size)
        self._normal = np.zeros((self._nodes.shape[0], 2), order='F')
        fgeo = base_geo()
        fgeo.set_dof(self.get_dof())
        fgeo.set_nodes(np.asfortranarray(np.vstack(fgeo_nodes)), deltalength=deltalength * epsilon,
                       resetVelocity=True)
        return fgeo

    def create_deltatheta(self, dth: float, radius: float, R1, R2, B, n_c, epsilon=0,
                          with_cover=False, factor=1, left_hand=False, theta0=0):
        # definition of parameters see self.__init__()
        # err_msg = 'current version only have right hand helix. '
        # assert not left_hand, err_msg
        err_msg = 'R1 must >= R2'
        assert R1 >= R2, err_msg

        self._R1 = R1
        self._R2 = R2
        self._rho = radius
        self._B = B
        self._n_c = n_c
        self._factor = factor
        self._left_hand = left_hand
        return self._create_deltatheta(dth, radius, epsilon, with_cover, theta0)


class slb_geo(base_geo):
    def __init__(self, rt2):
        super().__init__()
        self._rt2 = rt2
        self._s_list = np.ones(0)

    @property
    def rt2(self):
        return self._rt2

    @property
    def s_list(self):
        return self._s_list

    @abc.abstractmethod
    # xc = xc(s), the center line function of slender body.
    def xc_fun(self, s):
        return

    @abc.abstractmethod
    # xs = xs(s, theta), the surface function of slender body.
    def xs_fun(self, s, theta):
        return

    @abc.abstractmethod
    # tangent
    def t_fun(self, s):
        return

    @abc.abstractmethod
    # normal
    def n_fun(self, s):
        return

    @abc.abstractmethod
    # binormal
    def b_fun(self, s):
        return

    @abc.abstractmethod
    def arclength(self, s):
        return

    @abc.abstractmethod
    def rho_r(self, s):
        return

    # normal component of force
    def fn_matrix(self, s):
        t = self.t_fun(s)
        tm = np.eye(3) - np.outer(t, t)
        return tm

    def r0_fun(self, s1, s2):
        r0 = np.linalg.norm(self.xc_fun(s1) - self.xc_fun(s2), axis=-1)
        return r0

    def natu_cut(self, s):
        nc = self.rt2 * self.rho_r(s) * np.sqrt(np.e) / 2
        return nc


# archived
# class _self_repeat_geo(slb_geo):
#     def __init__(self, rt2):
#         super().__init__(rt2)
#         self._type = '_self_repeat_geo'  # geo type
#         # self._start_frenetFrame = (np.array([]).reshape((-1, 3)),
#         #                            np.array([]).reshape((-1, 3)),
#         #                            np.array([]).reshape((-1, 3)))
#         # self._body_frenetFrame = (np.array([]).reshape((-1, 3)),
#         #                           np.array([]).reshape((-1, 3)),
#         #                           np.array([]).reshape((-1, 3)))
#         # self._end_frenetFrame = (np.array([]).reshape((-1, 3)),
#         #                          np.array([]).reshape((-1, 3)),
#         #                          np.array([]).reshape((-1, 3)))
#         self._dth = 0  # anglar between nodes in a cycle.
#         self._axisNodes = np.array([]).reshape((-1, 3))
#         self._frenetFrame = (np.array([]).reshape((-1, 3)),
#                              np.array([]).reshape((-1, 3)),
#                              np.array([]).reshape((-1, 3)))
#         self._iscover = []  # start: -1, body: 0, end: 1
#         self._with_cover = 0
#         self._factor = 1e-5
#         self._left_hand = False
#         self._check_epsilon = True
#         self._type = '_ThickLine_geo'  # geo type
#         self._cover_strat_idx = np.array([])
#         self._body_idx_list = []
#         self._cover_end_idx = np.array([])
#
#     def _get_deltatheta_deltalength(self):
#         def eqr(dth, ds, r):
#             return (ds / (2 * r)) ^ 2 + np.sin(dth / 4) ** 2 - np.sin(dth / 2) ** 2
#
#         from scipy import optimize as sop
#         self._dth = sop.brentq(eqr, -1e-3 * np.pi, np.pi, args=(self.get_deltaLength(), self._r))
#         dl = 2 * self._rt2 * np.sin(self._dth / 2)
#         self.set_deltaLength(dl)
#         return self._dth, dl
#
#     @abc.abstractmethod
#     def _get_axis(self, s):
#         return
#
#     @abc.abstractmethod
#     def _get_fgeo_axis(self, s, epsilon):
#         return
#
#     def _create_deltatheta(self, dth: float,  # delta theta of the cycle for the mesh
#                            radius: float,  # radius of the cycle
#                            epsilon=0, with_cover=0):
#         # means of parameters see function _create_deltatheta(...).
#         # the tunnel is along z axis
#         err_msg = 'dth must less than pi'
#         assert dth < np.pi, err_msg
#         self._dth = dth
#         self._r = radius
#         deltalength = self._get_deltalength()
#         axisNodes, T_frame, N_frame, B_frame = self._get_axis()
#         fgeo_axisNodes, fgeo_T_frame, fgeo_N_frame, fgeo_B_frame = self._get_fgeo_axis(epsilon)
#         iscover = []
#         vgeo_nodes = []
#         fgeo_nodes = []
#         epsilon = (radius + epsilon * deltalength) / radius
#         if self.get_check_epsilon():
#             err_msg = 'epsilon > %f. ' % (-radius / deltalength)
#             assert epsilon > 0, err_msg
#         ai_para = 0
#         t_node_idx = 0
#
#     def _create_cover_deltatheta(self, s, dth: float, radius: float, epsilon=0, with_cover=0):
#         # s: arc length parameter of center line
#         # means of parameters see function _create_deltatheta(...).
#         # cover at start
#
#         nc = np.ceil(2 * np.pi / dth).astype(int)
#         angleCycle = np.linspace(0, 2 * np.pi, nc, endpoint=False)
#         deltalength = self._get_deltalength()
#         axisNodes, T_frame, N_frame, B_frame = self._get_axis(s)
#         fgeo_axisNodes, fgeo_T_frame, fgeo_N_frame, fgeo_B_frame = self._get_fgeo_axis(s, epsilon)
#         iscover = []
#         vgeo_nodes = []
#         fgeo_nodes = []
#         epsilon = (radius + epsilon * deltalength) / radius
#         if self.get_check_epsilon():
#             err_msg = 'epsilon > %f. ' % (-radius / deltalength)
#             assert epsilon > 0, err_msg
#         ai_para = 0
#         t_node_idx = 0
#
#         # 20170929, new version, cover is a hemisphere
#         assert with_cover == 2
#         vhsgeo = sphere_geo()
#         vhsgeo.create_half_delta(deltalength, radius)
#         vhsgeo.node_rotation((1, 0, 0), np.pi / 2 + ai_para)
#         t_nodes = axisNodes[0] + np.dot(vhsgeo.get_nodes(),
#                                         np.vstack((-T_frame[0], N_frame[0], B_frame[0])))
#         vgeo_nodes.append(t_nodes)
#         self._cover_strat_idx = np.arange(t_nodes.shape[0]) + t_node_idx
#         t_node_idx = self._cover_strat_idx[-1] + 1
#         fhsgeo = vhsgeo.copy()
#         fhsgeo.node_zoom(epsilon)
#         tf_nodes = fgeo_axisNodes[0] + np.dot(fhsgeo.get_nodes(),
#                                               np.vstack((-T_frame[0], N_frame[0], B_frame[0])))
#         fgeo_nodes.append(tf_nodes)
#         self._strat_pretreatment(t_nodes)
#         iscover.append(np.ones(vhsgeo.get_n_nodes()) * -1)
#
#         # body
#         for i0, nodei_line in enumerate(axisNodes):
#             ai_para = ai_para + 1
#             ai = angleCycle + (-1) ** ai_para * dth / 4
#             nodes_cycle = np.vstack((np.cos(ai) * radius, np.sin(ai) * radius, np.zeros_like(ai))).T
#             t_nodes = nodei_line + np.dot(nodes_cycle,
#                                           np.vstack((N_frame[i0], B_frame[i0],
#                                                      np.zeros_like(T_frame[i0]))))
#             vgeo_nodes.append(t_nodes)
#             self._body_idx_list.append(np.arange(t_nodes.shape[0]) + t_node_idx)
#             t_node_idx = self._body_idx_list[-1][-1] + 1
#             iscover.append(np.zeros_like(ai))
#             nodes_cycle = np.vstack(
#                     (np.cos(ai) * radius, np.sin(ai) * radius, np.zeros_like(ai))).T * epsilon
#             tf_nodes = fgeo_axisNodes[i0] + np.dot(nodes_cycle, np.vstack(
#                     (fgeo_N_frame[i0], fgeo_B_frame[i0], np.zeros_like(fgeo_T_frame[i0]))))
#             fgeo_nodes.append(tf_nodes)
#             self._body_pretreatment(t_nodes)
#
#         self._iscover.append(np.hstack(iscover))
#         vstart_nodes = np.asfortranarray(np.vstack(vgeo_nodes))
#         fstart_nodes = np.asfortranarray(np.vstack(fgeo_nodes))
#         return vstart_nodes, fstart_nodes
#
#     def _create_start_deltatheta(self, s, dth: float, radius: float, epsilon=0, with_cover=0):
#         # s: arc length parameter of center line
#         # means of parameters see function _create_deltatheta(...).
#         # cover at start
#
#         nc = np.ceil(2 * np.pi / dth).astype(int)
#         angleCycle = np.linspace(0, 2 * np.pi, nc, endpoint=False)
#         deltalength = self._get_deltalength()
#         axisNodes, T_frame, N_frame, B_frame = self._get_axis(s)
#         fgeo_axisNodes, fgeo_T_frame, fgeo_N_frame, fgeo_B_frame = self._get_fgeo_axis(s, epsilon)
#         iscover = []
#         vgeo_nodes = []
#         fgeo_nodes = []
#         epsilon = (radius + epsilon * deltalength) / radius
#         if self.get_check_epsilon():
#             err_msg = 'epsilon > %f. ' % (-radius / deltalength)
#             assert epsilon > 0, err_msg
#         ai_para = 0
#         t_node_idx = 0
#
#         # 20170929, new version, cover is a hemisphere
#         assert with_cover == 2
#         vhsgeo = sphere_geo()
#         vhsgeo.create_half_delta(deltalength, radius)
#         vhsgeo.node_rotation((1, 0, 0), np.pi / 2 + ai_para)
#         t_nodes = axisNodes[0] + np.dot(vhsgeo.get_nodes(),
#                                         np.vstack((-T_frame[0], N_frame[0], B_frame[0])))
#         vgeo_nodes.append(t_nodes)
#         self._cover_strat_idx = np.arange(t_nodes.shape[0]) + t_node_idx
#         t_node_idx = self._cover_strat_idx[-1] + 1
#         fhsgeo = vhsgeo.copy()
#         fhsgeo.node_zoom(epsilon)
#         tf_nodes = fgeo_axisNodes[0] + np.dot(fhsgeo.get_nodes(),
#                                               np.vstack((-T_frame[0], N_frame[0], B_frame[0])))
#         fgeo_nodes.append(tf_nodes)
#         self._strat_pretreatment(t_nodes)
#         iscover.append(np.ones(vhsgeo.get_n_nodes()) * -1)
#
#         # body
#         for i0, nodei_line in enumerate(axisNodes):
#             ai_para = ai_para + 1
#             ai = angleCycle + (-1) ** ai_para * dth / 4
#             nodes_cycle = np.vstack((np.cos(ai) * radius, np.sin(ai) * radius, np.zeros_like(ai))).T
#             t_nodes = nodei_line + np.dot(nodes_cycle,
#                                           np.vstack((N_frame[i0], B_frame[i0],
#                                                      np.zeros_like(T_frame[i0]))))
#             vgeo_nodes.append(t_nodes)
#             self._body_idx_list.append(np.arange(t_nodes.shape[0]) + t_node_idx)
#             t_node_idx = self._body_idx_list[-1][-1] + 1
#             iscover.append(np.zeros_like(ai))
#             nodes_cycle = np.vstack(
#                     (np.cos(ai) * radius, np.sin(ai) * radius, np.zeros_like(ai))).T * epsilon
#             tf_nodes = fgeo_axisNodes[i0] + np.dot(nodes_cycle, np.vstack(
#                     (fgeo_N_frame[i0], fgeo_B_frame[i0], np.zeros_like(fgeo_T_frame[i0]))))
#             fgeo_nodes.append(tf_nodes)
#             self._body_pretreatment(t_nodes)
#
#         self._iscover.append(np.hstack(iscover))
#         vstart_nodes = np.asfortranarray(np.vstack(vgeo_nodes))
#         fstart_nodes = np.asfortranarray(np.vstack(fgeo_nodes))
#         return vstart_nodes, fstart_nodes


class slb_helix(slb_geo):
    def __init__(self, ph, ch, rt1, rt2, theta0=0):
        super().__init__(rt2)
        self._ph = ph
        self._ch = ch
        self._rt1 = rt1
        self._arc_length = np.sqrt(ph ** 2 + 4 * np.pi ** 2 * rt1 ** 2)
        self._s0 = theta0 / (2 * np.pi) * self._arc_length
        self._hlx_th = np.arctan(2 * np.pi * rt1 / ph)

    @property
    def ph(self):
        return self._ph

    @property
    def ch(self):
        return self._ch

    @property
    def rt1(self):
        return self._rt1

    @property
    def theta0(self):
        return self._s0 * 2 * np.pi / self.arclength(0)

    def xc_fun(self, s):
        ph = self.ph
        rt1 = self.rt1
        theta = s * 2 * np.pi / self.arclength(0)
        theta1 = theta + self.theta0
        xc = np.array((rt1 * np.cos(theta1),
                       rt1 * np.sin(theta1),
                       ph * s / self.arclength(0))).T
        return xc

    def xs_fun(self, s, theta):
        xc = self.xc_fun(s)
        ep = np.cos(theta) * self.n_fun(s) + np.sin(theta) * self.b_fun(s)
        xs = xc + ep * self.rt2 * self.rho_r(s)
        return xs

    def t_fun(self, s):
        ph = self.ph
        rt1 = self.rt1
        arcl = self.arclength(s)
        theta = s * 2 * np.pi / self.arclength(0)
        theta1 = theta + self.theta0
        t = np.array(((-2 * np.pi * rt1 * np.sin(theta1)),
                      (2 * np.pi * rt1 * np.cos(theta1)),
                      np.ones_like(theta1) * ph)).T / arcl
        return t

    def n_fun(self, s):
        ph = self.ph
        rt1 = self.rt1
        arcl = self.arclength(s)
        theta = s * 2 * np.pi / self.arclength(0)
        theta1 = theta + self.theta0
        n = np.array(((ph * np.sin(theta1)),
                      -((ph * np.cos(theta1))),
                      np.ones_like(theta1) * (2 * np.pi * rt1))).T / arcl
        return n

    def b_fun(self, s):
        theta = s * 2 * np.pi / self.arclength(0)
        theta1 = theta + self.theta0
        b = np.array((-np.cos(theta1), -np.sin(theta1), np.zeros_like(theta1))).T
        return b

    def arclength(self, s):
        return np.ones_like(s) * self._arc_length

    def rho_r(self, s):
        return np.ones_like(s) * 1

    def create_nSegment(self, n=None, check_nth=True):
        ch = self.ch
        ch_min, ch_max = -1 * ch / 2, ch / 2
        # ch_min, ch_max = 0, ch
        ch_mid = (ch_min + ch_max) / 2
        s_mid = ch_mid * self.arclength(0)
        self._s_list = np.array((ch_min, ch_max)) * self.arclength(s_mid)
        max_n = np.floor(ch * self.arclength(s_mid) / (self.natu_cut(s_mid) * 2)).astype(int)
        if n is None:
            n = max_n
        if check_nth:
            err_msg = 'nth is too large. nth <= %d' % max_n
            assert n <= max_n, err_msg

        dth = 2 * np.pi * (ch_max - ch_min) / n
        ds = dth / (2 * np.pi) * self.arclength(s_mid)
        th_list = np.linspace(2 * np.pi * ch_min, 2 * np.pi * ch_max, n, endpoint=False) + dth / 2
        s_list = th_list / (2 * np.pi) * self.arclength(s_mid)
        nodes = self.xc_fun(s_list)
        self._s_list = s_list
        self.set_nodes(nodes, ds, resetVelocity=True)
        self.set_geo_norm((0, 0, 1))
        self.set_origin((0, 0, self.xc_fun(s_mid)[2]))
        return True


class Johnson_helix(slb_helix):
    def rho_r(self, s):
        s_list = self.s_list
        ds = self.get_deltaLength()
        l_min = s_list[0] - ds / 2
        l_max = s_list[-1] + ds / 2
        rho_r = (-l_max * l_min + (l_max + l_min) * s - s ** 2) / ((l_max - l_min) / 2) ** 2
        return rho_r


class expJohnson_helix(Johnson_helix):
    def rho_r(self, s):
        xfct = 10
        xmove = 0.7

        s_list = self.s_list
        ds = self.get_deltaLength()
        l_min = s_list[0] - ds / 2
        l_max = s_list[-1] + ds / 2
        l_mid = (l_max + l_min) / 2
        l_len = (l_max - l_min)
        tx = 2 * np.abs(s - l_mid) / l_len
        fct = -1 * np.arctan(xfct * (tx - xmove)) / np.pi + 0.5
        rho_ra = (-l_max * l_min + 2 * l_mid * s - s ** 2) / (l_len / 2) ** 2
        rho_rb = np.ones_like(rho_ra)
        rho_r = rho_rb * fct + rho_ra * (1 - fct)
        return rho_r


class regularizeDisk(base_geo):
    def __init__(self):
        super().__init__()
        self._type = 'regularizeDisk'  # geo type

    def create_ds(self, ds, r):
        tn = np.ceil(r / ds).astype(int)
        r_list = np.linspace(ds / 2, r - ds / 2, tn)

        # generate nodes.
        x = []
        y = []
        ai_para = 0
        for ri in r_list:
            ai_para = ai_para + 1
            ni = np.ceil(2 * np.pi * ri / ds).astype(int)
            ai, da = np.linspace(0, 2 * np.pi, ni, endpoint=False, retstep=True)
            ai = ai + (-1) ** ai_para * da / 4
            x.append(ri * np.cos(ai))
            y.append(ri * np.sin(ai))
        x = np.hstack(x)
        y = np.hstack(y)
        z = np.zeros_like(x)
        self.set_nodes(nodes=np.vstack((x, y, z)).T, deltalength=ds)
        self.set_dmda()
        self._u = np.zeros_like(self.get_nodes())
        self._normal = np.zeros((self.get_nodes().shape[0], 2), order='F')
        self._geo_norm = np.array((0, 0, 1))
        return True


class helicoid(base_geo):
    def __init__(self):
        super().__init__()
        self._type = 'helicoid'  # geo type

    def create(self, r1, r2, ds, th_loc=np.pi / 4, ndsk_each=4):
        tgeo = regularizeDisk()
        tgeo.create_ds(ds, r2)
        tgeo.node_rotation(norm=np.array([1, 0, 0]), theta=th_loc)
        tgeo.move(np.array((r1, 0, 0)))
        # tgeo.show_nodes()

        tgeo_list = []
        rot_dth = 2 * np.pi / ndsk_each
        for i0 in range(ndsk_each):
            rot_th = i0 * rot_dth + rot_dth / 2
            tgeo21 = tgeo.copy()
            tgeo21.node_rotation(norm=np.array([0, 0, 1]), theta=rot_th,
                                 rotation_origin=np.zeros(3))
            tgeo22 = tgeo21.copy()
            tgeo_list.append(tgeo21)
            tgeo22.node_rotation(norm=np.array([1, 0, 0]), theta=np.pi / 2,
                                 rotation_origin=np.zeros(3))
            tgeo23 = tgeo21.copy()
            tgeo_list.append(tgeo22)
            tgeo23.node_rotation(norm=np.array([0, 1, 0]), theta=np.pi / 2,
                                 rotation_origin=np.zeros(3))
            tgeo_list.append(tgeo23)
        self.combine(tgeo_list)
        return tgeo_list


# symmetric geo with infinity length, i.e. infinite long helix, infinite long tube.
class infgeo_1d(base_geo):
    # the system have a rotational symmetry.
    # currently, assume the symmetry is along the z axis.
    def __init__(self):
        super().__init__()
        self._max_period = 0  # cut off in the range (-max_period*2*pi, max_period*2*pi).
        self._nSegment = 0  # number of subsections of geo per period
        self._type = 'infgeo_1d'  # geo type
        self._phi = ...  # type: np.ndarray # define the coordinates of nodes at the reference cross section.
        self._ph = 0  # the length of the period of the infgeo, assume the system repeat itself at z axis infinitely.

    def get_nSegment(self):
        return self._nSegment

    def get_max_period(self):
        return self._max_period

    def get_phi(self):
        return self._phi

    @abc.abstractmethod
    def coord_x123(self, theta):
        return

    @abc.abstractmethod
    def Frenetframe(self, theta):
        return

    def rot_matrix(self, theta):
        # local -> reference
        # currently, assume the symmetry is along the z axis.
        Rmxt = np.identity(3)
        Rmxt[0][0] = np.cos(theta)
        Rmxt[0][1] = -np.sin(theta)
        Rmxt[1][0] = np.sin(theta)
        Rmxt[1][1] = np.cos(theta)
        return Rmxt

    def show_segment(self, linestyle='-'):
        return super().show_nodes(linestyle)

    def show_nodes(self, linestyle='-'):
        t_nodes = []
        for ni in np.arange(-self.get_max_period(), self.get_max_period()):
            for thi in np.linspace(0, 2 * np.pi, self.get_nSegment(), endpoint=False):
                th = ni * 2 * np.pi + thi
                t_nodes.append(self.coord_x123(th))
        t_nodes = np.vstack(t_nodes)
        t_geo = base_geo()
        t_geo.set_nodes(t_nodes, deltalength=0)
        return t_geo.show_nodes(linestyle)

    def print_info(self):
        PETSc.Sys.Print('    %s: # of Segment: %d' % (self.get_type(), self.get_nSegment()))
        return True


# a infinite long helix along z axis
class infHelix(infgeo_1d):
    def __init__(self):
        super().__init__()  # here max_theta means the cut off max theta of helix
        self._R = 0  # major radius of helix
        self._rho = 0  # minor radius of helix
        self._type = 'infHelix'  # geo type
        self._theta0 = 0  # define the reference location (original rotation) of the helix (for multi helix)

    def coord_x123(self, th):
        R = self._R
        rho = self._rho
        ph = self._ph
        phi = self._phi
        th1 = th % (2 * np.pi) + self._theta0
        # th = th + self._theta0

        # definition of parameters see __init__()
        # x1, x2, x3, coordinates of helix nodes
        x1 = lambda theta: np.cos(theta) * (R - rho * np.sin(phi)) + (
                ph * rho * np.cos(phi) * np.sin(
                theta)) / np.sqrt(ph ** 2 + 4 * np.pi ** 2 * R ** 2)
        x2 = lambda theta: - (ph * rho * np.cos(phi) * np.cos(theta)) / np.sqrt(
                ph ** 2 + 4 * np.pi ** 2 * R ** 2) + (R - rho * np.sin(phi)) * np.sin(theta)
        x3 = lambda theta: (ph * theta) / (2. * np.pi) + (
                2 * np.pi * R * rho * np.cos(phi)) / np.sqrt(
                ph ** 2 + 4 * np.pi ** 2 * R ** 2)
        return np.vstack((x1(th1), x2(th1), x3(th))).T

    def Frenetframe(self, th):
        th = th % (2 * np.pi) + self._theta0
        ph = self._ph
        lh = 2 * np.pi * self._R
        s = np.sqrt(lh ** 2 + ph ** 2)
        T = np.array((-lh * np.sin(th) / s, lh * np.cos(th) / s, ph / s))
        N = np.array((ph * np.sin(th) / s, -ph * np.cos(th) / s, lh / s))
        B = np.array((-np.cos(th), -np.sin(th), 0))
        return T, N, B

    def create_n(self, R, rho, ph, ch, nth, theta0=0, nSegment=None):
        self._max_period = ch
        if nSegment is None:
            nSegment = np.ceil(np.sqrt(ph ** 2 + (2 * np.pi * R) ** 2) / (2 * np.pi * rho)) * nth
        nSegment = int(nSegment)
        self._nSegment = nSegment
        self._R = R
        self._rho = rho
        self._ph = ph
        self._theta0 = theta0
        self._phi = np.linspace(0, 2 * np.pi, nth, endpoint=False)
        self._nodes = self.coord_x123(0)
        self.set_deltaLength(2 * np.pi * rho / nth)
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
        fgeo = infHelix()
        deltalength = self.get_deltaLength()
        f_rho = (self._rho + epsilon * deltalength)
        err_msg = 'epsilon > %f. ' % (-self._rho / deltalength)
        assert f_rho > 0, err_msg
        fgeo.create_n(self._R, f_rho, self._ph, self.get_max_period(), self.get_n_nodes(),
                      self._theta0, self.get_nSegment())
        return fgeo


# a infinite long pipe along z axis
class infPipe(infgeo_1d):
    def __init__(self):
        super().__init__()
        self._R = 0  # radius of pipe
        # self._theta = 0  # the angle between the cut plane and the z axis
        self._type = 'infPipe'  # geo type

    def coord_x123(self, th):
        # return coordinates of inf pipe nodes
        xz = th / (2 * np.pi) * self._ph
        R = self._R
        phi = (self._phi + th) % (2 * np.pi)
        return np.vstack((np.cos(phi) * R, np.sin(phi) * R, np.ones_like(phi) * xz)).T

    def create_n(self, R, ph, ch, nth, nSegment=None):
        deltaLength = 2 * np.pi * R / nth
        if nSegment is None:
            nSegment = np.ceil(ph / deltaLength)
        self._max_period = ch
        self._nSegment = nSegment
        self._R = R
        self._ph = ph
        self._phi = np.linspace(0, 2 * np.pi, nth, endpoint=False)
        self._nodes = self.coord_x123(0)
        self.set_deltaLength(deltaLength)
        self.set_origin((0, 0, 0))
        self._u = np.zeros(self._nodes.size)
        self.set_dmda()
        return True

    def create_fgeo(self, epsilon):
        fgeo = infPipe()
        deltalength = self.get_deltaLength()
        f_R = self._R + epsilon * deltalength
        err_msg = 'epsilon > %f. ' % (-self._R / deltalength)
        assert f_R > 0, err_msg
        fgeo.create_n(f_R, self._ph, self.get_max_period(), self.get_n_nodes(), self.get_nSegment())
        return fgeo


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
        [full_region_x, full_region_y, full_region_z] = \
            np.meshgrid(full_region_x, full_region_y, full_region_z, indexing='ij')

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
        [full_region_x, temp_r, temp_theta] = np.meshgrid(full_region_x, full_region_r,
                                                          full_region_theta,
                                                          indexing='ij')
        full_region_y = temp_r * np.cos(temp_theta)
        full_region_z = temp_r * np.sin(temp_theta)

        return full_region_x, full_region_y, full_region_z


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])

    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])

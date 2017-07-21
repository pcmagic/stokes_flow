# coding=utf-8
import copy
import numpy as np
from numpy import sin, cos
import scipy.io as sio
from petsc4py import PETSc
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from src.support_class import *
import abc

__all__ = ['geo', 'sphere_geo', 'ellipse_geo', 'tunnel_geo', 'stokeslets_tunnel_geo', 'pipe_cover_geo', 'supHelix',
           'region', 'createEcoli']


class geo():
    def __init__(self):
        self._nodes = np.array([])
        self._elems = np.array([])
        self._elemtype = ' '
        self._normal = np.array([])  # norm of surface at each point.
        self._origin = np.array((0, 0, 0))
        self._u = np.array([])
        self._deltaLength = 0
        self._dmda = None  # dof management
        self._stencil_width = 0  # --->>>if change in further version, deal with combine method.
        self._glbIdx = np.array([])  # global indices
        self._glbIdx_all = np.array([])  # global indices for all process.

    def mat_nodes(self,
                  filename: str = '..',
                  mat_handle: str = 'nodes'):
        err_msg = 'wrong mat file name. '
        assert filename != '..', err_msg

        mat_contents = sio.loadmat(filename)
        nodes = mat_contents[mat_handle].astype(np.float, order='F')
        err_msg = 'nodes is a n*3 numpy array containing x, y and z coordinates. '
        assert nodes.shape[1] == 3, err_msg
        self._nodes = nodes
        self._u = np.zeros(self._nodes.size)
        self.set_dmda()
        return True

    def mat_elmes(self,
                  filename: str = '..',
                  mat_handle: str = 'elmes',
                  elemtype: str = ' '):
        err_msg = 'wrong mat file name. '
        assert filename != '..', err_msg

        mat_contents = sio.loadmat(filename)
        elems = mat_contents[mat_handle].astype(np.int, order='F')
        elems = elems - elems.min()
        self._elems = elems
        self._elemtype = elemtype
        return True

    def text_nodes(self,
                   filename: str = '..'):
        err_msg = 'wrong mat file name. '
        assert filename != '..', err_msg
        nodes = np.loadtxt(filename)
        err_msg = 'nodes is a n*3 numpy array containing x, y and z coordinates. '
        assert nodes.shape[1] == 3, err_msg
        self._nodes = np.asfortranarray(nodes)
        self._u = np.zeros(self._nodes.size)
        self.set_dmda()
        return True

    def mat_origin(self,
                   filename: str = '..',
                   mat_handle: str = 'origin'):
        err_msg = 'wrong mat file name. '
        assert filename != '..', err_msg

        mat_contents = sio.loadmat(filename)
        self._origin = mat_contents[mat_handle].astype(np.float)
        return True

    def mat_velocity(self,
                     filename: str = '..',
                     mat_handle: str = 'U'):
        err_msg = 'wrong mat file name. '
        assert filename != '..', err_msg

        mat_contents = sio.loadmat(filename)
        self._u = mat_contents[mat_handle].flatten()
        return True

    def node_rotation(self, norm=np.array([0, 0, 1]), theta=0, rotation_origin=None):
        # The rotation is counterclockwise
        if rotation_origin is None:
            rotation_origin = self.get_origin()
        else:
            rotation_origin = np.array(rotation_origin).reshape((3,))

        norm = np.array(norm).reshape((3,))
        rotation_origin = np.array(rotation_origin).reshape((3,))
        norm = norm / np.linalg.norm(norm)
        a = norm[0]
        b = norm[1]
        c = norm[2]
        rotation = np.array([
            [a ** 2 + (1 - a ** 2) * cos(theta), a * b * (1 - cos(theta)) + c * sin(theta),
             a * c * (1 - cos(theta)) - b * sin(theta)],
            [a * b * (1 - cos(theta)) - c * sin(theta), b ** 2 + (1 - b ** 2) * cos(theta),
             b * c * (1 - cos(theta)) + a * sin(theta)],
            [a * c * (1 - cos(theta)) + b * sin(theta), b * c * (1 - cos(theta)) - a * sin(theta),
             c ** 2 + (1 - c ** 2) * cos(theta)]
        ])
        self._nodes = np.dot(rotation,
                             (self._nodes - rotation_origin).T).T + rotation_origin  # The rotation is counterclockwise
        return True

    def coord_rotation(self, norm=np.array([0, 0, 1]), theta=0):
        norm = norm / np.linalg.norm(norm)
        a = norm[0]
        b = norm[1]
        c = norm[2]
        # theta = -theta # The rotation is counterclockwise
        rotation = np.array([
            [a ** 2 + (1 - a ** 2) * cos(theta), a * b * (1 - cos(theta)) + c * sin(theta),
             a * c * (1 - cos(theta)) - b * sin(theta)],
            [a * b * (1 - cos(theta)) - c * sin(theta), b ** 2 + (1 - b ** 2) * cos(theta),
             b * c * (1 - cos(theta)) + a * sin(theta)],
            [a * c * (1 - cos(theta)) + b * sin(theta), b * c * (1 - cos(theta)) - a * sin(theta),
             c ** 2 + (1 - c ** 2) * cos(theta)]
        ])

        temp_u = self._u.reshape((3, -1), order='F')
        self._u = rotation.dot(temp_u).T.flatten()
        self._nodes = np.dot(rotation, (self._nodes).T).T
        return True

    def node_zoom(self, factor, zoom_origin=None):
        if zoom_origin is None:
            zoom_origin = self.get_origin()
        self._nodes = (self._nodes - zoom_origin) * factor + zoom_origin
        return True

    def get_nodes(self):
        return self._nodes

    def get_nodes_petsc(self):
        nodes_petsc = self.get_dmda().createGlobalVector()
        nodes_petsc[:] = self._nodes.reshape((3, -1))[:]
        nodes_petsc.assemble()
        return nodes_petsc

    def set_nodes(self, nodes: np.array, deltalength, resetVelocity=False):
        err_msg = 'nodes is a n*3 numpy array containing x, y and z coordinates. '
        assert nodes.shape[1] == 3, err_msg
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

        x_petsc = self._dmda.createGlobalVector()
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

    def set_rigid_velocity(self,
                           U=np.array((0, 0, 0, 0, 0, 0))):  # [u1, u2, u3, w1, w2, w3], velocity and angular velocity.
        """

        :type U: np.array
        """
        self._u = np.zeros(self._nodes.size)
        self._u[0::3] = U[0] + U[4] * self._nodes[:, 2] - U[5] * self._nodes[:, 1]
        self._u[1::3] = U[1] + U[5] * self._nodes[:, 0] - U[3] * self._nodes[:, 2]
        self._u[2::3] = U[2] + U[3] * self._nodes[:, 1] - U[4] * self._nodes[:, 0]
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

    def get_origin(self):
        return self._origin

    def set_origin(self, origin):
        self._origin = origin
        return True

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

    def show_velocity(self, length_factor=1, show_nodes=True):
        comm = PETSc.COMM_WORLD.tompi4py()
        rank = comm.Get_rank()
        if rank == 0:
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.set_aspect('equal')
            # Be carefull. the axis using in matplotlib is a left-handed coordinate system
            if show_nodes:
                ax.plot(self.get_nodes_x(), self.get_nodes_y(), self.get_nodes_z(), linestyle='None', c='b', marker='o')
            INDEX = np.zeros_like(self.get_nodes_z(), dtype=bool)
            INDEX[:] = True
            length = 1 / self._deltaLength * length_factor
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
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')

            plt.grid()
            plt.get_current_fig_manager().window.showMaximized()
            plt.show()
        return True

    def show_nodes(self, linestyle='-'):
        comm = PETSc.COMM_WORLD.tompi4py()
        rank = comm.Get_rank()
        if rank == 0:
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.set_aspect('equal')
            ax.plot(self.get_nodes_x(), self.get_nodes_y(), self.get_nodes_z(), linestyle=linestyle, c='b', marker='.')

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
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')

            plt.grid()
            plt.get_current_fig_manager().window.showMaximized()
            plt.show()
        return True

    def get_mesh(self):
        return self._elems, self._elemtype

    def get_dmda(self):
        return self._dmda

    def set_dmda(self):
        self._dmda = PETSc.DMDA().create(sizes=(self.get_n_nodes(),), dof=3, stencil_width=self._stencil_width,
                                         comm=PETSc.COMM_WORLD)
        self._dmda.setFromOptions()
        self._dmda.setUp()
        # self._dmda.createGlobalVector()
        return True

    def destroy_dmda(self):
        self._dmda.destroy()
        self._dmda = None
        return True

    def set_glbIdx(self, indices):
        comm = PETSc.COMM_WORLD.tompi4py()
        self._glbIdx = indices
        self._glbIdx_all = np.hstack(comm.allgather(indices))
        return True

    def get_glbIdx(self):
        return self._glbIdx, self._glbIdx_all

        # def _heaviside(self, n, factor):
        #     f = lambda x: 1 / (1 + np.exp(-factor * x))
        #     x = np.linspace(-0.5, 0.5, n)
        #     return (f(x) - f(-0.5)) / (f(0.5) - f(-0.5))


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

    def _get_theta(self):
        def eqr(dth, ds, r):
            return (ds / (2 * r)) ^ 2 + np.sin(dth / 4) ** 2 - np.sin(dth / 2) ** 2

        from scipy import optimize as sop
        self._dth = sop.brentq(eqr, -1e-3 * np.pi, np.pi, args=(self.get_deltaLength(), self._r))
        return self._dth

    def _get_deltalength(self):
        dl = 2 * self._r * np.sqrt(np.sin(self._dth / 2) ** 2 - np.sin(self._dth / 4) ** 2)
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
                           epsilon=1,
                           with_cover=False):
        # the tunnel is along z axis
        self._dth = dth
        self._r = radius
        nc = np.ceil(2 * np.pi / dth).astype(int)
        angleCycle = np.linspace(0, 2 * np.pi, nc, endpoint=False)
        axisNodes, T_frame, N_frame, B_frame = self._get_axis()
        fgeo_axisNodes, fgeo_T_frame, fgeo_N_frame, fgeo_B_frame = self._get_fgeo_axis(epsilon)
        deltalength = self.get_deltaLength()
        iscover = []
        vgeo_nodes = []
        fgeo_nodes = []
        epsilon = (radius + epsilon * deltalength) / radius
        err_msg = 'epsilon > %f. ' % (-radius / deltalength)
        assert epsilon > 0, err_msg
        ai_para = 0

        # cover at start
        if with_cover:
            nc = np.ceil((radius - deltalength) / deltalength).astype(int)
            ri = np.linspace(deltalength / 2, radius, nc, endpoint=False)
            # self
            for i0 in range(0, nc):
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
        if with_cover:
            nc = np.ceil((radius - deltalength) / deltalength).astype(int)
            ri = np.linspace(deltalength / 2, radius, nc, endpoint=False)[-1::-1]
            for i0 in range(0, nc):
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

        self._iscover = np.hstack(iscover)
        self._nodes = np.asfortranarray(np.vstack(vgeo_nodes))
        self.set_dmda()
        self._u = np.zeros(self._nodes.size)
        self._normal = np.zeros((self._nodes.shape[0], 2), order='F')
        self._tunnel_norm = np.array((0, 0, 1))
        fgeo = geo()
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


class sphere_geo(ellipse_geo):
    def create_n(self, n: int,  # number of nodes.
                 radius: float, *args):  # radius
        err_msg = 'aditional paramenters are useless.  '
        assert not args, err_msg
        self._deltaLength = np.sqrt(4 * np.pi * radius * radius / n)
        return super().create_n(n, radius, radius)

    def create_delta(self, deltaLength: float,  # length of the mesh
                     radius: float, *args):  # radius
        err_msg = 'aditional paramenters are useless.  '
        assert not args, err_msg
        return super().create_delta(deltaLength, radius, radius)

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
        self._tunnel_norm = np.zeros((0, 0, 0))  # describing the aspect of tunnel.
        self._length = 0
        self._cover_strat_list = uniqueList()
        self._cover_end_list = uniqueList()

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
        self._tunnel_norm = np.array((1, 0, 0))
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
        self._tunnel_norm = np.array((0, 0, 1))
        return True

    def create_deltatheta(self, dth: float,  # delta theta of the cycle for the mesh
                          radius: float,
                          length,
                          epsilon=1,
                          with_cover=False,
                          factor=1,
                          left_hand=False):
        self._length = length
        self._factor = factor
        self._left_hand = left_hand
        return self._create_deltatheta(dth, radius, epsilon, with_cover)

    def _get_axis(self):
        length = self._length
        factor = self._factor
        left_hand = self._left_hand
        nl = np.ceil(length / self._get_deltalength()).astype(int)
        z = self._factor_fun(nl, factor) * length - length / 2
        if left_hand:
            self._axisNodes = np.vstack((np.zeros_like(z), np.zeros_like(z), z)).T
            T_frame = np.vstack((np.zeros(nl), np.zeros(nl), np.ones(nl))).T  # (0, 0, 1)
            N_frame = np.vstack((np.ones(nl), np.zeros(nl), np.zeros(nl))).T  # (1, 0, 0)
            B_frame = np.vstack((np.zeros(nl), np.ones(nl), np.zeros(nl))).T  # (0, 1, 0)
        else:
            self._axisNodes = np.vstack((np.zeros_like(z), np.zeros_like(z), z)).T
            T_frame = np.vstack((np.zeros(nl), np.zeros(nl), np.ones(nl))).T  # (0, 0, 1)
            N_frame = np.vstack((np.zeros(nl), np.ones(nl), np.zeros(nl))).T  # (0, 1, 0)
            B_frame = np.vstack((np.ones(nl), np.zeros(nl), np.zeros(nl))).T  # (1, 0, 0)
        self._frenetFrame = (T_frame, N_frame, B_frame)
        return self._axisNodes, self._frenetFrame[0], self._frenetFrame[1], self._frenetFrame[2]

    def _get_fgeo_axis(self, epsilon):
        factor = self._factor
        left_hand = self._left_hand
        nl = self._axisNodes.shape[0]
        ds = -self.get_deltaLength() * epsilon / 4
        length = self._length
        z = self._factor_fun(nl, factor) * (length - ds * 2) - length / 2 + ds
        if left_hand:
            axisNodes = np.vstack((np.zeros_like(z), np.zeros_like(z), z)).T
            T_frame = np.vstack((np.zeros(nl), np.zeros(nl), np.ones(nl))).T  # (0, 0, 1)
            N_frame = np.vstack((np.ones(nl), np.zeros(nl), np.zeros(nl))).T  # (1, 0, 0)
            B_frame = np.vstack((np.zeros(nl), np.ones(nl), np.zeros(nl))).T  # (0, 1, 0)
        else:
            axisNodes = np.vstack((np.zeros_like(z), np.zeros_like(z), z)).T
            T_frame = np.vstack((np.zeros(nl), np.zeros(nl), np.ones(nl))).T  # (0, 0, 1)
            N_frame = np.vstack((np.zeros(nl), np.ones(nl), np.zeros(nl))).T  # (0, 1, 0)
            B_frame = np.vstack((np.ones(nl), np.zeros(nl), np.zeros(nl))).T  # (1, 0, 0)
        return axisNodes, self._frenetFrame[0], self._frenetFrame[1], self._frenetFrame[2]

    def _strat_pretreatment(self, nodes, **kwargs):
        def cart2pol(x, y):
            rho = np.sqrt(x ** 2 + y ** 2)
            phi = np.arctan2(y, x)
            return (rho, phi)

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

    def node_rotation(self, norm=np.array([0, 0, 1]), theta=0, rotation_origin=None):
        # The rotation is counterclockwise
        super().node_rotation(norm=norm, theta=theta, rotation_origin=rotation_origin)

        if rotation_origin is None:
            rotation_origin = self.get_origin()

        norm = norm / np.linalg.norm(norm)
        a = norm[0]
        b = norm[1]
        c = norm[2]
        rotation = np.array([
            [a ** 2 + (1 - a ** 2) * cos(theta), a * b * (1 - cos(theta)) + c * sin(theta),
             a * c * (1 - cos(theta)) - b * sin(theta)],
            [a * b * (1 - cos(theta)) - c * sin(theta), b ** 2 + (1 - b ** 2) * cos(theta),
             b * c * (1 - cos(theta)) + a * sin(theta)],
            [a * c * (1 - cos(theta)) + b * sin(theta), b * c * (1 - cos(theta)) - a * sin(theta),
             c ** 2 + (1 - c ** 2) * cos(theta)]
        ])
        self._tunnel_norm = np.dot(rotation, self._tunnel_norm)
        return True

    def node_zoom_radius(self, epsilon):
        def cart2pol(x, y):
            rho = np.sqrt(x ** 2 + y ** 2)
            phi = np.arctan2(y, x)
            return (rho, phi)

        def pol2cart(rho, phi):
            x = rho * np.cos(phi)
            y = rho * np.sin(phi)
            return (x, y)

        # zooming geo along radius of tunnel, keep longitude axis.
        # 1. copy
        temp_geo = geo()
        temp_nodes = self.get_nodes() - self.get_origin()
        temp_geo.set_nodes(temp_nodes, self.get_deltaLength())
        # temp_geo.show_nodes()
        # 2. rotation, tunnel center line along x axis.
        temp_norm = self._tunnel_norm
        rotation_norm = np.cross(temp_norm, [1, 0, 0])
        temp_theta = -np.arccos(temp_norm[0] / np.linalg.norm(temp_norm))
        doRotation = np.array_equal(rotation_norm, np.array((0, 0, 0))) and temp_theta != 0.
        if doRotation:
            temp_geo.node_rotation(rotation_norm, temp_theta)
        # 3. zooming
        temp_nodes = temp_geo.get_nodes()
        temp_R, temp_phi = cart2pol(temp_nodes[:, 1], temp_nodes[:, 2])
        factor = (np.max(temp_R) + epsilon * self._deltaLength) / max(temp_R)
        temp_R = temp_R * factor
        X1 = np.min(temp_nodes[:, 0])
        X2 = np.max(temp_nodes[:, 0])
        factor = 2 * epsilon * self._deltaLength / (X2 - X1)
        temp_nodes[:, 0] = (temp_nodes[:, 0] - (X1 + X2) / 2) * (1 + factor) + (X1 + X2) / 2
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
        self._tunnel_norm = np.array((1, 0, 0))
        self._cover_node_list = cover_node_list
        return True

    def get_cover_node_list(self):
        return self._cover_node_list


class stokeslets_tunnel_geo(tunnel_geo):
    from src import stokes_flow as sf
    def stokeslets_velocity(self, problem: 'sf.stokesletsProblem'):
        from src.StokesFlowMethod import light_stokeslets_matrix_3d

        stokeslets_post = problem.get_stokeslets_post()
        stokeslets_f = problem.get_stokeslets_f()
        m = light_stokeslets_matrix_3d(self.get_nodes(), stokeslets_post)
        self._u = -1 * m.dot(stokeslets_f)
        return True


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
        self._R = 0
        self._B = 0
        self._n_c = 0

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
                          epsilon=1,
                          with_cover=False,
                          factor=1,
                          left_hand=False):
        self._R = R
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
        length = np.sqrt(R ** 2 + B ** 2) * 2 * np.pi * n_c
        nl = np.ceil(length / self._get_deltalength()).astype(int)
        s = self._factor_fun(nl, factor) * length - length / 2
        if left_hand:
            self._frenetFrame = (self._T_frame_left_hand(R, B, s), self._N_frame_left_hand(R, B, s), self._B_frame_left_hand(R, B, s))
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
            frenetFrame = (self._T_frame_left_hand(R, B, s), self._N_frame_left_hand(R, B, s), self._B_frame_left_hand(R, B, s))
            axisNodes = self._helix_left_hand(R, B, s)
        else:
            frenetFrame = (self._T_frame(R, B, s), self._N_frame(R, B, s), self._B_frame(R, B, s))
            axisNodes = self._helix(R, B, s)
        return axisNodes, frenetFrame[0], frenetFrame[1], frenetFrame[2]


class region:
    def __init__(self):
        self.type = {'rectangle': self.rectangle,
                     'sector':    self.sector}

    def rectangle(self,
                  field_range: np.array,
                  n_grid: np.array):
        """

        :type self: stokesFlowProblem
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

        :type self: stokesFlowProblem
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


def createEcoli(objtype, **kwargs):
    nth = kwargs['nth']
    hfct = kwargs['hfct']
    eh = kwargs['eh']
    ch = kwargs['ch']
    rh1 = kwargs['rh1']
    rh2 = kwargs['rh2']
    ph = kwargs['ph']
    moveh = kwargs['moveh']
    with_cover = kwargs['with_cover']
    ds = kwargs['ds']
    rs1 = kwargs['rs1']
    rs2 = kwargs['rs2']
    es = kwargs['es']
    moves = kwargs['moves']
    left_hand = kwargs['left_hand']
    # sphere_rotation = kwargs['sphere_rotation'] if 'sphere_rotation' in kwargs.keys() else 0
    zoom_factor = kwargs['zoom_factor'] if 'zoom_factor' in kwargs.keys() else 1

    # create helix
    B = ph / (2 * np.pi)
    vhgeo0 = supHelix()  # velocity node geo of helix
    dth = 2 * np.pi / nth
    fhgeo0 = vhgeo0.create_deltatheta(dth=dth, radius=rh2, R=rh1, B=B, n_c=ch, epsilon=eh, with_cover=with_cover,
                                      factor=hfct, left_hand=left_hand)
    vhobj0 = objtype()
    vhobj0.set_data(fhgeo0, vhgeo0, name='helix_0')
    vhobj0.zoom(zoom_factor)
    vhobj0.move(moveh * zoom_factor)
    vhobj1 = vhobj0.copy()
    vhobj1.node_rotation(norm=(0, 0, 1), theta=np.pi, rotation_origin=(0, 0, 0))
    vhobj1.set_name('helix_1')

    # create sphere
    vsgeo = ellipse_geo()  # velocity node geo of sphere
    vsgeo.create_delta(ds, rs1, rs2)
    vsgeo.node_rotation(norm=np.array((0, 1, 0)), theta=np.pi / 2)
    fsgeo = vsgeo.copy()  # force node geo of sphere
    fsgeo.node_zoom(1 + ds / (0.5 * (rs1 + rs2)) * es)
    vsobj = objtype()
    vsobj.set_data(fsgeo, vsgeo, name='sphere_0')
    vsobj.zoom(zoom_factor)

    vsobj.move(moves * zoom_factor)
    return vsobj, vhobj0, vhobj1

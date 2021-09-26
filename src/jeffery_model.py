# coding=utf-8
"""
Classes for solving jeffery equations.
Zhang Ji, 20181219
"""

import numpy as np
from src.support_class import *
import abc
from scipy import interpolate, integrate
import os
import pickle
from petsc4py import PETSc
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook
from scipy.spatial.transform import Rotation as spR
from src.objComposite import create_ecoli_2part

# import quaternion

__all__ = ['JefferyObj',
           'ShearJefferyProblem', ]


class _JefferyProblem:
    def __init__(self, name='...', **kwargs):
        self._name = name
        self._kwargs = kwargs
        self._type = '_JefferyProblem'
        self._obj_list = uniqueList()  # contain objects

    def __repr__(self):
        return self._type

    def __str__(self):
        return self.get_name()

    def get_name(self):
        return self._name

    @abc.abstractmethod
    def flow_velocity(self, location):
        ...

    @abc.abstractmethod
    def flow_gradient(self, location):
        ...

    def flow_strain(self, location):
        Dij = self.flow_gradient(location)
        Eij = 0.5 * (Dij + Dij.T)
        return Eij

    def flow_rotation(self, location):
        Dij = self.flow_gradient(location)
        # print(Dij)
        Sij = 0.5 * (Dij - Dij.T)
        return Sij

    def flow_strain_rotation(self, location):
        Dij = self.flow_gradient(location)
        Eij = 0.5 * (Dij + Dij.T)
        Sij = 0.5 * (Dij - Dij.T)
        return Eij, Sij

    def _check_add_obj(self, obj):
        err_msg = 'only JefferyObj accept'
        assert isinstance(obj, JefferyObj), err_msg

    def add_obj(self, obj):
        """
        Add a new object to the problem.

        :type obj: JefferyObj
        :param obj: added object
        :return: none.
        """
        self._check_add_obj(obj)
        self._obj_list.append(obj)
        obj.index = self.get_n_obj()
        obj.father = self
        return True

    def get_n_obj(self):
        return len(self._obj_list)

    @property
    def obj_list(self):
        return self._obj_list

    def update_location(self, eval_dt, print_handle=''):
        for obj in self.obj_list:  # type: JefferyObj
            obj.update_location(eval_dt, print_handle)
        # return True

    def Rloc2glb(self, theta, phi, psi):
        Rloc2glb = np.array(
                ((np.cos(phi) * np.cos(psi) * np.cos(theta) - np.sin(phi) * np.sin(psi),
                  -(np.cos(psi) * np.sin(phi)) - np.cos(phi) * np.cos(theta) * np.sin(psi),
                  np.cos(phi) * np.sin(theta)),
                 (np.cos(psi) * np.cos(theta) * np.sin(phi) + np.cos(phi) * np.sin(psi),
                  np.cos(phi) * np.cos(psi) - np.cos(theta) * np.sin(phi) * np.sin(psi),
                  np.sin(phi) * np.sin(theta)),
                 (-(np.cos(psi) * np.sin(theta)),
                  np.sin(psi) * np.sin(theta),
                  np.cos(theta))))
        return Rloc2glb

    @abc.abstractmethod
    def MBF_U_b(self, X):
        # see my paper for detail
        ...

    @abc.abstractmethod
    def MBF_W_S(self, X):
        # see my paper for detail
        ...

    @abc.abstractmethod
    def MBF_beta(self, theta, phi, psi, X):
        # see my paper for detail
        ...

    def MBF_U_b_v2(self, X):
        U_b = self.flow_velocity(X)
        return U_b

    def MBF_W_S_v2(self, X):
        Sij = self.flow_rotation(X)
        # print(Sij)
        W = np.array((Sij[2, 1], Sij[2, 0], Sij[0, 1]))
        return W

    def MBF_beta_v2(self, theta, phi, psi, X):
        Eij = self.flow_strain(X)
        R = self.Rloc2glb(theta, phi, psi)
        Elc = np.dot(R.T, np.dot(Eij, R))
        beta = np.array((Elc[0, 0], Elc[2, 2], Elc[0, 1], Elc[0, 2], Elc[1, 2]))
        return beta

    def MBF_Ub_WS_beta_v2(self, theta, phi, psi, X):
        # print('dbg fix X=(0, 0, 0)')
        # X = np.zeros(3)

        Dij = self.flow_gradient(X)
        Eij = 0.5 * (Dij + Dij.T)
        Sij = 0.5 * (Dij - Dij.T)
        R = self.Rloc2glb(theta, phi, psi)

        U_b = self.flow_velocity(X)
        W = np.array((Sij[1, 2], Sij[2, 0], Sij[0, 1]))
        Elc = np.dot(R.T, np.dot(Eij, R))
        beta = np.array((Elc[0, 0], Elc[2, 2], Elc[0, 1], Elc[0, 2], Elc[1, 2]))

        # print('dbg theta, phi, psi', theta, phi, psi)
        # print('dbg W', W)
        # print('dbg Eij')
        # print(Eij)
        # print('dbg Elc')
        # print(Elc)
        # print('dbg beta', beta)
        return U_b, W, beta


# class _Jeffery3DProblem(_JefferyProblem):
#     def __init__(self, name='...', **kwargs):
#         super().__init__(name=name, **kwargs)
#         self._type = 'Jeffery3DProblem'


class ShearJefferyProblem(_JefferyProblem):
    # _planeShearRate = ...  # type: np.ndarray

    # current version the velocity of shear flow points to the x axis and only varys in the z axis.
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._type = 'ShearJefferyProblem'
        self._planeShearRate = np.array(kwargs['planeShearRate']).reshape((1, 3))
        err_msg = 'shear flow velocity is must vertical to (y, z) plane. '
        assert np.all(np.isclose(self._planeShearRate[0, -2:], (0, 0))), err_msg

    def flow_velocity(self, location):
        loc_z = location[-1]
        tao_x = self._planeShearRate[0, 0]
        given_u = np.array((tao_x * loc_z, 0, 0))
        return given_u

    def flow_gradient(self, location):
        tao_x = self._planeShearRate[0, 0]
        Dij = np.array(((0, 0, 0,),
                        (0, 0, 0,),
                        (tao_x, 0, 0,)))
        return Dij

    @property
    def planeShearRate(self):
        return self._planeShearRate

    def MBF_U_b(self, X):
        # see my paper for detail
        U_p = self.flow_velocity(X)
        return U_p

    def MBF_W_S(self, X):
        # see my paper for detail
        tao_x = self._planeShearRate[0, 0]
        W_p = np.hstack((0, 1 / 2 * tao_x, 0))
        return W_p

    def MBF_beta(self, theta, phi, psi, X):
        # see my paper for detail
        tao_x = self._planeShearRate[0, 0]
        beta = tao_x * np.array((np.cos(psi) * (-(np.cos(phi) * np.cos(psi) * np.cos(theta)) +
                                                np.sin(phi) * np.sin(psi)) * np.sin(theta),
                                 np.cos(phi) * np.cos(theta) * np.sin(theta),
                                 (2 * np.cos(2 * psi) * np.sin(phi) * np.sin(theta) +
                                  np.cos(phi) * np.sin(2 * psi) * np.sin(2 * theta)) / 4.,
                                 (np.cos(phi) * np.cos(psi) * np.cos(2 * theta) -
                                  np.cos(theta) * np.sin(phi) * np.sin(psi)) / 2.,
                                 (-(np.cos(psi) * np.cos(theta) * np.sin(phi)) -
                                  np.cos(phi) * np.cos(2 * theta) * np.sin(psi)) / 2.,))
        return beta


class ABCFlowProblem(_JefferyProblem):
    # see https://en.wikipedia.org/wiki/Arnold%E2%80%93Beltrami%E2%80%93Childress_flow
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._type = 'ABCFlowProblem'
        self._A = kwargs['ABC_A']
        self._B = kwargs['ABC_B']
        self._C = kwargs['ABC_C']

    @property
    def A(self):
        return self._A

    @property
    def B(self):
        return self._B

    @property
    def C(self):
        return self._C

    def flow_velocity(self, location):
        A, B, C = self.A, self.B, self.C
        x, y, z = location[0], location[1], location[2]
        Ub = np.array((A * np.sin(z) + C * np.cos(y),
                       B * np.sin(x) + A * np.cos(z),
                       C * np.sin(y) + B * np.cos(x)))
        return Ub

    def flow_gradient(self, location):
        A, B, C = self.A, self.B, self.C
        x, y, z = location[0], location[1], location[2]
        Dij = np.array(((0, B * np.cos(x), -B * np.sin(x),),
                        (-C * np.sin(y), 0, C * np.cos(y),),
                        (A * np.cos(z), -A * np.sin(z), 0,)))

        # print('dbg code 330')
        # Dij = np.array(((0, B * np.cos(x), -B * np.sin(x),),
        #                 (-C * np.sin(y), 0, C * np.cos(y),),
        #                 (A * np.cos(z), -A * np.sin(z), 0,))).T
        return Dij


class ABCFlowProblem_DEFHIJ(ABCFlowProblem):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._type = 'ABCFlowProblem'
        self._D = kwargs['ABC_D']
        self._E = kwargs['ABC_E']
        self._F = kwargs['ABC_F']
        self._G = kwargs['ABC_G']
        self._H = kwargs['ABC_H']
        self._I = kwargs['ABC_I']

    @property
    def D(self):
        return self._D

    @property
    def E(self):
        return self._E

    @property
    def F(self):
        return self._F

    @property
    def G(self):
        return self._G

    @property
    def H(self):
        return self._H

    @property
    def I(self):
        return self._I

    def flow_velocity(self, location):
        A, B, C = self.A, self.B, self.C
        D, E, F = self.D, self.E, self.F
        G, H, I = self.G, self.H, self.I
        x, y, z = location[0], location[1], location[2]
        Ub = np.array((A * np.sin(D * z + G) + C * np.cos(F * y + I),
                       B * np.sin(E * x + H) + A * np.cos(D * z + G),
                       C * np.sin(F * y + I) + B * np.cos(E * x + H)))
        return Ub

    def flow_gradient(self, location):
        A, B, C = self.A, self.B, self.C
        D, E, F = self.D, self.E, self.F
        G, H, I = self.G, self.H, self.I
        x, y, z = location[0], location[1], location[2]
        Dij = np.array(((0, B * E * np.cos(E * x + H), -B * E * np.sin(E * x + H),),
                        (-C * F * np.sin(F * y + I), 0, C * F * np.cos(F * y + I),),
                        (A * D * np.cos(D * z + G), -A * D * np.sin(D * z + G), 0,)))
        return Dij


class ShearTableProblem(ShearJefferyProblem):
    def _nothing(self):
        pass


class SingleStokesletsJefferyProblem(_JefferyProblem):
    _StokesletsStrength = ...  # type: np.ndarray

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._type = 'SingleStokesletsJefferyProblem'
        self._StokesletsStrength = np.array(kwargs['StokesletsStrength']).reshape((1, 3)).flatten()

    # def flow_strain(self, location):
    #     S_fun = lambda x0, x1, x2, f0, f1, f2: np.array(
    #             [[(-3.0 * x0 ** 2 * (x0 ** 2 + x1 ** 2 + x2 ** 2) ** 1.5 +
    #                1.0 * (x0 ** 2 + x1 ** 2 + x2 ** 2) ** 2.5) * (x0 ** 2 + x1 ** 2 + x2 ** 2) ** (
    #                   -4.0) * (
    #                       f0 * x0 + f1 * x1 + f2 * x2),
    #               -3.0 * x0 * x1 * (x0 ** 2 + x1 ** 2 + x2 ** 2) ** (-2.5) * (
    #                       f0 * x0 + f1 * x1 + f2 * x2),
    #               -3.0 * x0 * x2 * (x0 ** 2 + x1 ** 2 + x2 ** 2) ** (-2.5) * (
    #                       f0 * x0 + f1 * x1 + f2 * x2)],
    #              [-3.0 * x0 * x1 * (x0 ** 2 + x1 ** 2 + x2 ** 2) ** (-2.5) * (
    #                      f0 * x0 + f1 * x1 + f2 * x2),
    #               (-3.0 * x1 ** 2 * (x0 ** 2 + x1 ** 2 + x2 ** 2) ** 1.5 +
    #                1.0 * (x0 ** 2 + x1 ** 2 + x2 ** 2) ** 2.5) * (x0 ** 2 + x1 ** 2 + x2 ** 2) ** (
    #                   -4.0) * (
    #                       f0 * x0 + f1 * x1 + f2 * x2),
    #               -3.0 * x1 * x2 * (x0 ** 2 + x1 ** 2 + x2 ** 2) ** (-2.5) * (
    #                       f0 * x0 + f1 * x1 + f2 * x2)],
    #              [-3.0 * x0 * x2 * (x0 ** 2 + x1 ** 2 + x2 ** 2) ** (-2.5) * (
    #                      f0 * x0 + f1 * x1 + f2 * x2),
    #               -3.0 * x1 * x2 * (x0 ** 2 + x1 ** 2 + x2 ** 2) ** (-2.5) * (
    #                       f0 * x0 + f1 * x1 + f2 * x2),
    #               (-3.0 * x2 ** 2 * (x0 ** 2 + x1 ** 2 + x2 ** 2) ** 1.5 +
    #                1.0 * (x0 ** 2 + x1 ** 2 + x2 ** 2) ** 2.5) * (x0 ** 2 + x1 ** 2 + x2 ** 2) ** (
    #                   -4.0) * (
    #                       f0 * x0 + f1 * x1 + f2 * x2)]]) / (8 * np.pi)
    #     return S_fun(*location, *self._StokesletsStrength)
    #
    # def flow_rotation(self, location):
    #     Omega_fun = lambda x0, x1, x2, f0, f1, f2: np.array(
    #             [[np.zeros_like(x0),
    #               (-1.0 * f0 * x1 + f1 * x0) * (x0 ** 2 + x1 ** 2 + x2 ** 2) ** (-1.5),
    #               (-1.0 * f0 * x2 + f2 * x0) * (x0 ** 2 + x1 ** 2 + x2 ** 2) ** (-1.5)],
    #              [(f0 * x1 - 1.0 * f1 * x0) * (x0 ** 2 + x1 ** 2 + x2 ** 2) ** (-1.5),
    #               np.zeros_like(x0),
    #               (-1.0 * f1 * x2 + f2 * x1) * (x0 ** 2 + x1 ** 2 + x2 ** 2) ** (-1.5)],
    #              [(f0 * x2 - 1.0 * f2 * x0) * (x0 ** 2 + x1 ** 2 + x2 ** 2) ** (-1.5),
    #               (f1 * x2 - 1.0 * f2 * x1) * (x0 ** 2 + x1 ** 2 + x2 ** 2) ** (-1.5),
    #               np.zeros_like(x0), ]]) / (8 * np.pi)
    #     return Omega_fun(*location, *self._StokesletsStrength)

    def flow_gradient(self, location):
        Omega_fun = lambda x0, x1, x2, f0, f1, f2: np.array(
                [[np.zeros_like(x0),
                  (-1.0 * f0 * x1 + f1 * x0) * (x0 ** 2 + x1 ** 2 + x2 ** 2) ** (-1.5),
                  (-1.0 * f0 * x2 + f2 * x0) * (x0 ** 2 + x1 ** 2 + x2 ** 2) ** (-1.5)],
                 [(f0 * x1 - 1.0 * f1 * x0) * (x0 ** 2 + x1 ** 2 + x2 ** 2) ** (-1.5),
                  np.zeros_like(x0),
                  (-1.0 * f1 * x2 + f2 * x1) * (x0 ** 2 + x1 ** 2 + x2 ** 2) ** (-1.5)],
                 [(f0 * x2 - 1.0 * f2 * x0) * (x0 ** 2 + x1 ** 2 + x2 ** 2) ** (-1.5),
                  (f1 * x2 - 1.0 * f2 * x1) * (x0 ** 2 + x1 ** 2 + x2 ** 2) ** (-1.5),
                  np.zeros_like(x0), ]]) / (8 * np.pi)
        S_fun = lambda x0, x1, x2, f0, f1, f2: np.array(
                [[(-3.0 * x0 ** 2 * (x0 ** 2 + x1 ** 2 + x2 ** 2) ** 1.5 +
                   1.0 * (x0 ** 2 + x1 ** 2 + x2 ** 2) ** 2.5) * (x0 ** 2 + x1 ** 2 + x2 ** 2) ** (
                      -4.0) * (
                          f0 * x0 + f1 * x1 + f2 * x2),
                  -3.0 * x0 * x1 * (x0 ** 2 + x1 ** 2 + x2 ** 2) ** (-2.5) * (
                          f0 * x0 + f1 * x1 + f2 * x2),
                  -3.0 * x0 * x2 * (x0 ** 2 + x1 ** 2 + x2 ** 2) ** (-2.5) * (
                          f0 * x0 + f1 * x1 + f2 * x2)],
                 [-3.0 * x0 * x1 * (x0 ** 2 + x1 ** 2 + x2 ** 2) ** (-2.5) * (
                         f0 * x0 + f1 * x1 + f2 * x2),
                  (-3.0 * x1 ** 2 * (x0 ** 2 + x1 ** 2 + x2 ** 2) ** 1.5 +
                   1.0 * (x0 ** 2 + x1 ** 2 + x2 ** 2) ** 2.5) * (x0 ** 2 + x1 ** 2 + x2 ** 2) ** (
                      -4.0) * (
                          f0 * x0 + f1 * x1 + f2 * x2),
                  -3.0 * x1 * x2 * (x0 ** 2 + x1 ** 2 + x2 ** 2) ** (-2.5) * (
                          f0 * x0 + f1 * x1 + f2 * x2)],
                 [-3.0 * x0 * x2 * (x0 ** 2 + x1 ** 2 + x2 ** 2) ** (-2.5) * (
                         f0 * x0 + f1 * x1 + f2 * x2),
                  -3.0 * x1 * x2 * (x0 ** 2 + x1 ** 2 + x2 ** 2) ** (-2.5) * (
                          f0 * x0 + f1 * x1 + f2 * x2),
                  (-3.0 * x2 ** 2 * (x0 ** 2 + x1 ** 2 + x2 ** 2) ** 1.5 +
                   1.0 * (x0 ** 2 + x1 ** 2 + x2 ** 2) ** 2.5) * (x0 ** 2 + x1 ** 2 + x2 ** 2) ** (
                      -4.0) * (
                          f0 * x0 + f1 * x1 + f2 * x2)]]) / (8 * np.pi)
        Dij = S_fun(*location, *self._StokesletsStrength) \
              + Omega_fun(*location, *self._StokesletsStrength)
        return Dij

    def flow_velocity(self, location):
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
        return given_u_fun(*location, *self._StokesletsStrength)

    @property
    def StokesletsStrength(self):
        return self._StokesletsStrength


class HalfSpaceJefferyProblem(_JefferyProblem):
    _StokesletsStrength = ...  # type: np.ndarray

    def __init__(self, h, **kwargs):
        super().__init__(**kwargs)
        self._type = 'HalfSpaceJefferyProblem'
        self._StokesletsStrength = np.array(kwargs['StokesletsStrength']).reshape((1, 3)).flatten()
        self._h = h

    def flow_gradient(self, location):
        h = self._h
        x1, x2, x3 = location
        f1, f2, f3 = self.StokesletsStrength
        j00 = (1 / 8) * np.pi ** (-1) * (
                (h ** 2 + x1 ** 2 + x2 ** 2 + (-2) * h * x3 + x3 ** 2) ** (-3 / 2) * (
                4 * f1 * x1 + f2 * x2 + f3 * ((-1) * h + x3)) + 2 * h ** 2 * (
                        (-4) * f1 * x1 + (-3) * f2 * x2 + 3 * f3 * (h + x3)) * (
                        x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (-5 / 2) + (-1) * (
                        4 * f1 * x1 + f2 * x2 + f3 * (h + x3)) * (
                        x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (-3 / 2) + 2 * h * (
                        x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (-5 / 2) * (
                        (4 * f1 * x1 + 3 * f2 * x2) * (h + x3) + f3 * (
                        (-2) * h ** 2 + 3 * x1 ** 2 + x2 ** 2 + (-4) * h * x3 + (
                    -2) * x3 ** 2)) + (-3) * x1 * (
                        h ** 2 + x1 ** 2 + x2 ** 2 + (-2) * h * x3 + x3 ** 2) ** (
                        -5 / 2) * (f2 * x1 * x2 + f3 * x1 * ((-1) * h + x3) + f1 * (
                h ** 2 + 2 * x1 ** 2 + x2 ** 2 + (-2) * h * x3 + x3 ** 2)) + (
                    -10) * h ** 2 * x1 * (
                        x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (-7 / 2) * (
                        (-3) * f2 * x1 * x2 + 3 * f3 * x1 * (h + x3) + f1 * (h ** 2 + (
                    -2) * x1 ** 2 + x2 ** 2 + 2 * h * x3 + x3 ** 2)) + 3 * x1 * (
                        x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (-5 / 2) * (
                        f2 * x1 * x2 + f3 * x1 * (h + x3) + f1 * (
                        h ** 2 + 2 * x1 ** 2 + x2 ** 2 + 2 * h * x3 + x3 ** 2)) + 10 * h * x1 * (
                        x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (-7 / 2) * (f1 * (h + x3) * (
                h ** 2 + (-2) * x1 ** 2 + x2 ** 2 + 2 * h * x3 + x3 ** 2) + (-1) * x1 * (
                                                                                  3 * f2 * x2 * (
                                                                                  h + x3) + f3 * (
                                                                                          (
                                                                                              -2) * h ** 2 + x1 ** 2 + x2 ** 2 + (
                                                                                              -4) * h * x3 + (
                                                                                              -2) * x3 ** 2))))
        j01 = (1 / 8) * np.pi ** (-1) * (
                (f2 * x1 + 2 * f1 * x2) * (
                h ** 2 + x1 ** 2 + x2 ** 2 + (-2) * h * x3 + x3 ** 2) ** (
                        -3 / 2) + 2 * h ** 2 * ((-3) * f2 * x1 + 2 * f1 * x2) * (
                        x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (
                        -5 / 2) + 2 * h * (
                        2 * f3 * x1 * x2 + 3 * f2 * x1 * (h + x3) + (-2) * f1 * x2 * (h + x3)) * (
                        x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (-5 / 2) + (-1) * (
                        f2 * x1 + 2 * f1 * x2) * (
                        x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (-3 / 2) + (-3) * x2 * (
                        h ** 2 + x1 ** 2 + x2 ** 2 + (-2) * h * x3 + x3 ** 2) ** (-5 / 2) * (
                        f2 * x1 * x2 + f3 * x1 * ((-1) * h + x3) + f1 * (
                        h ** 2 + 2 * x1 ** 2 + x2 ** 2 + (-2) * h * x3 + x3 ** 2)) + (
                    -10) * h ** 2 * x2 * (
                        x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (-7 / 2) * (
                        (-3) * f2 * x1 * x2 + 3 * f3 * x1 * (h + x3) + f1 * (
                        h ** 2 + (-2) * x1 ** 2 + x2 ** 2 + 2 * h * x3 + x3 ** 2)) + 3 * x2 * (
                        x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (-5 / 2) * (
                        f2 * x1 * x2 + f3 * x1 * (h + x3) + f1 * (
                        h ** 2 + 2 * x1 ** 2 + x2 ** 2 + 2 * h * x3 + x3 ** 2)) + 10 * h * x2 * (
                        x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (-7 / 2) * (
                        f1 * (h + x3) * (
                        h ** 2 + (-2) * x1 ** 2 + x2 ** 2 + 2 * h * x3 + x3 ** 2) + (
                            -1) * x1 * (3 * f2 * x2 * (h + x3) + f3 * (
                        (-2) * h ** 2 + x1 ** 2 + x2 ** 2 + (-4) * h * x3 + (-2) * x3 ** 2))))
        j02 = (1 / 8) * np.pi ** (-1) * (((-2) * f1 * h + f3 * x1 + 2 * f1 * x3) * (
                h ** 2 + x1 ** 2 + x2 ** 2 + (-2) * h * x3 + x3 ** 2) ** (-3 / 2) + 2 * h ** 2 * (
                                                 3 * f3 * x1 + 2 * f1 * (h + x3)) * (
                                                 x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (-5 / 2) + (
                                                 (-1) * f3 * x1 + (-2) * f1 * (h + x3)) * (
                                                 x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (
                                                 -3 / 2) + 3 * (
                                                 h + (-1) * x3) * (
                                                 h ** 2 + x1 ** 2 + x2 ** 2 + (
                                             -2) * h * x3 + x3 ** 2) ** (
                                                 -5 / 2) * (f2 * x1 * x2 + f3 * x1 * (
                (-1) * h + x3) + f1 * (
                                                                    h ** 2 + 2 * x1 ** 2 + x2 ** 2 + (
                                                                -2) * h * x3 + x3 ** 2)) + (
                                             -10) * h ** 2 * (h + x3) * (
                                                 x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (-7 / 2) * (
                                                 (-3) * f2 * x1 * x2 + 3 * f3 * x1 * (
                                                 h + x3) + f1 * (h ** 2 + (
                                             -2) * x1 ** 2 + x2 ** 2 + 2 * h * x3 + x3 ** 2)) + 3 * (
                                                 h + x3) * (
                                                 x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (-5 / 2) * (
                                                 f2 * x1 * x2 + f3 * x1 * (h + x3) + f1 * (
                                                 h ** 2 + 2 * x1 ** 2 + x2 ** 2 + 2 * h * x3 + x3 ** 2)) + (
                                             -2) * h * (x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (
                                                 -5 / 2) * (
                                                 (-3) * f2 * x1 * x2 + 4 * f3 * x1 * (
                                                 h + x3) + f1 * (3 * h ** 2 + (
                                             -2) * x1 ** 2 + x2 ** 2 + 6 * h * x3 + 3 * x3 ** 2)) + 10 * h * (
                                                 h + x3) * (x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (
                                                 -7 / 2) * (
                                                 f1 * (h + x3) * (
                                                 h ** 2 + (
                                             -2) * x1 ** 2 + x2 ** 2 + 2 * h * x3 + x3 ** 2) + (
                                                     -1) * x1 * (3 * f2 * x2 * (h + x3) + f3 * (
                                                 (-2) * h ** 2 + x1 ** 2 + x2 ** 2 + (
                                             -4) * h * x3 + (
                                                     -2) * x3 ** 2))))
        j10 = (1 / 8) * np.pi ** (-1) * (
                (2 * f2 * x1 + f1 * x2) * (
                h ** 2 + x1 ** 2 + x2 ** 2 + (-2) * h * x3 + x3 ** 2) ** (
                        -3 / 2) + 2 * h ** 2 * (2 * f2 * x1 + (-3) * f1 * x2) * (
                        x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (
                        -5 / 2) + 2 * h * (
                        2 * f3 * x1 * x2 + (-2) * f2 * x1 * (h + x3) + 3 * f1 * x2 * (h + x3)) * (
                        x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (-5 / 2) + (-1) * (
                        2 * f2 * x1 + f1 * x2) * (
                        x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (-3 / 2) + (-3) * x1 * (
                        h ** 2 + x1 ** 2 + x2 ** 2 + (-2) * h * x3 + x3 ** 2) ** (-5 / 2) * (
                        f1 * x1 * x2 + f3 * x2 * ((-1) * h + x3) + f2 * (
                        h ** 2 + x1 ** 2 + 2 * x2 ** 2 + (-2) * h * x3 + x3 ** 2)) + (
                    -10) * h ** 2 * x1 * (
                        x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (-7 / 2) * (
                        (-3) * f1 * x1 * x2 + 3 * f3 * x2 * (h + x3) + f2 * (
                        h ** 2 + x1 ** 2 + (-2) * x2 ** 2 + 2 * h * x3 + x3 ** 2)) + 3 * x1 * (
                        x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (-5 / 2) * (
                        f1 * x1 * x2 + f3 * x2 * (h + x3) + f2 * (
                        h ** 2 + x1 ** 2 + 2 * x2 ** 2 + 2 * h * x3 + x3 ** 2)) + 10 * h * x1 * (
                        x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (-7 / 2) * (
                        f2 * (h + x3) * (
                        h ** 2 + x1 ** 2 + (-2) * x2 ** 2 + 2 * h * x3 + x3 ** 2) + (
                            -1) * x2 * (3 * f1 * x1 * (h + x3) + f3 * (
                        (-2) * h ** 2 + x1 ** 2 + x2 ** 2 + (-4) * h * x3 + (-2) * x3 ** 2))))
        j11 = (1 / 8) * np.pi ** (-1) * (
                (h ** 2 + x1 ** 2 + x2 ** 2 + (-2) * h * x3 + x3 ** 2) ** (-3 / 2) * (
                f1 * x1 + 4 * f2 * x2 + f3 * ((-1) * h + x3)) + 2 * h ** 2 * (
                        (-3) * f1 * x1 + (-4) * f2 * x2 + 3 * f3 * (h + x3)) * (
                        x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (-5 / 2) + (-1) * (
                        f1 * x1 + 4 * f2 * x2 + f3 * (h + x3)) * (
                        x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (-3 / 2) + 2 * h * (
                        x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (-5 / 2) * (
                        (3 * f1 * x1 + 4 * f2 * x2) * (h + x3) + f3 * (
                        (-2) * h ** 2 + x1 ** 2 + 3 * x2 ** 2 + (-4) * h * x3 + (
                    -2) * x3 ** 2)) + (-3) * x2 * (
                        h ** 2 + x1 ** 2 + x2 ** 2 + (-2) * h * x3 + x3 ** 2) ** (
                        -5 / 2) * (f1 * x1 * x2 + f3 * x2 * ((-1) * h + x3) + f2 * (
                h ** 2 + x1 ** 2 + 2 * x2 ** 2 + (-2) * h * x3 + x3 ** 2)) + (
                    -10) * h ** 2 * x2 * (
                        x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (-7 / 2) * (
                        (-3) * f1 * x1 * x2 + 3 * f3 * x2 * (h + x3) + f2 * (
                        h ** 2 + x1 ** 2 + (
                    -2) * x2 ** 2 + 2 * h * x3 + x3 ** 2)) + 3 * x2 * (
                        x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (-5 / 2) * (
                        f1 * x1 * x2 + f3 * x2 * (h + x3) + f2 * (
                        h ** 2 + x1 ** 2 + 2 * x2 ** 2 + 2 * h * x3 + x3 ** 2)) + 10 * h * x2 * (
                        x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (-7 / 2) * (f2 * (h + x3) * (
                h ** 2 + x1 ** 2 + (-2) * x2 ** 2 + 2 * h * x3 + x3 ** 2) + (-1) * x2 * (
                                                                                  3 * f1 * x1 * (
                                                                                  h + x3) + f3 * (
                                                                                          (
                                                                                              -2) * h ** 2 + x1 ** 2 + x2 ** 2 + (
                                                                                              -4) * h * x3 + (
                                                                                              -2) * x3 ** 2))))
        j12 = (1 / 8) * np.pi ** (-1) * (((-2) * f2 * h + f3 * x2 + 2 * f2 * x3) * (
                h ** 2 + x1 ** 2 + x2 ** 2 + (-2) * h * x3 + x3 ** 2) ** (-3 / 2) + 2 * h ** 2 * (
                                                 3 * f3 * x2 + 2 * f2 * (h + x3)) * (
                                                 x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (-5 / 2) + (
                                                 (-1) * f3 * x2 + (-2) * f2 * (h + x3)) * (
                                                 x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (
                                                 -3 / 2) + 3 * (
                                                 h + (-1) * x3) * (
                                                 h ** 2 + x1 ** 2 + x2 ** 2 + (
                                             -2) * h * x3 + x3 ** 2) ** (
                                                 -5 / 2) * (f1 * x1 * x2 + f3 * x2 * (
                (-1) * h + x3) + f2 * (
                                                                    h ** 2 + x1 ** 2 + 2 * x2 ** 2 + (
                                                                -2) * h * x3 + x3 ** 2)) + (
                                             -10) * h ** 2 * (h + x3) * (
                                                 x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (-7 / 2) * (
                                                 (-3) * f1 * x1 * x2 + 3 * f3 * x2 * (
                                                 h + x3) + f2 * (
                                                         h ** 2 + x1 ** 2 + (
                                                     -2) * x2 ** 2 + 2 * h * x3 + x3 ** 2)) + 3 * (
                                                 h + x3) * (
                                                 x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (-5 / 2) * (
                                                 f1 * x1 * x2 + f3 * x2 * (h + x3) + f2 * (
                                                 h ** 2 + x1 ** 2 + 2 * x2 ** 2 + 2 * h * x3 + x3 ** 2)) + (
                                             -2) * h * (x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (
                                                 -5 / 2) * (
                                                 (-3) * f1 * x1 * x2 + 4 * f3 * x2 * (
                                                 h + x3) + f2 * (
                                                         3 * h ** 2 + x1 ** 2 + (
                                                     -2) * x2 ** 2 + 6 * h * x3 + 3 * x3 ** 2)) + 10 * h * (
                                                 h + x3) * (
                                                 x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (-7 / 2) * (
                                                 f2 * (h + x3) * (
                                                 h ** 2 + x1 ** 2 + (
                                             -2) * x2 ** 2 + 2 * h * x3 + x3 ** 2) + (
                                                     -1) * x2 * (3 * f1 * x1 * (
                                                 h + x3) + f3 * ((
                                                                     -2) * h ** 2 + x1 ** 2 + x2 ** 2 + (
                                                                     -4) * h * x3 + (
                                                                     -2) * x3 ** 2))))
        j20 = (-1 / 8) * np.pi ** (-1) * (((-2) * f3 * x1 + f1 * (h + (-1) * x3)) * (
                h ** 2 + x1 ** 2 + x2 ** 2 + (-2) * h * x3 + x3 ** 2) ** (-3 / 2) + 2 * h ** 2 * (
                                                  2 * f3 * x1 + 3 * f1 * (h + x3)) * (
                                                  x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (-5 / 2) + (
                                                  2 * f3 * x1 + f1 * (h + x3)) * (
                                                  x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (-3 / 2) + (
                                              -10) * h ** 2 * x1 * (
                                                  x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (
                                                  -7 / 2) * (
                                                  3 * (f1 * x1 + f2 * x2) * (h + x3) + f3 * (
                                                  (-2) * h ** 2 + x1 ** 2 + x2 ** 2 + (
                                              -4) * h * x3 + (
                                                      -2) * x3 ** 2)) + 3 * x1 * (
                                                  h ** 2 + x1 ** 2 + x2 ** 2 + (
                                              -2) * h * x3 + x3 ** 2) ** (
                                                  -5 / 2) * ((-1) * (f1 * x1 + f2 * x2) * (
                h + (-1) * x3) + f3 * (
                                                                     2 * h ** 2 + x1 ** 2 + x2 ** 2 + (
                                                                 -4) * h * x3 + 2 * x3 ** 2)) + (
                                              -3) * x1 * (
                                                  x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (-5 / 2) * (
                                                  (f1 * x1 + f2 * x2) * (h + x3) + f3 * (
                                                  2 * h ** 2 + x1 ** 2 + x2 ** 2 + 4 * h * x3 + 2 * x3 ** 2)) + (
                                              -10) * h * x1 * (
                                                  x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (
                                                  -7 / 2) * (
                                                  f3 * (h + x3) * (2 * h ** 2 + (-1) * x1 ** 2 + (
                                              -1) * x2 ** 2 + 4 * h * x3 + 2 * x3 ** 2) + (-1) * (
                                                          f1 * x1 + f2 * x2) * (
                                                          4 * h ** 2 + x1 ** 2 + x2 ** 2 + 8 * h * x3 + 4 * x3 ** 2)) + (
                                              -2) * h * (x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (
                                                  -5 / 2) * (f1 * (
                4 * h ** 2 + 3 * x1 ** 2 + x2 ** 2 + 8 * h * x3 + 4 * x3 ** 2) + 2 * x1 * (
                                                                     f2 * x2 + f3 * (
                                                                     h + x3))))
        j21 = (-1 / 8) * np.pi ** (-1) * (((-2) * f3 * x2 + f2 * (h + (-1) * x3)) * (
                h ** 2 + x1 ** 2 + x2 ** 2 + (-2) * h * x3 + x3 ** 2) ** (-3 / 2) + 2 * h ** 2 * (
                                                  2 * f3 * x2 + 3 * f2 * (h + x3)) * (
                                                  x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (-5 / 2) + (
                                                  2 * f3 * x2 + f2 * (h + x3)) * (
                                                  x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (-3 / 2) + (
                                              -10) * h ** 2 * x2 * (
                                                  x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (
                                                  -7 / 2) * (
                                                  3 * (f1 * x1 + f2 * x2) * (h + x3) + f3 * (
                                                  (-2) * h ** 2 + x1 ** 2 + x2 ** 2 + (
                                              -4) * h * x3 + (
                                                      -2) * x3 ** 2)) + 3 * x2 * (
                                                  h ** 2 + x1 ** 2 + x2 ** 2 + (
                                              -2) * h * x3 + x3 ** 2) ** (
                                                  -5 / 2) * ((-1) * (f1 * x1 + f2 * x2) * (
                h + (-1) * x3) + f3 * (
                                                                     2 * h ** 2 + x1 ** 2 + x2 ** 2 + (
                                                                 -4) * h * x3 + 2 * x3 ** 2)) + (
                                              -3) * x2 * (
                                                  x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (-5 / 2) * (
                                                  (f1 * x1 + f2 * x2) * (h + x3) + f3 * (
                                                  2 * h ** 2 + x1 ** 2 + x2 ** 2 + 4 * h * x3 + 2 * x3 ** 2)) + (
                                              -10) * h * x2 * (
                                                  x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (
                                                  -7 / 2) * (
                                                  f3 * (h + x3) * (2 * h ** 2 + (-1) * x1 ** 2 + (
                                              -1) * x2 ** 2 + 4 * h * x3 + 2 * x3 ** 2) + (-1) * (
                                                          f1 * x1 + f2 * x2) * (
                                                          4 * h ** 2 + x1 ** 2 + x2 ** 2 + 8 * h * x3 + 4 * x3 ** 2)) + (
                                              -2) * h * (x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (
                                                  -5 / 2) * (f2 * (
                4 * h ** 2 + x1 ** 2 + 3 * x2 ** 2 + 8 * h * x3 + 4 * x3 ** 2) + 2 * x2 * (
                                                                     f1 * x1 + f3 * (
                                                                     h + x3))))
        j22 = (-1 / 8) * np.pi ** (-1) * (
                ((-1) * f1 * x1 + (-1) * f2 * x2 + 4 * f3 * (h + (-1) * x3)) * (
                h ** 2 + x1 ** 2 + x2 ** 2 + (-2) * h * x3 + x3 ** 2) ** (-3 / 2) + (
                    -2) * h ** 2 * (
                        (-3) * (f1 * x1 + f2 * x2) + 4 * f3 * (h + x3)) * (
                        x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (-5 / 2) + (
                        f1 * x1 + f2 * x2 + 4 * f3 * (h + x3)) * (
                        x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (-3 / 2) + (
                    -10) * h ** 2 * (h + x3) * (x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (
                        -7 / 2) * (3 * (f1 * x1 + f2 * x2) * (h + x3) + f3 * (
                (-2) * h ** 2 + x1 ** 2 + x2 ** 2 + (-4) * h * x3 + (-2) * x3 ** 2)) + (-3) * (
                        h + (-1) * x3) * (
                        h ** 2 + x1 ** 2 + x2 ** 2 + (-2) * h * x3 + x3 ** 2) ** (
                        -5 / 2) * ((-1) * (f1 * x1 + f2 * x2) * (h + (-1) * x3) + f3 * (
                2 * h ** 2 + x1 ** 2 + x2 ** 2 + (-4) * h * x3 + 2 * x3 ** 2)) + (-3) * (
                        h + x3) * (
                        x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (-5 / 2) * (
                        (f1 * x1 + f2 * x2) * (h + x3) + f3 * (
                        2 * h ** 2 + x1 ** 2 + x2 ** 2 + 4 * h * x3 + 2 * x3 ** 2)) + (
                    -10) * h * (h + x3) * (x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (-7 / 2) * (
                        f3 * (h + x3) * (2 * h ** 2 + (-1) * x1 ** 2 + (
                    -1) * x2 ** 2 + 4 * h * x3 + 2 * x3 ** 2) + (-1) * (
                                f1 * x1 + f2 * x2) * (
                                4 * h ** 2 + x1 ** 2 + x2 ** 2 + 8 * h * x3 + 4 * x3 ** 2)) + 2 * h * (
                        x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (-5 / 2) * (
                        (-8) * (f1 * x1 + f2 * x2) * (h + x3) + f3 * (
                        6 * h ** 2 + (-1) * x1 ** 2 + (
                    -1) * x2 ** 2 + 12 * h * x3 + 6 * x3 ** 2)))

        J = np.array(((j00, j01, j02),
                      (j10, j11, j12),
                      (j20, j21, j22),))
        return J

    def flow_velocity(self, location):
        u0_fun = lambda x0, x1, x2, f0, f1, f2, h: (
                0.75 * f0 * h * x0 ** 2 * x2 * (x0 ** 2 + x1 ** 2 + (h + x2) ** 2) ** (
            -2.5) / np.pi - 1 / 4 * f0 * h * x2 * (x0 ** 2 + x1 ** 2 + (h + x2) ** 2) ** (
                    -1.5) / np.pi + (
                        1 / 8) * f0 * x0 ** 2 * (
                        x0 ** 2 + x1 ** 2 + (h - x2) ** 2) ** (
                    -1.5) / np.pi - 1 / 8 * f0 * x0 ** 2 * (
                        x0 ** 2 + x1 ** 2 + (h + x2) ** 2) ** (-1.5) / np.pi + (
                        1 / 8) * f0 * (x0 ** 2 + x1 ** 2 + (h - x2) ** 2) ** (
                    -0.5) / np.pi - 1 / 8 * f0 * (
                        x0 ** 2 + x1 ** 2 + (h + x2) ** 2) ** (
                    -0.5) / np.pi + 0.75 * f1 * h * x0 * x1 * x2 * (
                        x0 ** 2 + x1 ** 2 + (h + x2) ** 2) ** (-2.5) / np.pi + (
                        1 / 8) * f1 * x0 * x1 * (
                        x0 ** 2 + x1 ** 2 + (h - x2) ** 2) ** (
                    -1.5) / np.pi - 1 / 8 * f1 * x0 * x1 * (
                        x0 ** 2 + x1 ** 2 + (h + x2) ** 2) ** (-1.5) / np.pi + (
                        3 / 4) * f2 * h ** 3 * x0 * (
                        x0 ** 2 + x1 ** 2 + (h + x2) ** 2) ** (-2.5) / np.pi + (
                        3 / 4) * f2 * h ** 2 * x0 * x2 * (
                        x0 ** 2 + x1 ** 2 + (h + x2) ** 2) ** (
                    -2.5) / np.pi - 0.75 * f2 * h * x0 * (h + x2) ** 2 * (
                        x0 ** 2 + x1 ** 2 + (h + x2) ** 2) ** (
                    -2.5) / np.pi - 1 / 8 * f2 * h * x0 * (
                        x0 ** 2 + x1 ** 2 + (h - x2) ** 2) ** (-1.5) / np.pi + (
                        1 / 8) * f2 * h * x0 * (
                        x0 ** 2 + x1 ** 2 + (h + x2) ** 2) ** (-1.5) / np.pi + (
                        1 / 8) * f2 * x0 * x2 * (
                        x0 ** 2 + x1 ** 2 + (h - x2) ** 2) ** (
                    -1.5) / np.pi - 1 / 8 * f2 * x0 * x2 * (
                        x0 ** 2 + x1 ** 2 + (h + x2) ** 2) ** (-1.5) / np.pi)

        u1_fun = lambda x0, x1, x2, f0, f1, f2, h: (
                0.75 * f0 * h * x0 * x1 * x2 * (x0 ** 2 + x1 ** 2 + (h + x2) ** 2) ** (
            -2.5) / np.pi + (
                        1 / 8) * f0 * x0 * x1 * (x0 ** 2 + x1 ** 2 + (h - x2) ** 2) ** (
                    -1.5) / np.pi - 1 / 8 * f0 * x0 * x1 * (x0 ** 2 + x1 ** 2 + (h + x2) ** 2) ** (
                    -1.5) / np.pi + 0.75 * f1 * h * x1 ** 2 * x2 * (
                        x0 ** 2 + x1 ** 2 + (h + x2) ** 2) ** (
                    -2.5) / np.pi - 1 / 4 * f1 * h * x2 * (x0 ** 2 + x1 ** 2 + (h + x2) ** 2) ** (
                    -1.5) / np.pi + (
                        1 / 8) * f1 * x1 ** 2 * (x0 ** 2 + x1 ** 2 + (h - x2) ** 2) ** (
                    -1.5) / np.pi - 1 / 8 * f1 * x1 ** 2 * (x0 ** 2 + x1 ** 2 + (h + x2) ** 2) ** (
                    -1.5) / np.pi + (
                        1 / 8) * f1 * (x0 ** 2 + x1 ** 2 + (h - x2) ** 2) ** (
                    -0.5) / np.pi - 1 / 8 * f1 * (
                        x0 ** 2 + x1 ** 2 + (h + x2) ** 2) ** (-0.5) / np.pi + (
                        3 / 4) * f2 * h ** 3 * x1 * (
                        x0 ** 2 + x1 ** 2 + (h + x2) ** 2) ** (-2.5) / np.pi + (
                        3 / 4) * f2 * h ** 2 * x1 * x2 * (x0 ** 2 + x1 ** 2 + (h + x2) ** 2) ** (
                    -2.5) / np.pi - 0.75 * f2 * h * x1 * (h + x2) ** 2 * (
                        x0 ** 2 + x1 ** 2 + (h + x2) ** 2) ** (
                    -2.5) / np.pi - 1 / 8 * f2 * h * x1 * (x0 ** 2 + x1 ** 2 + (h - x2) ** 2) ** (
                    -1.5) / np.pi + (
                        1 / 8) * f2 * h * x1 * (x0 ** 2 + x1 ** 2 + (h + x2) ** 2) ** (
                    -1.5) / np.pi + (
                        1 / 8) * f2 * x1 * x2 * (x0 ** 2 + x1 ** 2 + (h - x2) ** 2) ** (
                    -1.5) / np.pi - 1 / 8 * f2 * x1 * x2 * (x0 ** 2 + x1 ** 2 + (h + x2) ** 2) ** (
                    -1.5) / np.pi)

        u2_fun = lambda x0, x1, x2, f0, f1, f2, h: (
                -3 / 4 * f0 * h ** 3 * x0 * (x0 ** 2 + x1 ** 2 + (h + x2) ** 2) ** (
            -2.5) / np.pi - 3 / 4 * f0 * h ** 2 * x0 * x2 * (
                        x0 ** 2 + x1 ** 2 + (h + x2) ** 2) ** (
                    -2.5) / np.pi + 0.75 * f0 * h * x0 * (h + x2) ** 2 * (
                        x0 ** 2 + x1 ** 2 + (h + x2) ** 2) ** (
                    -2.5) / np.pi - 1 / 8 * f0 * h * x0 * (
                        x0 ** 2 + x1 ** 2 + (h - x2) ** 2) ** (
                    -1.5) / np.pi + 0.125 * f0 * h * x0 * (
                        x0 ** 2 + x1 ** 2 + (h + x2) ** 2) ** (-1.5) / np.pi + (
                        1 / 8) * f0 * x0 * x2 * (
                        x0 ** 2 + x1 ** 2 + (h - x2) ** 2) ** (
                    -1.5) / np.pi - 1 / 8 * f0 * x0 * x2 * (
                        x0 ** 2 + x1 ** 2 + (h + x2) ** 2) ** (
                    -1.5) / np.pi - 3 / 4 * f1 * h ** 3 * x1 * (
                        x0 ** 2 + x1 ** 2 + (h + x2) ** 2) ** (
                    -2.5) / np.pi - 3 / 4 * f1 * h ** 2 * x1 * x2 * (
                        x0 ** 2 + x1 ** 2 + (h + x2) ** 2) ** (
                    -2.5) / np.pi + 0.75 * f1 * h * x1 * (h + x2) ** 2 * (
                        x0 ** 2 + x1 ** 2 + (h + x2) ** 2) ** (
                    -2.5) / np.pi - 1 / 8 * f1 * h * x1 * (
                        x0 ** 2 + x1 ** 2 + (h - x2) ** 2) ** (
                    -1.5) / np.pi + 0.125 * f1 * h * x1 * (
                        x0 ** 2 + x1 ** 2 + (h + x2) ** 2) ** (-1.5) / np.pi + (
                        1 / 8) * f1 * x1 * x2 * (
                        x0 ** 2 + x1 ** 2 + (h - x2) ** 2) ** (
                    -1.5) / np.pi - 1 / 8 * f1 * x1 * x2 * (
                        x0 ** 2 + x1 ** 2 + (h + x2) ** 2) ** (
                    -1.5) / np.pi - 0.75 * f2 * h * x2 * (h + x2) ** 2 * (
                        x0 ** 2 + x1 ** 2 + (h + x2) ** 2) ** (
                    -2.5) / np.pi + 0.25 * f2 * h * x2 * (
                        x0 ** 2 + x1 ** 2 + (h + x2) ** 2) ** (-1.5) / np.pi + (
                        1 / 8) * f2 * (h - x2) ** 2 * (
                        x0 ** 2 + x1 ** 2 + (h - x2) ** 2) ** (
                    -1.5) / np.pi - 1 / 8 * f2 * (h + x2) ** 2 * (
                        x0 ** 2 + x1 ** 2 + (h + x2) ** 2) ** (-1.5) / np.pi + (
                        1 / 8) * f2 * (x0 ** 2 + x1 ** 2 + (h - x2) ** 2) ** (
                    -0.5) / np.pi - 1 / 8 * f2 * (
                        x0 ** 2 + x1 ** 2 + (h + x2) ** 2) ** (-0.5) / np.pi)
        return np.array((u0_fun(*location, *self.StokesletsStrength, self._h),
                         u1_fun(*location, *self.StokesletsStrength, self._h),
                         u2_fun(*location, *self.StokesletsStrength, self._h)))

    @property
    def StokesletsStrength(self):
        return self._StokesletsStrength

    @property
    def h(self):
        return self._h


class SingleDoubleletJefferyProblem(_JefferyProblem):
    _DoubleletStrength = ...  # type: np.ndarray
    _B = ...  # type: np.ndarray

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._type = 'SingleDoubleletJefferyProblem'
        self._DoubleletStrength = np.array(kwargs['DoubleletStrength']).reshape((1, 3)).flatten()
        self._B = np.array(kwargs['B']).reshape((1, 3)).flatten()

    def flow_gradient(self, location):
        b1, b2, b3 = self._B
        x1, x2, x3 = location
        f1, f2, f3 = self.DoubleletStrength
        j00 = (1 / 8) * np.pi ** (-1) * (x1 ** 2 + x2 ** 2 + x3 ** 2) ** (-7 / 2) * (
                3 * b3 * (f1 * x1 + f2 * x2) * x3 * (
                4 * x1 ** 2 + (-1) * x2 ** 2 + (-1) * x3 ** 2) + (
                    -1) * b3 * f3 * (2 * x1 ** 4 + (
            -1) * x2 ** 4 + x2 ** 2 * x3 ** 2 + 2 * x3 ** 4 + x1 ** 2 * (
                                             x2 ** 2 + (-11) * x3 ** 2)) + (-1) * b2 * (
                        3 * x2 * (f1 * x1 + f3 * x3) * ((-4) * x1 ** 2 + x2 ** 2 + x3 ** 2) + f2 * (
                        2 * x1 ** 4 + 2 * x2 ** 4 + x2 ** 2 * x3 ** 2 + (-1) * x3 ** 4 + x1 ** 2 * (
                        (-11) * x2 ** 2 + x3 ** 2))) + b1 * (
                        3 * x1 * (f2 * x2 + f3 * x3) * (
                        2 * x1 ** 2 + (-3) * (x2 ** 2 + x3 ** 2)) + f1 * (
                                4 * x1 ** 4 + (-10) * x1 ** 2 * (x2 ** 2 + x3 ** 2) + (
                                x2 ** 2 + x3 ** 2) ** 2)))

        j01 = (1 / 8) * np.pi ** (-1) * (x1 ** 2 + x2 ** 2 + x3 ** 2) ** (-7 / 2) * ((-3) * b3 * (
                f3 * x1 * x2 * (x1 ** 2 + x2 ** 2 + (-4) * x3 ** 2) + f2 * x1 * x3 * (
                x1 ** 2 + (-4) * x2 ** 2 + x3 ** 2) + (-1) * f1 * x2 * x3 * (
                        6 * x1 ** 2 + x2 ** 2 + x3 ** 2)) + (-1) * b1 * (3 * x2 * (
                f1 * x1 + f3 * x3) * (
                                                                                 (
                                                                                     -4) * x1 ** 2 + x2 ** 2 + x3 ** 2) + f2 * (
                                                                                 2 * x1 ** 4 + 2 * x2 ** 4 + x2 ** 2 * x3 ** 2 + (
                                                                             -1) * x3 ** 4 + x1 ** 2 * (
                                                                                         (
                                                                                             -11) * x2 ** 2 + x3 ** 2))) + (
                                                                                         -1) * b2 * (
                                                                                             3 * x1 * (
                                                                                             f3 * x3 * (
                                                                                             x1 ** 2 + (
                                                                                         -4) * x2 ** 2 + x3 ** 2) + f2 * x2 * (
                                                                                                     3 * x1 ** 2 + (
                                                                                                 -2) * x2 ** 2 + 3 * x3 ** 2)) + f1 * (
                                                                                                     4 * x1 ** 4 + (
                                                                                                 -2) * x2 ** 4 + (
                                                                                                         -1) * x2 ** 2 * x3 ** 2 + x3 ** 4 + x1 ** 2 * (
                                                                                                             (
                                                                                                                 -13) * x2 ** 2 + 5 * x3 ** 2))))

        j02 = (1 / 8) * np.pi ** (-1) * (x1 ** 2 + x2 ** 2 + x3 ** 2) ** (-7 / 2) * (
                3 * b1 * (f1 * x1 + f2 * x2) * x3 * (
                4 * x1 ** 2 + (-1) * x2 ** 2 + (-1) * x3 ** 2) + (
                    -1) * b1 * f3 * (2 * x1 ** 4 + (
            -1) * x2 ** 4 + x2 ** 2 * x3 ** 2 + 2 * x3 ** 4 + x1 ** 2 * (
                                             x2 ** 2 + (-11) * x3 ** 2)) + (-3) * b2 * (
                        f3 * x1 * x2 * (x1 ** 2 + x2 ** 2 + (-4) * x3 ** 2) + f2 * x1 * x3 * (
                        x1 ** 2 + (-4) * x2 ** 2 + x3 ** 2) + (-1) * f1 * x2 * x3 * (
                                6 * x1 ** 2 + x2 ** 2 + x3 ** 2)) + (-1) * b3 * (f1 * (
                4 * x1 ** 4 + x2 ** 4 + (-1) * x2 ** 2 * x3 ** 2 + (-2) * x3 ** 4 + x1 ** 2 * (
                5 * x2 ** 2 + (-13) * x3 ** 2)) + 3 * x1 * (f2 * x2 * (
                x1 ** 2 + x2 ** 2 + (-4) * x3 ** 2) + f3 * x3 * (3 * x1 ** 2 + 3 * x2 ** 2 + (
            -2) * x3 ** 2))))

        j10 = (1 / 8) * np.pi ** (-1) * (x1 ** 2 + x2 ** 2 + x3 ** 2) ** (-7 / 2) * ((-3) * b3 * (
                f3 * x1 * x2 * (x1 ** 2 + x2 ** 2 + (-4) * x3 ** 2) + f1 * x2 * x3 * (
                (-4) * x1 ** 2 + x2 ** 2 + x3 ** 2) + (-1) * f2 * x1 * x3 * (
                        x1 ** 2 + 6 * x2 ** 2 + x3 ** 2)) + (-1) * b2 * (3 * x1 * (
                f2 * x2 + f3 * x3) * (
                                                                                 x1 ** 2 + (
                                                                             -4) * x2 ** 2 + x3 ** 2) + f1 * (
                                                                                 2 * x1 ** 4 + 2 * x2 ** 4 + x2 ** 2 * x3 ** 2 + (
                                                                             -1) * x3 ** 4 + x1 ** 2 * (
                                                                                         (
                                                                                             -11) * x2 ** 2 + x3 ** 2))) + b1 * (
                                                                                             (
                                                                                                 -3) * x2 * (
                                                                                                     (
                                                                                                         -2) * f1 * x1 ** 3 + 3 * f1 * x1 * (
                                                                                                             x2 ** 2 + x3 ** 2) + f3 * x3 * (
                                                                                                             (
                                                                                                                 -4) * x1 ** 2 + x2 ** 2 + x3 ** 2)) + f2 * (
                                                                                                     2 * x1 ** 4 + (
                                                                                                 -4) * x2 ** 4 + (
                                                                                                         -5) * x2 ** 2 * x3 ** 2 + (
                                                                                                         -1) * x3 ** 4 + x1 ** 2 * (
                                                                                                             13 * x2 ** 2 + x3 ** 2))))

        j11 = (1 / 8) * np.pi ** (-1) * (x1 ** 2 + x2 ** 2 + x3 ** 2) ** (-7 / 2) * (
                3 * b2 * x2 * (f1 * x1 + f3 * x3) * (
                (-3) * x1 ** 2 + 2 * x2 ** 2 + (-3) * x3 ** 2) + (-3) * b3 * (
                        f1 * x1 + f2 * x2) * x3 * (x1 ** 2 + (-4) * x2 ** 2 + x3 ** 2) + b2 * f2 * (
                        x1 ** 4 + 4 * x2 ** 4 + (
                    -10) * x2 ** 2 * x3 ** 2 + x3 ** 4 + 2 * x1 ** 2 * (
                                (-5) * x2 ** 2 + x3 ** 2)) + b3 * f3 * (
                        x1 ** 4 + (-2) * x2 ** 4 + 11 * x2 ** 2 * x3 ** 2 + (-2) * x3 ** 4 + (
                    -1) * x1 ** 2 * (
                                x2 ** 2 + x3 ** 2)) + (-1) * b1 * (
                        3 * x1 * (f2 * x2 + f3 * x3) * (x1 ** 2 + (-4) * x2 ** 2 + x3 ** 2) + f1 * (
                        2 * x1 ** 4 + 2 * x2 ** 4 + x2 ** 2 * x3 ** 2 + (-1) * x3 ** 4 + x1 ** 2 * (
                        (-11) * x2 ** 2 + x3 ** 2))))

        j12 = (1 / 8) * np.pi ** (-1) * (x1 ** 2 + x2 ** 2 + x3 ** 2) ** (-7 / 2) * ((-3) * b1 * (
                f3 * x1 * x2 * (x1 ** 2 + x2 ** 2 + (-4) * x3 ** 2) + f1 * x2 * x3 * (
                (-4) * x1 ** 2 + x2 ** 2 + x3 ** 2) + (-1) * f2 * x1 * x3 * (
                        x1 ** 2 + 6 * x2 ** 2 + x3 ** 2)) + (-1) * b3 * (3 * x2 * (
                f1 * x1 * (x1 ** 2 + x2 ** 2 + (-4) * x3 ** 2) + f3 * x3 * (
                3 * x1 ** 2 + 3 * x2 ** 2 + (-2) * x3 ** 2)) + f2 * (x1 ** 4 + 4 * x2 ** 4 + (
            -13) * x2 ** 2 * x3 ** 2 + (-2) * x3 ** 4 + x1 ** 2 * (5 * x2 ** 2 + (
            -1) * x3 ** 2))) + b2 * ((-3) * (
                f1 * x1 + f2 * x2) * x3 * (x1 ** 2 + (-4) * x2 ** 2 + x3 ** 2) + f3 * (x1 ** 4 + (
            -2) * x2 ** 4 + 11 * x2 ** 2 * x3 ** 2 + (-2) * x3 ** 4 + (-1) * x1 ** 2 * (
                                                                                               x2 ** 2 + x3 ** 2))))

        j20 = (1 / 8) * np.pi ** (-1) * (x1 ** 2 + x2 ** 2 + x3 ** 2) ** (-7 / 2) * (
                3 * b2 * f3 * x1 * x2 * (x1 ** 2 + x2 ** 2 + 6 * x3 ** 2) + (-3) * b2 * x3 * (
                f2 * x1 * (x1 ** 2 + (-4) * x2 ** 2 + x3 ** 2) + f1 * x2 * (
                (-4) * x1 ** 2 + x2 ** 2 + x3 ** 2)) + (-3) * b1 * x3 * (
                        (-2) * f1 * x1 ** 3 + 3 * f1 * x1 * (x2 ** 2 + x3 ** 2) + f2 * x2 * (
                        (-4) * x1 ** 2 + x2 ** 2 + x3 ** 2)) + b1 * f3 * (
                        2 * x1 ** 4 + (-1) * x2 ** 4 + (-5) * x2 ** 2 * x3 ** 2 + (
                    -4) * x3 ** 4 + x1 ** 2 * (
                                x2 ** 2 + 13 * x3 ** 2)) + (-1) * b3 * (
                        3 * x1 * (f2 * x2 + f3 * x3) * (x1 ** 2 + x2 ** 2 + (-4) * x3 ** 2) + f1 * (
                        2 * x1 ** 4 + (-1) * x2 ** 4 + x2 ** 2 * x3 ** 2 + 2 * x3 ** 4 + x1 ** 2 * (
                        x2 ** 2 + (-11) * x3 ** 2))))

        j21 = (1 / 8) * np.pi ** (-1) * (x1 ** 2 + x2 ** 2 + x3 ** 2) ** (-7 / 2) * (
                3 * b1 * f3 * x1 * x2 * (x1 ** 2 + x2 ** 2 + 6 * x3 ** 2) + (-1) * b2 * f3 * (
                x1 ** 4 + (-2) * x2 ** 4 + (-13) * x2 ** 2 * x3 ** 2 + 4 * x3 ** 4 + (
            -1) * x1 ** 2 * (
                        x2 ** 2 + (-5) * x3 ** 2)) + (-3) * b1 * x3 * (
                        f2 * x1 * (x1 ** 2 + (-4) * x2 ** 2 + x3 ** 2) + f1 * x2 * (
                        (-4) * x1 ** 2 + x2 ** 2 + x3 ** 2)) + (-3) * b2 * x3 * (
                        f1 * x1 * (x1 ** 2 + (-4) * x2 ** 2 + x3 ** 2) + f2 * x2 * (
                        3 * x1 ** 2 + (-2) * x2 ** 2 + 3 * x3 ** 2)) + b3 * (
                        (-3) * x2 * (f1 * x1 + f3 * x3) * (
                        x1 ** 2 + x2 ** 2 + (-4) * x3 ** 2) + f2 * (
                                x1 ** 4 + (-2) * x2 ** 4 + 11 * x2 ** 2 * x3 ** 2 + (
                            -2) * x3 ** 4 + (
                                    -1) * x1 ** 2 * (x2 ** 2 + x3 ** 2))))

        j22 = (1 / 8) * np.pi ** (-1) * (x1 ** 2 + x2 ** 2 + x3 ** 2) ** (-7 / 2) * (
                (-3) * b2 * x2 * (f1 * x1 + f3 * x3) * (x1 ** 2 + x2 ** 2 + (-4) * x3 ** 2) + (
            -3) * b3 * (
                        f1 * x1 + f2 * x2) * x3 * (
                        3 * x1 ** 2 + 3 * x2 ** 2 + (-2) * x3 ** 2) + b3 * f3 * (
                        x1 ** 4 + x2 ** 4 + (
                    -10) * x2 ** 2 * x3 ** 2 + 4 * x3 ** 4 + 2 * x1 ** 2 * (
                                x2 ** 2 + (-5) * x3 ** 2)) + b2 * f2 * (
                        x1 ** 4 + (-2) * x2 ** 4 + 11 * x2 ** 2 * x3 ** 2 + (-2) * x3 ** 4 + (
                    -1) * x1 ** 2 * (
                                x2 ** 2 + x3 ** 2)) + (-1) * b1 * (
                        3 * x1 * (f2 * x2 + f3 * x3) * (x1 ** 2 + x2 ** 2 + (-4) * x3 ** 2) + f1 * (
                        2 * x1 ** 4 + (-1) * x2 ** 4 + x2 ** 2 * x3 ** 2 + 2 * x3 ** 4 + x1 ** 2 * (
                        x2 ** 2 + (-11) * x3 ** 2))))
        J = np.array(((j00, j01, j02),
                      (j10, j11, j12),
                      (j20, j21, j22),))
        return J

    def flow_velocity(self, location):
        b1, b2, b3 = self._B
        x1, x2, x3 = location
        f1, f2, f3 = self.DoubleletStrength
        u0 = (1 / 8) * np.pi ** (-1) * (x1 ** 2 + x2 ** 2 + x3 ** 2) ** (-5 / 2) * (
                b3 * f3 * x1 * (x1 ** 2 + x2 ** 2 + (-2) * x3 ** 2) + (-1) * b1 * (
                f1 * x1 + f2 * x2 + f3 * x3) * (
                        2 * x1 ** 2 + (-1) * x2 ** 2 + (-1) * x3 ** 2) + b2 * f2 * x1 * (
                        x1 ** 2 + (-2) * x2 ** 2 + x3 ** 2) + (-1) * b3 * x3 * (
                        3 * f2 * x1 * x2 + f1 * (4 * x1 ** 2 + x2 ** 2 + x3 ** 2)) + (
                    -1) * b2 * x2 * (
                        3 * f3 * x1 * x3 + f1 * (4 * x1 ** 2 + x2 ** 2 + x3 ** 2)))

        u1 = (1 / 8) * np.pi ** (-1) * (x1 ** 2 + x2 ** 2 + x3 ** 2) ** (-5 / 2) * (
                (-3) * b1 * f3 * x1 * x2 * x3 + b3 * f3 * x2 * (
                x1 ** 2 + x2 ** 2 + (-2) * x3 ** 2) + b2 * (
                        f1 * x1 + f2 * x2 + f3 * x3) * (
                        x1 ** 2 + (-2) * x2 ** 2 + x3 ** 2) + b1 * f1 * x2 * (
                        (-2) * x1 ** 2 + x2 ** 2 + x3 ** 2) + (-1) * b1 * f2 * x1 * (
                        x1 ** 2 + 4 * x2 ** 2 + x3 ** 2) + (-1) * b3 * x3 * (
                        3 * f1 * x1 * x2 + f2 * (x1 ** 2 + 4 * x2 ** 2 + x3 ** 2)))

        u2 = (1 / 8) * np.pi ** (-1) * (x1 ** 2 + x2 ** 2 + x3 ** 2) ** (-5 / 2) * (
                (-3) * b2 * f1 * x1 * x2 * x3 + (-3) * b1 * f2 * x1 * x2 * x3 + b3 * (
                f1 * x1 + f2 * x2 + f3 * x3) * (
                        x1 ** 2 + x2 ** 2 + (-2) * x3 ** 2) + b2 * f2 * x3 * (
                        x1 ** 2 + (-2) * x2 ** 2 + x3 ** 2) + b1 * f1 * x3 * (
                        (-2) * x1 ** 2 + x2 ** 2 + x3 ** 2) + (-1) * b1 * f3 * x1 * (
                        x1 ** 2 + x2 ** 2 + 4 * x3 ** 2) + (-1) * b2 * f3 * x2 * (
                        x1 ** 2 + x2 ** 2 + 4 * x3 ** 2))
        return np.array((u0, u1, u2))

    @property
    def DoubleletStrength(self):
        return self._DoubleletStrength

    @property
    def B(self):
        return self._B


class JefferyObj:
    def __init__(self, name='...', rot_v=0, **kwargs):
        self._type = 'JefferyObj'
        self._center = kwargs['center']  # type: np.ndarray
        self._norm = ...  # type: np.ndarray
        self._lateral_norm = ...  # type: np.ndarray
        self._set_norm(kwargs['norm'])
        self._set_lateral_norm(kwargs['lateral_norm'])
        self._speed = kwargs['speed']
        self._rot_v = rot_v
        # lbd = (a^2-1)/(a^2+1), a = rs2 / rs1, rs1(2) is the major (minor) axis.
        self._lbd = kwargs['lbd']  # type: np.ndarray
        self._index = -1
        self._father = None  # type: _JefferyProblem
        self._name = name
        # the following properties store the location history of the composite.
        self._update_fun = Adams_Moulton_Methods  # funHandle and order
        self._update_order = 1  # funHandle and order
        self._locomotion_fct = np.ones(3)
        self._center_hist = []
        self._norm_hist = []
        self._lateral_norm_hist = []
        self._U_hist = []
        self._dP_hist = []
        self._dP2_hist = []
        self._displace_hist = []
        self._rotation_hist = []

    def __repr__(self):
        return self._type + ' (index %d)' % self._index

    def __str__(self):
        return self._name

    @property
    def center(self):
        return self._center

    @center.setter
    def center(self, center: np.ndarray):
        err_msg = 'center=[x, y, z] has 3 components. '
        assert center.size == 3, err_msg
        self._center = center

    @property
    def norm(self):
        return self._norm

    @norm.setter
    def norm(self, norm: np.ndarray):
        self._set_norm(norm)

    @property
    def lateral_norm(self):
        return self._lateral_norm

    @lateral_norm.setter
    def lateral_norm(self, lateral_norm: np.ndarray):
        self._set_lateral_norm(lateral_norm)

    def _set_norm(self, norm: np.ndarray):
        err_msg = 'norm=[x, y, z] has 3 components and ||norm|| > 0. '
        assert norm.size == 3 and np.linalg.norm(norm) > 0, err_msg
        self._norm = norm / np.linalg.norm(norm)
        return True

    def _set_lateral_norm(self, lateral_norm: np.ndarray):
        err_msg = 'lateral_norm=%s should have 3 components and ||lateral_norm|| > 0. ' % str(
                lateral_norm)
        assert lateral_norm.size == 3 and np.linalg.norm(lateral_norm) > 0, err_msg
        self.check_orthogonality(self.norm, lateral_norm)
        self._lateral_norm = lateral_norm / np.linalg.norm(lateral_norm)
        return True

    def check_orthogonality(self, P, P2):
        err_msg = 'current norm %s and lateral norm %s is not orthogonality.' % (P, P2)
        assert np.isclose(np.dot(P, P2), 0), err_msg
        return True

    def _theta_phi_psi_v1(self, P, P2):
        t_theta_all = np.arccos(P[:, 2] / np.linalg.norm(P, axis=1))
        t_phi_all = np.arctan2(P[:, 1], P[:, 0])
        t_phi_all = np.hstack([t1 + 2 * np.pi if t1 < 0 else t1
                               for t1 in t_phi_all])  # (-pi,pi) -> (0, 2pi)

        t_psi_all = []
        ini_lateral_norm2 = self._ini_lateral_norm2
        for t_lateral_norm, t_theta, t_phi in zip(P2, t_theta_all, t_phi_all):
            t_lateral_norm = vector_rotation_norm(t_lateral_norm, norm=np.array((0, 0, 1)),
                                                  theta=-t_phi)
            t_lateral_norm = vector_rotation_norm(t_lateral_norm, norm=np.array((0, 1, 0)),
                                                  theta=- t_theta)
            sign = np.sign(np.dot(np.array((0, 0, 1)), np.cross(ini_lateral_norm2, t_lateral_norm)))
            t_psi = sign * np.arccos(np.clip(np.dot(ini_lateral_norm2, t_lateral_norm)
                                             / np.linalg.norm(t_lateral_norm)
                                             / np.linalg.norm(ini_lateral_norm2),
                                             -1, 1))
            tfct = np.zeros_like(t_psi)
            tfct[t_psi < 0] = 2
            t_psi = t_psi + tfct * np.pi  # (-pi,pi) -> (0, 2pi)
            t_psi_all.append(t_psi)
        t_psi_all = np.hstack(t_psi_all)
        return t_theta_all, t_phi_all, t_psi_all

    def _P2_psi(self, t_theta, t_phi, tP2):
        if np.isclose(t_theta, np.pi / 2):
            cos_psi = -1 * tP2[2] / np.sin(t_theta)
            if np.isclose(t_phi, np.pi / 2):
                sin_psi = -1 * (tP2[0] - np.cos(t_phi) * np.cos(t_theta) * cos_psi) / np.sin(t_phi)
            else:
                sin_psi = +1 * (tP2[1] - np.sin(t_phi) * np.cos(t_theta) * cos_psi) / np.cos(t_phi)
        else:
            tA = np.array(((-1 * np.sin(t_phi), np.cos(t_theta) * np.cos(t_phi)),
                           (+1 * np.cos(t_phi), np.cos(t_theta) * np.sin(t_phi))))
            tb = np.array((tP2[0], tP2[1]))
            sin_psi, cos_psi = np.linalg.solve(tA, tb)
        t_psi = np.arctan2(sin_psi, cos_psi)
        # t_psi = t_psi + 2 * np.pi if t_psi < 0 else t_psi  # (-pi,pi) -> (0, 2pi)
        t_psi = t_psi + np.pi * 1.5 if t_psi < np.pi / 2 \
            else t_psi - np.pi / 2  # (-pi,pi) -> (0, 2pi)
        return t_psi

    def _theta_phi_psi_v2(self, P1, P2):
        t_theta_all = np.arccos(P1[:, 2] / np.linalg.norm(P1, axis=1))
        t_phi_all = np.arctan2(P1[:, 1], P1[:, 0])
        t_phi_all = np.hstack([t1 + 2 * np.pi if t1 < 0 else t1
                               for t1 in t_phi_all])  # (-pi,pi) -> (0, 2pi)

        t_psi_all = []
        for t_theta, t_phi, tP2 in zip(t_theta_all, t_phi_all, P2):
            t_psi = self._P2_psi(t_theta, t_phi, tP2)
            t_psi_all.append(t_psi)
        t_psi_all = np.hstack(t_psi_all)
        return t_theta_all, t_phi_all, t_psi_all

    def _theta_phi_psi(self, P1, P2):
        return self._theta_phi_psi_v2(P1, P2)

    @property
    def theta_phi_psi(self):
        t_theta_all, t_phi_all, t_psi_all = self._theta_phi_psi(np.vstack(self.norm_hist),
                                                                np.vstack(self.lateral_norm_hist))
        return t_theta_all, t_phi_all, t_psi_all

    @property
    def speed(self):
        return self._speed

    @speed.setter
    def speed(self, speed):
        speed = np.array(speed).flatten()
        err_msg = 'speed is a scalar. '
        assert speed.size == 1, err_msg
        self._speed = speed

    @property
    def rot_v(self):
        return self._rot_v

    @rot_v.setter
    def rot_v(self, rot_v):
        rot_v = np.array(rot_v).flatten()
        err_msg = 'rot_v is a scalar. '
        assert rot_v.size == 1, err_msg
        self._rot_v = rot_v

    @property
    def lbd(self):
        return self._lbd

    @lbd.setter
    def lbd(self, lbd):
        lbd = np.array(lbd).flatten()
        err_msg = 'lbd is a scalar. '
        assert lbd.size == 1, err_msg
        self._lbd = lbd

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, idx):
        self._index = idx

    @property
    def father(self):
        return self._father

    @father.setter
    def father(self, problem):
        self._father = problem

    @property
    def center_hist(self):
        return self._center_hist

    @property
    def norm_hist(self):
        return self._norm_hist

    @property
    def lateral_norm_hist(self):
        return self._lateral_norm_hist

    @property
    def U_hist(self):
        return self._U_hist

    @property
    def displace_hist(self):
        return self._displace_hist

    @property
    def rotation_hist(self):
        return self._rotation_hist

    @property
    def update_order(self):
        return self._update_order

    def set_update_para(self, fix_x=False, fix_y=False, fix_z=False,
                        update_fun=Adams_Moulton_Methods, update_order=1, **kwargs):
        # for a cutoff infinity symmetric problem,
        #   each time step set the obj in the center of the cutoff region to improve the accuracy.
        self._locomotion_fct = np.array((not fix_x, not fix_y, not fix_z), dtype=np.float)
        self._update_fun = update_fun
        self._update_order = update_order
        return self._locomotion_fct

    def dbg_set_update_para(self, fix_x=1, fix_y=1, fix_z=1, update_fun=Adams_Moulton_Methods,
                            update_order=1):
        # for a cutoff infinity symmetric problem,
        #   each time step set the obj in the center of the cutoff region to improve the accuracy.
        self._locomotion_fct = np.array((fix_x, fix_y, fix_z))
        self._update_fun = update_fun
        self._update_order = update_order
        return self._locomotion_fct

    def move(self, displacement):
        self._center = self._center + displacement
        return True

    def rotate(self, rotation, rotation2):
        self._norm = self._norm + rotation
        self._norm = self._norm / np.linalg.norm(self._norm)
        self._lateral_norm = self._lateral_norm + rotation2
        self._lateral_norm = self._lateral_norm / np.linalg.norm(self._lateral_norm)
        # # dbg
        # print(self._norm, np.linalg.norm(self._norm))
        return True

    def get_dP_at(self, X, P, P2, rot_v):
        S, Omega = self.father.flow_strain_rotation(X)

        dP = np.dot(Omega, P) + self._lbd * (np.dot(S, P) - np.dot(P, np.dot(S, P)) * P) + \
             np.cross(P2 * rot_v, P)
        dP2 = np.dot(Omega, P2) - self._lbd * np.dot(P, np.dot(S, P2)) * P
        omega = np.cross(P, dP) / np.dot(P, P)
        return dP, dP2, omega

    def get_dX_at(self, X, P, trs_v):
        Ub = self.father.flow_velocity(X)  # background velocity
        dX = trs_v * P + Ub
        # print(trs_v, P, Ub)
        return dX

    def _get_velocity_at(self, X, P, P2, trs_v, rot_v):
        dX = self.get_dX_at(X, P, trs_v)
        dP, dP2, omega = self.get_dP_at(X, P, P2, rot_v)
        return dX, dP, dP2, omega

    def get_velocity_at(self, X, P, P2, trs_v=0, rot_v=0, check_orthogonality=True):
        if check_orthogonality:
            self.check_orthogonality(P, P2)
        return self._get_velocity_at(X, P, P2, trs_v, rot_v)

    def update_location(self, eval_dt, print_handle=''):
        P = self.norm
        P2 = self.lateral_norm
        X = self.center
        trs_v = self.speed
        rot_v = self.rot_v

        dX, dP, dP2, omega = self._get_velocity_at(X, P, P2, trs_v, rot_v)
        self._U_hist.append(np.hstack((dX, omega)))
        self._dP_hist.append(dP)
        self._dP2_hist.append(dP2)

        fct = self._locomotion_fct
        order = np.min((len(self.U_hist), self.update_order))
        dst_fct_list = [fct[:3] for fct in self.U_hist[-1:-(order + 1):-1]]
        rot_fct_list = [fct for fct in self._dP_hist[-1:-(order + 1):-1]]
        rot2_fct_list = [fct for fct in self._dP2_hist[-1:-(order + 1):-1]]
        distance_true = self._update_fun(order, dst_fct_list, eval_dt)
        rotation = self._update_fun(order, rot_fct_list, eval_dt)
        rotation2 = self._update_fun(order, rot2_fct_list, eval_dt)
        # # dbg
        # print(distance_true)
        # print(rotation, np.linalg.norm(rotation))
        # print()
        distance = distance_true * fct
        self.move(distance)
        self.rotate(rotation, rotation2)
        self._center_hist.append(self.center)
        self._norm_hist.append(self.norm)
        self._lateral_norm_hist.append(self.lateral_norm)
        self._displace_hist.append(distance_true)
        self._rotation_hist.append(rotation)

        # mpiprint('---->%s %s at %s' % (str(self), print_handle, self.center))
        # mpiprint('    U', U)
        # mpiprint('    norm', P)
        # mpiprint('    |ref_U|', np.hstack((np.linalg.norm(U[:3]), np.linalg.norm(U[3:]))))
        # tU = np.dot(U[:3], P) / np.dot(P, P)
        # tW = np.dot(U[3:], P) / np.dot(P, P)
        # mpiprint('    ref_U projection on norm', np.hstack((tU, tW)))

        # print('---->%s %s at %s' % (str(self), print_handle, self.center))
        # print('    U', U)
        # print('    norm', P)
        # print('    |ref_U|', np.hstack((np.linalg.norm(U[:3]), np.linalg.norm(U[3:]))))
        # tU = np.dot(U[:3], P) / np.dot(P, P)
        # tW = np.dot(U[3:], P) / np.dot(P, P)
        # print('    ref_U projection on norm', np.hstack((tU, tW)))
        return True


class TableObj(JefferyObj):
    def __init__(self, table_name, name='...', rot_v=0, ini_psi=0, **kwargs):
        super().__init__(name=name, rot_v=rot_v, **kwargs)
        err_msg = 'current version rot_v==0'
        assert rot_v == 0, err_msg

        self._type = 'TableObj'
        self._intp_fun_list = []
        self._intp_psi_list = []
        self.load_table(table_name=table_name)

        # !!! after rotate back.
        P0 = self.norm  # ini direction of norm
        theta0 = np.arccos(P0[2] / np.linalg.norm(P0))
        phi0 = np.arctan2(P0[1], P0[0])
        phi0 = phi0 + 2 * np.pi if phi0 < 0 else phi0  # (-pi,pi) -> (0, 2pi)
        tP = vector_rotation_norm(self.lateral_norm, norm=np.array((0, 0, 1)), theta=-phi0)
        tP = vector_rotation_norm(tP, norm=np.array((0, 1, 0)), theta=-theta0)
        self._ini_lateral_norm2 = tP / np.linalg.norm(tP)

        # # dbg
        # ini_lateral_norm = self.ini_lateral_norm
        # t1 = -1 * ini_lateral_norm[2] / np.sin(theta0)
        # t2 = -1 * (ini_lateral_norm[0] - np.cos(phi0) * np.cos(theta0) * t1) / np.sin(phi0)
        # psi0 = np.arctan2(t2, t1)
        # psi00 = psi0 + 2 * np.pi if psi0 < 0 else psi0  # (-pi,pi) -> (0, 2pi)
        # self._lateral_norm = vector_rotation_norm(self.lateral_norm,
        #                                           norm=self.norm, theta=ini_psi)
        # ini_lateral_norm = self.ini_lateral_norm
        # t1 = -1 * ini_lateral_norm[2] / np.sin(theta0)
        # t2 = -1 * (ini_lateral_norm[0] - np.cos(phi0) * np.cos(theta0) * t1) / np.sin(phi0)
        # psi0 = np.arctan2(t2, t1)
        # psi01 = psi0 + 2 * np.pi if psi0 < 0 else psi0  # (-pi,pi) -> (0, 2pi)
        # t1 = psi01 - psi00
        # t1 = t1 if t1 > 0 else (2 * np.pi + t1)
        # print(theta0, phi0, psi00, psi01, t1)
        # assert 1 == 2

        # rotate a ini psi
        self._lateral_norm = vector_rotation_norm(self.lateral_norm,
                                                  norm=self.norm, theta=ini_psi)
        # ini information
        self._ini_center = self.center.copy()
        self._ini_norm = self.norm.copy()
        self._ini_lateral_norm = self.lateral_norm.copy()

    @property
    def ini_center(self):
        return self._ini_center

    @property
    def ini_norm(self):
        return self._ini_norm

    @property
    def ini_lateral_norm(self):
        return self._ini_lateral_norm

    @property
    def intp_psi_list(self):
        return self._intp_psi_list

    def load_table(self, table_name):
        table_name = check_file_extension(table_name, extension='.pickle')
        t_path = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.normpath(t_path + '/' + table_name)
        with open(full_path, 'rb') as handle:
            table_data = pickle.load(handle)
        intp_fun_list = self._intp_fun_list
        intp_psi_list = self._intp_psi_list
        for tpsi, table_psi_data in table_data:
            tintp_fun_list = []
            intp_psi_list.append(tpsi)
            for ty, tx, tU in table_psi_data:
                tfun = interpolate.RectBivariateSpline(ty, tx, tU.values, kx=3, ky=3)
                # tfun = interpolate.interp2d(tx, ty, tU.values.T, kind='quintic', copy=False, )
                tintp_fun_list.append(tfun)
            intp_fun_list.append(tintp_fun_list)
        return True

    def intp_U_fun(self, t_theta, t_phi, t_psi):
        intp_fun_list = self._intp_fun_list
        intp_psi_list = self._intp_psi_list

        # # version 2
        # if 0 <= t_theta <= np.pi and 0 <= t_phi < np.pi:  # letf down
        #     sign_list = [1, 1, 1, 1, 1, 1]
        # elif 0 <= t_theta <= np.pi and np.pi <= t_phi <= 2 * np.pi:  # right down
        #     t_theta = t_theta
        #     t_phi = t_phi - np.pi
        #     sign_list = [1, 1, -1, 1, 1, -1]
        # else:
        #     err_msg = 'norm_theta %f and (or) norm_phi %f out of range (0, pi) * (0, 2pi)' % (t_theta, t_phi)
        #     raise Exception(err_msg)

        # version full table
        sign_list = np.ones(6)
        err_msg = 'norm_theta %f and (or) norm_phi %f out of range (0, pi) * (0, 2pi)' % (
            t_theta, t_phi)
        assert 0 <= t_theta <= np.pi and 0 <= t_phi <= 2 * np.pi, err_msg

        intp_U = []
        for tfun in intp_fun_list:
            t_U = []
            for intp_fun, sign in zip(tfun, sign_list):
                t_U.append(intp_fun(t_theta, t_phi) * sign)
            intp_U.append(np.hstack(t_U).flatten())
        intp_U.append(intp_U[0].copy())
        intp_U = np.vstack(intp_U)
        intp_psi = np.hstack([intp_psi_list, np.pi * 2])
        intp_fun1d = interpolate.interp1d(intp_psi, intp_U, kind='quadratic',
                                          copy=False, axis=0, bounds_error=True)
        return intp_fun1d(t_psi)

    def get_dP_at(self, X, P, P2, rot_v):
        raise Exception('This function do NOT work in %s Obj' % self._name)

    def get_dX_at(self, X, P, trs_v):
        raise Exception('This function do NOT work in %s Obj' % self._name)

    def _get_velocity_at(self, X, P, P2, trs_v, rot_v=0):
        # values associated with ini direction.
        P20 = self._ini_lateral_norm2  # ini direction of lateral norm

        t_theta = np.arccos(P[2] / np.linalg.norm(P))
        t_phi = np.arctan2(P[1], P[0])
        tfct = np.zeros_like(t_phi)
        tfct[t_phi < 0] = 2
        t_phi = t_phi + tfct * np.pi  # (-pi,pi) -> (0, 2pi)

        # rotate the lateral norm back (the direction that norm=(0, 0, 1),
        #   and compare with ini lateral norm to calculate psi.
        tP = vector_rotation_norm(P2, norm=np.array((0, 0, 1)), theta=-t_phi)
        tP = vector_rotation_norm(tP, norm=np.array((0, 1, 0)), theta=-t_theta)
        sign = np.sign(np.dot(np.array((0, 0, 1)), np.cross(P20, tP)))
        # print('P2', P20, tP, sign)
        t_psi = sign * np.arccos(
                np.clip(np.dot(tP, P20) / np.linalg.norm(tP) / np.linalg.norm(P20), -1, 1))
        tfct = np.zeros_like(t_psi)
        tfct[t_psi < 0] = 2
        t_psi = t_psi + tfct * np.pi  # (-pi,pi) -> (0, 2pi)

        # # old version, update use dX and dP
        # Ub = self.father.flow_velocity(X)  # background velocity
        # tU = self.intp_U_fun(t_theta, t_phi, t_psi)
        # dX = tU[:3] + Ub + trs_v * P
        # omega = tU[3:]
        # dP = np.cross(omega, P)
        # dP2 = np.cross(omega, P2)
        # return dX, dP, dP2, omega

        # new version, update use ref_U
        ref_U = self.intp_U_fun(t_theta, t_phi, t_psi)
        Ub = self.father.flow_velocity(X)  # background velocity
        ref_U = ref_U + np.hstack((Ub + trs_v * P, np.zeros(3)))
        # print()
        # print(X, P, P2, trs_v, rot_v)
        # print(t_theta, t_phi, t_psi, P2)
        # print(ref_U)
        return ref_U

    def get_velocity_at3(self, X, theta, phi, psi):
        ref_U = self.intp_U_fun(theta, phi, psi)
        Ub = self.father.flow_velocity(X)  # background velocity
        ref_U = ref_U + np.hstack((Ub, np.zeros(3)))
        return ref_U

    def get_velocity_at(self, X, P, P2, trs_v=0, rot_v=0, check_orthogonality=True):
        err_msg = 'current version rot_v==0'
        assert rot_v == 0, err_msg
        return super().get_velocity_at(X, P, P2, trs_v, rot_v, check_orthogonality)

    def node_rotation(self, norm=np.array([0, 0, 1]), theta=np.zeros(1), rotation_origin=None, ):
        rotation_origin = self._center if rotation_origin is None else rotation_origin
        rotation = get_rot_matrix(norm, theta)
        t_origin = self._center
        self._center = np.dot(rotation, (self._center - rotation_origin)) + rotation_origin
        self._norm = np.dot(rotation, (self._norm + t_origin - rotation_origin)) \
                     + rotation_origin - self._center
        self._norm = self._norm / np.linalg.norm(self._norm)
        self._lateral_norm = np.dot(rotation, (self._lateral_norm + t_origin - rotation_origin)) \
                             + rotation_origin - self._center
        self._lateral_norm = self._lateral_norm / np.linalg.norm(self._lateral_norm)
        return True

    def lateral_norm_rotation(self, eval_dt):
        pass

    def update_location(self, eval_dt, print_handle=''):
        fct = self._locomotion_fct
        P = self.norm
        P2 = self.lateral_norm
        X = self.center
        trs_v = self.speed
        rot_v = self.rot_v
        ref_U = self._get_velocity_at(X, P, P2, trs_v, rot_v)
        omega = ref_U[3:]
        dP = np.cross(omega, P)
        dP2 = np.cross(omega, P2)
        self._U_hist.append(ref_U)
        self._dP_hist.append(dP)
        self._dP2_hist.append(dP2)

        order = np.min((len(self.U_hist), self.update_order))
        fct_list = self.U_hist[-1:-(order + 1):-1]
        dst_fct_list = [fct[:3] for fct in fct_list]
        rot_fct_list = [fct[3:] for fct in fct_list]
        distance_true = self._update_fun(order, dst_fct_list, eval_dt)
        rotation = self._update_fun(order, rot_fct_list, eval_dt)
        distance = distance_true * fct
        self.move(distance)
        self.node_rotation(norm=rotation, theta=np.linalg.norm(rotation))
        self.lateral_norm_rotation(eval_dt)
        self._center_hist.append(self._center)
        self._norm_hist.append(self._norm)
        self._lateral_norm_hist.append(self.lateral_norm)
        self._displace_hist.append(distance_true)
        self._rotation_hist.append(rotation)
        return True


class TableEcoli(TableObj):
    def __init__(self, table_name, omega_tail, name='...', rot_v=0, ini_psi=0, **kwargs):
        super().__init__(table_name, name, rot_v, ini_psi, **kwargs)
        self._omega_tail = omega_tail  # the norm of the rotational velocity of tail.

    @property
    def omega_tail(self):
        return self._omega_tail

    @omega_tail.setter
    def omega_tail(self, omega_tail):
        self._set_omega_tail(omega_tail)

    def _set_omega_tail(self, omega_tail):
        omega_tail = np.array(omega_tail)
        # print('dbg', omega_tail)
        err_msg = 'omega_tail is a scale define the norm of the rotational velocity of tail. '
        assert omega_tail.shape == (1,), err_msg
        self._omega_tail = omega_tail
        return True

    def lateral_norm_rotation(self, eval_dt):
        # rotate by rel_U_tail.
        self.lateral_norm = vector_rotation_norm(self.lateral_norm, norm=self.norm,
                                                 theta=self.omega_tail * eval_dt)
        return True


class TableAvrObj(TableObj):
    def load_table(self, table_name):
        t_path = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.normpath(t_path + '/' + '%s.pickle' % table_name)
        with open(full_path, 'rb') as handle:
            table_data = pickle.load(handle)
        intp_fun_list = self._intp_fun_list
        for ty, tx, tU in table_data:
            tfun = interpolate.RectBivariateSpline(ty, tx, tU.values, kx=3, ky=3)
            # tfun = interpolate.interp2d(tx, ty, tU.values.T, kind='quintic', copy=False, )
            intp_fun_list.append(tfun)
        return True

    def intp_U_fun(self, t_theta, t_phi, t_psi):
        intp_fun_list = self._intp_fun_list
        err_msg = 'norm_theta %f and (or) norm_phi %f out of range (0, pi) * (0, 2pi)' % \
                  (t_theta, t_phi)
        assert 0 <= t_theta <= np.pi and 0 <= t_phi <= 2 * np.pi, err_msg

        t_U = []
        for intp_fun in intp_fun_list:
            t_U.append(intp_fun(t_theta, t_phi)[0])
        return np.hstack(t_U)


class TableRkObj(TableObj):
    def __init__(self, table_name, name='...', rot_v=0, ini_psi=0, **kwargs):
        super().__init__(table_name, name, rot_v, ini_psi, **kwargs)
        self._t_hist = []

    @property
    def t_hist(self):
        return self._t_hist

    def _get_velocity_at(self, X, P, P2, trs_v, rot_v=0):
        # values associated with ini direction.
        P20 = self._ini_lateral_norm2  # ini direction of lateral norm

        t_theta = np.arccos(P[2] / np.linalg.norm(P))
        t_phi = np.arctan2(P[1], P[0])
        tfct = np.zeros_like(t_phi)
        tfct[t_phi < 0] = 2
        t_phi = t_phi + tfct * np.pi  # (-pi,pi) -> (0, 2pi)

        # rotate the lateral norm back (the direction that norm=(0, 0, 1),
        #   and compare with ini lateral norm to calculate psi.
        tP = vector_rotation_norm(P2, norm=np.array((0, 0, 1)), theta=-t_phi)
        tP = vector_rotation_norm(tP, norm=np.array((0, 1, 0)), theta=-t_theta)
        sign = np.sign(np.dot(np.array((0, 0, 1)), np.cross(P20, tP)))
        t_psi = sign * np.arccos(
                np.clip(np.dot(tP, P20) / np.linalg.norm(tP) / np.linalg.norm(P20), -1, 1))
        tfct = np.zeros_like(t_psi)
        tfct[t_psi < 0] = 2
        t_psi = t_psi + tfct * np.pi  # (-pi,pi) -> (0, 2pi)

        Ub = self.father.flow_velocity(X)  # background velocity
        # print('dbg t_theta, t_phi, t_psi')
        # print(t_theta, t_phi, t_psi)
        tU = self.intp_U_fun(t_theta, t_phi, t_psi)
        dX = tU[:3] + Ub + trs_v * P
        omega = tU[3:]
        dP = np.cross(omega, P)
        dP2 = np.cross(omega, P2)
        return dX, dP, dP2, omega

    def _wrapper_solve_ivp(self, t, y):
        X = y[0:3]
        P = y[3:6]
        P2 = y[6:9]
        trs_v = self.speed
        rot_v = self.rot_v
        dX, dP, dP2, omega = self._get_velocity_at(X, P, P2, trs_v, rot_v)
        dX = dX * self._locomotion_fct
        return np.hstack((dX, dP, dP2))

    def set_update_para(self, fix_x=False, fix_y=False, fix_z=False,
                        update_fun=integrate.RK45, rtol=1e-6, atol=1e-9, **kwargs):
        # for a cutoff infinity symmetric problem,
        #   each time step set the obj in the center of the cutoff region to improve the accuracy.
        self._locomotion_fct = np.array((not fix_x, not fix_y, not fix_z), dtype=np.float)
        self._update_fun = update_fun
        self._update_order = (rtol, atol)
        return self._locomotion_fct

    def update_self(self, t1, t0=0):
        y0 = np.hstack((self.center, self.norm, self.lateral_norm))
        (rtol, atol) = self._update_order
        sol = integrate.solve_ivp(self._wrapper_solve_ivp, [t0, t1], y0,
                                  method=self._update_fun, rtol=rtol, atol=atol, vectorized=False)
        Table_t = sol.t
        ty = sol.y
        Table_X = ty[0:3].T
        Table_P = ty[3:6].T
        Table_P2 = ty[6:9].T
        Table_dt = np.hstack((np.diff(Table_t), 0))

        self._t_hist = Table_t
        self._center_hist = Table_X
        self._norm_hist = Table_P
        self._lateral_norm_hist = Table_P2
        return Table_t, Table_dt, Table_X, Table_P, Table_P2


class TableRk4nObj(TableRkObj):
    def __init__(self, table_name, name='...', rot_v=0, ini_psi=0, **kwargs):
        super().__init__(table_name, name, rot_v, ini_psi, **kwargs)
        self.q = self._get_q(self.norm, self.lateral_norm)

    def _get_q(self, P, P2):
        e0 = np.vstack((np.cross(P2, P), P2, P)).T
        tq1 = spR.from_matrix(e0).as_quat()
        q = Quaternion()
        q.set_wxyz(tq1[3], tq1[0], tq1[1], tq1[2])
        return q

    @property
    def q(self):
        return self._q

    @q.setter
    def q(self, q: Quaternion):
        norm_q = np.linalg.norm(q.q)
        err_msg = '||q|| should be 1, now is %f ' % norm_q
        assert np.isclose(norm_q, 1), err_msg
        self._q = q

    def _wrapper_solve_ivp(self, t, y):
        X = y[0:3]  # center (x, y, z)
        Q = y[3:7]  # quaternion (w, x, y, z), where w=cos(theta/2).
        tq = self.q
        tq.set_wxyz(*Q)
        trs_v = self.speed
        rot_v = self.rot_v
        R = tq.get_R()
        P = R[:, 2]
        P2 = R[:, 1]

        dX, dP, dP2, omega = self._get_velocity_at(X, P, P2, trs_v, rot_v)
        dX = dX * self._locomotion_fct
        G = tq.get_E()
        dQ = 0.5 * omega.dot(G)
        return np.hstack((dX, dQ))

    def update_self(self, t1, t0=0):
        tq = self.q
        y0 = np.hstack((self.center, tq.q))
        (rtol, atol) = self._update_order
        sol = integrate.solve_ivp(self._wrapper_solve_ivp, [t0, t1], y0,
                                  method=self._update_fun, rtol=rtol, atol=atol)
        Table_t = sol.t
        ty = sol.y
        Table_X = ty[0:3].T
        Table_q = ty[3:7].T
        Table_P = Table_X.copy()
        Table_P2 = Table_X.copy()
        for i0, tQ in enumerate(Table_q):
            tq.set_wxyz(*tQ)
            R = tq.get_R()
            tP = R[:, 2]
            tP2 = R[:, 1]
            Table_P[i0, :] = tP
            Table_P2[i0, :] = tP2
        Table_dt = np.hstack((np.diff(Table_t), 0))

        self._t_hist = Table_t
        self._center_hist = Table_X
        self._norm_hist = Table_P
        self._lateral_norm_hist = Table_P2
        return Table_t, Table_dt, Table_X, Table_P, Table_P2


class TableRkEcoli(TableEcoli, TableRkObj):
    def _get_velocity_at(self, X, P, P2, trs_v, rot_v=0):
        dX, dP, dP2, omega = super()._get_velocity_at(X, P, P2, trs_v, rot_v)
        omega_tail = P * self.omega_tail / np.linalg.norm(P)
        omega = omega + omega_tail
        return dX, dP, dP2, omega


class TableRk4nEcoli(TableRkEcoli, TableRk4nObj):
    def _nothing(self):
        pass


class TablePetscObj(TableRkObj):
    def __init__(self, table_name, name='...', rot_v=0, ini_psi=0, **kwargs):
        super().__init__(table_name, name, rot_v, ini_psi, **kwargs)
        self._comm = PETSc.COMM_SELF
        self._save_every = 1
        self._tqdm_fun = tqdm_notebook
        self._tqdm = None
        self._t1 = -1  # simulation time in the range (0, t1)
        self._max_it = -1  # iteration loop no more than max_it
        self._percentage = 0  # percentage of time depend solver.
        self._dt_hist = []
        self._tmp_idx = []  # temporary globe idx

    @property
    def dt_hist(self):
        return self._dt_hist

    def _rhsfunction(self, ts, t, Y, F):
        X, P, P2 = self._get_X_P_P2(Y)
        trs_v = self.speed
        rot_v = self.rot_v
        dX, dP, dP2, omega = self._get_velocity_at(X, P, P2, trs_v, rot_v)
        dX = dX * self._locomotion_fct
        F[:] = np.hstack((dX, dP, dP2))
        F.assemble()
        return True

    def _postfunction(self, ts):
        Y = ts.getSolution()
        X, P, P2 = self._get_X_P_P2(Y)

        P = P / np.linalg.norm(P)
        P2 = P2 / np.linalg.norm(P2)
        Y[:] = np.hstack((X, P, P2))
        Y.assemble()
        return True

    def set_update_para(self, fix_x=False, fix_y=False, fix_z=False, update_fun='3bs',
                        rtol=1e-6, atol=1e-9, save_every=1, tqdm_fun=tqdm_notebook, **kwargs):
        # for a cutoff infinity symmetric problem,
        #   each time step set the obj in the center of the cutoff region to improve the accuracy.
        self._locomotion_fct = np.array((not fix_x, not fix_y, not fix_z), dtype=np.float)
        self._update_fun = update_fun
        self._tqdm_fun = tqdm_fun
        self._update_order = (rtol, atol)
        self._save_every = save_every
        return self._locomotion_fct

    def _get_X_P_P2(self, Y):
        y = Y.getArray().copy()
        X = y[0:3]
        P = y[3:6]
        P2 = y[6:9]
        return X, P, P2

    def _do_store_data(self, ts, i, t, Y):
        X, P, P2 = self._get_X_P_P2(Y)
        dt = ts.getTimeStep()
        self.t_hist.append(t)
        self.dt_hist.append(dt)
        self.center_hist.append(X)
        self.norm_hist.append(P)
        self.lateral_norm_hist.append(P2)
        return True

    def _monitor(self, ts, i, t, Y):
        save_every = self._save_every
        # print(ts.getTimeStep())
        if not i % save_every:
            percentage = np.clip(t / self._t1 * 100, 0, 100)
            dp = int(percentage - self._percentage)
            if dp >= 1:
                self._tqdm.update(dp)
                self._percentage = self._percentage + dp
            self._do_store_data(ts, i, t, Y)
        return True

    def _get_y0(self):
        y0 = np.hstack((self.center, self.norm, self.lateral_norm))
        return y0

    def get_simulate_results(self):
        self.center = self.center_hist[-2]
        self.norm = self.norm_hist[-2]
        self.lateral_norm = self.lateral_norm_hist[-2]
        self._tmp_idx = np.hstack(self.t_hist) < self._t1
        self._t_hist = [j for (i, j) in zip(self._tmp_idx, self.t_hist) if i]
        self._dt_hist = [j for (i, j) in zip(self._tmp_idx, self.dt_hist) if i]
        self._center_hist = [j for (i, j) in zip(self._tmp_idx, self.center_hist) if i]
        self._norm_hist = [j for (i, j) in zip(self._tmp_idx, self.norm_hist) if i]
        self._lateral_norm_hist = [j for (i, j) in zip(self._tmp_idx, self.lateral_norm_hist) if i]
        Table_t = np.hstack(self.t_hist)
        Table_dt = np.hstack(self.dt_hist)
        Table_X = np.vstack(self.center_hist)
        Table_P = np.vstack(self.norm_hist)
        Table_P2 = np.vstack(self.lateral_norm_hist)
        return Table_t, Table_dt, Table_X, Table_P, Table_P2

    def update_self(self, t1, t0=0, max_it=10 ** 9, eval_dt=0.001):
        comm = self._comm
        (rtol, atol) = self._update_order
        update_fun = self._update_fun
        tqdm_fun = self._tqdm_fun
        self._tqdm = tqdm_fun(total=100)
        self._t1 = t1
        self._max_it = max_it
        self._percentage = 0

        # do simulation
        y0 = self._get_y0()
        y = PETSc.Vec().createWithArray(y0, comm=comm)
        f = y.duplicate()
        ts = PETSc.TS().create(comm=comm)
        ts.setProblemType(ts.ProblemType.NONLINEAR)
        ts.setType(ts.Type.RK)
        ts.setRKType(update_fun)
        ts.setRHSFunction(self._rhsfunction, f)
        ts.setTime(t0)
        ts.setMaxTime(t1)
        ts.setMaxSteps(max_it)
        ts.setTimeStep(eval_dt)
        ts.setMonitor(self._monitor)
        ts.setPostStep(self._postfunction)
        ts.setExactFinalTime(PETSc.TS.ExactFinalTime.MATCHSTEP)
        ts.setFromOptions()
        ts.setSolution(y)
        ts.setTolerances(rtol, atol)
        ts.setUp()
        self._do_store_data(ts, 0, 0, y)
        ts.solve(y)

        # finish simulation
        self._tqdm.update(100 - self._percentage)
        self._tqdm.close()
        i = ts.getStepNumber()
        t = ts.getTime()
        Y = ts.getSolution()
        self._do_store_data(ts, i, t, Y)
        return self.get_simulate_results()


class TablePetsc4nObj(TablePetscObj, TableRk4nObj):
    def _get_X_P_P2(self, Y):
        y = Y.getArray().copy()
        X = y[0:3]  # center (x, y, z)
        Q = y[3:7]  # quaternion (w, x, y, z), where w=cos(theta/2).
        tq = self.q
        tq.set_wxyz(*Q)
        R = tq.get_R()
        P = R[:, 2]
        P2 = R[:, 1]
        return X, P, P2

    def _get_y0(self):
        tq = self.q
        y0 = np.hstack((self.center, tq.q))
        return y0

    def _rhsfunction(self, ts, t, Y, F):
        # print('###################################################################')
        y = Y.getArray()
        X = y[0:3]  # center (x, y, z)
        Q = y[3:7]  # quaternion (w, x, y, z), where w=cos(theta/2).
        tq = self.q
        tq.set_wxyz(*Q)
        R = tq.get_R()
        P = R[:, 2]
        P2 = R[:, 1]

        trs_v = self.speed
        rot_v = self.rot_v
        dX, _, _, omega = self._get_velocity_at(X, P, P2, trs_v, rot_v)
        # print('dbg, dX', dX)
        # print('dbg, omega', omega)
        dX = dX * self._locomotion_fct
        dQ = 0.5 * omega.dot(tq.get_E())
        F[:] = np.hstack((dX, dQ))
        F.assemble()
        return True

    def _postfunction(self, ts):
        Y = ts.getSolution()
        y = Y.getArray()
        X = y[0:3]  # center (x, y, z)
        Q = y[3:7]  # quaternion (w, x, y, z), where w=cos(theta/2).

        # print('dbg Q', Q)
        Q = Q / np.linalg.norm(Q)
        Y[:] = np.hstack((X, Q))
        Y.assemble()
        return True


class TablePetsc4nEcoli(TableRkEcoli, TablePetsc4nObj):
    def _nothing(self):
        pass


class TablePetsc4nPsiEcoli(TablePetsc4nEcoli):
    def __init__(self, table_name, omega_tail, name='...', ini_psi=0, **kwargs):
        super().__init__(table_name, omega_tail, name, rot_v=0, ini_psi=ini_psi, **kwargs)
        self._type = 'TablePetsc4nPsiEcoli'
        self._psi = np.hstack((ini_psi,))
        self._ini_psi = self.psi.copy()[0]
        # the following properties store the location history of the composite.
        # self._q_hist = []
        self._psi_hist = []

    @property
    def psi(self):
        return self._psi

    @psi.setter
    def psi(self, psi):
        self._psi = psi

    @property
    def ini_psi(self):
        return self._ini_psi

    @ini_psi.setter
    def ini_psi(self, ini_psi):
        self._ini_psi = ini_psi

    # @property
    # def q_hist(self):
    #     return self._q_hist

    @property
    def psi_hist(self):
        return self._psi_hist

    def _theta_phi_psi2_v1(self, P1, P2):
        # angles of head
        t_theta = np.arccos(P1[2] / np.linalg.norm(P1))
        t_phi = np.arctan2(P1[1], P1[0])
        tfct = 2 if t_phi < 0 else 0
        t_phi = t_phi + tfct * np.pi  # (-pi,pi) -> (0, 2pi)
        # rotate the lateral norm back (the direction that norm=(0, 0, 1),
        #   and compare with ini lateral norm to calculate psi.
        tP = vector_rotation_norm(P2, norm=np.array((0, 0, 1)), theta=-t_phi)
        tP = vector_rotation_norm(tP, norm=np.array((0, 1, 0)), theta=-t_theta)
        P20 = self._ini_lateral_norm2  # ini direction of lateral norm
        sign = np.sign(np.dot(np.array((0, 0, 1)), np.cross(P20, tP)))
        t_psi = sign * np.arccos(np.clip(np.dot(tP, P20) / np.linalg.norm(tP)
                                         / np.linalg.norm(P20), -1, 1))
        tfct = 2 if t_psi < 0 else 0
        t_psi = t_psi + tfct * np.pi  # (-pi,pi) -> (0, 2pi)
        return t_theta, t_phi, t_psi

    def _theta_phi_psi2_v2(self, P1, P2):
        # angles of head
        t_theta = np.arccos(P1[2] / np.linalg.norm(P1))
        t_phi = np.arctan2(P1[1], P1[0])
        tfct = 2 if t_phi < 0 else 0
        t_phi = t_phi + tfct * np.pi  # (-pi,pi) -> (0, 2pi)
        t_psi = self._P2_psi(t_theta, t_phi, P2)
        return t_theta, t_phi, t_psi

    def _theta_phi_psi2(self, P, P2):
        return self._theta_phi_psi2_v2(P, P2)

    def theta_phi_psi2(self, P, P2):
        return self._theta_phi_psi2(P, P2)

    def _get_velocity_at2(self, X, P, P2, psi):
        # print('dbg', P, P2)
        t_theta, t_phi, t_psi = self._theta_phi_psi2(P, P2)
        t_psi = (t_psi + psi - self._ini_psi) % (2 * np.pi)
        Ub = self.father.flow_velocity(X)  # background velocity
        # print('dbg t_theta, t_phi, t_psi')
        # print(t_theta, t_phi, t_psi)
        tU = self.intp_U_fun(t_theta, t_phi, t_psi)
        dX = tU[:3] + Ub
        omega = tU[3:]
        return dX, omega

    def _rhsfunction(self, ts, t, Y, F):
        # print('###################################################################')
        y = Y.getArray()
        X = y[0:3]  # center (x, y, z)
        Q = y[3:7]  # quaternion (w, x, y, z), where w=cos(theta/2).
        psi = y[7]  # relative tail spin about head
        tq = self.q
        tq.set_wxyz(*Q)
        R = tq.get_R()
        P = R[:, 2]
        P2 = R[:, 1]
        # print('dbg, self.q', self.q)

        dX, omega = self._get_velocity_at2(X, P, P2, psi)
        # print('dbg, dX', dX)
        # print('dbg, omega', omega)
        dX = dX * self._locomotion_fct
        dQ = 0.5 * omega.dot(tq.get_E())
        F[:] = np.hstack((dX, dQ, self.omega_tail))
        F.assemble()
        return True

    def _do_store_data(self, ts, i, t, Y):
        y = Y.getArray().copy()
        X = y[0:3]  # center (x, y, z)
        Q = y[3:7]  # quaternion (w, x, y, z), where w=cos(theta/2).
        psi = y[7]
        tq = self.q
        tq.set_wxyz(*Q)
        R = tq.get_R()
        P = R[:, 2]
        P2 = R[:, 1]
        _, _, t_psi = self._theta_phi_psi2(P, P2)
        t_psi = (t_psi + psi - self._ini_psi) % (2 * np.pi)

        dt = ts.getTimeStep()
        self.t_hist.append(t)
        self.dt_hist.append(dt)
        self.center_hist.append(X)
        self.norm_hist.append(P)
        self.lateral_norm_hist.append(P2)
        self.psi_hist.append(t_psi)
        return True

    def _get_y0(self):
        tq = self.q
        y0 = np.hstack((self.center, tq.q, self.psi))
        return y0

    def get_simulate_results(self):
        Table_t, Table_dt, Table_X, Table_P, Table_P2 = super().get_simulate_results()
        self.psi[:] = self.psi_hist[-2]
        self.q = self._get_q(self.norm, self.lateral_norm)
        self._psi_hist = [j for (i, j) in zip(self._tmp_idx, self.psi_hist) if i]
        Table_psi = np.hstack(self.psi_hist)

        # # dbg
        # from matplotlib import pyplot as plt
        # plt.plot(Table_X[:, 0])
        return Table_t, Table_dt, Table_X, Table_P, Table_P2, Table_psi

    def _postfunction(self, ts):
        Y = ts.getSolution()
        y = Y.getArray()
        X = y[0:3]  # center (x, y, z)
        Q = y[3:7]  # quaternion (w, x, y, z), where w=cos(theta/2).
        psi = y[7]

        # print('dbg Q', Q)
        Q = Q / np.linalg.norm(Q)
        Y[:] = np.hstack((X, Q, psi))
        Y.assemble()
        return True


class TableAvrPetsc4nObj(TableAvrObj, TablePetsc4nObj):
    def _nothing(self):
        pass


class TableAvrPetsc4nEcoli(TableAvrObj, TablePetsc4nEcoli):
    def _nothing(self):
        pass


class GivenFlowPetsc4nPsiObj(TablePetsc4nPsiEcoli):
    def __init__(self, table_name, flow_strength, omega_tail, name='...', ini_psi=0, **kwargs):
        self._uEbase_list = []
        self._uSbase_list = []
        self._wEbase_list = []
        self._wSbase_list = []
        self._Ua_loc = np.zeros(6)  # active part of translational and rotational velocity
        super().__init__(table_name, omega_tail, name, ini_psi=ini_psi, **kwargs)
        self._flow_strength = flow_strength

    def Rloc2glb(self, theta, phi, psi):
        Rloc2glb = np.array(
                ((np.cos(phi) * np.cos(psi) * np.cos(theta) - np.sin(phi) * np.sin(psi),
                  -(np.cos(psi) * np.sin(phi)) - np.cos(phi) * np.cos(theta) * np.sin(psi),
                  np.cos(phi) * np.sin(theta)),
                 (np.cos(psi) * np.cos(theta) * np.sin(phi) + np.cos(phi) * np.sin(psi),
                  np.cos(phi) * np.cos(psi) - np.cos(theta) * np.sin(phi) * np.sin(psi),
                  np.sin(phi) * np.sin(theta)),
                 (-(np.cos(psi) * np.sin(theta)),
                  np.sin(psi) * np.sin(theta),
                  np.cos(theta))))
        return Rloc2glb

    @abc.abstractmethod
    def Eij_loc(self, theta, phi, psi):
        return

    @abc.abstractmethod
    def Sij_loc(self, theta, phi, psi):
        return

    def load_table(self, table_name):
        table_name = check_file_extension(table_name, extension='.pickle')
        t_path = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.normpath(t_path + '/' + table_name)
        with open(full_path, 'rb') as handle:
            pickle_dict = pickle.load(handle)
        uEbase_list = np.vstack(pickle_dict['uw_Base_list'])[1:6, 0:3]
        wEbase_list = np.vstack(pickle_dict['uw_Base_list'])[1:6, 3:6]
        # uSbase_list = np.vstack(pickle_dict['uw_Base_list'])[6:9, 0:3]
        # wSbase_list = np.vstack(pickle_dict['uw_Base_list'])[6:9, 3:6]
        uSbase_list = np.zeros((3, 3))
        wSbase_list = np.array(((1, 0, 0), (0, 1, 0), (0, 0, 1)))
        self._uEbase_list = uEbase_list
        self._uSbase_list = uSbase_list
        self._wEbase_list = wEbase_list
        self._wSbase_list = wSbase_list
        self._Ua_loc = pickle_dict['uw_Base_list'][9]
        self._pickle_kwargs = pickle_dict['problem_kwargs']
        return True

    @property
    def uEbase_list(self):
        return self._uEbase_list

    @property
    def uSbase_list(self):
        return self._uSbase_list

    @property
    def wEbase_list(self):
        return self._wEbase_list

    @property
    def wSbase_list(self):
        return self._wSbase_list

    @property
    def Ua_loc(self):
        return self._Ua_loc

    @property
    def flow_strength(self):
        return self._flow_strength

    # back flow induced passive velocity of the microswimmer
    def calc_Up_fun(self, t_theta, t_phi, t_psi, X, Rlog2glb):
        uEbase_list = self.uEbase_list
        wEbase_list = self.wEbase_list
        problem = self.father  # Type: JefferyProblem

        # # # version 1
        # MBF_U_b = problem.MBF_U_b(X)
        # MBF_W_S = problem.MBF_W_S(X)
        # MBF_beta = problem.MBF_beta(t_theta, t_phi, t_psi, X)
        # # print(MBF_W_S)
        # # # version 2
        # MBF_U_b = problem.MBF_U_b_v2(X)
        # MBF_W_S = problem.MBF_W_S_v2(X)
        # MBF_beta = problem.MBF_beta_v2(t_theta, t_phi, t_psi, X)
        # # print(MBF_W_S)
        # # version 2, speedup
        MBF_U_b, MBF_W_S, MBF_beta = problem.MBF_Ub_WS_beta_v2(t_theta, t_phi, t_psi, X)

        uE_loc = np.sum([a * b for a, b in zip(MBF_beta, uEbase_list)], axis=0)
        Up = MBF_U_b + np.dot(Rlog2glb, uE_loc)
        wE_loc = np.sum([a * b for a, b in zip(MBF_beta, wEbase_list)], axis=0)
        Wp = MBF_W_S + np.dot(Rlog2glb, wE_loc)

        np.set_printoptions(formatter={'float_kind': "{:10.5f}".format})
        # print('dbg Up', Up)
        # print('dbg wE_loc', wE_loc)
        # print('dbg Wp', Wp)
        # print()
        return Up, Wp

    def simu_Up_fun(self, t_theta, t_phi, t_psi, X, Rlog2glb):
        pass

    # active velocity
    def calc_Ua_fun(self, Rlog2glb):
        omega_tail = self.omega_tail
        Ua_loc = self._Ua_loc
        Ua = omega_tail * np.dot(Rlog2glb, Ua_loc[:3])
        Wa = omega_tail * np.dot(Rlog2glb, Ua_loc[3:])
        return Ua, Wa

    def _get_velocity_at_thphps(self, X, t_theta, t_phi, t_psi, psi):
        t_psi = (t_psi + psi - self._ini_psi) % (2 * np.pi)
        Rlog2glb = self.Rloc2glb(t_theta, t_phi, t_psi)
        Up, Wp = self.calc_Up_fun(t_theta, t_phi, t_psi, X, Rlog2glb)
        Ua, Wa = self.calc_Ua_fun(Rlog2glb)
        dX = Up + Ua
        omega = Wp + Wa
        # print(dX, omega)
        # print('dbg thphps, %.20f, %.20f, %.20f' % (t_theta, t_phi, t_psi))
        # print('dbg Rlog2glb')
        # print(Rlog2glb)
        # print('dbg dX, %.20f, %.20f, %.20f' % (dX[0], dX[1], dX[2]))
        # print('dbg omega, %.20f, %.20f, %.20f' % (omega[0], omega[1], omega[2]))
        # print()
        return dX, omega

    def _get_velocity_at2(self, X, P, P2, psi):
        t_theta, t_phi, t_psi = self._theta_phi_psi2(P, P2)
        # print('dbg, t_theta, t_phi, t_psi', t_theta, t_phi, t_psi)
        dX, omega = self._get_velocity_at_thphps(X, t_theta, t_phi, t_psi, psi)
        return dX, omega


class ShearFlowPetsc4nPsiObj(GivenFlowPetsc4nPsiObj):
    def Eij_loc(self, theta, phi, psi):
        Eij_loc = np.array(
                ((np.cos(psi) * (-(np.cos(phi) * np.cos(psi) * np.cos(theta)) +
                                 np.sin(phi) * np.sin(psi)) * np.sin(theta),
                  (2 * np.cos(2 * psi) * np.sin(phi) * np.sin(theta) +
                   np.cos(phi) * np.sin(2 * psi) * np.sin(2 * theta)) / 4.,
                  (np.cos(phi) * np.cos(psi) * np.cos(2 * theta) -
                   np.cos(theta) * np.sin(phi) * np.sin(psi)) / 2.),
                 ((2 * np.cos(2 * psi) * np.sin(phi) * np.sin(theta) +
                   np.cos(phi) * np.sin(2 * psi) * np.sin(2 * theta)) / 4.,
                  -(np.sin(psi) * (np.cos(psi) * np.sin(phi) +
                                   np.cos(phi) * np.cos(theta) * np.sin(psi)) * np.sin(theta)),
                  (-(np.cos(psi) * np.cos(theta) * np.sin(phi)) -
                   np.cos(phi) * np.cos(2 * theta) * np.sin(psi)) / 2.),
                 ((np.cos(phi) * np.cos(psi) * np.cos(2 * theta) -
                   np.cos(theta) * np.sin(phi) * np.sin(psi)) / 2.,
                  (-(np.cos(psi) * np.cos(theta) * np.sin(phi)) -
                   np.cos(phi) * np.cos(2 * theta) * np.sin(psi)) / 2.,
                  np.cos(phi) * np.cos(theta) * np.sin(theta))))
        return Eij_loc

    def Sij_loc(self, theta, phi, psi):
        Sij_loc = np.array(
                ((0,
                  -(np.sin(phi) * np.sin(theta)) / 2.,
                  (np.cos(phi) * np.cos(psi) - np.cos(theta) * np.sin(phi) * np.sin(psi)) / 2.),
                 ((np.sin(phi) * np.sin(theta)) / 2.,
                  0,
                  (-(np.cos(psi) * np.cos(theta) * np.sin(phi)) - np.cos(phi) * np.sin(psi)) / 2.),
                 ((-(np.cos(phi) * np.cos(psi)) + np.cos(theta) * np.sin(phi) * np.sin(psi)) / 2.,
                  (np.cos(psi) * np.cos(theta) * np.sin(phi) + np.cos(phi) * np.sin(psi)) / 2.,
                  0)))
        return Sij_loc


# rewrite the obj part using the base flow method
class GivenFlowObj(GivenFlowPetsc4nPsiObj):
    def __init__(self, ini_theta, ini_phi, ini_psi, table_name, omega_tail, name='...', **kwargs):
        # err_msg = 'this parameter is abandoned in current version'
        # assert np.isnan(flow_strength)
        # err_msg = 'this parameter is abandoned in current version'
        # assert np.isclose(ini_psi, 0)

        kwargs['norm'], kwargs['lateral_norm'] = self.thphps2P(ini_theta, ini_phi, ini_psi)
        self._pickle_kwargs = []
        self._simu_mode = False
        self._problem = None

        super().__init__(table_name=table_name, flow_strength=np.nan, omega_tail=omega_tail,
                         name=name, ini_psi=0, **kwargs)
        self._ini_theta = ini_theta
        self._ini_phi = ini_phi
        self._ini_psi = 0  # here psi is the pure tail rotation.
        self._theta = ini_theta
        self._phi = ini_phi
        self._psi = 0  # here psi is the pure tail rotation.
        self.q = self.thphps2q(ini_theta, ini_phi, ini_psi)
        self._q_hist = []
        # self._ini_center = kwargs['center']

        # the following parameters are abandoned in current version
        self._displace_hist = np.nan
        self._norm = np.nan
        self._lateral_norm = np.nan
        # self._ini_norm = np.nan
        self._norm_hist = np.nan
        # self._ini_lateral_norm = np.nan
        self._lateral_norm_hist = np.nan
        self._ini_lateral_norm2 = np.nan
        self._uSbase_list = np.nan
        self._wSbase_list = np.nan
        self._flow_strength = np.nan
        self._locomotion_fct = np.nan
        self._intp_fun_list = np.nan
        self._intp_psi_list = np.nan
        # self._tmp_idx = np.nan
        self._speed = np.nan
        self._dP_hist = np.nan
        self._dP2_hist = np.nan
        self._speed = np.nan
        self._lbd = np.nan
        self._rot_v = np.nan

    def set_update_para(self, fix_x=False, fix_y=False, fix_z=False, update_fun='3bs',
                        rtol=1e-6, atol=1e-9, save_every=1, tqdm_fun=tqdm_notebook,
                        simu_mode=False, **kwargs):
        # for a cutoff infinity symmetric problem,
        #   each time step set the obj in the center of the cutoff region to improve the accuracy.
        self._locomotion_fct = np.array((not fix_x, not fix_y, not fix_z), dtype=np.float)
        self._update_fun = update_fun
        self._tqdm_fun = tqdm_fun
        self._update_order = (rtol, atol)
        self._save_every = save_every
        self._simu_mode = simu_mode

        # if simu_mode:
        #     problem_kwargs = self._pickle_kwargs
        #     ecoli_comp = create_ecoli_2part(**problem_kwargs)
        #     # self._problem =
        return self._locomotion_fct

    @property
    def q_hist(self):
        return self._q_hist

    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, theta):
        self._theta = theta

    @property
    def phi(self):
        return self._phi

    @phi.setter
    def phi(self, phi):
        self._phi = phi

    @property
    def psi(self):
        return self._psi

    @psi.setter
    def psi(self, psi):
        self._psi = psi

    @property
    def theta_phi_psi(self):
        t_theta_all, t_phi_all, t_psi_all = [], [], []
        psi_t = self.psi
        for q in self.q_hist:
            t1 = q.get_thphps()
            t_theta_all.append(t1[0])
            t_phi_all.append(t1[1])
            t_psi_all.append(t1[2])
        t_theta_all = np.hstack(t_theta_all)
        t_phi_all = np.hstack(t_phi_all)
        t_psi_all = (np.hstack(t_psi_all) + psi_t) % (2 * np.pi)
        return t_theta_all, t_phi_all, t_psi_all

    def thphps2P(self, theta, phi, psi):
        rotM = self.Rloc2glb(theta, phi, psi)
        P0 = rotM[:, 2]
        P1 = rotM[:, 1]
        return P0, P1

    def thphps2q(self, theta, phi, psi):
        rotM = self.Rloc2glb(theta, phi, psi)
        tq1 = spR.from_matrix(rotM).as_quat()
        q = Quaternion()
        q.set_wxyz(tq1[3], tq1[0], tq1[1], tq1[2])
        return q

    def _rhsfunction(self, ts, t, Y, F):
        y = Y.getArray()
        X = y[0:3]  # center (x, y, z)
        Q = y[3:7]  # quaternion (w, x, y, z), where w=cos(theta/2).
        psi = y[7]  # relative tail spin about head
        tq = self.q
        tq.set_wxyz(*Q)
        tq.normalize()
        # print('dbg, self.q', self.q)

        t_theta, t_phi, t_psi = tq.get_thphps()
        # print('dbg, q', tq)
        # print('dbg, t_theta, t_phi, t_psi', t_theta, t_phi, t_psi)
        # print()
        dX, omega = self._get_velocity_at_thphps(X, t_theta, t_phi, t_psi, psi)
        # print('dbg X, %.2f, %.20f, %.20f, %.20f' % (t, X[0], X[1], X[2]))
        # print('dbg, omega', omega)
        dX = dX * self._locomotion_fct
        dQ = 0.5 * omega.dot(tq.get_E())
        F[:] = np.hstack((dX, dQ, self.omega_tail))
        F.assemble()
        return True

    # Todo: update this function
    def _monitor(self, ts, i, t, Y):
        save_every = self._save_every
        # print(ts.getTimeStep())
        if not i % save_every:
            percentage = np.clip(t / self._t1 * 100, 0, 100)
            dp = int(percentage - self._percentage)
            if dp >= 1:
                self._tqdm.update(dp)
                self._percentage = self._percentage + dp
            self._do_store_data(ts, i, t, Y)
        return True

    def _do_store_data(self, ts, i, t, Y):
        y = Y.getArray().copy()
        self.center = y[0:3]  # center (x, y, z)
        self.q.set_wxyz(*y[3:7])  # quaternion (w, x, y, z), where w=cos(theta/2).
        self.psi = y[7]
        self.theta, self.phi, tpsi = self.q.get_thphps()
        dX, omega = self._get_velocity_at_thphps(self.center, self.theta, self.phi, tpsi, self.psi)

        dt = ts.getTimeStep()
        self.t_hist.append(t)
        self.dt_hist.append(dt)
        self.center_hist.append(self.center)
        self.q_hist.append(self.q.copy())
        self.U_hist.append(dX)
        self.rotation_hist.append(omega)
        self.psi_hist.append(self.psi)
        return True

    def get_simulate_results(self):
        # self.center = self.center_hist[-2]
        # self.norm = self.norm_hist[-2]
        # self.lateral_norm = self.lateral_norm_hist[-2]
        self._tmp_idx = np.hstack(self.t_hist) < self._t1
        self._t_hist = [j for (i, j) in zip(self._tmp_idx, self.t_hist) if i]
        self._dt_hist = [j for (i, j) in zip(self._tmp_idx, self.dt_hist) if i]
        self._center_hist = [j for (i, j) in zip(self._tmp_idx, self.center_hist) if i]
        self._q_hist = [j for (i, j) in zip(self._tmp_idx, self.q_hist) if i]
        self._U_hist = [j for (i, j) in zip(self._tmp_idx, self.U_hist) if i]
        self._rotation_hist = [j for (i, j) in zip(self._tmp_idx, self.rotation_hist) if i]
        self._psi_hist = [j for (i, j) in zip(self._tmp_idx, self.psi_hist) if i]

        base_t = np.hstack(self.t_hist)
        base_dt = np.hstack(self.dt_hist)
        base_X = np.vstack(self.center_hist)
        base_thphps = np.vstack([q.get_thphps() for q in self.q_hist])
        base_U = np.vstack(self.U_hist)
        base_W = np.vstack(self.rotation_hist)
        # tail rotation respect to the global coordinate.
        base_psi_t = (np.hstack(self.psi_hist) + base_thphps[:, 2]) % (2 * np.pi)
        return base_t, base_dt, base_X, base_thphps, base_U, base_W, base_psi_t

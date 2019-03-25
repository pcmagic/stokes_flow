# coding=utf-8
"""
Classes for solving jeffery equations.
Zhang Ji, 20181219
"""

# import sys
# sys.path = ['..'] + sys.path
# from memory_profiler import profile
# from math import sin, cos
# import warnings

# from pyvtk import *
# import os
# import matplotlib.pyplot as plt
# import copy
# from scipy.io import savemat, loadmat
# from evtk.hl import pointsToVTK, gridToVTK
# from petsc4py import PETSc
# import pickle
# from time import time
# from tqdm import tqdm
# from src.geo import *
# from src.ref_solution import *

import numpy as np
from src.support_class import *
import abc
from scipy import interpolate
import os
import pickle

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
    def flow_strain(self, location):
        ...

    @abc.abstractmethod
    def flow_rotation(self, location):
        ...

    def flow_strain_rotation(self, location):
        S_ij = self.flow_strain(location)
        Omega_ij = self.flow_rotation(location)
        return S_ij, Omega_ij

    @abc.abstractmethod
    def flow_velocity(self, location):
        ...

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


# class _Jeffery3DProblem(_JefferyProblem):
#     def __init__(self, name='...', **kwargs):
#         super().__init__(name=name, **kwargs)
#         self._type = 'Jeffery3DProblem'


class ShearJefferyProblem(_JefferyProblem):
    _planeShearRate = ...  # type: np.ndarray

    # current version the velocity of shear flow points to the x axis and only varys in the z axis.
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._type = 'ShearJefferyProblem'
        self._planeShearRate = np.array(kwargs['planeShearRate']).reshape((1, 3))
        err_msg = 'shear flow velocity is must vertical to (y, z) plane. '
        assert np.all(np.isclose(self._planeShearRate[0, -2:], (0, 0))), err_msg

    def flow_strain(self, location):
        tao_x = self._planeShearRate[0, 0]
        S_ij = 1 / 2 * np.array(((0, 0, tao_x,),
                                 (0, 0, 0,),
                                 (tao_x, 0, 0,)))
        return S_ij

    def flow_rotation(self, location):
        tao_x = self._planeShearRate[0, 0]
        Omega_ij = 1 / 2 * np.array(((0, 0, tao_x,),
                                     (0, 0, 0,),
                                     (-tao_x, 0, 0,)))
        return Omega_ij

    def flow_velocity(self, location):
        loc_z = location[-1]
        tao_x = self._planeShearRate[0, 0]
        given_u = np.array((tao_x * loc_z, 0, 0))
        return given_u

    @property
    def planeShearRate(self):
        return self._planeShearRate


class ShearTableProblem(ShearJefferyProblem):
    def nothing(self):
        pass


class SingleStokesletsJefferyProblem(_JefferyProblem):
    _StokesletsStrength = ...  # type: np.ndarray

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._type = 'SingleStokesletsJefferyProblem'
        self._StokesletsStrength = np.array(kwargs['StokesletsStrength']).reshape((1, 3)).flatten()

    def flow_strain(self, location):
        S_fun = lambda x0, x1, x2, f0, f1, f2: np.array(
                [[(-3.0 * x0 ** 2 * (x0 ** 2 + x1 ** 2 + x2 ** 2) ** 1.5 +
                   1.0 * (x0 ** 2 + x1 ** 2 + x2 ** 2) ** 2.5) * (x0 ** 2 + x1 ** 2 + x2 ** 2) ** (-4.0) * (
                          f0 * x0 + f1 * x1 + f2 * x2),
                  -3.0 * x0 * x1 * (x0 ** 2 + x1 ** 2 + x2 ** 2) ** (-2.5) * (f0 * x0 + f1 * x1 + f2 * x2),
                  -3.0 * x0 * x2 * (x0 ** 2 + x1 ** 2 + x2 ** 2) ** (-2.5) * (f0 * x0 + f1 * x1 + f2 * x2)],
                 [-3.0 * x0 * x1 * (x0 ** 2 + x1 ** 2 + x2 ** 2) ** (-2.5) * (f0 * x0 + f1 * x1 + f2 * x2),
                  (-3.0 * x1 ** 2 * (x0 ** 2 + x1 ** 2 + x2 ** 2) ** 1.5 +
                   1.0 * (x0 ** 2 + x1 ** 2 + x2 ** 2) ** 2.5) * (x0 ** 2 + x1 ** 2 + x2 ** 2) ** (-4.0) * (
                          f0 * x0 + f1 * x1 + f2 * x2),
                  -3.0 * x1 * x2 * (x0 ** 2 + x1 ** 2 + x2 ** 2) ** (-2.5) * (f0 * x0 + f1 * x1 + f2 * x2)],
                 [-3.0 * x0 * x2 * (x0 ** 2 + x1 ** 2 + x2 ** 2) ** (-2.5) * (f0 * x0 + f1 * x1 + f2 * x2),
                  -3.0 * x1 * x2 * (x0 ** 2 + x1 ** 2 + x2 ** 2) ** (-2.5) * (f0 * x0 + f1 * x1 + f2 * x2),
                  (-3.0 * x2 ** 2 * (x0 ** 2 + x1 ** 2 + x2 ** 2) ** 1.5 +
                   1.0 * (x0 ** 2 + x1 ** 2 + x2 ** 2) ** 2.5) * (x0 ** 2 + x1 ** 2 + x2 ** 2) ** (-4.0) * (
                          f0 * x0 + f1 * x1 + f2 * x2)]]) / (8 * np.pi)
        return S_fun(*location, *self._StokesletsStrength)

    def flow_rotation(self, location):
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
        return Omega_fun(*location, *self._StokesletsStrength)
        # x0, x1, x2 = location
        # f0, f1, f2 = self._StokesletsStrength
        # o00 = np.ones_like(x0)
        # o01 = (-1.0 * f0 * x1 + f1 * x0) * (x0 ** 2 + x1 ** 2 + x2 ** 2) ** (-1.5)
        # o02 = (-1.0 * f0 * x2 + f2 * x0) * (x0 ** 2 + x1 ** 2 + x2 ** 2) ** (-1.5)
        # o10 = (f0 * x1 - 1.0 * f1 * x0) * (x0 ** 2 + x1 ** 2 + x2 ** 2) ** (-1.5)
        # o11 = np.ones_like(x0)
        # o12 = (-1.0 * f1 * x2 + f2 * x1) * (x0 ** 2 + x1 ** 2 + x2 ** 2) ** (-1.5)
        # o20 = (f0 * x2 - 1.0 * f2 * x0) * (x0 ** 2 + x1 ** 2 + x2 ** 2) ** (-1.5)
        # o21 = (f1 * x2 - 1.0 * f2 * x1) * (x0 ** 2 + x1 ** 2 + x2 ** 2) ** (-1.5)
        # o22 = np.ones_like(x0)
        # Omega = np.vstack(((o00, o01, o02),
        #                    (o10, o11, o12),
        #                    (o20, o21, o22)))

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

    def J_matrix(self, location):
        h = self._h
        x1, x2, x3 = location
        f1, f2, f3 = self.StokesletsStrength
        j00 = (1 / 8) * np.pi ** (-1) * ((h ** 2 + x1 ** 2 + x2 ** 2 + (-2) * h * x3 + x3 ** 2) ** (-3 / 2) * (
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
                h ** 2 + 2 * x1 ** 2 + x2 ** 2 + (-2) * h * x3 + x3 ** 2)) + (-10) * h ** 2 * x1 * (
                                                 x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (-7 / 2) * (
                                                 (-3) * f2 * x1 * x2 + 3 * f3 * x1 * (h + x3) + f1 * (h ** 2 + (
                                             -2) * x1 ** 2 + x2 ** 2 + 2 * h * x3 + x3 ** 2)) + 3 * x1 * (
                                                 x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (-5 / 2) * (
                                                 f2 * x1 * x2 + f3 * x1 * (h + x3) + f1 * (
                                                 h ** 2 + 2 * x1 ** 2 + x2 ** 2 + 2 * h * x3 + x3 ** 2)) + 10 * h * x1 * (
                                                 x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (-7 / 2) * (f1 * (h + x3) * (
                h ** 2 + (-2) * x1 ** 2 + x2 ** 2 + 2 * h * x3 + x3 ** 2) + (-1) * x1 * (3 * f2 * x2 * (
                h + x3) + f3 * ((-2) * h ** 2 + x1 ** 2 + x2 ** 2 + (-4) * h * x3 + (-2) * x3 ** 2))))
        j01 = (1 / 8) * np.pi ** (-1) * (
                (f2 * x1 + 2 * f1 * x2) * (h ** 2 + x1 ** 2 + x2 ** 2 + (-2) * h * x3 + x3 ** 2) ** (
                -3 / 2) + 2 * h ** 2 * ((-3) * f2 * x1 + 2 * f1 * x2) * (x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (
                        -5 / 2) + 2 * h * (
                        2 * f3 * x1 * x2 + 3 * f2 * x1 * (h + x3) + (-2) * f1 * x2 * (h + x3)) * (
                        x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (-5 / 2) + (-1) * (f2 * x1 + 2 * f1 * x2) * (
                        x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (-3 / 2) + (-3) * x2 * (
                        h ** 2 + x1 ** 2 + x2 ** 2 + (-2) * h * x3 + x3 ** 2) ** (-5 / 2) * (
                        f2 * x1 * x2 + f3 * x1 * ((-1) * h + x3) + f1 * (
                        h ** 2 + 2 * x1 ** 2 + x2 ** 2 + (-2) * h * x3 + x3 ** 2)) + (-10) * h ** 2 * x2 * (
                        x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (-7 / 2) * (
                        (-3) * f2 * x1 * x2 + 3 * f3 * x1 * (h + x3) + f1 * (
                        h ** 2 + (-2) * x1 ** 2 + x2 ** 2 + 2 * h * x3 + x3 ** 2)) + 3 * x2 * (
                        x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (-5 / 2) * (
                        f2 * x1 * x2 + f3 * x1 * (h + x3) + f1 * (
                        h ** 2 + 2 * x1 ** 2 + x2 ** 2 + 2 * h * x3 + x3 ** 2)) + 10 * h * x2 * (
                        x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (-7 / 2) * (
                        f1 * (h + x3) * (h ** 2 + (-2) * x1 ** 2 + x2 ** 2 + 2 * h * x3 + x3 ** 2) + (
                    -1) * x1 * (3 * f2 * x2 * (h + x3) + f3 * (
                        (-2) * h ** 2 + x1 ** 2 + x2 ** 2 + (-4) * h * x3 + (-2) * x3 ** 2))))
        j02 = (1 / 8) * np.pi ** (-1) * (((-2) * f1 * h + f3 * x1 + 2 * f1 * x3) * (
                h ** 2 + x1 ** 2 + x2 ** 2 + (-2) * h * x3 + x3 ** 2) ** (-3 / 2) + 2 * h ** 2 * (
                                                 3 * f3 * x1 + 2 * f1 * (h + x3)) * (
                                                 x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (-5 / 2) + (
                                                 (-1) * f3 * x1 + (-2) * f1 * (h + x3)) * (
                                                 x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (-3 / 2) + 3 * (
                                                 h + (-1) * x3) * (
                                                 h ** 2 + x1 ** 2 + x2 ** 2 + (-2) * h * x3 + x3 ** 2) ** (
                                                 -5 / 2) * (f2 * x1 * x2 + f3 * x1 * ((-1) * h + x3) + f1 * (
                h ** 2 + 2 * x1 ** 2 + x2 ** 2 + (-2) * h * x3 + x3 ** 2)) + (-10) * h ** 2 * (h + x3) * (
                                                 x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (-7 / 2) * (
                                                 (-3) * f2 * x1 * x2 + 3 * f3 * x1 * (h + x3) + f1 * (h ** 2 + (
                                             -2) * x1 ** 2 + x2 ** 2 + 2 * h * x3 + x3 ** 2)) + 3 * (h + x3) * (
                                                 x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (-5 / 2) * (
                                                 f2 * x1 * x2 + f3 * x1 * (h + x3) + f1 * (
                                                 h ** 2 + 2 * x1 ** 2 + x2 ** 2 + 2 * h * x3 + x3 ** 2)) + (
                                             -2) * h * (x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (-5 / 2) * (
                                                 (-3) * f2 * x1 * x2 + 4 * f3 * x1 * (h + x3) + f1 * (3 * h ** 2 + (
                                             -2) * x1 ** 2 + x2 ** 2 + 6 * h * x3 + 3 * x3 ** 2)) + 10 * h * (
                                                 h + x3) * (x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (-7 / 2) * (
                                                 f1 * (h + x3) * (
                                                 h ** 2 + (-2) * x1 ** 2 + x2 ** 2 + 2 * h * x3 + x3 ** 2) + (
                                                     -1) * x1 * (3 * f2 * x2 * (h + x3) + f3 * (
                                                 (-2) * h ** 2 + x1 ** 2 + x2 ** 2 + (-4) * h * x3 + (
                                             -2) * x3 ** 2))))
        j10 = (1 / 8) * np.pi ** (-1) * (
                (2 * f2 * x1 + f1 * x2) * (h ** 2 + x1 ** 2 + x2 ** 2 + (-2) * h * x3 + x3 ** 2) ** (
                -3 / 2) + 2 * h ** 2 * (2 * f2 * x1 + (-3) * f1 * x2) * (x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (
                        -5 / 2) + 2 * h * (
                        2 * f3 * x1 * x2 + (-2) * f2 * x1 * (h + x3) + 3 * f1 * x2 * (h + x3)) * (
                        x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (-5 / 2) + (-1) * (2 * f2 * x1 + f1 * x2) * (
                        x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (-3 / 2) + (-3) * x1 * (
                        h ** 2 + x1 ** 2 + x2 ** 2 + (-2) * h * x3 + x3 ** 2) ** (-5 / 2) * (
                        f1 * x1 * x2 + f3 * x2 * ((-1) * h + x3) + f2 * (
                        h ** 2 + x1 ** 2 + 2 * x2 ** 2 + (-2) * h * x3 + x3 ** 2)) + (-10) * h ** 2 * x1 * (
                        x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (-7 / 2) * (
                        (-3) * f1 * x1 * x2 + 3 * f3 * x2 * (h + x3) + f2 * (
                        h ** 2 + x1 ** 2 + (-2) * x2 ** 2 + 2 * h * x3 + x3 ** 2)) + 3 * x1 * (
                        x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (-5 / 2) * (
                        f1 * x1 * x2 + f3 * x2 * (h + x3) + f2 * (
                        h ** 2 + x1 ** 2 + 2 * x2 ** 2 + 2 * h * x3 + x3 ** 2)) + 10 * h * x1 * (
                        x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (-7 / 2) * (
                        f2 * (h + x3) * (h ** 2 + x1 ** 2 + (-2) * x2 ** 2 + 2 * h * x3 + x3 ** 2) + (
                    -1) * x2 * (3 * f1 * x1 * (h + x3) + f3 * (
                        (-2) * h ** 2 + x1 ** 2 + x2 ** 2 + (-4) * h * x3 + (-2) * x3 ** 2))))
        j11 = (1 / 8) * np.pi ** (-1) * ((h ** 2 + x1 ** 2 + x2 ** 2 + (-2) * h * x3 + x3 ** 2) ** (-3 / 2) * (
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
                h ** 2 + x1 ** 2 + 2 * x2 ** 2 + (-2) * h * x3 + x3 ** 2)) + (-10) * h ** 2 * x2 * (
                                                 x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (-7 / 2) * (
                                                 (-3) * f1 * x1 * x2 + 3 * f3 * x2 * (h + x3) + f2 * (
                                                 h ** 2 + x1 ** 2 + (
                                             -2) * x2 ** 2 + 2 * h * x3 + x3 ** 2)) + 3 * x2 * (
                                                 x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (-5 / 2) * (
                                                 f1 * x1 * x2 + f3 * x2 * (h + x3) + f2 * (
                                                 h ** 2 + x1 ** 2 + 2 * x2 ** 2 + 2 * h * x3 + x3 ** 2)) + 10 * h * x2 * (
                                                 x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (-7 / 2) * (f2 * (h + x3) * (
                h ** 2 + x1 ** 2 + (-2) * x2 ** 2 + 2 * h * x3 + x3 ** 2) + (-1) * x2 * (3 * f1 * x1 * (
                h + x3) + f3 * ((-2) * h ** 2 + x1 ** 2 + x2 ** 2 + (-4) * h * x3 + (-2) * x3 ** 2))))
        j12 = (1 / 8) * np.pi ** (-1) * (((-2) * f2 * h + f3 * x2 + 2 * f2 * x3) * (
                h ** 2 + x1 ** 2 + x2 ** 2 + (-2) * h * x3 + x3 ** 2) ** (-3 / 2) + 2 * h ** 2 * (
                                                 3 * f3 * x2 + 2 * f2 * (h + x3)) * (
                                                 x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (-5 / 2) + (
                                                 (-1) * f3 * x2 + (-2) * f2 * (h + x3)) * (
                                                 x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (-3 / 2) + 3 * (
                                                 h + (-1) * x3) * (
                                                 h ** 2 + x1 ** 2 + x2 ** 2 + (-2) * h * x3 + x3 ** 2) ** (
                                                 -5 / 2) * (f1 * x1 * x2 + f3 * x2 * ((-1) * h + x3) + f2 * (
                h ** 2 + x1 ** 2 + 2 * x2 ** 2 + (-2) * h * x3 + x3 ** 2)) + (-10) * h ** 2 * (h + x3) * (
                                                 x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (-7 / 2) * (
                                                 (-3) * f1 * x1 * x2 + 3 * f3 * x2 * (h + x3) + f2 * (
                                                 h ** 2 + x1 ** 2 + (
                                             -2) * x2 ** 2 + 2 * h * x3 + x3 ** 2)) + 3 * (h + x3) * (
                                                 x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (-5 / 2) * (
                                                 f1 * x1 * x2 + f3 * x2 * (h + x3) + f2 * (
                                                 h ** 2 + x1 ** 2 + 2 * x2 ** 2 + 2 * h * x3 + x3 ** 2)) + (
                                             -2) * h * (x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (-5 / 2) * (
                                                 (-3) * f1 * x1 * x2 + 4 * f3 * x2 * (h + x3) + f2 * (
                                                 3 * h ** 2 + x1 ** 2 + (
                                             -2) * x2 ** 2 + 6 * h * x3 + 3 * x3 ** 2)) + 10 * h * (h + x3) * (
                                                 x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (-7 / 2) * (f2 * (h + x3) * (
                h ** 2 + x1 ** 2 + (-2) * x2 ** 2 + 2 * h * x3 + x3 ** 2) + (-1) * x2 * (3 * f1 * x1 * (
                h + x3) + f3 * ((-2) * h ** 2 + x1 ** 2 + x2 ** 2 + (-4) * h * x3 + (-2) * x3 ** 2))))
        j20 = (-1 / 8) * np.pi ** (-1) * (((-2) * f3 * x1 + f1 * (h + (-1) * x3)) * (
                h ** 2 + x1 ** 2 + x2 ** 2 + (-2) * h * x3 + x3 ** 2) ** (-3 / 2) + 2 * h ** 2 * (
                                                  2 * f3 * x1 + 3 * f1 * (h + x3)) * (
                                                  x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (-5 / 2) + (
                                                  2 * f3 * x1 + f1 * (h + x3)) * (
                                                  x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (-3 / 2) + (
                                              -10) * h ** 2 * x1 * (x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (-7 / 2) * (
                                                  3 * (f1 * x1 + f2 * x2) * (h + x3) + f3 * (
                                                  (-2) * h ** 2 + x1 ** 2 + x2 ** 2 + (-4) * h * x3 + (
                                              -2) * x3 ** 2)) + 3 * x1 * (
                                                  h ** 2 + x1 ** 2 + x2 ** 2 + (-2) * h * x3 + x3 ** 2) ** (
                                                  -5 / 2) * ((-1) * (f1 * x1 + f2 * x2) * (h + (-1) * x3) + f3 * (
                2 * h ** 2 + x1 ** 2 + x2 ** 2 + (-4) * h * x3 + 2 * x3 ** 2)) + (-3) * x1 * (
                                                  x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (-5 / 2) * (
                                                  (f1 * x1 + f2 * x2) * (h + x3) + f3 * (
                                                  2 * h ** 2 + x1 ** 2 + x2 ** 2 + 4 * h * x3 + 2 * x3 ** 2)) + (
                                              -10) * h * x1 * (x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (-7 / 2) * (
                                                  f3 * (h + x3) * (2 * h ** 2 + (-1) * x1 ** 2 + (
                                              -1) * x2 ** 2 + 4 * h * x3 + 2 * x3 ** 2) + (-1) * (
                                                          f1 * x1 + f2 * x2) * (
                                                          4 * h ** 2 + x1 ** 2 + x2 ** 2 + 8 * h * x3 + 4 * x3 ** 2)) + (
                                              -2) * h * (x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (-5 / 2) * (f1 * (
                4 * h ** 2 + 3 * x1 ** 2 + x2 ** 2 + 8 * h * x3 + 4 * x3 ** 2) + 2 * x1 * (f2 * x2 + f3 * (
                h + x3))))
        j21 = (-1 / 8) * np.pi ** (-1) * (((-2) * f3 * x2 + f2 * (h + (-1) * x3)) * (
                h ** 2 + x1 ** 2 + x2 ** 2 + (-2) * h * x3 + x3 ** 2) ** (-3 / 2) + 2 * h ** 2 * (
                                                  2 * f3 * x2 + 3 * f2 * (h + x3)) * (
                                                  x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (-5 / 2) + (
                                                  2 * f3 * x2 + f2 * (h + x3)) * (
                                                  x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (-3 / 2) + (
                                              -10) * h ** 2 * x2 * (x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (-7 / 2) * (
                                                  3 * (f1 * x1 + f2 * x2) * (h + x3) + f3 * (
                                                  (-2) * h ** 2 + x1 ** 2 + x2 ** 2 + (-4) * h * x3 + (
                                              -2) * x3 ** 2)) + 3 * x2 * (
                                                  h ** 2 + x1 ** 2 + x2 ** 2 + (-2) * h * x3 + x3 ** 2) ** (
                                                  -5 / 2) * ((-1) * (f1 * x1 + f2 * x2) * (h + (-1) * x3) + f3 * (
                2 * h ** 2 + x1 ** 2 + x2 ** 2 + (-4) * h * x3 + 2 * x3 ** 2)) + (-3) * x2 * (
                                                  x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (-5 / 2) * (
                                                  (f1 * x1 + f2 * x2) * (h + x3) + f3 * (
                                                  2 * h ** 2 + x1 ** 2 + x2 ** 2 + 4 * h * x3 + 2 * x3 ** 2)) + (
                                              -10) * h * x2 * (x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (-7 / 2) * (
                                                  f3 * (h + x3) * (2 * h ** 2 + (-1) * x1 ** 2 + (
                                              -1) * x2 ** 2 + 4 * h * x3 + 2 * x3 ** 2) + (-1) * (
                                                          f1 * x1 + f2 * x2) * (
                                                          4 * h ** 2 + x1 ** 2 + x2 ** 2 + 8 * h * x3 + 4 * x3 ** 2)) + (
                                              -2) * h * (x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (-5 / 2) * (f2 * (
                4 * h ** 2 + x1 ** 2 + 3 * x2 ** 2 + 8 * h * x3 + 4 * x3 ** 2) + 2 * x2 * (f1 * x1 + f3 * (
                h + x3))))
        j22 = (-1 / 8) * np.pi ** (-1) * (((-1) * f1 * x1 + (-1) * f2 * x2 + 4 * f3 * (h + (-1) * x3)) * (
                h ** 2 + x1 ** 2 + x2 ** 2 + (-2) * h * x3 + x3 ** 2) ** (-3 / 2) + (-2) * h ** 2 * (
                                                  (-3) * (f1 * x1 + f2 * x2) + 4 * f3 * (h + x3)) * (
                                                  x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (-5 / 2) + (
                                                  f1 * x1 + f2 * x2 + 4 * f3 * (h + x3)) * (
                                                  x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (-3 / 2) + (
                                              -10) * h ** 2 * (h + x3) * (x1 ** 2 + x2 ** 2 + (h + x3) ** 2) ** (
                                                  -7 / 2) * (3 * (f1 * x1 + f2 * x2) * (h + x3) + f3 * (
                (-2) * h ** 2 + x1 ** 2 + x2 ** 2 + (-4) * h * x3 + (-2) * x3 ** 2)) + (-3) * (h + (-1) * x3) * (
                                                  h ** 2 + x1 ** 2 + x2 ** 2 + (-2) * h * x3 + x3 ** 2) ** (
                                                  -5 / 2) * ((-1) * (f1 * x1 + f2 * x2) * (h + (-1) * x3) + f3 * (
                2 * h ** 2 + x1 ** 2 + x2 ** 2 + (-4) * h * x3 + 2 * x3 ** 2)) + (-3) * (h + x3) * (
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

    def flow_strain(self, location):
        J = self.J_matrix(location)
        S_ij = 1 / 2 * (J + J.T)
        return S_ij

    def flow_rotation(self, location):
        J = self.J_matrix(location)
        Omega_ij = 1 / 2 * (J - J.T)
        return Omega_ij

    def flow_strain_rotation(self, location):
        J = self.J_matrix(location)
        S_ij = 1 / 2 * (J + J.T)
        Omega_ij = 1 / 2 * (J - J.T)
        return S_ij, Omega_ij

    def flow_velocity(self, location):
        u0_fun = lambda x0, x1, x2, f0, f1, f2, h: (
                0.75 * f0 * h * x0 ** 2 * x2 * (x0 ** 2 + x1 ** 2 + (h + x2) ** 2) ** (
            -2.5) / np.pi - 1 / 4 * f0 * h * x2 * (x0 ** 2 + x1 ** 2 + (h + x2) ** 2) ** (-1.5) / np.pi + (
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
                0.75 * f0 * h * x0 * x1 * x2 * (x0 ** 2 + x1 ** 2 + (h + x2) ** 2) ** (-2.5) / np.pi + (
                1 / 8) * f0 * x0 * x1 * (x0 ** 2 + x1 ** 2 + (h - x2) ** 2) ** (
                    -1.5) / np.pi - 1 / 8 * f0 * x0 * x1 * (x0 ** 2 + x1 ** 2 + (h + x2) ** 2) ** (
                    -1.5) / np.pi + 0.75 * f1 * h * x1 ** 2 * x2 * (x0 ** 2 + x1 ** 2 + (h + x2) ** 2) ** (
                    -2.5) / np.pi - 1 / 4 * f1 * h * x2 * (x0 ** 2 + x1 ** 2 + (h + x2) ** 2) ** (-1.5) / np.pi + (
                        1 / 8) * f1 * x1 ** 2 * (x0 ** 2 + x1 ** 2 + (h - x2) ** 2) ** (
                    -1.5) / np.pi - 1 / 8 * f1 * x1 ** 2 * (x0 ** 2 + x1 ** 2 + (h + x2) ** 2) ** (-1.5) / np.pi + (
                        1 / 8) * f1 * (x0 ** 2 + x1 ** 2 + (h - x2) ** 2) ** (-0.5) / np.pi - 1 / 8 * f1 * (
                        x0 ** 2 + x1 ** 2 + (h + x2) ** 2) ** (-0.5) / np.pi + (3 / 4) * f2 * h ** 3 * x1 * (
                        x0 ** 2 + x1 ** 2 + (h + x2) ** 2) ** (-2.5) / np.pi + (
                        3 / 4) * f2 * h ** 2 * x1 * x2 * (x0 ** 2 + x1 ** 2 + (h + x2) ** 2) ** (
                    -2.5) / np.pi - 0.75 * f2 * h * x1 * (h + x2) ** 2 * (x0 ** 2 + x1 ** 2 + (h + x2) ** 2) ** (
                    -2.5) / np.pi - 1 / 8 * f2 * h * x1 * (x0 ** 2 + x1 ** 2 + (h - x2) ** 2) ** (-1.5) / np.pi + (
                        1 / 8) * f2 * h * x1 * (x0 ** 2 + x1 ** 2 + (h + x2) ** 2) ** (-1.5) / np.pi + (
                        1 / 8) * f2 * x1 * x2 * (x0 ** 2 + x1 ** 2 + (h - x2) ** 2) ** (
                    -1.5) / np.pi - 1 / 8 * f2 * x1 * x2 * (x0 ** 2 + x1 ** 2 + (h + x2) ** 2) ** (-1.5) / np.pi)

        u2_fun = lambda x0, x1, x2, f0, f1, f2, h: (-3 / 4 * f0 * h ** 3 * x0 * (x0 ** 2 + x1 ** 2 + (h + x2) ** 2) ** (
            -2.5) / np.pi - 3 / 4 * f0 * h ** 2 * x0 * x2 * (x0 ** 2 + x1 ** 2 + (h + x2) ** 2) ** (
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

    def J_matrix(self, location):
        b1, b2, b3 = self._B
        x1, x2, x3 = location
        f1, f2, f3 = self.DoubleletStrength
        j00 = (1 / 8) * np.pi ** (-1) * (x1 ** 2 + x2 ** 2 + x3 ** 2) ** (-7 / 2) * (
                3 * b3 * (f1 * x1 + f2 * x2) * x3 * (4 * x1 ** 2 + (-1) * x2 ** 2 + (-1) * x3 ** 2) + (
            -1) * b3 * f3 * (2 * x1 ** 4 + (-1) * x2 ** 4 + x2 ** 2 * x3 ** 2 + 2 * x3 ** 4 + x1 ** 2 * (
                x2 ** 2 + (-11) * x3 ** 2)) + (-1) * b2 * (
                        3 * x2 * (f1 * x1 + f3 * x3) * ((-4) * x1 ** 2 + x2 ** 2 + x3 ** 2) + f2 * (
                        2 * x1 ** 4 + 2 * x2 ** 4 + x2 ** 2 * x3 ** 2 + (-1) * x3 ** 4 + x1 ** 2 * (
                        (-11) * x2 ** 2 + x3 ** 2))) + b1 * (
                        3 * x1 * (f2 * x2 + f3 * x3) * (2 * x1 ** 2 + (-3) * (x2 ** 2 + x3 ** 2)) + f1 * (
                        4 * x1 ** 4 + (-10) * x1 ** 2 * (x2 ** 2 + x3 ** 2) + (x2 ** 2 + x3 ** 2) ** 2)))

        j01 = (1 / 8) * np.pi ** (-1) * (x1 ** 2 + x2 ** 2 + x3 ** 2) ** (-7 / 2) * ((-3) * b3 * (
                f3 * x1 * x2 * (x1 ** 2 + x2 ** 2 + (-4) * x3 ** 2) + f2 * x1 * x3 * (
                x1 ** 2 + (-4) * x2 ** 2 + x3 ** 2) + (-1) * f1 * x2 * x3 * (
                        6 * x1 ** 2 + x2 ** 2 + x3 ** 2)) + (-1) * b1 * (3 * x2 * (f1 * x1 + f3 * x3) * (
                (-4) * x1 ** 2 + x2 ** 2 + x3 ** 2) + f2 * (2 * x1 ** 4 + 2 * x2 ** 4 + x2 ** 2 * x3 ** 2 + (
            -1) * x3 ** 4 + x1 ** 2 * ((-11) * x2 ** 2 + x3 ** 2))) + (-1) * b2 * (3 * x1 * (
                f3 * x3 * (x1 ** 2 + (-4) * x2 ** 2 + x3 ** 2) + f2 * x2 * (
                3 * x1 ** 2 + (-2) * x2 ** 2 + 3 * x3 ** 2)) + f1 * (4 * x1 ** 4 + (-2) * x2 ** 4 + (
            -1) * x2 ** 2 * x3 ** 2 + x3 ** 4 + x1 ** 2 * ((-13) * x2 ** 2 + 5 * x3 ** 2))))

        j02 = (1 / 8) * np.pi ** (-1) * (x1 ** 2 + x2 ** 2 + x3 ** 2) ** (-7 / 2) * (
                3 * b1 * (f1 * x1 + f2 * x2) * x3 * (4 * x1 ** 2 + (-1) * x2 ** 2 + (-1) * x3 ** 2) + (
            -1) * b1 * f3 * (2 * x1 ** 4 + (-1) * x2 ** 4 + x2 ** 2 * x3 ** 2 + 2 * x3 ** 4 + x1 ** 2 * (
                x2 ** 2 + (-11) * x3 ** 2)) + (-3) * b2 * (
                        f3 * x1 * x2 * (x1 ** 2 + x2 ** 2 + (-4) * x3 ** 2) + f2 * x1 * x3 * (
                        x1 ** 2 + (-4) * x2 ** 2 + x3 ** 2) + (-1) * f1 * x2 * x3 * (
                                6 * x1 ** 2 + x2 ** 2 + x3 ** 2)) + (-1) * b3 * (f1 * (
                4 * x1 ** 4 + x2 ** 4 + (-1) * x2 ** 2 * x3 ** 2 + (-2) * x3 ** 4 + x1 ** 2 * (
                5 * x2 ** 2 + (-13) * x3 ** 2)) + 3 * x1 * (f2 * x2 * (
                x1 ** 2 + x2 ** 2 + (-4) * x3 ** 2) + f3 * x3 * (3 * x1 ** 2 + 3 * x2 ** 2 + (-2) * x3 ** 2))))

        j10 = (1 / 8) * np.pi ** (-1) * (x1 ** 2 + x2 ** 2 + x3 ** 2) ** (-7 / 2) * ((-3) * b3 * (
                f3 * x1 * x2 * (x1 ** 2 + x2 ** 2 + (-4) * x3 ** 2) + f1 * x2 * x3 * (
                (-4) * x1 ** 2 + x2 ** 2 + x3 ** 2) + (-1) * f2 * x1 * x3 * (
                        x1 ** 2 + 6 * x2 ** 2 + x3 ** 2)) + (-1) * b2 * (3 * x1 * (f2 * x2 + f3 * x3) * (
                x1 ** 2 + (-4) * x2 ** 2 + x3 ** 2) + f1 * (2 * x1 ** 4 + 2 * x2 ** 4 + x2 ** 2 * x3 ** 2 + (
            -1) * x3 ** 4 + x1 ** 2 * ((-11) * x2 ** 2 + x3 ** 2))) + b1 * ((-3) * x2 * (
                (-2) * f1 * x1 ** 3 + 3 * f1 * x1 * (x2 ** 2 + x3 ** 2) + f3 * x3 * (
                (-4) * x1 ** 2 + x2 ** 2 + x3 ** 2)) + f2 * (2 * x1 ** 4 + (-4) * x2 ** 4 + (
            -5) * x2 ** 2 * x3 ** 2 + (-1) * x3 ** 4 + x1 ** 2 * (13 * x2 ** 2 + x3 ** 2))))

        j11 = (1 / 8) * np.pi ** (-1) * (x1 ** 2 + x2 ** 2 + x3 ** 2) ** (-7 / 2) * (
                3 * b2 * x2 * (f1 * x1 + f3 * x3) * ((-3) * x1 ** 2 + 2 * x2 ** 2 + (-3) * x3 ** 2) + (-3) * b3 * (
                f1 * x1 + f2 * x2) * x3 * (x1 ** 2 + (-4) * x2 ** 2 + x3 ** 2) + b2 * f2 * (
                        x1 ** 4 + 4 * x2 ** 4 + (-10) * x2 ** 2 * x3 ** 2 + x3 ** 4 + 2 * x1 ** 2 * (
                        (-5) * x2 ** 2 + x3 ** 2)) + b3 * f3 * (
                        x1 ** 4 + (-2) * x2 ** 4 + 11 * x2 ** 2 * x3 ** 2 + (-2) * x3 ** 4 + (-1) * x1 ** 2 * (
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
            -13) * x2 ** 2 * x3 ** 2 + (-2) * x3 ** 4 + x1 ** 2 * (5 * x2 ** 2 + (-1) * x3 ** 2))) + b2 * ((-3) * (
                f1 * x1 + f2 * x2) * x3 * (x1 ** 2 + (-4) * x2 ** 2 + x3 ** 2) + f3 * (x1 ** 4 + (
            -2) * x2 ** 4 + 11 * x2 ** 2 * x3 ** 2 + (-2) * x3 ** 4 + (-1) * x1 ** 2 * (x2 ** 2 + x3 ** 2))))

        j20 = (1 / 8) * np.pi ** (-1) * (x1 ** 2 + x2 ** 2 + x3 ** 2) ** (-7 / 2) * (
                3 * b2 * f3 * x1 * x2 * (x1 ** 2 + x2 ** 2 + 6 * x3 ** 2) + (-3) * b2 * x3 * (
                f2 * x1 * (x1 ** 2 + (-4) * x2 ** 2 + x3 ** 2) + f1 * x2 * (
                (-4) * x1 ** 2 + x2 ** 2 + x3 ** 2)) + (-3) * b1 * x3 * (
                        (-2) * f1 * x1 ** 3 + 3 * f1 * x1 * (x2 ** 2 + x3 ** 2) + f2 * x2 * (
                        (-4) * x1 ** 2 + x2 ** 2 + x3 ** 2)) + b1 * f3 * (
                        2 * x1 ** 4 + (-1) * x2 ** 4 + (-5) * x2 ** 2 * x3 ** 2 + (-4) * x3 ** 4 + x1 ** 2 * (
                        x2 ** 2 + 13 * x3 ** 2)) + (-1) * b3 * (
                        3 * x1 * (f2 * x2 + f3 * x3) * (x1 ** 2 + x2 ** 2 + (-4) * x3 ** 2) + f1 * (
                        2 * x1 ** 4 + (-1) * x2 ** 4 + x2 ** 2 * x3 ** 2 + 2 * x3 ** 4 + x1 ** 2 * (
                        x2 ** 2 + (-11) * x3 ** 2))))

        j21 = (1 / 8) * np.pi ** (-1) * (x1 ** 2 + x2 ** 2 + x3 ** 2) ** (-7 / 2) * (
                3 * b1 * f3 * x1 * x2 * (x1 ** 2 + x2 ** 2 + 6 * x3 ** 2) + (-1) * b2 * f3 * (
                x1 ** 4 + (-2) * x2 ** 4 + (-13) * x2 ** 2 * x3 ** 2 + 4 * x3 ** 4 + (-1) * x1 ** 2 * (
                x2 ** 2 + (-5) * x3 ** 2)) + (-3) * b1 * x3 * (
                        f2 * x1 * (x1 ** 2 + (-4) * x2 ** 2 + x3 ** 2) + f1 * x2 * (
                        (-4) * x1 ** 2 + x2 ** 2 + x3 ** 2)) + (-3) * b2 * x3 * (
                        f1 * x1 * (x1 ** 2 + (-4) * x2 ** 2 + x3 ** 2) + f2 * x2 * (
                        3 * x1 ** 2 + (-2) * x2 ** 2 + 3 * x3 ** 2)) + b3 * (
                        (-3) * x2 * (f1 * x1 + f3 * x3) * (x1 ** 2 + x2 ** 2 + (-4) * x3 ** 2) + f2 * (
                        x1 ** 4 + (-2) * x2 ** 4 + 11 * x2 ** 2 * x3 ** 2 + (-2) * x3 ** 4 + (
                    -1) * x1 ** 2 * (x2 ** 2 + x3 ** 2))))

        j22 = (1 / 8) * np.pi ** (-1) * (x1 ** 2 + x2 ** 2 + x3 ** 2) ** (-7 / 2) * (
                (-3) * b2 * x2 * (f1 * x1 + f3 * x3) * (x1 ** 2 + x2 ** 2 + (-4) * x3 ** 2) + (-3) * b3 * (
                f1 * x1 + f2 * x2) * x3 * (3 * x1 ** 2 + 3 * x2 ** 2 + (-2) * x3 ** 2) + b3 * f3 * (
                        x1 ** 4 + x2 ** 4 + (-10) * x2 ** 2 * x3 ** 2 + 4 * x3 ** 4 + 2 * x1 ** 2 * (
                        x2 ** 2 + (-5) * x3 ** 2)) + b2 * f2 * (
                        x1 ** 4 + (-2) * x2 ** 4 + 11 * x2 ** 2 * x3 ** 2 + (-2) * x3 ** 4 + (-1) * x1 ** 2 * (
                        x2 ** 2 + x3 ** 2)) + (-1) * b1 * (
                        3 * x1 * (f2 * x2 + f3 * x3) * (x1 ** 2 + x2 ** 2 + (-4) * x3 ** 2) + f1 * (
                        2 * x1 ** 4 + (-1) * x2 ** 4 + x2 ** 2 * x3 ** 2 + 2 * x3 ** 4 + x1 ** 2 * (
                        x2 ** 2 + (-11) * x3 ** 2))))
        J = np.array(((j00, j01, j02),
                      (j10, j11, j12),
                      (j20, j21, j22),))
        return J

    def flow_strain(self, location):
        J = self.J_matrix(location)
        S_ij = 1 / 2 * (J + J.T)
        return S_ij

    def flow_rotation(self, location):
        J = self.J_matrix(location)
        Omega_ij = 1 / 2 * (J - J.T)
        return Omega_ij

    def flow_strain_rotation(self, location):
        J = self.J_matrix(location)
        S_ij = 1 / 2 * (J + J.T)
        Omega_ij = 1 / 2 * (J - J.T)
        return S_ij, Omega_ij

    def flow_velocity(self, location):
        b1, b2, b3 = self._B
        x1, x2, x3 = location
        f1, f2, f3 = self.DoubleletStrength
        u0 = (1 / 8) * np.pi ** (-1) * (x1 ** 2 + x2 ** 2 + x3 ** 2) ** (-5 / 2) * (
                b3 * f3 * x1 * (x1 ** 2 + x2 ** 2 + (-2) * x3 ** 2) + (-1) * b1 * (f1 * x1 + f2 * x2 + f3 * x3) * (
                2 * x1 ** 2 + (-1) * x2 ** 2 + (-1) * x3 ** 2) + b2 * f2 * x1 * (
                        x1 ** 2 + (-2) * x2 ** 2 + x3 ** 2) + (-1) * b3 * x3 * (
                        3 * f2 * x1 * x2 + f1 * (4 * x1 ** 2 + x2 ** 2 + x3 ** 2)) + (-1) * b2 * x2 * (
                        3 * f3 * x1 * x3 + f1 * (4 * x1 ** 2 + x2 ** 2 + x3 ** 2)))

        u1 = (1 / 8) * np.pi ** (-1) * (x1 ** 2 + x2 ** 2 + x3 ** 2) ** (-5 / 2) * (
                (-3) * b1 * f3 * x1 * x2 * x3 + b3 * f3 * x2 * (x1 ** 2 + x2 ** 2 + (-2) * x3 ** 2) + b2 * (
                f1 * x1 + f2 * x2 + f3 * x3) * (x1 ** 2 + (-2) * x2 ** 2 + x3 ** 2) + b1 * f1 * x2 * (
                        (-2) * x1 ** 2 + x2 ** 2 + x3 ** 2) + (-1) * b1 * f2 * x1 * (
                        x1 ** 2 + 4 * x2 ** 2 + x3 ** 2) + (-1) * b3 * x3 * (
                        3 * f1 * x1 * x2 + f2 * (x1 ** 2 + 4 * x2 ** 2 + x3 ** 2)))

        u2 = (1 / 8) * np.pi ** (-1) * (x1 ** 2 + x2 ** 2 + x3 ** 2) ** (-5 / 2) * (
                (-3) * b2 * f1 * x1 * x2 * x3 + (-3) * b1 * f2 * x1 * x2 * x3 + b3 * (
                f1 * x1 + f2 * x2 + f3 * x3) * (x1 ** 2 + x2 ** 2 + (-2) * x3 ** 2) + b2 * f2 * x3 * (
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
    def __init__(self, name='...', rot_speed=0, **kwargs):
        self._type = 'JefferyObj'
        self._center = kwargs['center']  # type: np.ndarray
        self._norm = ...  # type: np.ndarray
        self._lateral_norm = ...  # type: np.ndarray
        self._set_norm(kwargs['norm'])
        self._set_lateral_norm(kwargs['lateral_norm'])
        self._speed = kwargs['speed']
        self._rot_speed = rot_speed
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
        err_msg = 'lateral_norm=[x, y, z] has 3 components and ||lateral_norm|| > 0. '
        assert lateral_norm.size == 3 and np.linalg.norm(lateral_norm) > 0, err_msg
        self._lateral_norm = lateral_norm / np.linalg.norm(lateral_norm)
        return True

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
    def rot_speed(self):
        return self._rot_speed

    @rot_speed.setter
    def rot_speed(self, rot_speed):
        rot_speed = np.array(rot_speed).flatten()
        err_msg = 'rot_speed is a scalar. '
        assert rot_speed.size == 1, err_msg
        self._rot_speed = rot_speed

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
                        update_fun=Adams_Moulton_Methods, update_order=1):
        # for a cutoff infinity symmetric problem,
        #   each time step set the obj in the center of the cutoff region to improve the accuracy.
        self._locomotion_fct = np.array((not fix_x, not fix_y, not fix_z), dtype=np.float)
        self._update_fun = update_fun
        self._update_order = update_order
        return self._locomotion_fct

    def dbg_set_update_para(self, fix_x=1, fix_y=1, fix_z=1, update_fun=Adams_Moulton_Methods, update_order=1):
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

    def get_velocity_at(self, X, P, P2, trs_v, rot_v):
        dX = self.get_dX_at(X, P, trs_v)
        dP, dP2, omega = self.get_dP_at(X, P, P2, rot_v)
        return dX, dP, dP2, omega

    def update_location(self, eval_dt, print_handle=''):
        P = self.norm
        P2 = self.lateral_norm
        X = self.center
        trs_v = self.speed
        rot_v = self.rot_speed

        dX, dP, dP2, omega = self.get_velocity_at(X, P, P2, trs_v, rot_v)
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


class TableObj(JefferyObj):
    def __init__(self, table_name, name='...', rot_speed=0, **kwargs):
        self._lateral_norm0 = ...  # type: np.ndarray # ini direction of lateral norm
        super().__init__(name=name, rot_speed=rot_speed, **kwargs)
        self._type = 'TableObj'
        self._intp_fun_list = []
        self._intp_psi_list = []
        self.load_table(table_name=table_name)

    def _set_lateral_norm(self, lateral_norm: np.ndarray):
        err_msg = 'lateral_norm=[x, y, z] has 3 components and ||lateral_norm|| > 0. '
        assert lateral_norm.size == 3 and np.linalg.norm(lateral_norm) > 0, err_msg
        self._lateral_norm = lateral_norm / np.linalg.norm(lateral_norm)
        self._lateral_norm0 = lateral_norm / np.linalg.norm(lateral_norm)
        return True

    def load_table(self, table_name):
        t_path = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.normpath(t_path + '/' + '%s.pickle' % table_name)
        with open(full_path, 'rb') as handle:
            table_data = pickle.load(handle)
        intp_fun_list = self._intp_fun_list
        intp_psi_list = self._intp_psi_list
        for tpsi, table_psi_data in table_data:
            tintp_fun_list = []
            intp_psi_list.append(tpsi)
            for tx, ty, tU in table_psi_data:
                tfun = interpolate.RectBivariateSpline(tx, ty, tU, kx=3, ky=3)
                # tfun = interpolate.interp2d(tx, ty, tU.T, kind='quintic', copy=False, )
                tintp_fun_list.append(tfun)
            intp_fun_list.append(tintp_fun_list)
        return True

    def intp_U_fun(self, t_theta, t_phi, t_psi):
        # if 0 <= t_theta < np.pi and 0 <= t_phi < np.pi:  # letf down
        #     sign_list = [1, 1, 1, 1, 1, 1]
        # elif 0 <= t_theta < np.pi and np.pi <= t_phi <= 2 * np.pi:  # right down
        #     t_theta = t_theta
        #     t_phi = t_phi - np.pi
        #     sign_list = [1, 1, -1, 1, 1, -1]
        # elif np.pi <= t_theta <= 2 * np.pi and np.pi <= t_phi <= 2 * np.pi:  # right up
        #     t_theta = t_theta - np.pi
        #     t_phi = 2 * np.pi - t_phi
        #     sign_list = [-1, 1, -1, -1, 1, -1]
        # elif np.pi <= t_theta <= 2 * np.pi and 0 <= t_phi < np.pi:  # left up
        #     t_theta = t_theta - np.pi
        #     t_phi = np.pi - t_phi
        #     sign_list = [-1, 1, 1, -1, 1, 1]
        # else:
        #     raise Exception('norm_theta %f and (or) norm_phi %f out of range (0, 2*pi)' % (t_theta, t_phi))
        if 0 <= t_theta <= np.pi and 0 <= t_phi < np.pi:  # letf down
            sign_list = [1, 1, 1, 1, 1, 1]
        elif 0 <= t_theta <= np.pi and np.pi <= t_phi <= 2 * np.pi:  # right down
            t_theta = t_theta
            t_phi = t_phi - np.pi
            sign_list = [1, 1, -1, 1, 1, -1]
        else:
            raise Exception('norm_theta %f and (or) norm_phi %f out of range (0, pi) * (0, 2pi)' % (t_theta, t_phi))

        intp_U = []
        for tfun in self._intp_fun_list:
            t_U = []
            for intp_fun, sign in zip(tfun, sign_list):
                t_U.append(intp_fun(t_theta, t_phi) * sign)
            intp_U.append(np.hstack(t_U).flatten())
        intp_U.append(intp_U[0].copy())
        intp_U = np.vstack(intp_U)
        intp_psi = np.hstack([self._intp_psi_list, np.pi * 2])
        intp_fun1d = interpolate.interp1d(intp_psi, intp_U, kind='quadratic', copy=False, axis=0, bounds_error=True)
        return intp_fun1d(t_psi)

    def get_dP_at(self, X, P, P2, rot_v):
        raise Exception('This function do NOT work in %s Obj' % self._name)

    def get_dX_at(self, X, P, trs_v):
        raise Exception('This function do NOT work in %s Obj' % self._name)

    def get_velocity_at(self, X, P, P2, trs_v, rot_v=0):
        err_msg = 'current version rot_v==0'
        assert rot_v == 0, err_msg

        Ub = self.father.flow_velocity(X)  # background velocity
        P20 = self._lateral_norm0  # ini direction of lateral norm
        t_theta = np.arccos(P[2] / np.linalg.norm(P))
        t_phi = np.arctan2(P[1], P[0])
        t_phi = t_phi + 2 * np.pi if t_phi < 0 else t_phi
        rot_mtx = np.array(((np.cos(-t_phi), np.sin(-t_phi), 0), (np.sin(t_phi), np.cos(-t_phi), 0), (0, 0, 1)))
        P21 = np.dot(rot_mtx, P2)
        rot_mtx = np.array(((np.cos(-t_phi), 0, np.sin(-t_phi)), (0, 1, 0), (np.sin(t_phi), 0, np.cos(-t_phi))))
        P21 = np.dot(rot_mtx, P21)
        sign = np.sign(np.cross(P20, P21)[2])
        t_psi = sign * np.arccos(np.clip(np.dot(P20, P21) / np.linalg.norm(P20) / np.linalg.norm(P21), -1, 1))
        t_psi = 2 * np.pi * (sign < 0) + t_psi  # (-pi,pi) -> (0, 2pi)
        tU = self.intp_U_fun(t_theta, t_phi, t_psi)
        dX = tU[:3] + Ub + trs_v * P
        omega = tU[3:]
        dP = np.cross(omega, P)
        dP2 = np.cross(omega, P2)
        return dX, dP, dP2, omega

# class JefferyObj3D(JefferyObj):
#     def __init__(self, name='...', **kwargs):
#         super().__init__(name, **kwargs)
#         self._type = 'JefferyObj3D'
#         self._norm = ...  # type: np.ndarray
#         self._set_norm(kwargs['norm'])
#         # lbd   = (a^2-1)/(a^2+1), a = rs2 / rs1
#         # kappa = (b^2-1)/(b^2+1), b = rs3 / rs1
#         # rs1, rs2, rs3 are 3 half length of the ellipse.
#         self._lbd = kwargs['lbd']  # type: np.ndarray
#         self._kappa = kwargs['kappa']  # type: np.ndarray
#         # the following properties store the location history of the composite.
#         self._norm_hist = []  # each element is a (9,) array contain 3 norms.
#
#     def nothint(self):
#         pass

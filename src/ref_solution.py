# coding=utf-8
"""
    Contain reference solutions of different problems, i.e., analytical solution of sphere case.
     Zhang Ji, 20161204

"""

from src import stokes_flow as sf
import numpy as np
from src.geo import *
from numpy import sin, cos, sqrt

__all__ = ['slt', 'sphere_slt']

# Todo: warnning what need I know!!!
np.seterr(all='ignore')


class slt():
    def __init__(self, problem: 'sf.StokesFlowProblem', **kwargs):
        self._problem = problem
        self._kwargs = kwargs

    def get_solution(self, node_geo: 'base_geo'):
        pass

    def get_errNorm(self, node_geo: base_geo, num_u):
        u = self.get_solution(node_geo)
        err = sqrt(np.sum((u - num_u) ** 2) / np.sum(u ** 2))
        return err


class sphere_slt(slt):
    def __init__(self, problem, **kwargs):
        super().__init__(problem, **kwargs)
        problem_kwargs = problem.get_kwargs()
        self._radius = problem_kwargs['radius']
        self._u = problem_kwargs['u']

    def get_solution(self, node_geo: base_geo):
        err_msg = 'input a geo objects interesting nodes, not an array of nodes directy. '
        assert isinstance(node_geo, base_geo), err_msg

        a = self._radius
        u0 = -self._u
        x0 = node_geo.get_nodes_x() - node_geo.get_origin()[0]
        y0 = node_geo.get_nodes_y() - node_geo.get_origin()[1]
        z0 = node_geo.get_nodes_z() - node_geo.get_origin()[2]

        rx2 = y0 ** 2 + z0 ** 2
        rx = sqrt(rx2)
        r2 = rx2 + x0 ** 2
        r = sqrt(r2)
        sinTheta = rx / r
        cosTheta = x0 / r
        sinPhi = z0 / rx
        cosPhi = y0 / rx
        ux = 1 + (1 / 8) * (a + (-1) * r) * r ** (-3) * u0 * (a ** 2 + a * r + (-8) * r ** 2 + 3 * a * (a + r) * (1 - 2 * sinTheta ** 2))
        uy = (3 / 4) * r ** (-3) * (a ** 3 + (-1) * a * r ** 2) * u0 * cosPhi * cosTheta * sinTheta
        uz = (3 / 4) * r ** (-3) * (a ** 3 + (-1) * a * r ** 2) * u0 * cosTheta * sinPhi * sinTheta
        u = np.dstack((ux, uy, uz)).flatten()

        return u

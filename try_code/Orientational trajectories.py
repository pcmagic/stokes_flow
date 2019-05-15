import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

# my add
import quaternion
from src.support_class import *
import importlib
from codeStore import support_fun_table as spf_tb
from time import time


class Quaternion(object):
    """docstring for Quaternion"""

    def __init__(self, axis=None, angle=0):
        super(Quaternion, self).__init__()

        if axis is None:
            axis = np.array([0, 0, 1.0])
        axis = np.array(axis)

        xyz = np.sin(.5 * angle) * axis / np.linalg.norm(axis)
        self.q = np.array([
            np.cos(.5 * angle),
            xyz[0],
            xyz[1],
            xyz[2]
        ])

    def __add__(self, other):
        Q = Quaternion()
        Q.q = self.q + other
        return Q

    def mul(self, other):
        assert (type(other) is Quaternion)

        W = self.q[0]
        X = self.q[1]
        Y = self.q[2]
        Z = self.q[3]

        w = other.q[0]
        x = other.q[1]
        y = other.q[2]
        z = other.q[3]

        Q = Quaternion()
        Q.q = np.array([
            w * W - x * X - y * Y - z * Z,
            W * x + w * X + Y * z - y * Z,
            W * y + w * Y - X * z + x * Z,
            X * y - x * Y + W * z + w * Z
        ])
        return Q

    def set_wxyz(self, w, x, y, z):
        self.q = np.array([w, x, y, z])

    def __str__(self):
        return str(self.q)

    def normalize(self):
        self.q = self.q / np.linalg.norm(self.q)

    def get_E(self):
        W = self.q[0]
        X = self.q[1]
        Y = self.q[2]
        Z = self.q[3]

        return np.array([
            [-X, W, -Z, Y],
            [-Y, Z, W, -X],
            [-Z, -Y, X, W]
        ])

    def get_G(self):
        W = self.q[0]
        X = self.q[1]
        Y = self.q[2]
        Z = self.q[3]

        return np.array([
            [-X, W, Z, -Y],
            [-Y, -Z, W, X],
            [-Z, Y, -X, W]
        ])

    def get_R(self):
        return np.matmul(self.get_E(), self.get_G().T)


# # Functions for angular velocity & integration
# 
# The particle is an ellipsoid. The reference state (corresponding to no rotation) is that the ellipsoid is axis-aligned and the axis lengths are (a_x, a_y, a_z). The shape parameters in the code below are
# 
# ```
# l = a_z/a_x
# k = a_y/a_x
# ```
# 
# Its orientation is represented by the rotation (as a Quaternion) from the reference state. See Appendix A of https://arxiv.org/abs/1705.06997 for the quaternion convention.

# In[3]:


def jeffery_omega(L, K, n1, n2, n3, Omega, E):
    """
    Compute Jeffery angular velocity
    
    L: (lambda^2-1)/(lambda^2+1)
    K: (kappa^2-1)/(kappa^2+1)
    n1,n2,n3: vector triplet representing current orientation
    Omega: vorticity (lab frame) 
    E: strain matrix (lab frame)

    Returns (3,) ndarray with angular velocity of particle (body frame)
    See Appendix A in http://hdl.handle.net/2077/40830
    """

    omega1 = n1.dot(Omega) + (L - K) / (L * K - 1.) * (n2.dot(E.dot(n3)))
    omega2 = n2.dot(Omega) + L * (n1.dot(E.dot(n3)))
    omega3 = n3.dot(Omega) - K * (n1.dot(E.dot(n2)))

    return np.array([omega1, omega2, omega3])


def jeffery_numerical(L, K, q0, Omega, E, max_t=None, dt=1e-3, return_n1s=False):
    """
    Integrate one trajectory according to Jeffery's equations.
    
    L: (lambda^2-1)/(lambda^2+1) shape parameter 1
    K: (kappa^2-1)/(kappa^2+1) shape parameter 2  
    q0: quaternion representing initial orientation
    Omega: vorticity (lab frame) 
    E: strain matrix (lab frame)
    max_t: Max time of trajectory, defaults to 2 Jeffery periods based on L
    dt: Integration timestep
    
    See Appendix A in https://arxiv.org/abs/1705.06997 for quaternion convention.
    
    Returns (ts, qs, n2s, n3s) where
        ts is (N,1) ndarray with timestamps (starting at 0) for N steps
        qs is (N,4) ndarray with orientations (quaternions) for N steps
        n2s is (N,3) ndarray with n2 vector for N steps
        n3s is (N,3) ndarray with n3 vector for N steps        
    """

    if max_t is None:
        maxKL = max(abs(L), abs(K))
        jeffery_T = 4 * np.pi / np.sqrt(1 - maxKL * maxKL)
        max_t = 2 * jeffery_T

    N = int(max_t / dt)

    ts = np.zeros((N, 1))
    n1s = np.zeros((N, 3))
    n2s = np.zeros((N, 3))
    n3s = np.zeros((N, 3))
    qs = np.zeros((N, 4))
    q = q0
    t = 0
    for n in range(N):
        R = q.get_R()
        n1 = R[:, 0]
        n2 = R[:, 1]
        n3 = R[:, 2]

        ts[n] = n * dt
        n1s[n, :] = n1
        n2s[n, :] = n2
        n3s[n, :] = n3
        qs[n, :] = q.q

        omega = jeffery_omega(L, K, n1, n2, n3, Omega, E)
        print(omega, np.linalg.norm(omega))
        omg_norm = np.dot(omega, n2)*n2/np.dot(n2, n2)
        omg_tang = omega - omg_norm
        qdot = 0.5 * omega.dot(q.get_G())
        #         # dbg
        #         qdot1 = Quaternion()
        #         qdot1.set_wxyz(*qdot)
        #         print(omega)
        #         print(qdot1)
        #         print(qdot1.get_R())
        #         print()
        q = q + dt * qdot
        q.normalize()

    if return_n1s:
        return ts, qs, n1s, n2s, n3s
    else:
        return ts, qs, n2s, n3s


def jeffery_axisymmetric_exact(L, q0, Omega, E, max_t=None, dt=1e-1):
    """
    Generate one exact trajectory for axisymmetric particle ('Jeffery orbit')
    
    L: (lambda^2-1)/(lambda^2+1) shape parameter
    q0: quaternion representing initial orientation
    Omega: vorticity (lab frame) 
    E: strain matrix (lab frame)
    max_t: Max time of trajectory, defaults to 2 Jeffery periods based on L
    dt: Sample spacing
    
    See Appendix A in https://arxiv.org/abs/1705.06997 for quaternion convention.
    
    Returns (ts, qs, n2s, n3s) where
        ts is (N,1) ndarray with timestamps (starting at 0) for N steps
        n3s is (N,3) ndarray with n3 vector for N steps        
    """
    if max_t is None:
        jeffery_T = 4 * np.pi / np.sqrt(1 - L * L)
        max_t = 2 * jeffery_T

    N = int(max_t / dt)

    levi_civita = np.zeros((3, 3, 3))
    levi_civita[0, 1, 2] = levi_civita[1, 2, 0] = levi_civita[2, 0, 1] = 1
    levi_civita[0, 2, 1] = levi_civita[2, 1, 0] = levi_civita[1, 0, 2] = -1
    O = -np.einsum('ijk,k', levi_civita, Omega)

    B = O + L * E
    n30 = q0.get_R().dot(np.array([0, 0, 1]))

    ts = np.zeros((N, 1))
    n3s = np.zeros((N, 3))
    for n in range(N):

        t = dt * n

        M = scipy.linalg.expm(B * t)
        n3 = M.dot(n30)
        n3 = n3 / np.linalg.norm(n3)

        ts[n] = t
        n3s[n, :] = n3

    return (ts, n3s)


# Omega & E (strain) for simple shear flow


Omega = np.array([0, 0, -.5])
E = np.array([
    [0, .5, 0],
    [.5, 0, 0],
    [0, 0, 0]
])

# dbg my code, compare with this code. 
max_t = 0.01
eval_dt = 1e-2
# norm = np.array((0, 1, 0))
# norm = norm / np.linalg.norm(norm)
# # np.random.seed(0)
# # tlateral_norm = np.random.sample(3)
# tlateral_norm = np.array((2, 0, 1))
# tlateral_norm = tlateral_norm / np.linalg.norm(tlateral_norm)
# tlateral_norm = tlateral_norm - norm * np.dot(norm, tlateral_norm)
# tlateral_norm = tlateral_norm / np.linalg.norm(tlateral_norm)
# P0 = norm / np.linalg.norm(norm)
# P20 = tlateral_norm / np.linalg.norm(tlateral_norm)
# e = np.identity(3)
# e0 = np.vstack((np.cross(P0, P20), P0, P20)).T
# tR = rotMatrix_DCM(*e0, *e)
# q0 = Quaternion()
# q0.set_wxyz(*quaternion.as_float_array(quaternion.from_rotation_matrix(tR)))
angle = np.random.sample(1)[0]
# angle = 0
q0 = Quaternion(axis=[0,1,0], angle=angle)

l = 1
k = 3
L = (l ** 2 - 1) / (l ** 2 + 1)
K = (k ** 2 - 1) / (k ** 2 + 1)

(ts, qs, n1s, n2s, n3s) = jeffery_numerical(L, K, q0, Omega, E, dt=eval_dt, max_t=max_t, return_n1s=True)
t_theta_all = np.arccos(n2s[:, 1] / np.linalg.norm(n2s, axis=1))
t_phi_all = np.arctan2(-n2s[:, 2], n2s[:, 0])
t_phi_all = np.hstack([t1 + 2 * np.pi if t1 < 0 else t1 for t1 in t_phi_all])  # (-pi,pi) -> (0, 2pi)

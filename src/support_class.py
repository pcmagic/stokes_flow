import sys
from collections import UserList
import numpy as np

__all__ = ['uniqueList', 'typeList', 'intList', 'floatList',
           'abs_comp', 'abs_construct_matrix',
           'check_file_extension', 'mpiprint', 'fullprint',
           'coordinate_transformation',
           'tube_flatten',
           'get_rot_matrix', 'rot_vec2rot_mtx', 'vector_rotation_norm', 'vector_rotation',
           'rotMatrix_DCM',
           'Adams_Moulton_Methods', 'Adams_Bashforth_Methods',
           'Quaternion']


class uniqueList(UserList):
    def __init__(self, liste=[], acceptType=None):
        self._acceptType = acceptType
        super().__init__(liste)

    def check(self, other):
        err_msg = 'only type %s is accepted. ' % (self._acceptType)
        assert self._acceptType is None or isinstance(other, self._acceptType), err_msg

        err_msg = 'item ' + repr(other) + ' add muilt times. '
        assert self.count(other) == 0, err_msg

    def __add__(self, other):
        self.check(other)
        super().__add__(other)

    def append(self, item):
        self.check(item)
        super().append(item)


class typeList(UserList):
    def __init__(self, acceptType):
        self._acceptType = acceptType
        super().__init__()

    def check(self, other):
        err_msg = 'only type %s is accepted. ' % (self._acceptType)
        assert self._acceptType is None or isinstance(other, self._acceptType), err_msg

    def __add__(self, other):
        self.check(other)
        super().__add__(other)

    def append(self, item):
        self.check(item)
        super().append(item)


class intList(typeList):
    def __init__(self):
        super().__init__(int)


class floatList(typeList):
    def __init__(self):
        super().__init__(float)


class abs_comp:
    def __init__(self, **kwargs):
        need_args = ['name']
        opt_args = {'childType': abs_comp}
        self._kwargs = kwargs
        self.check_args(need_args, opt_args)

        self._father = None
        self._name = kwargs['name']
        self._child_list = uniqueList(acceptType=kwargs['childType'])
        self._create_finished = False
        self._index = -1

    def checkmyself(self):
        for child in self._child_list:
            self._create_finished = child.checkmyself and \
                                    self._create_finished
        return self._create_finished

    def myself_info(self):
        if self._create_finished:
            str = ('%s: %d, %s, create sucessed' % (
                self.__class__.__name__, self._index, self._name))
        else:
            str = ('%s: %d, %s, create not finished' % (
                self.__class__.__name__, self._index, self._name))
        return str

    def printmyself(self):
        PETSc.Sys.Print(self.myself_info())
        return True

    def savemyself(self, file_name):
        fid = open(file_name, 'w')
        fid.write(self.myself_info())
        return True

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, index):
        self._index = index

    @property
    def type(self):
        return self.__class__.__name__

    @property
    def father(self):
        return self._father

    @father.setter
    def father(self, father):
        self._father = father

    def save_context(self):
        pass

    def restore_context(self):
        pass

    def check_child_index(self, index):
        err_msg = 'wrong index %d, last index is %d. ' % (index, len(self._child_list))
        assert index <= len(self._child_list), err_msg

        err_msg = 'wrong index %d, first index is 0' % (index)
        assert index > 0, err_msg
        return True

    def get_child(self, index):
        self.check_child_index(index)
        return self._child_list[index]

    @property
    def child_list(self):
        return self._child_list

    def __repr__(self):
        return self.myself_info()

    def check_need_args(self, need_args=list()):
        kwargs = self._kwargs
        for key in need_args:
            err_msg = "information about '%s' is necessary for %s.%s. " \
                      % (key, self.__class__.__name__, sys._getframe(2).f_code.co_name)
            assert key in kwargs, err_msg
        return True

    def check_opt_args(self, opt_args=dict()):
        kwargs = self._kwargs
        for key, value in opt_args.items():
            if not key in kwargs:
                kwargs[key] = opt_args[key]
        self._kwargs = kwargs
        return kwargs

    def check_args(self, need_args=list(), opt_args=dict()):
        self.check_need_args(need_args)
        kwargs = self.check_opt_args(opt_args)
        self._kwargs = kwargs
        return kwargs


# abstract class for matrix construction.
class abs_construct_matrix(abs_comp):
    def __init__(self):
        super().__init__(childType=abs_comp)


def check_file_extension(filename, extension):
    if filename[-len(extension):] != extension:
        filename = filename + extension
    return filename


def tube_flatten(container):
    for i in container:
        if isinstance(i, (uniqueList, list, tuple)):
            for j in tube_flatten(i):
                yield j
        else:
            yield i


def rot_vec2rot_mtx(rot_vct):
    rot_vct = np.array(rot_vct).flatten()
    err_msg = 'rot_vct is a numpy array contain three components. '
    assert rot_vct.size == 3, err_msg

    def S(n):
        Sn = np.array([[0, -n[2], n[1]],
                       [n[2], 0, -n[0]],
                       [-n[1], n[0], 0]])
        return Sn

    theta = np.linalg.norm(rot_vct)
    if theta > 1e-6:
        n = rot_vct / theta
        Sn = S(n)
        R = np.eye(3) + np.sin(theta) * Sn + (1 - np.cos(theta)) * np.dot(Sn, Sn)
    else:
        Sr = S(rot_vct)
        theta2 = theta ** 2
        R = np.eye(3) + (1 - theta2 / 6.) * Sr + (.5 - theta2 / 24.) * np.dot(Sr, Sr)
    return R


def get_rot_matrix(norm=np.array([0, 0, 1]), theta=0):
    norm = np.array(norm).reshape((3,))
    theta = -1 * float(theta)
    if np.linalg.norm(norm) > 0:
        norm = norm / np.linalg.norm(norm)
    a = norm[0]
    b = norm[1]
    c = norm[2]
    rotation = np.array([
        [a ** 2 + (1 - a ** 2) * np.cos(theta),
         a * b * (1 - np.cos(theta)) + c * np.sin(theta),
         a * c * (1 - np.cos(theta)) - b * np.sin(theta)],
        [a * b * (1 - np.cos(theta)) - c * np.sin(theta),
         b ** 2 + (1 - b ** 2) * np.cos(theta),
         b * c * (1 - np.cos(theta)) + a * np.sin(theta)],
        [a * c * (1 - np.cos(theta)) + b * np.sin(theta),
         b * c * (1 - np.cos(theta)) - a * np.sin(theta),
         c ** 2 + (1 - c ** 2) * np.cos(theta)]])
    return rotation


def vector_rotation_norm(P2, norm=np.array([0, 0, 1]), theta=0, rotation_origin=np.zeros(3)):
    rotation = get_rot_matrix(norm, theta)
    P20 = np.dot(rotation, (P2 - rotation_origin)) + rotation_origin
    P20 = P20 / np.linalg.norm(P20)
    return P20


def vector_rotation(P2, norm=np.array([0, 0, 1]), theta=0, rotation_origin=np.zeros(3)):
    rotation = get_rot_matrix(norm, theta)
    P20 = np.dot(rotation, (P2 - rotation_origin)) + rotation_origin
    return P20


def rotMatrix_DCM(x0, y0, z0, x, y, z):
    # Diebel, James. "Representing attitude: Euler angles, unit quaternions, and rotation vectors."
    #  Matrix 58.15-16 (2006): 1-35.
    # eq. 17
    # https://arxiv.org/pdf/1705.06997.pdf
    # appendix B
    # Graf, Basile. "Quaternions and dynamics." arXiv preprint arXiv:0811.2889 (2008).
    #
    # A rotation matrix may also be referred to as a direction
    # cosine matrix, because the elements of this matrix are the
    # cosines of the unsigned angles between the body-¯xed axes
    # and the world axes. Denoting the world axes by (x; y; z)
    # and the body-fixed axes by (x0; y0; z0), let \theta_{x';y} be,
    # for example, the unsigned angle between the x'-axis and the y-axis
    # (x0, y0, z0)^T = dot(R, (x, y, z)^T )

    R = np.array(((np.dot(x0, x), np.dot(x0, y), np.dot(x0, z)),
                  (np.dot(y0, x), np.dot(y0, y), np.dot(y0, z)),
                  (np.dot(z0, x), np.dot(z0, y), np.dot(z0, z))))
    return R


class coordinate_transformation:
    @staticmethod
    def vector_rotation(f, R):
        fx, fy, fz = f.T[[0, 1, 2]]
        fx1 = R[0][0] * fx + R[0][1] * fy + R[0][2] * fz
        fy1 = R[1][0] * fx + R[1][1] * fy + R[1][2] * fz
        fz1 = R[2][0] * fx + R[2][1] * fy + R[2][2] * fz
        return np.dstack((fx1, fy1, fz1))[0]


class fullprint:
    'context manager for printing full numpy arrays'
    def __init__(self, **kwargs):
        kwargs.setdefault('threshold', np.inf)
        self.opt = kwargs

    def __enter__(self):
        self._opt = np.get_printoptions()
        np.set_printoptions(**self.opt)

    def __exit__(self, type, value, traceback):
        np.set_printoptions(**self._opt)


def Adams_Bashforth_Methods(order, f_list, eval_dt):
    def o1(f_list, eval_dt):
        delta = eval_dt * f_list[-1]
        return delta

    def o2(f_list, eval_dt):
        delta = eval_dt * (3 / 2 * f_list[-1] - 1 / 2 * f_list[-2])
        return delta

    def o3(f_list, eval_dt):
        delta = eval_dt * (23 / 12 * f_list[-1] - 16 / 12 * f_list[-2] + 5 / 12 * f_list[-3])
        return delta

    def o4(f_list, eval_dt):
        delta = eval_dt * (
                55 / 24 * f_list[-1] - 59 / 24 * f_list[-2] + 37 / 24 * f_list[-3] - 9 / 24 *
                f_list[-4])
        return delta

    def o5(f_list, eval_dt):
        delta = eval_dt * (
                1901 / 720 * f_list[-1] - 2774 / 720 * f_list[-2] + 2616 / 720 * f_list[-3]
                - 1274 / 720 * f_list[-4] + 251 / 720 * f_list[-5])
        return delta

    def get_order(order):
        return dict([(1, o1),
                     (2, o2),
                     (3, o3),
                     (4, o4),
                     (5, o5),
                     ]).get(order, o1)

    return get_order(order)(f_list, eval_dt)


def Adams_Moulton_Methods(order, f_list, eval_dt):
    def o1(f_list, eval_dt):
        delta = eval_dt * f_list[-1]
        return delta

    def o2(f_list, eval_dt):
        delta = eval_dt * (1 / 2 * f_list[-1] + 1 / 2 * f_list[-2])
        return delta

    def o3(f_list, eval_dt):
        delta = eval_dt * (5 / 12 * f_list[-1] + 8 / 12 * f_list[-2] - 1 / 12 * f_list[-3])
        return delta

    def o4(f_list, eval_dt):
        delta = eval_dt * (
                9 / 24 * f_list[-1] + 19 / 24 * f_list[-2] - 5 / 24 * f_list[-3] + 1 / 24 *
                f_list[-4])
        return delta

    def o5(f_list, eval_dt):
        delta = eval_dt * (251 / 720 * f_list[-1] + 646 / 720 * f_list[-2] - 264 / 720 * f_list[-3]
                           + 106 / 720 * f_list[-4] - 19 / 720 * f_list[-5])
        return delta

    def get_order(order):
        return dict([(1, o1),
                     (2, o2),
                     (3, o3),
                     (4, o4),
                     (5, o5),
                     ]).get(order, o1)

    return get_order(order)(f_list, eval_dt)


def mpiprint(*args, **kwargs):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    if comm.rank == 0:
        print(*args, **kwargs)


class Quaternion:
    """docstring for Quaternion"""

    def __init__(self, axis=np.array([0, 0, 1.0]), angle=0):
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

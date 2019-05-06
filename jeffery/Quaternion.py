import numpy as np


class Quaternion(object):
    """docstring for Quaternion"""
    def __init__(self, axis = None, angle = 0):
        super(Quaternion, self).__init__()

        if axis is None:
            axis = np.array([0,0,1.0])
        axis = np.array(axis)

        xyz= np.sin(.5*angle)*axis/np.linalg.norm(axis)
        self.q = np.array([
                np.cos(.5*angle),
                xyz[0],
                xyz[1],
                xyz[2]
            ])


    def __add__(self, other):
        Q = Quaternion()
        Q.q = self.q + other
        return Q

    def mul(self, other):
        assert(type(other) is Quaternion)

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



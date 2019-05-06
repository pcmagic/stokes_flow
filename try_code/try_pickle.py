# coding=utf-8
import pickle
from petsc4py import PETSc
import numpy as np
from scipy.io import savemat

# filename = 'sphere'
# with open(filename + '_pick.bin', 'rb') as input:
#     unpick = pickle.Unpickler(input)
#
# viewer = PETSc.Viewer().createBinary(filename + '_M.bin', 'r')
# M = PETSc.Mat().create(comm=PETSc.COMM_WORLD)
# M.setType('dense')
# M = M.load(viewer)
#
# viewer = PETSc.Viewer().createBinary(filename + '_F.bin', 'r')
# F = PETSc.Vec().create(comm=PETSc.COMM_WORLD)
# F = F.load(viewer)

# deltaLength = 0.05 ** np.arange(0.25, 1.05, 0.1)
# epsilon = np.arange(0.1, 2, 0.2)
# deltaLength, epsilon = np.meshgrid(deltaLength, epsilon)
# deltaLength = deltaLength.flatten()
# epsilon = epsilon.flatten()
# sphere_err = epsilon.copy()
# for i0 in range(sphere_err.size):
#     d = deltaLength[i0]
#     e = epsilon[i0]
#     fileName = 'sphere_%d_%f_%f' % (i0, d, e)
#     PETSc.Sys.Print(fileName)
# pass



# class a():
#     def printme(self):
#         PETSc.Sys.Print('a')
#
# class b(a):
#     def printme(self):
#         PETSc.Sys.Print('b')
#
# class c(a):
#     def printme(self):
#         PETSc.Sys.Print('c')
#
# class d(c, b):
#     def notiong(self):
#         pass
#
# if __name__ == '__main__':
#     d1 = d()
#     d1.printme()

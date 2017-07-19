import sys
import petsc4py

petsc4py.init(sys.argv)

from petsc4py import PETSc
import numpy as np
from scipy.io import loadmat
from scipy.io import savemat


def main_fun():
    # filename = 'sphereInPipe/symmetry_rho05'
    filename = 'symmetry_rho08_th0400_l016_c050'
    solve_method = 'gmres'
    precondition_method = 'none'

    mat_contents = loadmat(filename + '.mat')
    M = mat_contents['M'].astype(np.float)
    U = mat_contents['U'].astype(np.float)

    comm = PETSc.COMM_WORLD.tompi4py()
    rank = comm.Get_rank()
    M_petsc = PETSc.Mat().create(comm=PETSc.COMM_WORLD)
    M_petsc.setSizes(((None, M.shape[0]), (None, M.shape[1])))
    M_petsc.setType('dense')
    M_petsc.setFromOptions()
    M_petsc.setUp()
    temp_m_start, temp_m_end = M_petsc.getOwnershipRange()
    for k in range(temp_m_start, temp_m_end):
        M_petsc.setValues(k, np.arange(0, M.shape[1], dtype='int32'), M[k, :])
    M_petsc.assemble()

    F_petsc, U_petsc = M_petsc.createVecs()
    U_petsc[:] = U[:]
    U_petsc.assemble()
    F_petsc.set(0)

    ksp = PETSc.KSP()
    ksp.create(comm=PETSc.COMM_WORLD)
    ksp.setType(solve_method)
    ksp.getPC().setType(precondition_method)
    ksp.setOperators(M_petsc)
    ksp.setFromOptions()
    ksp.setInitialGuessNonzero(True)
    ksp.solve(U_petsc, F_petsc)

    F = self.vec_scatter(F_petsc)
    residualNorm = ksp.getResidualNorm()
    if rank == 0:
        savemat(filename + 'F' + '.mat',
                {'F':            F,
                 'residualNorm': residualNorm},
                oned_as='column')


if __name__ == '__main__':
    main_fun()

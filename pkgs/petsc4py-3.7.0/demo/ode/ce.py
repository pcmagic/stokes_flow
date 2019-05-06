# Stiff scalar valued ODE problem with an exact solution

import sys, petsc4py

petsc4py.init(sys.argv)

from petsc4py import PETSc
from math import sin, cos, exp
import numpy as np

class Monitor(object):
    def __init__(self, ode):
        self.ode = ode
        self.x = PETSc.Vec().createSeq(ode.n, comm=ode.comm)

    def __call__(self, ts, k, t, x):
        ode.evalSolution(t, self.x)
        self.x.axpy(-1, x)
        e = self.x.norm()
        h = ts.getTimeStep()
        PETSc.Sys.Print("step %3d t=%8.2e x=%8.2e h=%8.2e error=%8.2e" %
                        (k, t, x.getArray(), h, e), comm=self.ode.comm)


def monitor(self, ts, i, t, x):
    if self.history:
        lasti, lastt, lastx = self.history[-1]
        if i < lasti + 4 or t < lastt + 1e-4:
            return
    self.tozero.scatter(x, self.zvec, PETSc.InsertMode.INSERT)
    xx = self.zvec[:].tolist()
    self.history.append((i, t, xx))


class CE(object):
    n = 1
    comm = PETSc.COMM_SELF

    def __init__(self, lambda_=1.0):
        self.lambda_ = lambda_

    def evalSolution(self, t, x):
        l = self.lambda_
        x[0] = l / (l * l + 1) * (l * cos(t) + sin(t)) - l * l / (l * l + 1) * exp(-l * t)
        x.assemble()

    def evalFunction(self, ts, t, x, xdot, f):
        l = self.lambda_
        f[0] = xdot[0] + l * (x[0] - cos(t))
        f.assemble()

    def rhsfunction(self, ts, t, u, f):
        l = self.lambda_
        f[0] = l * (u[0] - cos(t))
        f.assemble()
        # print(1)
        # return f

    # def rhsjacobian(self, ts, t, u, J, P):
    #     # print ('MyODE.rhsjacobian()')
    #     # self.rhsjacobian_calls += 1
    #     P.zeroEntries()
    #     diag = -2 * u
    #     P.setDiagonal(diag)
    #     P.assemble()
    #     if J != P:
    #         J.assemble()
    #     print(2)
    #     return True  # same_nz

    def evalJacobian(self, ts, t, x, xdot, a, A, B):
        J = B
        l = self.lambda_
        J[0, 0] = a + l
        J.assemble()
        if A != B:
            A.assemble()
        return True  # same nonzero pattern


OptDB = PETSc.Options()

lambda_ = OptDB.getScalar('lambda', 10.0)
ode = CE(lambda_)

x0 = np.zeros(1)
x = PETSc.Vec().createWithArray(x0, comm=ode.comm)
f = x.duplicate()
ts = PETSc.TS().create(comm=ode.comm)
ts.setProblemType(ts.ProblemType.NONLINEAR)
ts.setType(ts.Type.RK)
ts.setRKType('3bs')
ts.setRHSFunction(ode.rhsfunction, f)
ts.setTime(0)
ts.setMaxSteps(3)
ts.setMaxTime(0.0001)
ts.setExactFinalTime(PETSc.TS.ExactFinalTime.INTERPOLATE)
ts.setMonitor(Monitor(ode))
ts.setFromOptions()
ts.setSolution(x)
ts.setTolerances(1e-9, 1e-12)
ts.setUp()
ts.setSaveTrajectory()
ts.solve(x)
# for i0 in range(10 ** 1):
#     ts.step()
#     print(x.getArray())
#     print(ts.getTime())
#     print()


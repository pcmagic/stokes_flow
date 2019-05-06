import solveu
import numpy as np


def myfun():
    a = 1.1
    out = np.ones((1))
    solveu.try1(a, out)
    PETSc.Sys.Print(out)


if __name__ == '__main__':
    myfun()

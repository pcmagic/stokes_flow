import sys

import petsc4py

petsc4py.init(sys.argv)

from petsc4py import PETSc
from ecoli_in_pipe import single_ecoli, head_tail, ecoli_U

head_U, tail_U, ref_U = single_ecoli.main_fun()

PETSc.Sys.Print('###################################################')
t_kwargs = {'head_U': head_U,
            'tail_U': tail_U, }
head_tail.main_fun(**t_kwargs)

PETSc.Sys.Print('###################################################')
t_kwargs = {'ecoli_U': ref_U}
ecoli_U.main_fun(**t_kwargs)


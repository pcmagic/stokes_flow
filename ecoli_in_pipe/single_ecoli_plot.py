import sys
import petsc4py

petsc4py.init(sys.argv)

from petsc4py import PETSc
from src import stokes_flow as sf
from src.myio import *
from src.objComposite import createEcoliComp_tunnel
from src.myvtk import save_singleEcoli_vtk
from ecoli_in_pipe.ecoli_common import *


# @profile
def main_fun(**main_kwargs):
    OptDB = PETSc.Options()
    fileHeadle = OptDB.getString('f', 'singleEcoliPro')
    OptDB.setValue('f', fileHeadle)
    main_kwargs['fileHeadle'] = fileHeadle
    problem_kwargs = get_problem_kwargs(**main_kwargs)

    print_case_info(**problem_kwargs)
    ecoli_comp = createEcoliComp_tunnel(name='ecoli_0', **problem_kwargs)
    ecoli_comp.show_u_nodes(linestyle=' ')
    ecoli_comp.vtk(fileHeadle)

    return True


if __name__ == '__main__':
    main_fun()

import sys
import petsc4py

petsc4py.init(sys.argv)

from petsc4py import PETSc
from src.objComposite import createEcoliComp_tunnel
from codeStore.ecoli_common import *


# @profile
def main_fun(**main_kwargs):
    OptDB = PETSc.Options()
    fileHandle = OptDB.getString('f', 'singleEcoliPro')
    OptDB.setValue('f', fileHandle)
    main_kwargs['fileHandle'] = fileHandle
    problem_kwargs = get_problem_kwargs(**main_kwargs)

    print_case_info(**problem_kwargs)
    ecoli_comp = createEcoliComp_tunnel(name='ecoli_0', **problem_kwargs)
    ecoli_comp.show_u_nodes(linestyle=' ')
    for tobj in ecoli_comp.get_obj_list():
        tobj.set_matrix_method(problem_kwargs['matrix_method'])
    ecoli_comp.vtk(fileHandle)

    return True


if __name__ == '__main__':
    main_fun()

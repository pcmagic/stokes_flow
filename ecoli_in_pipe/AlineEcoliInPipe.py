# coding=utf-8

import sys

import petsc4py

petsc4py.init(sys.argv)

import numpy as np
# import pickle
# from time import time
# from scipy.io import loadmat
# from src.stokes_flow import problem_dic, obj_dic
# from src.geo import *
from petsc4py import PETSc
from src import stokes_flow as sf
from src.myio import *
from src.support_class import *
from src.objComposite import createEcoliComp_tunnel
from src.myvtk import save_singleEcoli_vtk
from ecoli_in_pipe.ecoli_common import *


# @profile
def main_fun(**main_kwargs):
    OptDB = PETSc.Options()
    fileHandle = OptDB.getString('f', 'AlineEcoliInPipe')
    OptDB.setValue('f', fileHandle)
    main_kwargs['fileHandle'] = fileHandle
    problem_kwargs = get_problem_kwargs(**main_kwargs)

    if not problem_kwargs['restart']:
        forcepipe = problem_kwargs['forcepipe']
        print_case_info(**problem_kwargs)
        ecoli_comp0 = createEcoliComp_tunnel(name='ecoli_0', **problem_kwargs)
        ecoli_comp1 = createEcoliComp_tunnel(name='ecoli_1', **problem_kwargs)
        ecoli_length = (problem_kwargs['ls'] + problem_kwargs['dist_hs'] +
                        problem_kwargs['ph'] * problem_kwargs['ch']) * problem_kwargs['zoom_factor']
        ecoli_comp1.move(np.array((0, 0, 1 * ecoli_length)))
        problem = sf.stokesletsInPipeforcefreeProblem(**problem_kwargs)
        problem.set_prepare(forcepipe)
        problem.add_obj(ecoli_comp0)
        problem.add_obj(ecoli_comp1)
        # problem.show_u_nodes()
        problem.print_info()
        problem.create_matrix()
        problem.solve()

        # post process
        head_U, tail_U = print_single_ecoli_forcefree_result(ecoli_comp0, **problem_kwargs)
        ecoli_U = ecoli_comp0.get_ref_U()
        save_singleEcoli_vtk(problem, createHandle=createEcoliComp_tunnel)
    else:
        head_U, tail_U, ecoli_U = ecoli_restart(**main_kwargs)
    return head_U, tail_U, ecoli_U


if __name__ == '__main__':
    main_fun()

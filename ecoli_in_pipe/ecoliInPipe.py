# coding=utf-8

import sys

import petsc4py

petsc4py.init(sys.argv)

# from time import time
# from scipy.io import loadmat
# from src.stokes_flow import problem_dic, obj_dic
# from src.geo import *
from petsc4py import PETSc
from src import stokes_flow as sf
from src.myio import *
from src.objComposite import createEcoliComp_tunnel
from src.myvtk import save_singleEcoli_vtk
from codeStore.ecoli_common import *


# def get_problem_kwargs(**main_kwargs):
#     problem_kwargs = get_solver_kwargs()
#     OptDB = PETSc.Options()
#     fileHandle = OptDB.getString('f', 'ecoliInPipe')
#     OptDB.setValue('f', fileHandle)
#     problem_kwargs['fileHandle'] = fileHandle
#
#     kwargs_list = (get_vtk_tetra_kwargs(), get_ecoli_kwargs(), get_forcefree_kwargs(), main_kwargs,)
#     for t_kwargs in kwargs_list:
#         for key in t_kwargs:
#             problem_kwargs[key] = t_kwargs[key]
#     return problem_kwargs
#
#
# def print_case_info(**problem_kwargs):
#     fileHandle = problem_kwargs['fileHandle']
#     PETSc.Sys.Print('-->Ecoli in pipe case, force free case.')
#     print_solver_info(**problem_kwargs)
#     print_forcefree_info(**problem_kwargs)
#     print_ecoli_info(fileHandle, **problem_kwargs)
#     return True


# @profile
def main_fun(**main_kwargs):
    OptDB = PETSc.Options()
    fileHandle = OptDB.getString('f', 'ecoliInPipe')
    OptDB.setValue('f', fileHandle)
    main_kwargs['fileHandle'] = fileHandle
    problem_kwargs = get_problem_kwargs(**main_kwargs)

    if not problem_kwargs['restart']:
        forcepipe = problem_kwargs['forcepipe']
        print_case_info(**problem_kwargs)
        ecoli_comp = createEcoliComp_tunnel(name='ecoli_0', **problem_kwargs)
        problem = sf.StokesletsInPipeforcefreeProblem(**problem_kwargs)
        problem.set_prepare(forcepipe)
        problem.do_solve_process((ecoli_comp,), pick_M=False)
        # post process
        head_U, tail_U = print_single_ecoli_forcefree_result(ecoli_comp, **problem_kwargs)
        ecoli_U = ecoli_comp.get_ref_U()
        save_singleEcoli_vtk(problem, createHandle=createEcoliComp_tunnel)
    else:
        head_U, tail_U, ecoli_U = ecoli_restart(**main_kwargs)
    return head_U, tail_U, ecoli_U
    # t_name = check_file_extension(fileHandle, '_pick.bin')
    # with open(t_name, 'rb') as myinput:
    #     unpick = pickle.Unpickler(myinput)
    #     problem = unpick.load()
    # problem.unpick_myself()
    # ecoli_comp = problem.get_obj_list()[0]
    #
    # problem_kwargs = problem.get_kwargs()
    # problem_kwargs1 = get_problem_kwargs(**main_kwargs)
    # problem_kwargs['matname'] = problem_kwargs1['matname']
    # problem_kwargs['bnodesHeadle'] = problem_kwargs1['bnodesHeadle']
    # problem_kwargs['belemsHeadle'] = problem_kwargs1['belemsHeadle']
    # problem_kwargs['ffweightx'] = problem_kwargs1['ffweightx']
    # problem_kwargs['ffweighty'] = problem_kwargs1['ffweighty']
    # problem_kwargs['ffweightz'] = problem_kwargs1['ffweightz']
    # problem_kwargs['ffweightT'] = problem_kwargs1['ffweightT']
    # # PETSc.Sys.Print([attr for attr in dir(problem) if not attr.startswith('__')])
    # # PETSc.Sys.Print(problem_kwargs1['ffweightT'])
    #
    # problem.set_kwargs(**problem_kwargs)
    # print_case_info(**problem_kwargs)
    # problem.print_info()
    # problem.set_force_free()
    # problem.solve()


if __name__ == '__main__':
    main_fun()

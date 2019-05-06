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
# from src.support_class import *
from src.objComposite import *
from src.myvtk import save_singleEcoli_vtk
from ecoli_in_pipe.ecoli_common import *


# @profile
def main_fun(**main_kwargs):
    OptDB = PETSc.Options()
    fileHandle = OptDB.getString('f', 'AlineEcoliNoPipe')
    OptDB.setValue('f', fileHandle)
    main_kwargs['fileHandle'] = fileHandle
    problem_kwargs = get_problem_kwargs(**main_kwargs)

    if not problem_kwargs['restart']:
        print_case_info(**problem_kwargs)
        ecoliHeadType = OptDB.getString('ecoliHeadType', 'tunnel')
        if 'ellipse' in ecoliHeadType:
            ecoli_comp0 = createEcoliComp_ellipse(name='ecoli_0', **problem_kwargs)
            ecoli_comp1 = createEcoliComp_ellipse(name='ecoli_1', **problem_kwargs)
            ecoli_length = (2 * problem_kwargs['rs1'] + problem_kwargs['dist_hs'] +
                            problem_kwargs['ph'] * problem_kwargs['ch']) * problem_kwargs['zoom_factor']
        elif 'tunnel' in ecoliHeadType:
            ecoli_comp0 = createEcoliComp_tunnel(name='ecoli_0', **problem_kwargs)
            ecoli_comp1 = createEcoliComp_tunnel(name='ecoli_1', **problem_kwargs)
            ecoli_length = (problem_kwargs['ls'] + problem_kwargs['dist_hs'] +
                            problem_kwargs['ph'] * problem_kwargs['ch']) * problem_kwargs['zoom_factor']
        else:
            err_msg = 'wrong ecoliHeadType'
            raise ValueError(err_msg)
        ecoli_comp1.move(np.array((0, 0, 1 * ecoli_length)))

        problem = sf.ForceFreeProblem(**problem_kwargs)
        problem.add_obj(ecoli_comp0)
        problem.add_obj(ecoli_comp1)
        # problem.show_u_nodes()
        problem.print_info()
        problem.create_matrix()
        problem.solve()

        # post process
        # head_U, tail_U = print_single_ecoli_forcefree_result(ecoli_comp0, **problem_kwargs)
        rh1 = problem_kwargs['rh1']
        rel_Uh = problem_kwargs['rel_Uh']
        rel_Us = problem_kwargs['rel_Us']
        zoom_factor = problem_kwargs['zoom_factor']
        t_nondim = np.sqrt(np.sum((rel_Uh[-3:] + rel_Us[-3:]) ** 2))
        for obj0 in problem.get_obj_list():
            if isinstance(obj0, sf.ForceFreeComposite):
                ecoli_U = obj0.get_ref_U()
                non_dim_U = ecoli_U / t_nondim / np.hstack((zoom_factor * rh1 * np.ones(3), np.ones(3)))
                PETSc.Sys.Print(str(obj0), non_dim_U)
        # save_singleEcoli_vtk(problem, createHandle=createEcoliComp_tunnel)
    else:
        pass
        # head_U, tail_U, ecoli_U = ecoli_restart(**main_kwargs)
    return True


if __name__ == '__main__':
    main_fun()

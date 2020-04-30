import sys

import petsc4py

petsc4py.init(sys.argv)

import numpy as np
import pickle
# from time import time
# from scipy.io import loadmat
# from src.stokes_flow import problem_dic, obj_dic
# from src.geo import *
from petsc4py import PETSc
from src import stokes_flow as sf
# from src.myio import *
from src.objComposite import *
# from src.myvtk import *
# from src.support_class import *
from codeStore.ecoli_common import *


def do_solve_base_flow(basei, problem, obj_comp, uw_Base_list, sumFT_Base_list):
    problem.set_basei(basei)
    problem.create_F_U()
    problem.solve()
    PETSc.Sys.Print('---> basei %d' % basei)
    PETSc.Sys.Print(obj_comp.get_total_force())
    PETSc.Sys.Print(obj_comp.get_ref_U())
    uw_Base_list.append(obj_comp.get_ref_U())
    sumFT_Base_list.append(obj_comp.get_total_force())
    return uw_Base_list, sumFT_Base_list


def do_solve_base_flow_iter(basei, problem, obj_comp, uw_Base_list, sumFT_Base_list):
    problem.set_basei(basei)
    problem.create_F_U()
    problem.do_iterate3()
    PETSc.Sys.Print('---> basei %d' % basei)
    PETSc.Sys.Print(obj_comp.get_total_force())
    PETSc.Sys.Print(obj_comp.get_ref_U())
    uw_Base_list.append(obj_comp.get_ref_U())
    sumFT_Base_list.append(obj_comp.get_total_force())
    return uw_Base_list, sumFT_Base_list


# @profile
def main_fun(**main_kwargs):
    OptDB = PETSc.Options()
    fileHandle = OptDB.getString('f', 'ecoli_strain_rate')
    OptDB.setValue('f', fileHandle)
    main_kwargs['fileHandle'] = fileHandle
    # field_range = np.array([[-3, -3, -3], [3, 3, 3]])
    # n_grid = np.array([1, 1, 1]) * OptDB.getInt('n_grid', 10)
    # main_kwargs['field_range'] = field_range
    # main_kwargs['n_grid'] = n_grid
    # main_kwargs['region_type'] = 'rectangle'
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    # matrix_method = problem_kwargs['matrix_method']
    # pickProblem = problem_kwargs['pickProblem']
    # fileHandle = problem_kwargs['fileHandle']
    # save_vtk = problem_kwargs['save_vtk']
    problem_kwargs['basei'] = 1

    if not problem_kwargs['restart']:
        print_case_info(**problem_kwargs)

        # # check and dbg
        # problem_kwargs['planeShearNorm'] = np.array((0, 0, 1))
        # problem_kwargs['planeShearRate'] = np.array((1, 0, 0))
        # problem_kwargs['rel_Us'] = np.array((0, 0, 0, 0, 0, 0))
        # problem_kwargs['rel_Uh'] = np.array((0, 0, 0, 0, 0, 0))
        # ecoli_comp = create_ecoli_2part(**problem_kwargs)
        # problem_ff = sf.ShearFlowForceFreeProblem(**problem_kwargs)
        # problem_ff.add_obj(ecoli_comp)
        # problem_ff.create_matrix()
        # problem_ff.solve()
        # PETSc.Sys.Print('dbg')
        # PETSc.Sys.Print(ecoli_comp.get_total_force())
        # PETSc.Sys.Print(ecoli_comp.get_ref_U())
        # problem_kwargs['rel_Us'] = np.array((0, 0, 0, 0, 0, 0))
        # problem_kwargs['rel_Uh'] = np.array((0, 0, 0, 0, 0, 0))

        ecoli_comp = create_ecoli_2part(**problem_kwargs)
        ecoli_comp.set_rel_U_list([np.zeros(6), np.zeros(6)])
        # for tobj in ecoli_comp.get_obj_list():
        #     tobj.get_u_geo().mirrorImage(norm=np.array((0, 0, 1)), rotation_origin=np.zeros(3))
        #     tobj.get_f_geo().mirrorImage(norm=np.array((0, 0, 1)), rotation_origin=np.zeros(3))

        problem = sf.StrainRateBaseForceFreeProblem(**problem_kwargs)
        problem.add_obj(ecoli_comp)
        problem.print_info()
        problem.create_matrix()
        uw_Base_list = []
        sumFT_Base_list = []

        # passive cases
        for basei in (0, 1, 2, 3, 4, 5, 6, 7, 8):
            uw_Base_list, sumFT_Base_list = do_solve_base_flow(basei, problem, ecoli_comp,
                                                               uw_Base_list, sumFT_Base_list)
        # active case
        ecoli_comp.set_rel_U_list([np.zeros(6), np.array((0, 0, 0, 0, 0, 1))])
        basei = 9
        uw_Base_list, sumFT_Base_list = do_solve_base_flow(basei, problem, ecoli_comp,
                                                           uw_Base_list, sumFT_Base_list)

        pickle_dict = {'problem_kwargs':  problem_kwargs,
                       'u_nodes':         ecoli_comp.get_u_nodes(),
                       'f_nodes':         ecoli_comp.get_f_nodes(),
                       'uw_Base_list':    uw_Base_list,
                       'sumFT_Base_list': sumFT_Base_list, }
        with open('%s.pickle' % fileHandle, 'wb') as handle:
            pickle.dump(pickle_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        PETSc.Sys.Print('save table_data to %s.pickle' % fileHandle)
        # print_single_ecoli_force_result(problem, part='tail', prefix='tran', **problem_kwargs)
    return True


def main_fun_iter(**main_kwargs):
    OptDB = PETSc.Options()
    fileHandle = OptDB.getString('f', 'ecoli_strain_rate')
    OptDB.setValue('f', fileHandle)
    main_kwargs['fileHandle'] = fileHandle
    field_range = np.array([[-3, -3, -3], [3, 3, 3]])
    n_grid = np.array([1, 1, 1]) * OptDB.getInt('n_grid', 10)
    main_kwargs['field_range'] = field_range
    main_kwargs['n_grid'] = n_grid
    main_kwargs['region_type'] = 'rectangle'
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    # matrix_method = problem_kwargs['matrix_method']
    # pickProblem = problem_kwargs['pickProblem']
    # fileHandle = problem_kwargs['fileHandle']
    # save_vtk = problem_kwargs['save_vtk']
    problem_kwargs['basei'] = 1

    if not problem_kwargs['restart']:
        print_case_info(**problem_kwargs)
        ecoli_comp = create_ecoli_2part(**problem_kwargs)
        problem = sf.StrainRateBaseForceFreeIterateProblem(**problem_kwargs)
        problem.add_obj(ecoli_comp)
        problem.set_iterate_comp(ecoli_comp)
        problem.print_info()
        problem.create_matrix()
        uw_Base_list = []
        sumFT_Base_list = []

        # passive cases
        for basei in (0, 1, 2, 3, 4, 5, 6, 7, 8):
            uw_Base_list, sumFT_Base_list = do_solve_base_flow_iter(basei, problem, ecoli_comp,
                                                                    uw_Base_list, sumFT_Base_list)
        # active case
        ecoli_comp.set_rel_U_list([np.zeros(6), np.array((0, 0, 0, 0, 0, 1))])
        basei = 9
        uw_Base_list, sumFT_Base_list = do_solve_base_flow_iter(basei, problem, ecoli_comp,
                                                                uw_Base_list, sumFT_Base_list)

        pickle_dict = {'problem_kwargs':  problem_kwargs,
                       'u_nodes':         ecoli_comp.get_u_nodes(),
                       'f_nodes':         ecoli_comp.get_f_nodes(),
                       'uw_Base_list':    uw_Base_list,
                       'sumFT_Base_list': sumFT_Base_list, }
        with open('%s.pickle' % fileHandle, 'wb') as handle:
            pickle.dump(pickle_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        PETSc.Sys.Print('save table_data to %s.pickle' % fileHandle)
        # print_single_ecoli_force_result(problem, part='tail', prefix='tran', **problem_kwargs)
    return True


if __name__ == '__main__':
    OptDB = PETSc.Options()
    if OptDB.getBool('main_fun_iter', False):
        OptDB.setValue('main_fun', False)
        main_fun_iter()

    if OptDB.getBool('main_fun', True):
        main_fun()

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
from src.myio import *
from src.objComposite import *
# from src.myvtk import *
# from src.support_class import *
from codeStore import helix_common


def get_problem_kwargs(**main_kwargs):
    problem_kwargs = get_solver_kwargs()
    OptDB = PETSc.Options()
    fileHandle = OptDB.getString('f', 'helicoid_strain_rate')
    OptDB.setValue('f', fileHandle)
    problem_kwargs['fileHandle'] = fileHandle

    kwargs_list = (get_helicoid_kwargs(), get_forcefree_kwargs(), main_kwargs,)
    for t_kwargs in kwargs_list:
        for key in t_kwargs:
            problem_kwargs[key] = t_kwargs[key]
    return problem_kwargs


def print_case_info(**problem_kwargs):
    fileHandle = problem_kwargs['fileHandle']
    print_solver_info(**problem_kwargs)
    print_forcefree_info(**problem_kwargs)
    print_helicoid_info(**problem_kwargs)
    return True


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
    fileHandle = OptDB.getString('f', 'helicoid_strain_rate')
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
    problem_kwargs['zoom_factor'] = 1

    if not problem_kwargs['restart']:
        print_case_info(**problem_kwargs)
        helicoid_comp = create_helicoid_comp(namehandle='helicoid', **problem_kwargs)

        problem = sf.StrainRateBaseForceFreeProblem(**problem_kwargs)
        problem.add_obj(helicoid_comp)
        problem.print_info()
        problem.create_matrix()
        uw_Base_list = []
        sumFT_Base_list = []

        # passive cases
        for basei in (0, 1, 2, 3, 4, 5, 6, 7, 8):
            uw_Base_list, sumFT_Base_list = do_solve_base_flow(basei, problem, helicoid_comp,
                                                               uw_Base_list, sumFT_Base_list)
        # active case
        helicoid_comp.set_rel_U_list([np.zeros(6), ] * len(helicoid_comp.get_obj_list()))
        basei = 9
        uw_Base_list, sumFT_Base_list = do_solve_base_flow(basei, problem, helicoid_comp,
                                                           uw_Base_list, sumFT_Base_list)

        pickle_dict = {'problem_kwargs':  problem_kwargs,
                       'u_nodes':         helicoid_comp.get_u_nodes(),
                       'f_nodes':         helicoid_comp.get_f_nodes(),
                       'uw_Base_list':    uw_Base_list,
                       'sumFT_Base_list': sumFT_Base_list, }
        with open('%s.pickle' % fileHandle, 'wb') as handle:
            pickle.dump(pickle_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        PETSc.Sys.Print('save table_data to %s.pickle' % fileHandle)
    return True


def main_fun_E(**main_kwargs):
    OptDB = PETSc.Options()
    fileHandle = OptDB.getString('f', 'helicoid_strain_rate')
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
    problem_kwargs['zoom_factor'] = 1

    if not problem_kwargs['restart']:
        print_case_info(**problem_kwargs)
        helicoid_comp = create_helicoid_comp(namehandle='helicoid', **problem_kwargs)


        problem = sf.StrainRateBaseForceFreeProblem(**problem_kwargs)
        problem.add_obj(helicoid_comp)
        problem.print_info()
        problem.create_matrix()
        uw_Base_list = []
        sumFT_Base_list = []

        # passive cases
        for basei in (1, 2, 3, 4, 5):
            uw_Base_list, sumFT_Base_list = do_solve_base_flow(basei, problem, helicoid_comp,
                                                               uw_Base_list, sumFT_Base_list)
    return True


def main_fun_iter(**main_kwargs):
    err_msg = 'main_fun_iter() do NOT finish yet. '
    assert 1 == 2, err_msg

    OptDB = PETSc.Options()
    fileHandle = OptDB.getString('f', 'helicoid_strain_rate')
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
    hlx_ini_rot_theta = problem_kwargs['hlx_ini_rot_theta']

    if not problem_kwargs['restart']:
        print_case_info(**problem_kwargs)
        tail_obj_list = create_ecoli_tail(moveh=np.zeros(3), **problem_kwargs)
        tail_comp = sf.ForceFreeComposite(center=np.zeros(3), norm=np.array((0, 0, 1)),
                                          name='tail_comp')
        for tobj in tail_obj_list:
            tobj.node_rotation(norm=np.array([0, 1, 0]), theta=hlx_ini_rot_theta)
            tail_comp.add_obj(obj=tobj, rel_U=np.zeros(6))

        problem = sf.StrainRateBaseForceFreeIterateProblem(**problem_kwargs)
        problem.add_obj(tail_comp)
        problem.set_iterate_comp(tail_comp)
        problem.print_info()
        problem.create_matrix()
        uw_Base_list = []
        sumFT_Base_list = []

        # passive cases
        for basei in (0, 1, 2, 3, 4, 5, 6, 7, 8):
            uw_Base_list, sumFT_Base_list = do_solve_base_flow_iter(basei, problem, tail_comp,
                                                                    uw_Base_list, sumFT_Base_list)
        # active case
        tail_comp.set_rel_U_list([np.zeros(6), ] * len(tail_obj_list))
        basei = 9
        uw_Base_list, sumFT_Base_list = do_solve_base_flow_iter(basei, problem, tail_comp,
                                                                uw_Base_list, sumFT_Base_list)

        pickle_dict = {'problem_kwargs':  problem_kwargs,
                       'u_nodes':         tail_comp.get_u_nodes(),
                       'f_nodes':         tail_comp.get_f_nodes(),
                       'uw_Base_list':    uw_Base_list,
                       'sumFT_Base_list': sumFT_Base_list, }
        with open('%s.pickle' % fileHandle, 'wb') as handle:
            pickle.dump(pickle_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        PETSc.Sys.Print('save table_data to %s.pickle' % fileHandle)
        # print_single_ecoli_force_result(problem, part='tail', prefix='tran', **problem_kwargs)
    return True


if __name__ == '__main__':
    OptDB = PETSc.Options()
    # if OptDB.getBool('main_fun_iter', False):
    #     OptDB.setValue('main_fun', False)
    #     main_fun_iter()

    if OptDB.getBool('main_fun_E', False):
        OptDB.setValue('main_fun', False)
        main_fun_E()

    if OptDB.getBool('main_fun', True):
        main_fun()

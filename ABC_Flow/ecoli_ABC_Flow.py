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
from codeStore import ecoli_common


def get_problem_kwargs(**main_kwargs):
    problem_kwargs = ecoli_common.get_problem_kwargs(**main_kwargs)

    OptDB = PETSc.Options()
    hlx_ini_rot_theta = OptDB.getReal('hlx_ini_rot_theta', 0)
    problem_kwargs['hlx_ini_rot_theta'] = hlx_ini_rot_theta
    hlx_ini_rot_phi = OptDB.getReal('hlx_ini_rot_phi', 0)
    problem_kwargs['hlx_ini_rot_phi'] = hlx_ini_rot_phi

    ini_theta = OptDB.getReal('ini_theta', 0)
    ini_phi = OptDB.getReal('ini_phi', 0)
    ini_psi = OptDB.getReal('ini_psi', 0)
    problem_kwargs['ini_theta'] = ini_theta
    problem_kwargs['ini_phi'] = ini_phi
    problem_kwargs['ini_psi'] = ini_psi
    return problem_kwargs


def print_case_info(**problem_kwargs):
    t1 = ecoli_common.print_case_info(**problem_kwargs)
    hlx_ini_rot_theta = problem_kwargs['hlx_ini_rot_theta']
    PETSc.Sys.Print('  hlx_ini_rot_theta is %f' % hlx_ini_rot_theta)
    hlx_ini_rot_phi = problem_kwargs['hlx_ini_rot_phi']
    PETSc.Sys.Print('  hlx_ini_rot_phi is %f' % hlx_ini_rot_phi)

    ini_theta = problem_kwargs['ini_theta']
    ini_phi = problem_kwargs['ini_theta']
    ini_psi = problem_kwargs['ini_psi']
    PETSc.Sys.Print('  ini_theta %d, ini_theta %d. ini_psi %f' %
                    (ini_theta, ini_theta, ini_psi))
    return t1


def do_solve_base_flow(basei, problem, obj_comp, uw_Base_list, sumFT_Base_list):
    problem.set_basei(basei)
    problem.create_F_U()
    problem.solve()
    PETSc.Sys.Print('---> basei %s' % basei)
    PETSc.Sys.Print(obj_comp.get_total_force())
    ref_U = obj_comp.get_ref_U()
    PETSc.Sys.Print('ref_u: %f %f %f' % (ref_U[0], ref_U[1], ref_U[2]))
    PETSc.Sys.Print('ref_w: %f %f %f' % (ref_U[3], ref_U[4], ref_U[5]))
    uw_Base_list.append(obj_comp.get_ref_U())
    sumFT_Base_list.append(obj_comp.get_total_force())
    return uw_Base_list, sumFT_Base_list


def do_solve_base_flow_iter(basei, problem, obj_comp, uw_Base_list, sumFT_Base_list):
    problem.set_basei(basei)
    problem.create_F_U()
    problem.do_iterate3()
    PETSc.Sys.Print('---> basei %s' % basei)
    PETSc.Sys.Print(obj_comp.get_total_force())
    ref_U = obj_comp.get_ref_U()
    PETSc.Sys.Print('ref_u: %f %f %f' % (ref_U[0], ref_U[1], ref_U[2]))
    PETSc.Sys.Print('ref_w: %f %f %f' % (ref_U[3], ref_U[4], ref_U[5]))
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
            pickle.dump(pickle_dict, handle, protocol=4)
        PETSc.Sys.Print('save table_data to %s.pickle' % fileHandle)
        # print_single_ecoli_force_result(problem, part='tail', prefix='tran', **problem_kwargs)
    return True


def main_fun_E(**main_kwargs):
    OptDB = PETSc.Options()
    fileHandle = OptDB.getString('f', 'ecoli_strain_rate')
    OptDB.setValue('f', fileHandle)
    main_kwargs['fileHandle'] = fileHandle
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    problem_kwargs['basei'] = 1
    hlx_ini_rot_theta = problem_kwargs['hlx_ini_rot_theta']
    hlx_ini_rot_phi = problem_kwargs['hlx_ini_rot_phi']

    if not problem_kwargs['restart']:
        print_case_info(**problem_kwargs)

        ecoli_comp = create_ecoli_2part(**problem_kwargs)
        ecoli_comp.node_rotation(norm=np.array([0, 1, 0]), theta=hlx_ini_rot_theta)
        ecoli_comp.node_rotation(norm=np.array([0, 0, 1]), theta=hlx_ini_rot_phi)
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
        for basei in (1, 2, 3, 4, 5,):
            uw_Base_list, sumFT_Base_list = do_solve_base_flow(basei, problem, ecoli_comp,
                                                               uw_Base_list, sumFT_Base_list)
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
            pickle.dump(pickle_dict, handle, protocol=4)
        PETSc.Sys.Print('save table_data to %s.pickle' % fileHandle)
        # print_single_ecoli_force_result(problem, part='tail', prefix='tran', **problem_kwargs)
    return True


# @profile
def main_fun_ABC(**main_kwargs):
    OptDB = PETSc.Options()
    fileHandle = OptDB.getString('f', 'ecoli_ABC')
    OptDB.setValue('f', fileHandle)
    main_kwargs['fileHandle'] = fileHandle
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    problem_kwargs['basei'] = 1
    ini_theta = problem_kwargs['ini_theta']
    ini_phi = problem_kwargs['ini_phi']
    ini_psi = problem_kwargs['ini_psi']

    if not problem_kwargs['restart']:
        print_case_info(**problem_kwargs)

        ecoli_comp = create_ecoli_2part(**problem_kwargs)
        ecoli_comp.node_rotation(ecoli_comp.get_norm(), ini_psi)
        ecoli_comp.node_rotation(np.array((0, 1, 0)), ini_theta)
        ecoli_comp.node_rotation(np.array((0, 0, 1)), ini_phi)
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
        basei = 'ABCDEFHIJ'
        do_solve_base_flow(basei, problem, ecoli_comp, uw_Base_list, sumFT_Base_list)
    return True


def main_fun_shear(**main_kwargs):
    OptDB = PETSc.Options()
    fileHandle = OptDB.getString('f', 'ecoli_shear')
    OptDB.setValue('f', fileHandle)
    main_kwargs['fileHandle'] = fileHandle
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    problem_kwargs['basei'] = 1

    if not problem_kwargs['restart']:
        print_case_info(**problem_kwargs)

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
        basei = 'shear'
        do_solve_base_flow(basei, problem, ecoli_comp, uw_Base_list, sumFT_Base_list)
    return True


def main_fun_plot(**main_kwargs):
    OptDB = PETSc.Options()
    fileHandle = OptDB.getString('f', 'ecoli_strain_rate')
    OptDB.setValue('f', fileHandle)
    main_kwargs['fileHandle'] = fileHandle
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    problem_kwargs['basei'] = 1

    if not problem_kwargs['restart']:
        print_case_info(**problem_kwargs)
        ecoli_comp = create_ecoli_2part(**problem_kwargs)
        ecoli_comp.set_rel_U_list([np.zeros(6), np.zeros(6)])
        # for tobj in ecoli_comp.get_obj_list():
        #     tobj.get_u_geo().mirrorImage(norm=np.array((0, 0, 1)), rotation_origin=np.zeros(3))
        #     tobj.get_f_geo().mirrorImage(norm=np.array((0, 0, 1)), rotation_origin=np.zeros(3))
        ecoli_comp.show_u_nodes(linestyle='')
    return True


# mpirun -n 4 python ../ecoli_ABC_Flow.py -main_fun 1 -sm lg_rs -legendre_m 3 -legendre_k 2 -epsilon 3.000000 -rh11 0.100000  -rh12 0.100000  -rh2 0.030000  -ch 1.000000  -nth 12  -eh 0  -ph 0.666667  -hfct 1.000000  -n_tail 1  -with_cover 2  -left_hand 0  -rs1 0.5  -rs2 0.5  -ds 0.05  -es 0  -with_T_geo 0  -dist_hs 0.500000  -ksp_max_it 100  -plot_geo 0  -ffweight 2.000000  -f ABC_dbg_baseFlow

if __name__ == '__main__':
    OptDB = PETSc.Options()

    if OptDB.getBool('main_fun_ABC', False):
        OptDB.setValue('main_fun', False)
        main_fun_ABC()

    if OptDB.getBool('main_fun_shear', False):
        OptDB.setValue('main_fun', False)
        main_fun_shear()

    if OptDB.getBool('main_fun_iter', False):
        OptDB.setValue('main_fun', False)
        main_fun_iter()

    if OptDB.getBool('main_fun_E', False):
        OptDB.setValue('main_fun', False)
        main_fun_E()

    if OptDB.getBool('main_fun_plot', False):
        OptDB.setValue('main_fun', False)
        main_fun_plot()

    if OptDB.getBool('main_fun', True):
        main_fun()

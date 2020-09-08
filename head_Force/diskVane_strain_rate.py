import sys

import petsc4py

petsc4py.init(sys.argv)

import numpy as np
import pickle
# from time import time
# from scipy.io import loadmat
# from src.stokes_flow import problem_dic, obj_dic
from src.geo import *
from petsc4py import PETSc
from src import stokes_flow as sf
from src import slender_body as slb
from src.myio import *
from src.objComposite import *
# from src.myvtk import *
# from src.support_class import *
from codeStore import helix_common


def get_problem_kwargs(**main_kwargs):
    problem_kwargs = get_solver_kwargs()

    kwargs_list = (get_vtk_tetra_kwargs(), get_one_ellipse_kwargs(), get_diskVane_kwargs(),
                   get_forcefree_kwargs(), main_kwargs,)
    for t_kwargs in kwargs_list:
        for key in t_kwargs:
            problem_kwargs[key] = t_kwargs[key]
    return problem_kwargs


def print_case_info(**problem_kwargs):
    caseIntro = '-->ellipsoid3d_strain_rate. '
    fileHandle = problem_kwargs['fileHandle']
    PETSc.Sys.Print(caseIntro)
    print_solver_info(**problem_kwargs)
    print_forcefree_info(**problem_kwargs)
    print_one_ellipse_info(fileHandle, **problem_kwargs)
    print_diskVane_kwargs(**problem_kwargs)
    return True


def do_solve_base_flow(basei, problem, obj_comp, uw_Base_list, sumFT_Base_list):
    problem.set_basei(basei)
    problem.create_F_U()
    problem.solve()
    PETSc.Sys.Print('---> basei %d' % basei)
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
    PETSc.Sys.Print('---> basei %d' % basei)
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
    fileHandle = OptDB.getString('f', 'diskVane_strain_rate')
    OptDB.setValue('f', fileHandle)
    main_kwargs['fileHandle'] = fileHandle
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    problem_kwargs['basei'] = 1
    # rs1 = problem_kwargs['rs1']
    # rs2 = problem_kwargs['rs2']
    # rs3 = problem_kwargs['rs3']
    # velocity = problem_kwargs['velocity']
    # ds = problem_kwargs['ds']
    # es = problem_kwargs['es']
    # center = problem_kwargs['center']

    # r1 = problem_kwargs['diskVane_r1']
    # rz = problem_kwargs['diskVane_rz']
    # r2 = problem_kwargs['diskVane_r2']
    # ds = problem_kwargs['diskVane_ds']
    # th_loc = problem_kwargs['diskVane_th_loc']
    # ph_loc = problem_kwargs['diskVane_ph_loc']
    # nr = problem_kwargs['diskVane_nr']
    # nz = problem_kwargs['diskVane_nz']

    if not problem_kwargs['restart']:
        print_case_info(**problem_kwargs)
        tgeo_list = create_diskVane_tail(0, **problem_kwargs)

        obj_comp = sf.ForceFreeComposite(center=np.zeros(3), norm=np.array((0, 0, 1)),
                                         name='diskVane_comp')
        for tgeo in tgeo_list:
            tobj = sf.StokesFlowObj()
            tobj.set_data(tgeo, tgeo, name='diskVane_obj')
            obj_comp.add_obj(obj=tobj, rel_U=np.zeros(6))

        problem = sf.StrainRateBaseForceFreeProblem(**problem_kwargs)
        problem.add_obj(obj_comp)
        problem.print_info()
        problem.create_matrix()
        uw_Base_list = []
        sumFT_Base_list = []

        # passive cases
        for basei in (0, 1, 2, 3, 4, 5, 6, 7, 8):
            uw_Base_list, sumFT_Base_list = do_solve_base_flow(basei, problem, obj_comp,
                                                               uw_Base_list, sumFT_Base_list)
        # active case
        obj_comp.set_rel_U_list([np.zeros(6), ] * len(tgeo_list))
        basei = 9
        uw_Base_list, sumFT_Base_list = do_solve_base_flow(basei, problem, obj_comp,
                                                           uw_Base_list, sumFT_Base_list)

        pickle_dict = {'problem_kwargs':  problem_kwargs,
                       'u_nodes':         obj_comp.get_u_nodes(),
                       'f_nodes':         obj_comp.get_f_nodes(),
                       'uw_Base_list':    uw_Base_list,
                       'sumFT_Base_list': sumFT_Base_list, }
        with open('%s.pickle' % fileHandle, 'wb') as handle:
            pickle.dump(pickle_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        PETSc.Sys.Print('save table_data to %s.pickle' % fileHandle)
    return True


def main_fun_E(**main_kwargs):
    OptDB = PETSc.Options()
    fileHandle = OptDB.getString('f', 'diskVane_strain_rate')
    OptDB.setValue('f', fileHandle)
    main_kwargs['fileHandle'] = fileHandle
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    problem_kwargs['basei'] = 1

    if not problem_kwargs['restart']:
        print_case_info(**problem_kwargs)
        tgeo_list = create_diskVane_tail(0, **problem_kwargs)

        obj_comp = sf.ForceFreeComposite(center=np.zeros(3), norm=np.array((0, 0, 1)),
                                         name='diskVane_comp')
        for tgeo in tgeo_list:
            tobj = sf.StokesFlowObj()
            tobj.set_data(tgeo, tgeo, name='diskVane_obj')
            obj_comp.add_obj(obj=tobj, rel_U=np.zeros(6))

        problem = sf.StrainRateBaseForceFreeProblem(**problem_kwargs)
        problem.add_obj(obj_comp)
        problem.print_info()
        problem.create_matrix()
        uw_Base_list = []
        sumFT_Base_list = []

        # passive cases
        for basei in (1, 2, 3, 4, 5,):
            uw_Base_list, sumFT_Base_list = do_solve_base_flow(basei, problem, obj_comp,
                                                               uw_Base_list, sumFT_Base_list)
        wE41 = (uw_Base_list[3][3] + uw_Base_list[4][4]) / 2
        wE42 = (uw_Base_list[3][4] - uw_Base_list[4][3]) / 2
        wE23 = (uw_Base_list[0][5] - 2 * uw_Base_list[1][5]) / 2
        PETSc.Sys.Print('%f, %f, %f' % (wE23, wE41, wE42))
    return True


if __name__ == '__main__':
    OptDB = PETSc.Options()
    # if OptDB.getBool('main_fun_iter', False):
    #     OptDB.setValue('main_fun', False)
    #     main_fun_iter()
    #
    # if OptDB.getBool('main_fun_rote', False):
    #     OptDB.setValue('main_fun', False)
    #     main_fun_rote()
    #
    if OptDB.getBool('main_fun_E', False):
        OptDB.setValue('main_fun', False)
        main_fun_E()
    #
    # if OptDB.getBool('main_fun_SLB_E', False):
    #     OptDB.setValue('main_fun', False)
    #     main_fun_SLB_E()

    if OptDB.getBool('main_fun', True):
        main_fun()

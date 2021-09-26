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

    kwargs_list = (get_vtk_tetra_kwargs(), get_one_ellipse_kwargs(),
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
    fileHandle = OptDB.getString('f', 'ellipsoid3d_strain_rate')
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
    rs1 = problem_kwargs['rs1']
    rs2 = problem_kwargs['rs2']
    rs3 = problem_kwargs['rs3']
    velocity = problem_kwargs['velocity']
    ds = problem_kwargs['ds']
    es = problem_kwargs['es']
    center = problem_kwargs['center']


    if not problem_kwargs['restart']:
        print_case_info(**problem_kwargs)
        ellipsoid_geo = ellipse_3d_geo()
        ellipsoid_geo.create_delta(ds, rs1, rs2, rs3)
        ellipsoid_obj = sf.StokesFlowObj()
        ellipsoid_obj.set_data(ellipsoid_geo, ellipsoid_geo, name='ellipsoid3d_obj')
        ellipsoid_obj.node_rotation(norm=np.array((0, 1, 0)), theta=-np.pi/2)

        ellipsoid3d_comp = sf.ForceFreeComposite(center=np.zeros(3), norm=np.array((0, 0, 1)),
                                                 name='ellipsoid3d_comp')
        ellipsoid3d_comp.add_obj(obj=ellipsoid_obj, rel_U=np.zeros(6))

        problem = sf.StrainRateBaseForceFreeProblem(**problem_kwargs)
        problem.add_obj(ellipsoid3d_comp)
        problem.print_info()
        problem.create_matrix()
        uw_Base_list = []
        sumFT_Base_list = []

        # passive cases
        for basei in (0, 1, 2, 3, 4, 5, 6, 7, 8):
            uw_Base_list, sumFT_Base_list = do_solve_base_flow(basei, problem, ellipsoid3d_comp,
                                                               uw_Base_list, sumFT_Base_list)
        # active case
        ellipsoid3d_comp.set_rel_U_list([np.zeros(6), ])
        basei = 9
        uw_Base_list, sumFT_Base_list = do_solve_base_flow(basei, problem, ellipsoid3d_comp,
                                                           uw_Base_list, sumFT_Base_list)

        pickle_dict = {'problem_kwargs':  problem_kwargs,
                       'u_nodes':         ellipsoid3d_comp.get_u_nodes(),
                       'f_nodes':         ellipsoid3d_comp.get_f_nodes(),
                       'uw_Base_list':    uw_Base_list,
                       'sumFT_Base_list': sumFT_Base_list, }
        with open('%s.pickle' % fileHandle, 'wb') as handle:
            pickle.dump(pickle_dict, handle, protocol=4)
        PETSc.Sys.Print('save table_data to %s.pickle' % fileHandle)
    return True


# def main_fun_E(**main_kwargs):
#     OptDB = PETSc.Options()
#     fileHandle = OptDB.getString('f', 'helix_strain_rate')
#     OptDB.setValue('f', fileHandle)
#     main_kwargs['fileHandle'] = fileHandle
#     problem_kwargs = get_problem_kwargs(**main_kwargs)
#     problem_kwargs['basei'] = 1
#     hlx_ini_rot_theta = problem_kwargs['hlx_ini_rot_theta']
#
#     if not problem_kwargs['restart']:
#         print_case_info(**problem_kwargs)
#         tail_obj_list = create_ecoli_tail(moveh=np.zeros(3), **problem_kwargs)
#         # PETSc.Sys.Print(problem_kwargs)
#         tail_comp = sf.ForceFreeComposite(center=np.zeros(3), norm=np.array((0, 0, 1)),
#                                           name='tail_comp')
#         for tobj in tail_obj_list:
#             tobj.node_rotation(norm=np.array([0, 1, 0]), theta=hlx_ini_rot_theta)
#             # tobj.node_rotation(norm=np.array([0, 0, 1]), theta=np.pi)
#             tail_comp.add_obj(obj=tobj, rel_U=np.zeros(6))
#
#         problem = sf.StrainRateBaseForceFreeProblem(**problem_kwargs)
#         problem.add_obj(tail_comp)
#         problem.print_info()
#         problem.create_matrix()
#         uw_Base_list = []
#         sumFT_Base_list = []
#         # passive cases
#         for basei in (1, 2, 3, 4, 5, ):
#             uw_Base_list, sumFT_Base_list = do_solve_base_flow(basei, problem, tail_comp,
#                                                                uw_Base_list, sumFT_Base_list)
#     return True
#
#
# def main_fun_rote(**main_kwargs):
#     err_msg = 'force free part do not finish yet'
#     assert 1 == 2, err_msg
#
#     OptDB = PETSc.Options()
#     fileHandle = OptDB.getString('f', 'helix_strain_rate')
#     OptDB.setValue('f', fileHandle)
#     main_kwargs['fileHandle'] = fileHandle
#     field_range = np.array([[-3, -3, -3], [3, 3, 3]])
#     n_grid = np.array([1, 1, 1]) * OptDB.getInt('n_grid', 10)
#     main_kwargs['field_range'] = field_range
#     main_kwargs['n_grid'] = n_grid
#     main_kwargs['region_type'] = 'rectangle'
#     problem_kwargs = get_problem_kwargs(**main_kwargs)
#     matrix_method = problem_kwargs['matrix_method']
#     # pickProblem = problem_kwargs['pickProblem']
#     # fileHandle = problem_kwargs['fileHandle']
#     # save_vtk = problem_kwargs['save_vtk']
#     problem_kwargs['basei'] = 1
#     hlx_ini_rot_theta = problem_kwargs['hlx_ini_rot_theta']
#     assert matrix_method == 'pf_selfRepeat'
#
#     if not problem_kwargs['restart']:
#         print_case_info(**problem_kwargs)
#         tail_obj_list = create_ecoli_tail(moveh=np.zeros(3), **problem_kwargs)
#         # PETSc.Sys.Print(problem_kwargs)
#         tail_comp = sf.ForceFreeComposite(center=np.zeros(3), norm=np.array((0, 0, 1)),
#                                           name='tail_comp')
#         tobj = tail_obj_list[0]
#         tobj.node_rotation(norm=np.array([0, 1, 0]), theta=hlx_ini_rot_theta)
#         # tobj.node_rotation(norm=np.array([0, 0, 1]), theta=np.pi)
#         tail_comp.add_obj(obj=tobj, rel_U=np.zeros(6))
#
#         problem = sf.StrainRateBaseForceFreeProblem(**problem_kwargs)
#         problem.add_obj(tail_comp)
#         problem.print_info()
#         problem.create_matrix()
#         uw_Base_list = []
#         sumFT_Base_list = []
#
#         # passive cases
#         for basei in (0, 1, 2, 3, 4, 5, 6, 7, 8):
#             uw_Base_list, sumFT_Base_list = do_solve_base_flow(basei, problem, tail_comp,
#                                                                uw_Base_list, sumFT_Base_list)
#         # active case
#         tail_comp.set_rel_U_list([np.zeros(6), ])
#         basei = 9
#         uw_Base_list, sumFT_Base_list = do_solve_base_flow(basei, problem, tail_comp,
#                                                            uw_Base_list, sumFT_Base_list)
#
#         pickle_dict = {'problem_kwargs':  problem_kwargs,
#                        'u_nodes':         tail_comp.get_u_nodes(),
#                        'f_nodes':         tail_comp.get_f_nodes(),
#                        'uw_Base_list':    uw_Base_list,
#                        'sumFT_Base_list': sumFT_Base_list, }
#         with open('%s.pickle' % fileHandle, 'wb') as handle:
#             pickle.dump(pickle_dict, handle, protocol=4)
#         PETSc.Sys.Print('save table_data to %s.pickle' % fileHandle)
#     return True
#
#
# def main_fun_iter(**main_kwargs):
#     OptDB = PETSc.Options()
#     fileHandle = OptDB.getString('f', 'helix_strain_rate')
#     OptDB.setValue('f', fileHandle)
#     main_kwargs['fileHandle'] = fileHandle
#     field_range = np.array([[-3, -3, -3], [3, 3, 3]])
#     n_grid = np.array([1, 1, 1]) * OptDB.getInt('n_grid', 10)
#     main_kwargs['field_range'] = field_range
#     main_kwargs['n_grid'] = n_grid
#     main_kwargs['region_type'] = 'rectangle'
#     problem_kwargs = get_problem_kwargs(**main_kwargs)
#     # matrix_method = problem_kwargs['matrix_method']
#     # pickProblem = problem_kwargs['pickProblem']
#     # fileHandle = problem_kwargs['fileHandle']
#     # save_vtk = problem_kwargs['save_vtk']
#     problem_kwargs['basei'] = 1
#     hlx_ini_rot_theta = problem_kwargs['hlx_ini_rot_theta']
#
#     if not problem_kwargs['restart']:
#         print_case_info(**problem_kwargs)
#         tail_obj_list = create_ecoli_tail(moveh=np.zeros(3), **problem_kwargs)
#         tail_comp = sf.ForceFreeComposite(center=np.zeros(3), norm=np.array((0, 0, 1)),
#                                           name='tail_comp')
#         for tobj in tail_obj_list:
#             tobj.node_rotation(norm=np.array([0, 1, 0]), theta=hlx_ini_rot_theta)
#             tail_comp.add_obj(obj=tobj, rel_U=np.zeros(6))
#
#         problem = sf.StrainRateBaseForceFreeIterateProblem(**problem_kwargs)
#         problem.add_obj(tail_comp)
#         problem.set_iterate_comp(tail_comp)
#         problem.print_info()
#         problem.create_matrix()
#         uw_Base_list = []
#         sumFT_Base_list = []
#
#         # passive cases
#         for basei in (0, 1, 2, 3, 4, 5, 6, 7, 8):
#             uw_Base_list, sumFT_Base_list = do_solve_base_flow_iter(basei, problem, tail_comp,
#                                                                     uw_Base_list, sumFT_Base_list)
#         # active case
#         tail_comp.set_rel_U_list([np.zeros(6), ] * len(tail_obj_list))
#         basei = 9
#         uw_Base_list, sumFT_Base_list = do_solve_base_flow_iter(basei, problem, tail_comp,
#                                                                 uw_Base_list, sumFT_Base_list)
#
#         pickle_dict = {'problem_kwargs':  problem_kwargs,
#                        'u_nodes':         tail_comp.get_u_nodes(),
#                        'f_nodes':         tail_comp.get_f_nodes(),
#                        'uw_Base_list':    uw_Base_list,
#                        'sumFT_Base_list': sumFT_Base_list, }
#         with open('%s.pickle' % fileHandle, 'wb') as handle:
#             pickle.dump(pickle_dict, handle, protocol=4)
#         PETSc.Sys.Print('save table_data to %s.pickle' % fileHandle)
#         # print_single_ecoli_force_result(problem, part='tail', prefix='tran', **problem_kwargs)
#     return True
#
#
# def main_fun_SLB_E(**main_kwargs):
#     OptDB = PETSc.Options()
#     fileHandle = OptDB.getString('f', 'helix_strain_rate')
#     OptDB.setValue('f', fileHandle)
#     main_kwargs['fileHandle'] = fileHandle
#     problem_kwargs = get_problem_kwargs(**main_kwargs)
#     problem_kwargs['basei'] = 1
#     hlx_ini_rot_theta = problem_kwargs['hlx_ini_rot_theta']
#     ph = problem_kwargs['ph']
#     ch = problem_kwargs['ch']
#     rt1 = problem_kwargs['rh11']
#     rt2 = problem_kwargs['rh2']
#     n_sgm = OptDB.getInt('n_sgm', 10)
#     n_segment = int(np.ceil(n_sgm * ch))
#     n_hlx = problem_kwargs['n_tail']
#     matrix_method = problem_kwargs['matrix_method']
#     problem_kwargs['basei'] = 1
#     # slb_epsabs = OptDB.getReal('slb_epsabs', 1e-200)
#     # slb_epsrel = OptDB.getReal('slb_epsrel', 1e-8)
#     # slb_limit = OptDB.getReal('slb_limit', 10000)
#
#     if not problem_kwargs['restart']:
#         print_case_info(**problem_kwargs)
#         tail_comp = sf.ForceFreeComposite(center=np.zeros(3), norm=np.array((0, 0, 1)),
#                                           name='tail_comp')
#         check_nth = matrix_method == 'lightill_slb'
#         slb_geo_fun = slb_helix if matrix_method == 'lightill_slb' else Johnson_helix
#         for i0, theta0 in enumerate(np.linspace(0, 2 * np.pi, n_hlx, endpoint=False)):
#             hlx1_geo = slb_geo_fun(ph, ch, rt1, rt2, theta0=theta0)
#             hlx1_geo.create_nSegment(n_segment, check_nth=check_nth)
#             hlx1_obj = sf.StokesFlowObj()
#             obj_name = 'helix%d' % i0
#             hlx1_obj.set_data(hlx1_geo, hlx1_geo, name=obj_name)
#             hlx1_obj.node_rotation(norm=np.array([0, 1, 0]), theta=hlx_ini_rot_theta)
#             tail_comp.add_obj(hlx1_obj, rel_U=np.zeros(6))
#
#         problem = slb.StrainRateBaseForceFreeProblem(**problem_kwargs)
#         problem.add_obj(tail_comp)
#         problem.print_info()
#         problem.create_matrix()
#         uw_Base_list = []
#         sumFT_Base_list = []
#         # passive cases
#         for basei in (1, 2, 3, 4, 5, ):
#             uw_Base_list, sumFT_Base_list = do_solve_base_flow(basei, problem, tail_comp,
#                                                                uw_Base_list, sumFT_Base_list)
#     return True
#


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
    # if OptDB.getBool('main_fun_E', False):
    #     OptDB.setValue('main_fun', False)
    #     main_fun_E()
    #
    # if OptDB.getBool('main_fun_SLB_E', False):
    #     OptDB.setValue('main_fun', False)
    #     main_fun_SLB_E()

    if OptDB.getBool('main_fun', True):
        main_fun()

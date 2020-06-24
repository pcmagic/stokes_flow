import sys

import petsc4py

petsc4py.init(sys.argv)
# from scipy.io import savemat, loadmat
# from src.ref_solution import *
# import warnings
# from memory_profiler import profile
# from time import time
from src.myio import *
from src.objComposite import *
from src.StokesFlowMethod import *
from src.geo import *
from src import stokes_flow as sf
from src import slender_body as slb
from codeStore.helix_common import AtBtCt


def get_problem_kwargs(**main_kwargs):
    problem_kwargs = get_solver_kwargs()
    OptDB = PETSc.Options()
    fileHandle = OptDB.getString('f', 'obj_helicoid')
    OptDB.setValue('f', fileHandle)
    problem_kwargs['fileHandle'] = fileHandle

    kwargs_list = (get_obj_helicoid_kwargs(), get_helix_SLB_kwargs(),
                   get_forcefree_kwargs(), main_kwargs,)
    for t_kwargs in kwargs_list:
        for key in t_kwargs:
            problem_kwargs[key] = t_kwargs[key]
    return problem_kwargs


def print_case_info(**problem_kwargs):
    fileHandle = problem_kwargs['fileHandle']
    print_solver_info(**problem_kwargs)
    print_forcefree_info(**problem_kwargs)
    print_helix_SLB_info(fileHandle, **problem_kwargs)
    print_obj_helicoid_info(**problem_kwargs)
    return True


def main_resistanceMatrix_SLB(**main_kwargs):
    # OptDB = PETSc.Options()
    main_kwargs['zoom_factor'] = 1
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    matrix_method = problem_kwargs['matrix_method']
    fileHandle = problem_kwargs['fileHandle']
    pickProblem = problem_kwargs['pickProblem']
    print_case_info(**problem_kwargs)

    # create SLB_helix
    rh1 = problem_kwargs['rh1']
    rh2 = problem_kwargs['rh2']
    ch = problem_kwargs['ch']
    center = problem_kwargs['center']
    ph = problem_kwargs['ph']
    n_segment = problem_kwargs['n_segment']
    n_tail = problem_kwargs['n_tail']
    # left_hand = problem_kwargs['left_hand']
    rel_Uh = problem_kwargs['rel_Uh']
    check_nth = problem_kwargs['check_nth']
    slb_geo_fun = problem_kwargs['slb_geo_fun']
    slb.check_matrix_method(matrix_method)

    i0, theta0 = 0, 0
    hlx1_geo = slb_geo_fun(ph, ch, rh1, rh2, theta0=theta0)
    hlx1_geo.create_nSegment(n_segment, check_nth=check_nth)
    hlx1_obj = sf.StokesFlowObj()
    obj_name = 'helix%d' % i0
    hlx1_obj.set_data(hlx1_geo, hlx1_geo, name=obj_name)
    # hlx1_obj.show_f_u_nodes()
    # assert 1==2
    # hlx1_obj.node_rotation(norm=np.array([0, 1, 0]), theta=hlx_ini_rot_theta)
    helicoid_comp = obj2helicoid_comp(hlx1_obj, **problem_kwargs)
    # hlx1_obj.show_f_u_nodes()
    # helicoid_comp.show_f_u_nodes(linestyle='-')
    # assert 1 == 2
    helicoid_obj_list = helicoid_comp.get_obj_list()
    helicoid_center = helicoid_comp.get_center()

    problem = slb.problem_dic[matrix_method](**problem_kwargs)
    for tobj in helicoid_obj_list:
        problem.add_obj(tobj)
    problem.print_info()
    problem.create_matrix()
    At, Bt, Ct, ftr_info, frt_info = AtBtCt(problem, save_vtk=False, pick_M=False)
    PETSc.Sys.Print(At, Bt, Ct)

    #
    # # solve
    # problem = slb.SlenderBodyProblem(**problem_kwargs)
    # for tobj in helicoid_obj_list:
    #     problem.add_obj(tobj)
    # # problem.show_f_nodes()
    # # problem.show_u_nodes()
    # # assert 1 == 2
    #
    # if pickProblem:
    #     problem.pickmyself('%s_tran' % fileHandle, ifcheck=True)
    # problem.print_info()
    # problem.create_matrix()
    #
    # # 1. translation
    # for tobj in helicoid_obj_list:
    #     tobj.set_rigid_velocity(np.array((0, 0, 1, 0, 0, 0)), center=helicoid_center)
    # problem.create_matrix()
    # problem.solve()
    # if problem_kwargs['pickProblem']:
    #     problem.pickmyself('%s_tran' % fileHandle, pick_M=False, mat_destroy=False)
    # total_force = problem.get_total_force()
    # PETSc.Sys.Print('translation total_force', total_force)
    # # problem.vtk_self('%s_tran' % fileHandle)
    #
    # # 2. rotation
    # for tobj in helicoid_obj_list:
    #     tobj.set_rigid_velocity(np.array((0, 0, 0, 0, 0, 1)), center=helicoid_center)
    # problem.create_matrix()
    # problem.solve()
    # if problem_kwargs['pickProblem']:
    #     problem.pickmyself('%s_rota' % fileHandle, pick_M=False, mat_destroy=False)
    # total_force = problem.get_total_force()
    # PETSc.Sys.Print('rotation total_force', total_force)
    # # problem.vtk_self('%s_rota' % fileHandle)
    #
    # # # 1. translation
    # # for tobj in helicoid_obj_list:
    # #     tobj.set_rigid_velocity(np.array((0, 1, 0, 0, 0, 0)), center=helicoid_center)
    # # problem.create_F_U()
    # # problem.solve()
    # # if problem_kwargs['pickProblem']:
    # #     problem.pickmyself('%s_tran' % fileHandle, pick_M=False, mat_destroy=False)
    # # total_force = problem.get_total_force()
    # # PETSc.Sys.Print('translation total_force', total_force)
    # # # problem.vtk_self('%s_tran' % fileHandle)
    # #
    # # # 2. rotation
    # # for tobj in helicoid_obj_list:
    # #     tobj.set_rigid_velocity(np.array((0, 0, 0, 0, 1, 0)), center=helicoid_center)
    # # problem.create_F_U()
    # # problem.solve()
    # # if problem_kwargs['pickProblem']:
    # #     problem.pickmyself('%s_rota' % fileHandle, pick_M=False, mat_destroy=False)
    # # total_force = problem.get_total_force()
    # # PETSc.Sys.Print('rotation total_force', total_force)
    # # # problem.vtk_self('%s_rota' % fileHandle)
    # #
    # # # 1. translation
    # # for tobj in helicoid_obj_list:
    # #     tobj.set_rigid_velocity(np.array((1, 0, 0, 0, 0, 0)), center=helicoid_center)
    # # problem.create_F_U()
    # # problem.solve()
    # # if problem_kwargs['pickProblem']:
    # #     problem.pickmyself('%s_tran' % fileHandle, pick_M=False, mat_destroy=False)
    # # total_force = problem.get_total_force()
    # # PETSc.Sys.Print('translation total_force', total_force)
    # # # problem.vtk_self('%s_tran' % fileHandle)
    # #
    # # # 2. rotation
    # # for tobj in helicoid_obj_list:
    # #     tobj.set_rigid_velocity(np.array((0, 0, 0, 1, 0, 0)), center=helicoid_center)
    # # problem.create_F_U()
    # # problem.solve()
    # # if problem_kwargs['pickProblem']:
    # #     problem.pickmyself('%s_rota' % fileHandle, pick_M=False, mat_destroy=False)
    # # total_force = problem.get_total_force()
    # # PETSc.Sys.Print('rotation total_force', total_force)
    # # # problem.vtk_self('%s_rota' % fileHandle)
    return True

def main_dbg(**main_kwargs):
    # OptDB = PETSc.Options()
    main_kwargs['zoom_factor'] = 1
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    matrix_method = problem_kwargs['matrix_method']
    fileHandle = problem_kwargs['fileHandle']
    pickProblem = problem_kwargs['pickProblem']
    print_case_info(**problem_kwargs)

    # create SLB_helix
    rh1 = problem_kwargs['rh1']
    rh2 = problem_kwargs['rh2']
    ch = problem_kwargs['ch']
    center = problem_kwargs['center']
    ph = problem_kwargs['ph']
    n_segment = problem_kwargs['n_segment']
    n_tail = problem_kwargs['n_tail']
    # left_hand = problem_kwargs['left_hand']
    rel_Uh = problem_kwargs['rel_Uh']
    check_nth = problem_kwargs['check_nth']
    slb_geo_fun = problem_kwargs['slb_geo_fun']
    slb.check_matrix_method(matrix_method)

    i0, theta0 = 0, 0
    hlx1_geo = slb_geo_fun(ph, ch, rh1, rh2, theta0=theta0)
    hlx1_geo.create_nSegment(n_segment, check_nth=check_nth)
    hlx1_obj = sf.StokesFlowObj()
    obj_name = 'helix%d' % i0
    hlx1_obj.set_data(hlx1_geo, hlx1_geo, name=obj_name)
    # hlx1_obj.show_f_u_nodes()
    # assert 1==2
    # hlx1_obj.node_rotation(norm=np.array([0, 1, 0]), theta=hlx_ini_rot_theta)

    problem = slb.problem_dic[matrix_method](**problem_kwargs)
    problem.add_obj(hlx1_obj)
    problem.print_info()
    problem.create_matrix()
    At, Bt, Ct, ftr_info, frt_info = AtBtCt(problem, save_vtk=False, pick_M=False)
    PETSc.Sys.Print(At, Bt, Ct)
    return True

if __name__ == '__main__':
    OptDB = PETSc.Options()
    # pythonmpi helicoid.py -sm lg_rs -legendre_m 3 -legendre_k 2 -epsilon 3 -ffweight 2 -main_fun_noIter 1 -vortexStrength 1 -helicoid_r1 1 -helicoid_r2 0.3 -helicoid_ds 0.03
    # if OptDB.getBool('main_fun_noIter', False):
    #     OptDB.setValue('main_fun', False)
    #     main_fun_noIter()

    if OptDB.getBool('main_resistanceMatrix_SLB', False):
        OptDB.setValue('main_fun', False)
        main_resistanceMatrix_SLB()

    if OptDB.getBool('main_dbg', False):
        OptDB.setValue('main_fun', False)
        main_dbg()

    # if OptDB.getBool('main_fun', True):
    #     main_fun()

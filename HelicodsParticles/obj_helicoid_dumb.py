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
from codeStore.helix_common import *


def get_problem_kwargs(**main_kwargs):
    problem_kwargs = get_solver_kwargs()
    OptDB = PETSc.Options()
    fileHandle = OptDB.getString('f', 'obj_helicoid')
    OptDB.setValue('f', fileHandle)
    problem_kwargs['fileHandle'] = fileHandle
    dumb_d = OptDB.getReal('dumb_d', 5)
    problem_kwargs['dumb_d'] = dumb_d
    dumb_theta = OptDB.getReal('dumb_theta', np.pi / 3)
    problem_kwargs['dumb_theta'] = dumb_theta

    kwargs_list = (get_obj_helicoid_kwargs(), get_sphere_kwargs(),
                   get_forcefree_kwargs(), main_kwargs,)
    for t_kwargs in kwargs_list:
        for key in t_kwargs:
            problem_kwargs[key] = t_kwargs[key]
    return problem_kwargs


def print_case_info(**problem_kwargs):
    fileHandle = problem_kwargs['fileHandle']
    print_solver_info(**problem_kwargs)
    print_forcefree_info(**problem_kwargs)
    print_sphere_info(fileHandle, **problem_kwargs)
    dumb_d = problem_kwargs['dumb_d']
    dumb_theta = problem_kwargs['dumb_theta']
    PETSc.Sys.Print('  dumb_d: %f, dumb_theta: %f' % (dumb_d, dumb_theta))
    print_obj_helicoid_info(**problem_kwargs)
    if 'center_sphere_rs' in problem_kwargs.keys():
        center_sphere_rs = problem_kwargs['center_sphere_rs']
        center_sphere_ds = problem_kwargs['center_sphere_ds']
        PETSc.Sys.Print('  center_sphere_rs: %f, center_sphere_ds: %f' %
                        (center_sphere_rs, center_sphere_ds))
    return True


def main_resistanceMatrix_dumb(**main_kwargs):
    # OptDB = PETSc.Options()
    main_kwargs['zoom_factor'] = 1
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    matrix_method = problem_kwargs['matrix_method']
    assert '_selfRotate' not in matrix_method
    fileHandle = problem_kwargs['fileHandle']
    # pickProblem = problem_kwargs['pickProblem']
    print_case_info(**problem_kwargs)

    helicoid_comp = creat_helicoid_dumb(**problem_kwargs)
    helicoid_obj_list = np.array(helicoid_comp.get_obj_list())
    helicoid_center = helicoid_comp.get_center()
    # helicoid_comp.show_f_u_nodes()
    # assert 1 == 2

    problem = sf.problem_dic[matrix_method](**problem_kwargs)
    for tobj in helicoid_obj_list:
        problem.add_obj(tobj)
        # f_geo = tobj.get_f_geo()
        # PETSc.Sys.Print(f_geo.get_deltaLength())
    problem.print_info()
    problem.create_matrix()
    # PETSc.Sys.Print(problem.get_obj_list()[0].get_u_nodes()[:10])
    # PETSc.Sys.Print(problem.get_M()[:5, :5])
    # PETSc.Sys.Print(helicoid_center)
    At, Bt1, Bt2, Ct = AtBtCt_full(problem, save_vtk=False, pick_M=False, print_each=True,
                                   center=helicoid_center, save_name=fileHandle)
    PETSc.Sys.Print(np.trace(Bt1), np.trace(Bt2))
    return True


def main_resistanceMatrix_selfRotate(**main_kwargs):
    # OptDB = PETSc.Options()
    main_kwargs['zoom_factor'] = 1
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    matrix_method = problem_kwargs['matrix_method']
    fileHandle = problem_kwargs['fileHandle']
    # pickProblem = problem_kwargs['pickProblem']
    print_case_info(**problem_kwargs)

    helicoid_comp = creat_helicoid_dumb_selfRotate(**problem_kwargs)
    helicoid_obj_list = np.array(helicoid_comp.get_obj_list())
    helicoid_center = helicoid_comp.get_center()
    helicoid_norm = helicoid_comp.get_norm()
    problem_kwargs['problem_center'] = helicoid_center
    problem_kwargs['problem_norm'] = helicoid_norm
    problem_kwargs['problem_n_copy'] = problem_kwargs['helicoid_ndsk_each']

    problem = sf.problem_dic[matrix_method](**problem_kwargs)
    for tobj in helicoid_obj_list:
        problem.add_obj(tobj)
    problem.print_info()
    # problem.show_u_nodes(linestyle='')
    # problem.show_all_u_nodes(linestyle='')
    # assert 1 == 2
    problem.create_matrix()
    AtBtCt_selfRotate(problem, save_vtk=False, pick_M=False,
                      center=helicoid_center, save_name=fileHandle)

    # # 1. translation
    # problem.set_rigid_velocity(1, 0)
    # problem.create_F_U()
    # problem.solve()
    # total_ft = np.sum([t_obj.get_total_force(center=helicoid_center)
    #                    for t_obj in problem.get_obj_list()], axis=0)
    # PETSc.Sys.Print(total_ft)
    # # 2. rotation
    # problem.set_rigid_velocity(0, 1)
    # problem.create_F_U()
    # problem.solve()
    # total_ft = np.sum([t_obj.get_total_force(center=helicoid_center)
    #                    for t_obj in problem.get_obj_list()], axis=0)
    # PETSc.Sys.Print(total_ft)
    return True


def main_resistanceMatrix_dumb_sphere(**main_kwargs):
    OptDB = PETSc.Options()
    center_sphere_rs = OptDB.getReal('center_sphere_rs', 0.1)
    center_sphere_ds = OptDB.getReal('center_sphere_ds', 0.02)
    main_kwargs['center_sphere_rs'] = center_sphere_rs
    main_kwargs['center_sphere_ds'] = center_sphere_ds
    main_kwargs['zoom_factor'] = 1
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    matrix_method = problem_kwargs['matrix_method']
    fileHandle = problem_kwargs['fileHandle']
    helicoid_r = problem_kwargs['helicoid_r']
    rs = problem_kwargs['rs']
    # pickProblem = problem_kwargs['pickProblem']
    print_case_info(**problem_kwargs)
    assert center_sphere_rs < (helicoid_r - rs)

    helicoid_comp = creat_helicoid_dumb(**problem_kwargs)
    helicoid_obj_list = np.array(helicoid_comp.get_obj_list())
    helicoid_center = helicoid_comp.get_center()
    # helicoid_comp.show_f_u_nodes()
    # assert 1 == 2
    center_geo = sphere_geo()
    center_geo.create_delta(center_sphere_ds, center_sphere_rs)
    center_geo.move(helicoid_center - center_geo.get_center())
    center_obj = sf.StokesFlowObj()
    center_obj.set_data(center_geo, center_geo, 'helicoid_center_sphere')

    problem = sf.problem_dic[matrix_method](**problem_kwargs)
    for tobj in helicoid_obj_list:
        problem.add_obj(tobj)
        # f_geo = tobj.get_f_geo()
        # PETSc.Sys.Print(f_geo.get_deltaLength())
    problem.add_obj(center_obj)
    problem.print_info()
    problem.create_matrix()
    # PETSc.Sys.Print(problem.get_obj_list()[0].get_u_nodes()[:10])
    # PETSc.Sys.Print(problem.get_M()[:5, :5])
    # PETSc.Sys.Print(helicoid_center)
    # u_use, w_use = 1, 1
    u_use, w_use = 1, 1 / helicoid_r
    At, Bt1, Bt2, Ct = AtBtCt_full(problem, save_vtk=False, pick_M=False, print_each=False,
                                   center=helicoid_center, save_name=fileHandle,
                                   u_use=u_use, w_use=w_use)
    PETSc.Sys.Print(np.trace(Bt1), np.trace(Bt2))
    return True


if __name__ == '__main__':
    # code resluts are wrong.
    OptDB = PETSc.Options()
    # pythonmpi helicoid.py -sm lg_rs -legendre_m 3 -legendre_k 2 -epsilon 3 -ffweight 2 -main_fun_noIter 1 -vortexStrength 1 -helicoid_r1 1 -helicoid_r2 0.3 -helicoid_ds 0.03
    # if OptDB.getBool('main_fun_noIter', False):
    #     OptDB.setValue('main_fun', False)
    #     main_fun_noIter()

    matrix_method = OptDB.getString('sm', 'pf')
    if OptDB.getBool('main_resistanceMatrix_dumb', False):
        assert '_selfRotate' not in matrix_method
        OptDB.setValue('main_fun', False)
        main_resistanceMatrix_dumb()


    if OptDB.getBool('main_resistanceMatrix_dumb_sphere', False):
        assert '_selfRotate' not in matrix_method
        OptDB.setValue('main_fun', False)
        main_resistanceMatrix_dumb_sphere()

    if OptDB.getBool('main_resistanceMatrix_selfRotate', False):
        assert '_selfRotate' in matrix_method
        OptDB.setValue('main_fun', False)
        main_resistanceMatrix_selfRotate()
    # if OptDB.getBool('main_fun', True):
    #     main_fun()

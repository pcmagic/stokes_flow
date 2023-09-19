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
from codeStore.helix_common import *


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
    # code results are wrong.
    assert 1 == 2

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

    # tobj_list = uniqueList()
    # for i0, theta0 in enumerate(np.linspace(0, 2 * np.pi, n_tail, endpoint=False)):
    #     hlx1_geo = slb_geo_fun(ph, ch, rh1, rh2, theta0=theta0)
    #     hlx1_geo.create_nSegment(n_segment, check_nth=check_nth)
    #     hlx1_obj = sf.StokesFlowObj()
    #     obj_name = 'helix%d' % i0
    #     hlx1_obj.set_data(hlx1_geo, hlx1_geo, name=obj_name)
    #     tobj_list.append(hlx1_obj)
    err_msg = 'current version, n_tail==1'
    assert n_tail == 1, err_msg
    hlx1_geo = slb_geo_fun(ph, ch, rh1, rh2, theta0=0)
    hlx1_geo.create_nSegment(n_segment, check_nth=check_nth)
    hlx1_obj = sf.StokesFlowObj()
    hlx1_obj.set_data(hlx1_geo, hlx1_geo, name='helix0')
    helicoid_comp = obj2helicoid_comp(hlx1_obj, **problem_kwargs)
    # helicoid_obj_list = np.array(helicoid_comp.get_obj_list())[[0, 3, 6, 9]]
    helicoid_obj_list = np.array(helicoid_comp.get_obj_list())
    helicoid_center = helicoid_comp.get_center()

    problem = slb.problem_dic[matrix_method](**problem_kwargs)
    for tobj in helicoid_obj_list:
        problem.add_obj(tobj)
    problem.print_info()
    problem.create_matrix()
    # PETSc.Sys.Print(problem.get_obj_list()[0].get_u_nodes()[:10])
    # PETSc.Sys.Print(problem.get_M()[:5, :5])
    # PETSc.Sys.Print(helicoid_center)
    AtBtCt_full(problem, save_vtk=False, pick_M=False, print_each=False,
                center=helicoid_center)
    return True


def main_resistanceMatrix_part(**main_kwargs):
    OptDB = PETSc.Options()
    main_kwargs['zoom_factor'] = 1
    save_vtk = OptDB.getBool('save_vtk', False)
    main_kwargs['save_vtk'] = save_vtk

    problem_kwargs = get_problem_kwargs(**main_kwargs)
    matrix_method = problem_kwargs['matrix_method']
    fileHandle = problem_kwargs['fileHandle']
    # pickProblem = problem_kwargs['pickProblem']
    print_case_info(**problem_kwargs)

    # create SLB_helix
    rh1 = problem_kwargs['rh1']
    rh2 = problem_kwargs['rh2']
    ch = problem_kwargs['ch']
    center = problem_kwargs['center']
    ph = problem_kwargs['ph']
    hlx_theta0 = problem_kwargs['hlx_theta0']
    n_segment = problem_kwargs['n_segment']
    n_tail = problem_kwargs['n_tail']
    # left_hand = problem_kwargs['left_hand']
    # rel_Uh = problem_kwargs['rel_Uh']
    check_nth = problem_kwargs['check_nth']
    slb_geo_fun = problem_kwargs['slb_geo_fun']
    slb.check_matrix_method(matrix_method)

    err_msg = 'current version, n_tail==1'
    assert n_tail == 1, err_msg
    hlx1_geo = slb_geo_fun(ph, ch, rh1, rh2, theta0=hlx_theta0)
    hlx1_geo.create_nSegment(n_segment, check_nth=check_nth)
    hlx1_obj = slb.obj_dic[matrix_method]()
    hlx1_obj.set_data(hlx1_geo, hlx1_geo, name='helix0')

    # solve
    problem = slb.problem_dic[matrix_method](**problem_kwargs)
    problem.add_obj(hlx1_obj)
    problem.print_info()
    problem.create_matrix()
    At, Bt1, Bt2, Ct = AtBtCt_pickInfo(problem=problem, pick_M=False, save_vtk=save_vtk,
                                       center=center, print_each=False, save_name=fileHandle)
    # At, Bt, Ct, ftr_info, frt_info = AtBtCt_SBT(problem=problem, pick_M=False, save_vtk=save_vtk,
    #                                             center=center, print_each=False)
    PETSc.Sys.Print('Tr(A)=%f, Tr(B1)=%f, Tr(B2)=%f, ' %
                    (np.trace(At), np.trace(Bt1), np.trace(Bt2)))
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
    problem = slb.problem_dic[matrix_method](**problem_kwargs)

    for i0, theta0 in enumerate(np.linspace(0, 2 * np.pi, n_tail, endpoint=False)):
        hlx1_geo = slb_geo_fun(ph, ch, rh1, rh2, theta0=theta0)
        hlx1_geo.create_nSegment(n_segment, check_nth=check_nth)
        hlx1_obj = sf.StokesFlowObj()
        obj_name = 'helix%d' % i0
        hlx1_obj.set_data(hlx1_geo, hlx1_geo, name=obj_name)
        problem.add_obj(hlx1_obj)

    # for tobj in problem.get_obj_list():
    #     tobj.node_rotation(norm=np.array((1, 0, 0)), theta=np.pi / 2)
    problem.print_info()
    problem.create_matrix()
    # PETSc.Sys.Print(problem.get_obj_list()[0].get_u_nodes()[:10])
    # PETSc.Sys.Print(problem.get_M()[:5, :5])
    AtBtCt_full(problem, save_vtk=False, pick_M=False, print_each=False,
                center=np.array((-1, 0, 0)))
    return True


if __name__ == '__main__':
    OptDB = PETSc.Options()
    if OptDB.getBool('main_resistanceMatrix_SLB', False):
        OptDB.setValue('main_fun', False)
        main_resistanceMatrix_SLB()

    if OptDB.getBool('main_resistanceMatrix_part', False):
        OptDB.setValue('main_fun', False)
        main_resistanceMatrix_part()

    if OptDB.getBool('main_dbg', False):
        OptDB.setValue('main_fun', False)
        main_dbg()

    # if OptDB.getBool('main_fun', True):
    #     main_fun()

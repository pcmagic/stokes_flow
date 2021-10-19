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
# from src import slender_body as slb
from codeStore.helix_common import *


def get_problem_kwargs(**main_kwargs):
    problem_kwargs = get_solver_kwargs()
    OptDB = PETSc.Options()
    fileHandle = OptDB.getString('f', 'obj_helicoid')
    OptDB.setValue('f', fileHandle)
    problem_kwargs['fileHandle'] = fileHandle

    kwargs_list = (get_obj_helicoid_kwargs(), get_helix_kwargs(),
                   get_forcefree_kwargs(), main_kwargs,)
    for t_kwargs in kwargs_list:
        for key in t_kwargs:
            problem_kwargs[key] = t_kwargs[key]
    return problem_kwargs


def print_case_info(**problem_kwargs):
    fileHandle = problem_kwargs['fileHandle']
    print_solver_info(**problem_kwargs)
    print_forcefree_info(**problem_kwargs)
    print_helix_info(fileHandle, **problem_kwargs)
    print_obj_helicoid_info(**problem_kwargs)
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


def create_helicoid_hlx_comp(**problem_kwargs):
    tail_obj_list = create_ecoli_tail(moveh=np.zeros(3), **problem_kwargs)
    tobj = sf.obj_dic[matrix_method]()
    tobj.combine(tail_obj_list)
    tobj.set_name('helicoid_hlx')
    # tobj.node_rotation(norm=np.array([1, 0, 0]), theta=th_loc)

    helicoid_list = obj2helicoid_list_v3(tobj, **problem_kwargs)
    helicoid_comp = sf.ForceFreeComposite(center=np.zeros(3), norm=np.array((1, 0, 0)),
                                          name='helicoid_comp')
    for tobj in helicoid_list:
        helicoid_comp.add_obj(obj=tobj, rel_U=np.zeros(6))
    # helicoid_comp.set_update_para(fix_x=False, fix_y=False, fix_z=False,
    #                               update_fun=update_fun, update_order=update_order)
    return helicoid_comp


def create_helicoid_hlx_selfRotate(**problem_kwargs):
    tail_obj_list = create_ecoli_tail(moveh=np.zeros(3), **problem_kwargs)
    tobj = sf.obj_dic[matrix_method]()
    tobj.combine(tail_obj_list)
    tobj.set_name('helicoid_hlx_selfRotate')
    # tobj.node_rotation(norm=np.array([1, 0, 0]), theta=th_loc)

    helicoid_list = obj2helicoid_list_selfRotate(tobj, **problem_kwargs)
    helicoid_comp = sf.ForceFreeComposite(center=np.zeros(3), norm=np.array((0, 0, 1)),
                                          name='helicoid_comp')
    for tobj in helicoid_list:
        helicoid_comp.add_obj(obj=tobj, rel_U=np.zeros(6))
    # helicoid_comp.set_update_para(fix_x=False, fix_y=False, fix_z=False,
    #                               update_fun=update_fun, update_order=update_order)
    return helicoid_comp


def main_resistanceMatrix_hlx_old(**main_kwargs):
    # OptDB = PETSc.Options()
    main_kwargs['zoom_factor'] = 1
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    matrix_method = problem_kwargs['matrix_method']
    fileHandle = problem_kwargs['fileHandle']
    pickProblem = problem_kwargs['pickProblem']
    print_case_info(**problem_kwargs)

    # create helicoid
    # # dbg, sub_geos are disk.
    # namehandle = 'helicoid'
    # r2 = 0.3
    # ds = 0.03
    # th_loc = 0.7853981633974483
    # tgeo = regularizeDisk()
    # tgeo.create_ds(ds, r2)
    # tgeo.node_rotation(norm=np.array([1, 0, 0]), theta=th_loc)
    # tobj = sf.StokesFlowObj()
    # tobj.set_matrix_method(matrix_method)  # the geo is regularizeDisk
    # tobj.set_data(f_geo=tgeo, u_geo=tgeo, name=namehandle)
    # helicoid_comp = obj2helicoid_comp(tobj, **problem_kwargs)
    # # helicoid_comp.show_u_nodes(linestyle='')
    # # assert 1 == 2

    tail_obj_list = create_ecoli_tail(moveh=np.zeros(3), **problem_kwargs)
    tobj = sf.StokesFlowObj()
    tobj.combine(tail_obj_list)
    tobj.set_name('helicoid_hlx')
    # tobj.node_rotation(norm=np.array([1, 0, 0]), theta=th_loc)
    helicoid_comp = obj2helicoid_comp(tobj, **problem_kwargs)
    # helicoid_comp.show_u_nodes(linestyle='')
    # assert 1 == 2
    helicoid_obj_list = helicoid_comp.get_obj_list()
    helicoid_center = helicoid_comp.get_center()

    # solve
    problem = sf.StokesFlowProblem(**problem_kwargs)
    for tobj in helicoid_obj_list:
        problem.add_obj(tobj)
    if pickProblem:
        problem.pickmyself('%s_tran' % fileHandle, ifcheck=True)
    problem.print_info()
    problem.create_matrix()

    # 1. translation
    for tobj in helicoid_obj_list:
        tobj.set_rigid_velocity(np.array((0, 0, 1, 0, 0, 0)), center=helicoid_center)
    problem.create_F_U()
    problem.solve()
    if problem_kwargs['pickProblem']:
        problem.pickmyself('%s_tran' % fileHandle, pick_M=False, mat_destroy=False)
    total_force = problem.get_total_force()
    PETSc.Sys.Print('translation total_force', total_force)
    # problem.vtk_self('%s_tran' % fileHandle)

    # 2. rotation
    for tobj in helicoid_obj_list:
        tobj.set_rigid_velocity(np.array((0, 0, 0, 0, 0, 1)), center=helicoid_center)
    problem.create_F_U()
    problem.solve()
    if problem_kwargs['pickProblem']:
        problem.pickmyself('%s_rota' % fileHandle, pick_M=False, mat_destroy=False)
    total_force = problem.get_total_force()
    PETSc.Sys.Print('rotation total_force', total_force)
    # problem.vtk_self('%s_rota' % fileHandle)

    # 1. translation
    for tobj in helicoid_obj_list:
        tobj.set_rigid_velocity(np.array((0, 1, 0, 0, 0, 0)), center=helicoid_center)
    problem.create_F_U()
    problem.solve()
    if problem_kwargs['pickProblem']:
        problem.pickmyself('%s_tran' % fileHandle, pick_M=False, mat_destroy=False)
    total_force = problem.get_total_force()
    PETSc.Sys.Print('translation total_force', total_force)
    # problem.vtk_self('%s_tran' % fileHandle)

    # 2. rotation
    for tobj in helicoid_obj_list:
        tobj.set_rigid_velocity(np.array((0, 0, 0, 0, 1, 0)), center=helicoid_center)
    problem.create_F_U()
    problem.solve()
    if problem_kwargs['pickProblem']:
        problem.pickmyself('%s_rota' % fileHandle, pick_M=False, mat_destroy=False)
    total_force = problem.get_total_force()
    PETSc.Sys.Print('rotation total_force', total_force)
    # problem.vtk_self('%s_rota' % fileHandle)

    # 1. translation
    for tobj in helicoid_obj_list:
        tobj.set_rigid_velocity(np.array((1, 0, 0, 0, 0, 0)), center=helicoid_center)
    problem.create_F_U()
    problem.solve()
    if problem_kwargs['pickProblem']:
        problem.pickmyself('%s_tran' % fileHandle, pick_M=False, mat_destroy=False)
    total_force = problem.get_total_force()
    PETSc.Sys.Print('translation total_force', total_force)
    # problem.vtk_self('%s_tran' % fileHandle)

    # 2. rotation
    for tobj in helicoid_obj_list:
        tobj.set_rigid_velocity(np.array((0, 0, 0, 1, 0, 0)), center=helicoid_center)
    problem.create_F_U()
    problem.solve()
    if problem_kwargs['pickProblem']:
        problem.pickmyself('%s_rota' % fileHandle, pick_M=False, mat_destroy=False)
    total_force = problem.get_total_force()
    PETSc.Sys.Print('rotation total_force', total_force)
    # problem.vtk_self('%s_rota' % fileHandle)
    return True


def main_resistanceMatrix_hlx(**main_kwargs):
    # OptDB = PETSc.Options()
    main_kwargs['zoom_factor'] = 1
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    matrix_method = problem_kwargs['matrix_method']
    fileHandle = problem_kwargs['fileHandle']
    pickProblem = problem_kwargs['pickProblem']
    print_case_info(**problem_kwargs)

    helicoid_comp = create_helicoid_hlx_comp(**problem_kwargs)
    # helicoid_comp.show_u_nodes(linestyle='')
    # assert 1 == 2
    helicoid_obj_list = helicoid_comp.get_obj_list()
    helicoid_center = helicoid_comp.get_center()

    # solve
    problem = sf.StokesFlowProblem(**problem_kwargs)
    for tobj in helicoid_obj_list:
        problem.add_obj(tobj)
    if pickProblem:
        problem.pickmyself('%s_tran' % fileHandle, ifcheck=True)
    problem.print_info()
    problem.create_matrix()
    AtBtCt_full(problem, save_vtk=False, pick_M=False,
                center=helicoid_center, save_name=fileHandle)
    return True


def main_resistanceMatrix_part(**main_kwargs):
    OptDB = PETSc.Options()
    main_kwargs['zoom_factor'] = 1
    # helicoid_part_idx = OptDB.getInt('helicoid_part_idx', 0)
    # main_kwargs['helicoid_part_idx'] = helicoid_part_idx
    save_vtk = OptDB.getBool('save_vtk', False)
    main_kwargs['save_vtk'] = save_vtk

    problem_kwargs = get_problem_kwargs(**main_kwargs)
    # matrix_method = problem_kwargs['matrix_method']
    fileHandle = problem_kwargs['fileHandle']
    pickProblem = problem_kwargs['pickProblem']
    center = problem_kwargs['center']
    # helicoid_part_idx = problem_kwargs['helicoid_part_idx']
    print_case_info(**problem_kwargs)

    # helicoid_comp = create_helicoid_hlx_comp(**problem_kwargs)
    # # helicoid_comp.show_u_nodes(linestyle='')
    # # assert 1 == 2
    # helicoid_obj_list = helicoid_comp.get_obj_list()
    # # helicoid_center = helicoid_comp.get_center()
    # tobj = helicoid_obj_list[helicoid_part_idx]
    # tobj_center = tobj.get_u_geo().get_center()
    # # PETSc.Sys.Print(tobj.get_u_geo().get_center())
    # # PETSc.Sys.Print(helicoid_center)

    tail_obj_list = create_ecoli_tail(moveh=np.zeros(3), **problem_kwargs)

    # solve
    problem = sf.StokesFlowProblem(**problem_kwargs)
    for tobj in tail_obj_list:
        problem.add_obj(tobj)
    if pickProblem:
        problem.pickmyself('%s_tran' % fileHandle, ifcheck=True)
    problem.print_info()
    problem.create_matrix()
    # At, Bt1, Bt2, Ct = AtBtCt_full(problem, save_vtk=save_vtk, pick_M=False, print_each=False,
    #                                center=tobj_center, save_name=fileHandle)
    At, Bt1, Bt2, Ct = AtBtCt_pickInfo(problem=problem, pick_M=False, save_vtk=save_vtk,
                                       center=center, print_each=False, save_name=fileHandle)
    PETSc.Sys.Print('Tr(A)=%f, Tr(B1)=%f, Tr(B2)=%f, ' %
                    (np.trace(At), np.trace(Bt1), np.trace(Bt2)))
    return True


def main_resistanceMatrix_selfRotate(**main_kwargs):
    # OptDB = PETSc.Options()
    main_kwargs['zoom_factor'] = 1
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    matrix_method = problem_kwargs['matrix_method']
    fileHandle = problem_kwargs['fileHandle']
    pickProblem = problem_kwargs['pickProblem']
    print_case_info(**problem_kwargs)

    helicoid_comp = create_helicoid_hlx_selfRotate(**problem_kwargs)
    # helicoid_comp.show_u_nodes(linestyle='')
    # assert 1 == 2
    helicoid_obj_list = helicoid_comp.get_obj_list()
    helicoid_center = helicoid_comp.get_center()
    helicoid_norm = helicoid_comp.get_norm()
    problem_kwargs['problem_center'] = helicoid_center
    problem_kwargs['problem_norm'] = helicoid_norm
    problem_kwargs['problem_n_copy'] = problem_kwargs['helicoid_ndsk_each']

    problem = sf.problem_dic[matrix_method](**problem_kwargs)
    for tobj in helicoid_obj_list:
        problem.add_obj(tobj)
    # problem.show_all_u_nodes(linestyle='')
    problem.print_info()
    problem.create_matrix()
    AtBtCt_selfRotate(problem, save_vtk=False, pick_M=False,
                      center=helicoid_center, save_name=fileHandle)
    return True


def main_fun_E(**main_kwargs):
    OptDB = PETSc.Options()
    fileHandle = OptDB.getString('f', 'helicoid_strain_rate')
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
    problem_kwargs['zoom_factor'] = 1

    if not problem_kwargs['restart']:
        print_case_info(**problem_kwargs)
        helicoid_comp = create_helicoid_hlx_comp(namehandle='helicoid', **problem_kwargs)

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


if __name__ == '__main__':
    OptDB = PETSc.Options()
    # pythonmpi helicoid.py -sm lg_rs -legendre_m 3 -legendre_k 2 -epsilon 3 -ffweight 2 -main_fun_noIter 1 -vortexStrength 1 -helicoid_r1 1 -helicoid_r2 0.3 -helicoid_ds 0.03
    # if OptDB.getBool('main_fun_noIter', False):
    #     OptDB.setValue('main_fun', False)
    #     main_fun_noIter()

    matrix_method = OptDB.getString('sm', 'pf')
    if OptDB.getBool('main_fun_E', False):
        assert '_selfRotate' not in matrix_method
        OptDB.setValue('main_fun', False)
        main_fun_E()

    matrix_method = OptDB.getString('sm', 'pf')
    if OptDB.getBool('main_resistanceMatrix_hlx', False):
        assert '_selfRotate' not in matrix_method
        OptDB.setValue('main_fun', False)
        main_resistanceMatrix_hlx()

    if OptDB.getBool('main_resistanceMatrix_part', False):
        assert '_selfRotate' not in matrix_method
        OptDB.setValue('main_fun', False)
        main_resistanceMatrix_part()

    if OptDB.getBool('main_resistanceMatrix_selfRotate', False):
        assert '_selfRotate' in matrix_method
        OptDB.setValue('main_fun', False)
        main_resistanceMatrix_selfRotate()

    # if OptDB.getBool('main_fun', True):
    #     main_fun()

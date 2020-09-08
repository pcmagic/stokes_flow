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
from codeStore.helix_common import AtBtCt_full


def get_problem_kwargs(**main_kwargs):
    problem_kwargs = get_solver_kwargs()
    OptDB = PETSc.Options()
    fileHandle = OptDB.getString('f', 'obj_helicoid')
    OptDB.setValue('f', fileHandle)
    problem_kwargs['fileHandle'] = fileHandle

    kwargs_list = (get_helicoid_kwargs(), get_obj_helicoid_kwargs(),
                   get_forcefree_kwargs(), main_kwargs,)
    for t_kwargs in kwargs_list:
        for key in t_kwargs:
            problem_kwargs[key] = t_kwargs[key]
    return problem_kwargs


def print_case_info(**problem_kwargs):
    fileHandle = problem_kwargs['fileHandle']
    print_solver_info(**problem_kwargs)
    print_forcefree_info(**problem_kwargs)
    print_helicoid_info(**problem_kwargs)
    print_obj_helicoid_info(**problem_kwargs)
    return True


def create_helicoid_dsk_comp(**problem_kwargs):
    r2 = problem_kwargs['helicoid_r2']
    ds = problem_kwargs['helicoid_ds']
    th_loc = problem_kwargs['helicoid_th_loc']
    namehandle = 'helicoid'

    tgeo = regularizeDisk()
    tgeo.create_ds(ds, r2)
    tgeo.node_rotation(norm=np.array([1, 0, 0]), theta=th_loc)
    tobj = sf.StokesFlowObj()
    tobj.set_data(f_geo=tgeo, u_geo=tgeo, name=namehandle)

    helicoid_list = obj2helicoid_list_v2(tobj, **problem_kwargs)
    # helicoid_list = obj2helicoid_list(tobj, **problem_kwargs)
    helicoid_comp = sf.ForceFreeComposite(center=np.zeros(3), norm=np.array((1, 0, 0)),
                                          name='helicoid_comp')
    for tobj in helicoid_list:
        helicoid_comp.add_obj(obj=tobj, rel_U=np.zeros(6))
    # helicoid_comp.set_update_para(fix_x=False, fix_y=False, fix_z=False,
    #                               update_fun=update_fun, update_order=update_order)
    return helicoid_comp


def main_resistanceMatrix(**main_kwargs):
    # OptDB = PETSc.Options()
    main_kwargs['zoom_factor'] = 1
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    matrix_method = problem_kwargs['matrix_method']
    fileHandle = problem_kwargs['fileHandle']
    pickProblem = problem_kwargs['pickProblem']
    print_case_info(**problem_kwargs)

    helicoid_comp = create_helicoid_dsk_comp(**problem_kwargs)
    # helicoid_comp = create_helicoid_comp(**problem_kwargs)
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
    AtBtCt_full(problem, save_vtk=False, pick_M=False, print_each=False,
                center=helicoid_center, save_name=fileHandle)
    return True


def main_resistanceMatrix_part(**main_kwargs):
    OptDB = PETSc.Options()
    main_kwargs['zoom_factor'] = 1
    helicoid_part_idx = OptDB.getInt('helicoid_part_idx', 0)
    main_kwargs['helicoid_part_idx'] = helicoid_part_idx

    problem_kwargs = get_problem_kwargs(**main_kwargs)
    matrix_method = problem_kwargs['matrix_method']
    fileHandle = problem_kwargs['fileHandle']
    pickProblem = problem_kwargs['pickProblem']
    helicoid_part_idx = problem_kwargs['helicoid_part_idx']
    print_case_info(**problem_kwargs)

    helicoid_comp = create_helicoid_dsk_comp(**problem_kwargs)
    # helicoid_comp.show_u_nodes(linestyle='')
    # assert 1 == 2
    helicoid_obj_list = helicoid_comp.get_obj_list()
    helicoid_center = helicoid_comp.get_center()
    tobj = helicoid_obj_list[helicoid_part_idx]
    # PETSc.Sys.Print(tobj.get_u_geo().get_center())
    # PETSc.Sys.Print(helicoid_center)

    # solve
    problem = sf.StokesFlowProblem(**problem_kwargs)
    problem.add_obj(tobj)
    if pickProblem:
        problem.pickmyself('%s_tran' % fileHandle, ifcheck=True)
    problem.print_info()
    problem.create_matrix()
    AtBtCt_full(problem, save_vtk=False, pick_M=False, print_each=False,
                center=helicoid_center, save_name=fileHandle)
    return True


if __name__ == '__main__':
    OptDB = PETSc.Options()
    # pythonmpi helicoid.py -sm lg_rs -legendre_m 3 -legendre_k 2 -epsilon 3 -ffweight 2 -main_fun_noIter 1 -vortexStrength 1 -helicoid_r1 1 -helicoid_r2 0.3 -helicoid_ds 0.03
    # if OptDB.getBool('main_fun_noIter', False):
    #     OptDB.setValue('main_fun', False)
    #     main_fun_noIter()

    if OptDB.getBool('main_resistanceMatrix', False):
        OptDB.setValue('main_fun', False)
        main_resistanceMatrix()

    if OptDB.getBool('main_resistanceMatrix_part', False):
        OptDB.setValue('main_fun', False)
        main_resistanceMatrix_part()

    # if OptDB.getBool('main_fun', True):
    #     main_fun()

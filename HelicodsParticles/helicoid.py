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
from codeStore.helix_common import AtBtCt_full


def get_problem_kwargs(**main_kwargs):
    problem_kwargs = get_solver_kwargs()
    OptDB = PETSc.Options()
    fileHandle = OptDB.getString('f', 'helicoid')
    OptDB.setValue('f', fileHandle)
    problem_kwargs['fileHandle'] = fileHandle

    kwargs_list = (get_shearFlow_kwargs(), get_freeVortex_kwargs(), get_helicoid_kwargs(),
                   get_forcefree_kwargs(),
                   main_kwargs,)
    for t_kwargs in kwargs_list:
        for key in t_kwargs:
            problem_kwargs[key] = t_kwargs[key]
    return problem_kwargs


def print_case_info(**problem_kwargs):
    fileHandle = problem_kwargs['fileHandle']
    print_solver_info(**problem_kwargs)
    print_forcefree_info(**problem_kwargs)
    print_shearFlow_info(**problem_kwargs)
    print_freeVortex_info(**problem_kwargs)
    print_helicoid_info(**problem_kwargs)
    return True


def main_fun_noIter(**main_kwargs):
    # OptDB = PETSc.Options()
    main_kwargs['zoom_factor'] = 1
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    matrix_method = problem_kwargs['matrix_method']
    fileHandle = problem_kwargs['fileHandle']
    print_case_info(**problem_kwargs)

    helicoid_comp = create_helicoid_comp(namehandle='helicoid', **problem_kwargs)
    # helicoid_comp.show_u_nodes()

    problem_ff = sf.LambOseenVortexForceFreeProblem(**problem_kwargs)
    problem_ff.add_obj(helicoid_comp)
    problem_ff.print_info()
    problem_ff.create_matrix()
    problem_ff.solve()
    ref_U = helicoid_comp.get_ref_U()
    PETSc.Sys.Print('  ref_U in Lamb–Oseen vortex', ref_U)
    # spheroid_comp_F = helicoid_comp.get_total_force()
    # spheroid0_F = spheroid0.get_total_force(center=np.zeros(3))
    # # spheroid0_F = tail_list[0].get_total_force(center=np.zeros(3))
    # PETSc.Sys.Print('  spheroid_comp_F %s' % str(spheroid_comp_F))
    # PETSc.Sys.Print('  spheroid0_F %s' % str(spheroid0_F))
    # PETSc.Sys.Print('  non dimensional (F, T) err %s' % str(spheroid_comp_F / spheroid0_F))
    return True


def main_fun(**main_kwargs):
    # OptDB = PETSc.Options()
    main_kwargs['zoom_factor'] = 1
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    matrix_method = problem_kwargs['matrix_method']
    fileHandle = problem_kwargs['fileHandle']
    print_case_info(**problem_kwargs)

    helicoid_comp = create_helicoid_comp(namehandle='helicoid', **problem_kwargs)
    # helicoid_comp.show_u_nodes()

    problem = sf.LambOseenVortexForceFreeIterateProblem(**problem_kwargs)
    problem.add_obj(helicoid_comp)
    problem.set_iterate_comp(helicoid_comp)
    problem.print_info()
    problem.create_matrix()
    problem.do_iterate3(rtol=1e-10, atol=1e-20)
    ref_U = helicoid_comp.get_ref_U()
    PETSc.Sys.Print('  final ref_U in Lamb–Oseen vortex', ref_U)
    return True


def main_resistanceMatrix(**main_kwargs):
    # OptDB = PETSc.Options()
    main_kwargs['zoom_factor'] = 1
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    matrix_method = problem_kwargs['matrix_method']
    fileHandle = problem_kwargs['fileHandle']
    pickProblem = problem_kwargs['pickProblem']
    print_case_info(**problem_kwargs)

    helicoid_comp = create_helicoid_comp(namehandle='helicoid', **problem_kwargs)
    # helicoid_comp.show_u_nodes()
    helicoid_obj_list = helicoid_comp.get_obj_list()
    helicoid_center = helicoid_comp.get_center()

    problem = sf.StokesFlowProblem(**problem_kwargs)
    for tobj in helicoid_obj_list:
        problem.add_obj(tobj)
    if pickProblem:
        problem.pickmyself('%s_tran' % fileHandle, ifcheck=True)
    problem.print_info()
    problem.create_matrix()
    AtBtCt_full(problem, save_vtk=False, pick_M=False,
                center=helicoid_center, save_name=fileHandle)

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

    problem.vtk_self('%s_rota' % fileHandle)

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


if __name__ == '__main__':
    OptDB = PETSc.Options()
    # pythonmpi helicoid.py -sm lg_rs -legendre_m 3 -legendre_k 2 -epsilon 3 -ffweight 2 -main_fun_noIter 1 -vortexStrength 1 -helicoid_r1 1 -helicoid_r2 0.3 -helicoid_ds 0.03
    if OptDB.getBool('main_fun_noIter', False):
        OptDB.setValue('main_fun', False)
        main_fun_noIter()

    if OptDB.getBool('main_resistanceMatrix', False):
        OptDB.setValue('main_fun', False)
        main_resistanceMatrix()

    if OptDB.getBool('main_fun', True):
        main_fun()

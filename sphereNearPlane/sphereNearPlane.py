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
    fileHandle = OptDB.getString('f', 'sphereNearPlane')
    OptDB.setValue('f', fileHandle)
    problem_kwargs['fileHandle'] = fileHandle

    kwargs_list = (get_sphere_kwargs(), get_forcefree_kwargs(), main_kwargs,)
    for t_kwargs in kwargs_list:
        for key in t_kwargs:
            problem_kwargs[key] = t_kwargs[key]

    err_msg = 'current version only accept single sphere. '
    assert problem_kwargs['sphere_coord'].size == 3, err_msg
    h2Plane = OptDB.getReal('h2Plane', 100)
    problem_kwargs['sphere_coord'] = problem_kwargs['sphere_coord'] + (0, 0, h2Plane)
    return problem_kwargs


def get_twoSphereProblem_kwargs(**main_kwargs):
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    rs = problem_kwargs['rs']
    move_x = OptDB.getReal('move_x', rs * 2)
    move_y = OptDB.getReal('move_y', 0)
    move_z = OptDB.getReal('move_z', 0)
    move_r = np.linalg.norm(np.array((move_x, move_y, move_z)))
    err_msg = 'move_r > rs'
    assert move_r > rs, err_msg
    sphere_coord = problem_kwargs['sphere_coord'][0]
    sphere_coord0 = sphere_coord - (move_x, move_y, move_z)
    sphere_coord1 = sphere_coord + (move_x, move_y, move_z)
    problem_kwargs['sphere_coord'] = np.vstack((sphere_coord0, sphere_coord1))
    sphere_velocity = problem_kwargs['sphere_velocity'][0]
    problem_kwargs['sphere_velocity'] = np.vstack((sphere_velocity, sphere_velocity))
    return problem_kwargs


def print_case_info(**problem_kwargs):
    fileHandle = problem_kwargs['fileHandle']
    print_solver_info(**problem_kwargs)
    print_forcefree_info(**problem_kwargs)
    print_sphere_info(fileHandle, **problem_kwargs)
    return True


def main_resistanceMatrix(**main_kwargs):
    # OptDB = PETSc.Options()
    main_kwargs['zoom_factor'] = 1
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    matrix_method = problem_kwargs['matrix_method']
    fileHandle = problem_kwargs['fileHandle']
    rs = problem_kwargs['rs']
    print_case_info(**problem_kwargs)

    # create sphere
    sphere_obj = create_sphere(**problem_kwargs)[0]
    sphere_center = sphere_obj.get_u_geo().get_center()
    sphere_norm = sphere_obj.get_u_geo().get_geo_norm()
    # sphere_comp = sf.ForceFreeComposite(center=sphere_center, norm=sphere_norm, name='sphereComp')
    # sphere_comp.add_obj(sphere_obj, rel_U=np.zeros(6))
    # sphere_comp.show_u_nodes()

    problem = sf.problem_dic[matrix_method](**problem_kwargs)
    # problem = sf.ForceFreeProblem(**problem_kwargs)
    problem.add_obj(sphere_obj)
    problem.print_info()
    problem.create_matrix()
    # AtBtCt_full(problem, save_vtk=False, pick_M=False, center=sphere_center, save_name=fileHandle,
    #             uNormFct=(6 * np.pi * rs), wNormFct=(6 * np.pi * rs ** 3),
    #             uwNormFct=(6 * np.pi * rs ** 2), )
    AtBtCt_multiObj(problem, save_vtk=False, pick_M=False, save_name=fileHandle,
                    uNormFct=(6 * np.pi * rs), wNormFct=(6 * np.pi * rs ** 3),
                    uwNormFct=(6 * np.pi * rs ** 2), )
    return True


def main_resistanceMatrix_twoSphere(**main_kwargs):
    # OptDB = PETSc.Options()
    main_kwargs['zoom_factor'] = 1
    problem_kwargs = get_twoSphereProblem_kwargs(**main_kwargs)
    matrix_method = problem_kwargs['matrix_method']
    fileHandle = problem_kwargs['fileHandle']
    rs = problem_kwargs['rs']
    print_case_info(**problem_kwargs)

    # create sphere
    # sphere0_obj = create_sphere(**problem_kwargs)[0]
    # sphere1_obj = sphere0_obj.copy()
    # sphere0_obj.set_name('sphereObj_0')
    # sphere1_obj.set_name('sphereObj_1')
    # sphere0_obj.move((-move_x, 0, 0))
    # sphere1_obj.move((move_x, 0, 0))
    # problem_center = (sphere0_obj.get_u_geo().get_center()
    #                   + sphere1_obj.get_u_geo().get_center()) / 2
    # # objtype = sf.obj_dic[matrix_method]
    # # all_obj = objtype()
    # # all_obj.combine((sphere0_obj, sphere1_obj), set_re_u=True, set_force=True)
    sphere0_obj, sphere1_obj = create_sphere(**problem_kwargs)
    # sphere0_obj.show_u_nodes()
    # sphere1_obj.show_u_nodes()

    problem = sf.problem_dic[matrix_method](**problem_kwargs)
    problem.add_obj(sphere0_obj)
    problem.add_obj(sphere1_obj)
    problem.print_info()
    problem.create_matrix()
    AtBtCt_multiObj(problem, save_vtk=False, pick_M=False, save_name=fileHandle,
                    print_each=False,
                    uNormFct=1, wNormFct=1, uwNormFct=1, )
    return True

def main_xu20201104(**main_kwargs):
    # OptDB = PETSc.Options()
    main_kwargs['zoom_factor'] = 1
    problem_kwargs = get_twoSphereProblem_kwargs(**main_kwargs)
    matrix_method = problem_kwargs['matrix_method']
    fileHandle = problem_kwargs['fileHandle']
    rs = problem_kwargs['rs']
    print_case_info(**problem_kwargs)

    # create sphere
    # sphere0_obj = create_sphere(**problem_kwargs)[0]
    # sphere1_obj = sphere0_obj.copy()
    # sphere0_obj.set_name('sphereObj_0')
    # sphere1_obj.set_name('sphereObj_1')
    # sphere0_obj.move((-move_x, 0, 0))
    # sphere1_obj.move((move_x, 0, 0))
    # problem_center = (sphere0_obj.get_u_geo().get_center()
    #                   + sphere1_obj.get_u_geo().get_center()) / 2
    # # objtype = sf.obj_dic[matrix_method]
    # # all_obj = objtype()
    # # all_obj.combine((sphere0_obj, sphere1_obj), set_re_u=True, set_force=True)
    sphere0_obj, _ = create_sphere(**problem_kwargs)
    problem_kwargs['rs'] = problem_kwargs['rs'] / 4
    problem_kwargs['ds'] = problem_kwargs['ds'] / 4
    _, sphere1_obj = create_sphere(**problem_kwargs)
    # sphere0_obj.show_u_nodes()
    # sphere1_obj.show_u_nodes()

    problem = sf.problem_dic[matrix_method](**problem_kwargs)
    problem.add_obj(sphere0_obj)
    problem.add_obj(sphere1_obj)
    # problem.show_u_nodes()
    problem.print_info()
    problem.create_matrix()
    AtBtCt_multiObj(problem, save_vtk=False, pick_M=False, save_name=fileHandle,
                    print_each=False,
                    uNormFct=1, wNormFct=1, uwNormFct=1, )
    return True


if __name__ == '__main__':
    OptDB = PETSc.Options()
    # mpirun -np 4 python ../sphereNearPlane.py  -main_resistanceMatrix_twoSphere 1  -sm rs_plane -epsilon 0.3 -rs 1.000000 -ds 0.05 -es 0.000000 -h2Plane 1.5  -ksp_max_it 1000 -ksp_rtol 1.000000e-10 -ksp_atol 1.000000e-100 -move_x 2 -f h1.50_movex2

    matrix_method = OptDB.getString('sm', 'pf')
    # assert matrix_method == 'rs_plane', matrix_method

    if OptDB.getBool('main_resistanceMatrix', False):
        OptDB.setValue('main_fun', False)
        main_resistanceMatrix()

    if OptDB.getBool('main_resistanceMatrix_twoSphere', False):
        OptDB.setValue('main_fun', False)
        main_resistanceMatrix_twoSphere()

    if OptDB.getBool('main_xu20201104', False):
        OptDB.setValue('main_fun', False)
        main_xu20201104()

    # if OptDB.getBool('main_fun', True):
    #     main_fun()

# coding=utf-8
# main codes, call functions at stokes_flow.py
# Zhang Ji, 20170518

import sys
import petsc4py

petsc4py.init(sys.argv)

# from scipy.io import savemat, loadmat
# from src.ref_solution import *
# import warnings
# from memory_profiler import profile
# from time import time
import pickle
import numpy as np
from src import stokes_flow as sf
from src.stokes_flow import problem_dic, obj_dic
from petsc4py import PETSc
from src.geo import *
from src.myio import *
from src.objComposite import *
from src.myvtk import *


def print_case_info(**problem_kwargs):
    fileHandle = problem_kwargs['fileHandle']
    print_solver_info(**problem_kwargs)
    print_sphere_info(fileHandle, **problem_kwargs)
    return True


def get_problem_kwargs(**main_kwargs):
    problem_kwargs = get_solver_kwargs()
    OptDB = PETSc.Options()
    fileHandle = OptDB.getString('f', 'sphereTwoPlane')
    matrix_method = problem_kwargs['matrix_method']
    if matrix_method in ('pf_stokesletsTwoPlane',):
        twoPlateHeight = problem_kwargs['twoPlateHeight']
        movez = OptDB.getReal('movez', twoPlateHeight / 2)
    else:
        movez = 0
    OptDB.setValue('f', fileHandle)
    problem_kwargs['fileHandle'] = fileHandle
    problem_kwargs['movez'] = movez

    kwargs_list = (main_kwargs, get_vtk_tetra_kwargs(), get_sphere_kwargs(), )
    for t_kwargs in kwargs_list:
        for key in t_kwargs:
            problem_kwargs[key] = t_kwargs[key]
    return problem_kwargs


def main_fun(**main_kwargs):
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    print_case_info(**problem_kwargs)
    rs = problem_kwargs['rs']
    matrix_method = problem_kwargs['matrix_method']
    sphere_velocity = problem_kwargs['sphere_velocity']

    obj_sphere = create_move_single_sphere(**problem_kwargs)[0]
    problem = problem_dic[matrix_method](**problem_kwargs)
    problem.do_solve_process((obj_sphere, ))

    force_sphere = obj_sphere.get_total_force()
    temp_F = (6 * rs, 6 * rs, 6 * rs, 8 * rs ** 3, 8 * rs ** 3, 8 * rs ** 3)
    force_sphere = force_sphere / temp_F / np.pi
    PETSc.Sys.Print('---->>>%s: Resultant is %s' % (str(problem), str(force_sphere)))

    save_grid_sphere_vtk(problem, )

    return True


def create_M(**main_kwargs):
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    print_case_info(**problem_kwargs)

    fileHandle = problem_kwargs['fileHandle']
    radius = problem_kwargs['radius']
    deltaLength = problem_kwargs['deltaLength']
    matrix_method = problem_kwargs['matrix_method']
    movez = problem_kwargs['movez']
    epsilon = problem_kwargs['epsilon']
    sphere_velocity = problem_kwargs['sphere_velocity']

    sphere_geo0 = sphere_geo()  # force geo
    sphere_geo0.create_delta(deltaLength=deltaLength, radius=radius)
    sphere_geo0.set_rigid_velocity(sphere_velocity)

    obj_sphere = obj_dic[matrix_method]()
    sphere_geo1 = sphere_geo0.copy()
    if matrix_method in ('pf', 'pf_ShearFlow',):
        sphere_geo1.node_zoom((radius + deltaLength * epsilon) / radius)
    obj_sphere.set_data(sphere_geo1, sphere_geo0, name='sphereObj_0_0')
    obj_sphere.move((0, 0, movez))

    problem = problem_dic[matrix_method](**problem_kwargs)
    problem.add_obj(obj_sphere)
    problem.print_info()
    # problem.show_f_u_nodes()
    problem.pickmyself(fileHandle, check=True)
    problem.create_matrix()
    problem.solve()
    problem.pickmyself(fileHandle, pick_M=True)
    return True


def restart_solve(**main_kwargs):
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    fileHandle = problem_kwargs['fileHandle']
    sphere_velocity = problem_kwargs['sphere_velocity']
    PETSc.Sys.Print('sphere velocity: %s' % str(sphere_velocity))
    with open(fileHandle + '_pick.bin', 'rb') as input:
        unpick = pickle.Unpickler(input)
        problem = unpick.load()
        problem.unpickmyself()

    problem_kwargs_old = problem.get_kwargs()
    radius = problem_kwargs_old['radius']
    deltaLength = problem_kwargs_old['deltaLength']
    movez = problem_kwargs_old['movez']
    matrix_method = problem_kwargs_old['matrix_method']

    obj_sphere = problem.get_obj_list()[0]
    sphere_geo0 = obj_sphere.get_u_geo()
    sphere_geo0.set_rigid_velocity(sphere_velocity)
    problem.create_F_U()
    problem.print_info()
    problem.solve()

    problem.vtk_self(fileHandle)
    force_sphere = obj_sphere.get_total_force()
    force_sphere = force_sphere / np.pi / \
                   (6 * radius, 6 * radius, 6 * radius, 8 * radius ** 3, 8 * radius ** 3, 8 * radius ** 3)
    PETSc.Sys.Print('---->>>%s: Resultant is %s' % (str(problem), str(force_sphere)))

    geo_check = sphere_geo()  # force geo
    n = int(16 * radius * radius / deltaLength / deltaLength)
    geo_check.create_n(n, radius)
    geo_check.set_rigid_velocity(sphere_velocity)
    geo_check.move((0, 0, movez))
    obj_check = obj_dic[matrix_method]()
    obj_check.set_data(geo_check, geo_check, name='sphereObj_check')
    err = problem.vtk_check(fileHandle + '_check', obj_check)
    PETSc.Sys.Print('err: ', err)


def export_data(**main_kwargs):
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    fileHandle = problem_kwargs['fileHandle']
    with open(fileHandle + '_pick.bin', 'rb') as input:
        unpick = pickle.Unpickler(input)
        problem = unpick.load()
        problem.unpickmyself()

    obj_sphere = problem.get_obj_list()[0]
    obj_sphere.save_mat()
    # problem.saveM_mat(fileHandle + '_M')
    problem.saveM_ASCII(fileHandle + '_M')
    # problem.saveM_HDF5(fileHandle + '_M')
    PETSc.Sys.Print('finish')


if __name__ == '__main__':
    OptDB = PETSc.Options()
    mode = OptDB.getString('mode', 'main')
    if mode == 'main':
        main_fun()
    elif mode == 'createM':
        create_M()
    elif mode == 'resolve':
        restart_solve()
    elif mode == 'export':
        export_data()
    else:
        err_msg = 'wrong program mode'
        raise err_msg

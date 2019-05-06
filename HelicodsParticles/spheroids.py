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
from src.stokes_flow import problem_dic, obj_dic, StokesFlowObj
from petsc4py import PETSc
from src.geo import *
from src.myio import *
from src.objComposite import *
from src.myvtk import *
from src.StokesFlowMethod import *
from src.stokesletsInPipe import *


def get_problem_kwargs(**main_kwargs):
    problem_kwargs = get_solver_kwargs()
    OptDB = PETSc.Options()
    fileHandle = OptDB.getString('f', 'spheroids_3')
    OptDB.setValue('f', fileHandle)
    problem_kwargs['fileHandle'] = fileHandle

    kwargs_list = (get_shearFlow_kwargs(), get_freeVortex_kwargs(),
                   get_ecoli_kwargs(), get_one_ellipse_kwargs(), get_forcefree_kwargs(), main_kwargs,)
    for t_kwargs in kwargs_list:
        for key in t_kwargs:
            problem_kwargs[key] = t_kwargs[key]
    return problem_kwargs


def print_case_info(**problem_kwargs):
    fileHandle = problem_kwargs['fileHandle']
    print_solver_info(**problem_kwargs)
    print_shearFlow_info(**problem_kwargs)
    print_freeVortex_info(**problem_kwargs)
    print_one_ellipse_info(fileHandle, **problem_kwargs)
    return True


def main_fun_noIter(**main_kwargs):
    # OptDB = PETSc.Options()
    main_kwargs['zoom_factor'] = 1
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    matrix_method = problem_kwargs['matrix_method']
    fileHandle = problem_kwargs['fileHandle']
    print_case_info(**problem_kwargs)
    iter_tor = 1e-3

    # location and norm of spheroids
    rc1 = np.array((0, 1, 0))
    rc2 = np.array((-np.sqrt(3) / 2, - 1 / 2, 0))
    rc3 = np.array((np.sqrt(3) / 2, - 1 / 2, 0))
    nc1 = np.array((-1, np.sqrt(3), np.sqrt(5))) / 3
    nc2 = np.array((-1, -np.sqrt(3), np.sqrt(5))) / 3
    nc3 = np.array((2, 0, np.sqrt(5))) / 3
    rc_all = np.vstack((rc1, rc2, rc3))
    nc_all = np.vstack((nc1, nc2, nc3))

    spheroid0 = create_one_ellipse(namehandle='spheroid0', **problem_kwargs)
    spheroid0.move(rc1)
    spheroid0_norm = spheroid0.get_u_geo().get_geo_norm()
    rot_norm = np.cross(nc1, spheroid0_norm)
    rot_theta = np.arccos(np.dot(nc1, spheroid0_norm) / np.linalg.norm(nc1) / np.linalg.norm(spheroid0_norm))
    spheroid0.node_rotation(rot_norm, rot_theta)
    spheroid1 = spheroid0.copy()
    spheroid1.node_rotation(np.array((0, 0, 1)), 2 * np.pi / 3, rotation_origin=np.zeros(3))
    spheroid2 = spheroid0.copy()
    spheroid2.node_rotation(np.array((0, 0, 1)), 4 * np.pi / 3, rotation_origin=np.zeros(3))
    spheroid_comp = sf.ForceFreeComposite(center=np.zeros(3), norm=np.array((0, 0, 1)), name='spheroid_comp')
    spheroid_comp.add_obj(obj=spheroid0, rel_U=np.zeros(6))
    spheroid_comp.add_obj(obj=spheroid1, rel_U=np.zeros(6))
    spheroid_comp.add_obj(obj=spheroid2, rel_U=np.zeros(6))
    # spheroid_comp.node_rotation(np.array((1, 0, 0)), np.pi / 2)

    # # dbg
    # tail_list = create_ecoli_tail(np.zeros(3), **problem_kwargs)
    # spheroid_comp = sf.ForceFreeComposite(center=np.zeros(3), norm=np.array((0, 0, 1)), name='spheroid_comp')
    # for tobj in tail_list:
    #     spheroid_comp.add_obj(obj=tobj, rel_U=np.zeros(6))

    problem_ff = sf.LambOseenVortexForceFreeProblem(**problem_kwargs)
    problem_ff.add_obj(spheroid_comp)
    problem_ff.print_info()
    problem_ff.create_matrix()
    problem_ff.solve()
    ref_U = spheroid_comp.get_ref_U()
    PETSc.Sys.Print('  ref_U in shear flow', ref_U)
    spheroid_comp_F = spheroid_comp.get_total_force()
    spheroid0_F = spheroid0.get_total_force(center=np.zeros(3))
    # spheroid0_F = tail_list[0].get_total_force(center=np.zeros(3))
    PETSc.Sys.Print('  spheroid_comp_F %s' % str(spheroid_comp_F))
    PETSc.Sys.Print('  spheroid0_F %s' % str(spheroid0_F))
    PETSc.Sys.Print('  non dimensional (F, T) err %s' % str(spheroid_comp_F / spheroid0_F))

    return True


def main_fun_fix(**main_kwargs):
    # OptDB = PETSc.Options()
    main_kwargs['zoom_factor'] = 1
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    matrix_method = problem_kwargs['matrix_method']
    fileHandle = problem_kwargs['fileHandle']
    print_case_info(**problem_kwargs)
    iter_tor = 1e-3

    # location and norm of spheroids
    rc1 = np.array((0, 1, 0))
    rc2 = np.array((-np.sqrt(3) / 2, - 1 / 2, 0))
    rc3 = np.array((np.sqrt(3) / 2, - 1 / 2, 0))
    nc1 = np.array((-1, np.sqrt(3), np.sqrt(5))) / 3
    nc2 = np.array((-1, -np.sqrt(3), np.sqrt(5))) / 3
    nc3 = np.array((2, 0, np.sqrt(5))) / 3
    rc_all = np.vstack((rc1, rc2, rc3))
    nc_all = np.vstack((nc1, nc2, nc3))

    spheroid0 = create_one_ellipse(namehandle='spheroid0', **problem_kwargs)
    spheroid0.move(rc1)
    spheroid0_norm = spheroid0.get_u_geo().get_geo_norm()
    rot_norm = np.cross(nc1, spheroid0_norm)
    rot_theta = np.arccos(np.dot(nc1, spheroid0_norm) / np.linalg.norm(nc1) / np.linalg.norm(spheroid0_norm))
    spheroid0.node_rotation(rot_norm, rot_theta)
    spheroid0.set_rigid_velocity(np.zeros(6))
    spheroid1 = spheroid0.copy()
    spheroid1.node_rotation(np.array((0, 0, 1)), 2 * np.pi / 3, rotation_origin=np.zeros(3))
    spheroid2 = spheroid0.copy()
    spheroid2.node_rotation(np.array((0, 0, 1)), 4 * np.pi / 3, rotation_origin=np.zeros(3))
    spheroid_comp = sf.ForceFreeComposite(center=np.zeros(3), norm=np.array((0, 0, 1)), name='spheroid_comp')
    spheroid_comp.add_obj(obj=spheroid0, rel_U=np.zeros(6))
    spheroid_comp.add_obj(obj=spheroid1, rel_U=np.zeros(6))
    spheroid_comp.add_obj(obj=spheroid2, rel_U=np.zeros(6))
    # spheroid_comp.node_rotation(np.array((1, 0, 0)), np.pi / 2)

    # # dbg
    # tail_list = create_ecoli_tail(np.zeros(3), **problem_kwargs)
    # spheroid_comp = sf.ForceFreeComposite(center=np.zeros(3), norm=np.array((0, 0, 1)), name='spheroid_comp')
    # for tobj in tail_list:
    #     spheroid_comp.add_obj(obj=tobj, rel_U=np.zeros(6))

    problem_ff = sf.LambOseenVortexProblem(**problem_kwargs)
    problem_ff.add_obj(spheroid0)
    problem_ff.add_obj(spheroid1)
    problem_ff.add_obj(spheroid2)
    problem_ff.print_info()
    problem_ff.create_matrix()
    problem_ff.solve()
    PETSc.Sys.Print('  spheroid0_F %s' % str(spheroid0.get_total_force()))
    PETSc.Sys.Print('  spheroid1_F %s' % str(spheroid1.get_total_force()))
    PETSc.Sys.Print('  spheroid2_F %s' % str(spheroid2.get_total_force()))
    spheroid_comp_F = spheroid_comp.get_total_force()
    PETSc.Sys.Print('  spheroid_comp_F %s' % str(spheroid_comp_F))
    return True


def main_fun(**main_kwargs):
    # OptDB = PETSc.Options()
    main_kwargs['zoom_factor'] = 1
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    matrix_method = problem_kwargs['matrix_method']
    fileHandle = problem_kwargs['fileHandle']
    print_case_info(**problem_kwargs)
    iter_tor = 1e-1

    # location and norm of spheroids
    rc1 = np.array((0, 1, 0))
    rc2 = np.array((-np.sqrt(3) / 2, - 1 / 2, 0))
    rc3 = np.array((np.sqrt(3) / 2, - 1 / 2, 0))
    nc1 = np.array((-1, np.sqrt(3), np.sqrt(5))) / 3
    nc2 = np.array((-1, -np.sqrt(3), np.sqrt(5))) / 3
    nc3 = np.array((2, 0, np.sqrt(5))) / 3
    rc_all = np.vstack((rc1, rc2, rc3))
    nc_all = np.vstack((nc1, nc2, nc3))

    spheroid0 = create_one_ellipse(namehandle='spheroid0', **problem_kwargs)
    spheroid0.move(rc1)
    spheroid0_norm = spheroid0.get_u_geo().get_geo_norm()
    rot_norm = np.cross(nc1, spheroid0_norm)
    rot_theta = np.arccos(np.dot(nc1, spheroid0_norm) / np.linalg.norm(nc1) / np.linalg.norm(spheroid0_norm))
    spheroid0.node_rotation(rot_norm, rot_theta)
    spheroid1 = spheroid0.copy()
    spheroid1.node_rotation(np.array((0, 0, 1)), 2 * np.pi / 3, rotation_origin=np.zeros(3))
    spheroid2 = spheroid0.copy()
    spheroid2.node_rotation(np.array((0, 0, 1)), 4 * np.pi / 3, rotation_origin=np.zeros(3))
    spheroid_comp = sf.ForceFreeComposite(center=np.zeros(3), norm=np.array((0, 0, 1)), name='spheroid_comp')
    spheroid_comp.add_obj(obj=spheroid0, rel_U=np.zeros(6))
    spheroid_comp.add_obj(obj=spheroid1, rel_U=np.zeros(6))
    spheroid_comp.add_obj(obj=spheroid2, rel_U=np.zeros(6))

    # dbg
    tail_list = create_ecoli_tail(np.zeros(3), **problem_kwargs)
    spheroid_comp = sf.ForceFreeComposite(center=np.zeros(3), norm=np.array((0, 0, 1)), name='spheroid_comp')
    for tobj in tail_list:
        spheroid_comp.add_obj(obj=tobj, rel_U=np.zeros(6))
    spheroid_comp.show_u_nodes()

    # 1). ini guess
    problem_ff = sf.FreeVortexForceFreeProblem(**problem_kwargs)
    problem_ff.add_obj(spheroid_comp)
    problem_ff.print_info()
    problem_ff.create_matrix()
    problem_ff.solve()
    ref_U = spheroid_comp.get_ref_U()
    PETSc.Sys.Print('  ini ref_U in free vortex flow is ', ref_U)
    # 2). optimize force free
    problem = sf.FreeVortexForceFreeIterateProblem(**problem_kwargs)
    problem.add_obj(spheroid_comp)
    problem.set_iterate_comp(spheroid_comp)
    problem.create_matrix()
    ref_U, _, _ = problem.do_iterate2(ini_refU1=ref_U, tolerate=iter_tor)
    spheroid_comp.set_ref_U(ref_U)
    PETSc.Sys.Print('  ref_U in free vortex flow is ', ref_U)
    spheroid_comp_F = spheroid_comp.get_total_force()
    spheroid0_F = spheroid0.get_total_force(center=np.zeros(3))
    # spheroid0_F = tail_list[0].get_total_force(center=np.zeros(3))
    PETSc.Sys.Print('  spheroid_comp_F %s' % str(spheroid_comp_F))
    PETSc.Sys.Print('  spheroid0_F %s' % str(spheroid0_F))
    PETSc.Sys.Print('  non dimensional (F, T) err %s' % str(spheroid_comp_F / spheroid0_F))

    return True


if __name__ == '__main__':
    # main_fun_noIter()
    # main_fun_fix()
    main_fun()

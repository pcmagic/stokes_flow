from time import time
import numpy as np
from petsc4py import PETSc
from src import stokes_flow as sf
from src.objComposite import *
from src.stokes_flow import obj_dic
from src.ref_solution import *
from src.geo import *

__all__ = ['save_singleEcoli_vtk', 'save_singleEcoli_U_vtk', 'save_singleEcoli_U_4part_vtk',
           'save_grid_sphere_vtk',
           'save_singleRod_vtk', ]


def save_singleEcoli_vtk(problem: sf.StokesFlowProblem, createHandle=createEcoliComp_tunnel):
    # force free
    OptDB = PETSc.Options()
    if not OptDB.getBool('save_singleEcoli_vtk', True):
        return False

    t0 = time()
    problem_kwargs = problem.get_kwargs()
    fileHandle = problem_kwargs['fileHandle']
    # with_T_geo = len(problem.get_all_obj_list()) == 4
    with_T_geo = problem_kwargs['with_T_geo']
    ref_U = problem.get_obj_list()[0].get_ref_U()

    # problem.vtk_obj(fileHandle)
    problem.vtk_self(fileHandle)

    # bgeo = geo()
    # bnodesHeadle = problem_kwargs['bnodesHeadle']
    # matname = problem_kwargs['matname']
    # bgeo.mat_nodes(filename=matname, mat_handle=bnodesHeadle)
    # belemsHeadle = problem_kwargs['belemsHeadle']
    # bgeo.mat_elmes(filename=matname, mat_handle=belemsHeadle, elemtype='tetra')
    # problem.vtk_tetra(fileHandle + '_Velocity', bgeo)

    # create check obj
    check_kwargs = problem_kwargs.copy()
    check_kwargs['nth'] = problem_kwargs['nth'] - 2 if problem_kwargs['nth'] >= 10 else problem_kwargs['nth'] + 1
    check_kwargs['ds'] = problem_kwargs['ds'] * 1.2
    check_kwargs['hfct'] = 1
    check_kwargs['Tfct'] = 1
    ecoli_comp_check = createHandle(**check_kwargs)
    ecoli_comp_check.set_ref_U(ref_U)
    # # dbg
    # for obj in ecoli_comp_check.get_obj_list():
    #     filename = fileHandle + '_check_' + str(obj)
    #     obj.get_u_geo().save_nodes(filename + '_U')
    #     obj.get_f_geo().save_nodes(filename + '_f')

    velocity_err_list = problem.vtk_check(fileHandle, ecoli_comp_check)
    PETSc.Sys.Print('velocity error of sphere (total, x, y, z): ', next(velocity_err_list))
    PETSc.Sys.Print('velocity error of helix0 (total, x, y, z): ', next(velocity_err_list))
    PETSc.Sys.Print('velocity error of helix1 (total, x, y, z): ', next(velocity_err_list))
    if with_T_geo:
        PETSc.Sys.Print('velocity error of Tgeo (total, x, y, z): ', next(velocity_err_list))

    t1 = time()
    PETSc.Sys.Print('%s: write vtk files use: %fs' % (str(problem), (t1 - t0)))
    return True


# given velocity case
def save_singleEcoli_U_vtk(problem: sf.StokesFlowProblem,
                           createHandle=createEcoliComp_tunnel, part='full'):
    def save_head():
        vsobj = createHandle(**check_kwargs).get_obj_list()[0]
        vsobj.set_rigid_velocity(rel_Us + ecoli_U, center=center)
        velocity_err_sphere = next(problem.vtk_check(fileHandle, vsobj))
        PETSc.Sys.Print('velocity error of sphere (total, x, y, z): ', velocity_err_sphere)

    def save_tail():
        tail_obj_list = createHandle(**check_kwargs)[1]
        for tail_obj in tail_obj_list:
            tail_obj.set_rigid_velocity(rel_Uh + ecoli_U, center=center)
        velocity_err_list = problem.vtk_check(fileHandle, tail_obj_list)
        PETSc.Sys.Print('velocity error of helix0 (total, x, y, z): ', next(velocity_err_list))
        PETSc.Sys.Print('velocity error of helix1 (total, x, y, z): ', next(velocity_err_list))
        if with_T_geo:
            PETSc.Sys.Print('velocity error of Tgeo (total, x, y, z): ', next(velocity_err_list))

    def save_full():
        save_head()
        save_tail()

    def do_save_part():
        return {'head': save_head,
                'tail': save_tail,
                'full': save_full}[part]

    OptDB = PETSc.Options()
    if not OptDB.getBool('save_singleEcoli_vtk', True):
        return False

    t0 = time()
    problem_kwargs = problem.get_kwargs()
    fileHandle = problem_kwargs['fileHandle']
    ecoli_U = problem_kwargs['ecoli_U']
    rel_Us = problem_kwargs['rel_Us']
    rel_Uh = problem_kwargs['rel_Uh']
    center = problem_kwargs['center']
    with_T_geo = problem_kwargs['with_T_geo'] if 'with_T_geo' in problem_kwargs.keys() else 0

    # problem.vtk_obj(fileHandle)
    problem.vtk_self(fileHandle)

    # bgeo = geo()
    # bnodesHeadle = problem_kwargs['bnodesHeadle']
    # matname = problem_kwargs['matname']
    # bgeo.mat_nodes(filename=matname, mat_handle=bnodesHeadle)
    # belemsHeadle = problem_kwargs['belemsHeadle']
    # bgeo.mat_elmes(filename=matname, mat_handle=belemsHeadle, elemtype='tetra')
    # problem.vtk_tetra(fileHandle + '_Velocity', bgeo)

    # create check obj
    check_kwargs = problem_kwargs.copy()
    check_kwargs['nth'] = problem_kwargs['nth'] - 2 if problem_kwargs['nth'] >= 6 else problem_kwargs['nth'] + 1
    check_kwargs['ds'] = problem_kwargs['ds'] * 1.2
    check_kwargs['hfct'] = 1
    check_kwargs['Tfct'] = 1
    check_kwargs['eh'] = 0
    check_kwargs['es'] = 0
    check_kwargs['eT'] = 0
    do_save_part()()

    t1 = time()
    PETSc.Sys.Print('%s: write vtk files use: %fs' % (str(problem), (t1 - t0)))
    return True


# given velocity case,
#  consider the ecoli constituted by four separate part: head, helix0, helix1, and Tgeo.
#  each part have its own velocity U=[ux, uy, uz, wx, wy ,wz]
def save_singleEcoli_U_4part_vtk(problem: sf.StokesFlowProblem, U_list, createHandle=createEcoliComp_tunnel):
    OptDB = PETSc.Options()
    if not OptDB.getBool('save_singleEcoli_vtk', True):
        return False

    t0 = time()
    problem_kwargs = problem.get_kwargs()
    fileHandle = problem_kwargs['fileHandle']
    center = problem_kwargs['center']
    # with_T_geo = len(problem.get_all_obj_list()) == 4
    with_T_geo = problem_kwargs['with_T_geo'] if 'with_T_geo' in problem_kwargs.keys() else 0

    # problem.vtk_obj(fileHandle)
    problem.vtk_self(fileHandle)

    # bgeo = geo()
    # bnodesHeadle = problem_kwargs['bnodesHeadle']
    # matname = problem_kwargs['matname']
    # bgeo.mat_nodes(filename=matname, mat_handle=bnodesHeadle)
    # belemsHeadle = problem_kwargs['belemsHeadle']
    # bgeo.mat_elmes(filename=matname, mat_handle=belemsHeadle, elemtype='tetra')
    # problem.vtk_tetra(fileHandle + '_Velocity', bgeo)

    # create check obj
    check_kwargs = problem_kwargs.copy()
    check_kwargs['nth'] = problem_kwargs['nth'] - 2 if problem_kwargs['nth'] >= 6 else problem_kwargs['nth'] + 1
    check_kwargs['ds'] = problem_kwargs['ds'] * 1.2
    check_kwargs['hfct'] = 1
    check_kwargs['Tfct'] = 1

    obj_list = createHandle(**check_kwargs)
    for obj, t_U in zip(sf.tube_flatten(obj_list), U_list):
        obj.set_rigid_velocity(t_U, center=center)
    velocity_err_list = problem.vtk_check(fileHandle, obj_list)
    PETSc.Sys.Print('velocity error of sphere (total, x, y, z): ', next(velocity_err_list))
    PETSc.Sys.Print('velocity error of helix0 (total, x, y, z): ', next(velocity_err_list))
    PETSc.Sys.Print('velocity error of helix1 (total, x, y, z): ', next(velocity_err_list))
    if with_T_geo:
        PETSc.Sys.Print('velocity error of Tgeo (total, x, y, z): ', next(velocity_err_list))

    cbd_obj = sf.StokesFlowObj()
    cbd_obj.combine(obj_list)
    velocity_err = problem.vtk_check(fileHandle, cbd_obj)
    PETSc.Sys.Print('velocity error of ecoli (total, x, y, z): ', next(velocity_err))
    t1 = time()
    PETSc.Sys.Print('%s: write vtk files use: %fs' % (str(problem), (t1 - t0)))
    return True


def save_grid_sphere_vtk(problem: sf.StokesFlowProblem, createHandle=create_sphere):
    OptDB = PETSc.Options()
    if not OptDB.getBool('save_grid_sphere_vtk', True):
        return False

    t0 = time()
    problem_kwargs = problem.get_kwargs()
    fileHandle = problem_kwargs['fileHandle']

    # problem.vtk_obj(fileHandle)
    # problem.vtk_velocity('%s_Velocity' % fileHandle)
    problem.vtk_self(fileHandle)

    check_kwargs = problem_kwargs.copy()
    check_kwargs['ds'] = problem_kwargs['ds'] * 1.2
    obj_sphere_check = sf.obj_dic[problem_kwargs['matrix_method']]()
    obj_sphere_check.combine(createHandle(**check_kwargs))
    obj_sphere_check.set_name('fullPro')
    velocity_err = problem.vtk_check(fileHandle, obj_sphere_check)
    PETSc.Sys.Print('velocity error (total, x, y, z): ', next(velocity_err))

    t1 = time()
    PETSc.Sys.Print('%s: write vtk files use: %fs' % (str(problem), (t1 - t0)))

    return velocity_err


def save_singleRod_vtk(problem: sf.StokesFlowProblem, ref_U=None, createHandle=create_rod):
    OptDB = PETSc.Options()
    if not OptDB.getBool('save_singleRod_vtk', True):
        return False

    t0 = time()
    problem_kwargs = problem.get_kwargs()
    fileHandle = problem_kwargs['fileHandle']
    rod_comp = problem.get_obj_list()[0]
    ref_U = rod_comp.get_ref_U() if ref_U is None else ref_U

    # create check obj
    check_kwargs = problem_kwargs.copy()
    check_kwargs['ntRod'] = 13 if np.abs(problem_kwargs['ntRod'] - 13) > 1 else 17
    rod_comp_check = createHandle(**check_kwargs)[0]
    rod_comp_check.set_ref_U(ref_U)

    problem.vtk_obj(fileHandle)
    velocity_err_rod = problem.vtk_check(fileHandle, rod_comp_check)
    PETSc.Sys.Print('velocity error of rod (total, x, y, z): ', velocity_err_rod)

    t1 = time()
    PETSc.Sys.Print('%s: write vtk files use: %fs' % (str(problem), (t1 - t0)))
    return True

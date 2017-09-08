from time import time
import numpy as np
from petsc4py import PETSc
from src import stokes_flow as sf
from src.objComposite import *
from src.stokes_flow import obj_dic
from src.ref_solution import *
from src.geo import *

__all__ = ['save_singleEcoli_vtk',
           'save_grid_sphere_vtk',
           'save_singleRod_vtk', ]


def save_singleEcoli_vtk(problem: sf.stokesFlowProblem, ref_U=None, createHandle=createEcoliComp_tunnel):
    OptDB = PETSc.Options()
    if not OptDB.getBool('save_singleEcoli_vtk', True):
        return False

    t0 = time()
    problem_kwargs = problem.get_kwargs()
    fileHeadle = problem_kwargs['fileHeadle']
    ecoli_comp = problem.get_obj_list()[0]
    with_T_geo = len(ecoli_comp.get_obj_list()) == 4
    ref_U = ecoli_comp.get_ref_U() if ref_U is None else ref_U

    # problem.vtk_obj(fileHeadle)
    problem.vtk_self(fileHeadle)

    # bgeo = geo()
    # bnodesHeadle = problem_kwargs['bnodesHeadle']
    # matname = problem_kwargs['matname']
    # bgeo.mat_nodes(filename=matname, mat_handle=bnodesHeadle)
    # belemsHeadle = problem_kwargs['belemsHeadle']
    # bgeo.mat_elmes(filename=matname, mat_handle=belemsHeadle, elemtype='tetra')
    # problem.vtk_tetra(fileHeadle + '_Velocity', bgeo)

    # create check obj
    check_kwargs = problem_kwargs.copy()
    check_kwargs['nth'] = problem_kwargs['nth'] - 2 if problem_kwargs['nth'] >= 6 else problem_kwargs['nth'] + 1
    check_kwargs['ds'] = problem_kwargs['ds'] * 1.2
    check_kwargs['hfct'] = 1
    check_kwargs['Tfct'] = 1
    ecoli_comp_check = createHandle(**check_kwargs)

    ecoli_comp_check.set_ref_U(ref_U)
    if with_T_geo:
        velocity_err_sphere, velocity_err_helix0, velocity_err_helix1, velocity_err_Tgeo = \
            problem.vtk_check(fileHeadle, ecoli_comp_check)
        PETSc.Sys.Print('velocity error of sphere (total, x, y, z): ', velocity_err_sphere)
        PETSc.Sys.Print('velocity error of helix0 (total, x, y, z): ', velocity_err_helix0)
        PETSc.Sys.Print('velocity error of helix1 (total, x, y, z): ', velocity_err_helix1)
        PETSc.Sys.Print('velocity error of Tgeo (total, x, y, z): ', velocity_err_Tgeo)
    else:
        velocity_err_sphere, velocity_err_helix0, velocity_err_helix1 = \
            problem.vtk_check(fileHeadle, ecoli_comp_check)
        PETSc.Sys.Print('velocity error of sphere (total, x, y, z): ', velocity_err_sphere)
        PETSc.Sys.Print('velocity error of helix0 (total, x, y, z): ', velocity_err_helix0)
        PETSc.Sys.Print('velocity error of helix1 (total, x, y, z): ', velocity_err_helix1)

    t1 = time()
    PETSc.Sys.Print('%s: write vtk files use: %fs' % (str(problem), (t1 - t0)))
    return True


def save_grid_sphere_vtk(problem: sf.stokesFlowProblem, createHandle=create_sphere):
    OptDB = PETSc.Options()
    if not OptDB.getBool('save_grid_sphere_vtk', True):
        return False

    t0 = time()
    problem_kwargs = problem.get_kwargs()
    fileHeadle = problem_kwargs['fileHeadle']

    # problem.vtk_obj(fileHeadle)
    # problem.vtk_velocity('%s_Velocity' % fileHeadle)
    problem.vtk_self(fileHeadle)

    check_kwargs = problem_kwargs.copy()
    check_kwargs['ds'] = problem_kwargs['ds'] * 1.2
    obj_sphere_check = sf.stokesFlowObj()
    obj_sphere_check.combine(createHandle(**check_kwargs))
    velocity_err = problem.vtk_check(fileHeadle, obj_sphere_check)
    PETSc.Sys.Print('velocity error (total, x, y, z): ', velocity_err)

    t1 = time()
    PETSc.Sys.Print('%s: write vtk files use: %fs' % (str(problem), (t1 - t0)))

    return velocity_err


def save_singleRod_vtk(problem: sf.stokesFlowProblem, ref_U=None, createHandle=create_rod):
    OptDB = PETSc.Options()
    if not OptDB.getBool('save_singleRod_vtk', True):
        return False

    t0 = time()
    problem_kwargs = problem.get_kwargs()
    fileHeadle = problem_kwargs['fileHeadle']
    rod_comp = problem.get_obj_list()[0]
    ref_U = rod_comp.get_ref_U() if ref_U is None else ref_U

    # create check obj
    check_kwargs = problem_kwargs.copy()
    check_kwargs['ntRod'] = 13 if np.abs(problem_kwargs['ntRod'] - 13) > 1 else 17
    rod_comp_check = createHandle(**check_kwargs)[0]
    rod_comp_check.set_ref_U(ref_U)

    problem.vtk_obj(fileHeadle)
    velocity_err_rod = problem.vtk_check(fileHeadle, rod_comp_check)
    PETSc.Sys.Print('velocity error of rod (total, x, y, z): ', velocity_err_rod)

    t1 = time()
    PETSc.Sys.Print('%s: write vtk files use: %fs' % (str(problem), (t1 - t0)))
    return True

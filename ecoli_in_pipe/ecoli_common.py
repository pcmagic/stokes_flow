import sys

import petsc4py

petsc4py.init(sys.argv)

from petsc4py import PETSc
import pickle
from src.support_class import *
from src.myio import *
from src.myvtk import save_singleEcoli_vtk
from src.objComposite import *
import numpy as np
import src.stokes_flow as sf

__all__ = ['get_problem_kwargs', 'print_case_info', 'ecoli_restart',
           'shear_sovle_velocity_1step', ]


def get_problem_kwargs(**main_kwargs):
    problem_kwargs = get_solver_kwargs()

    kwargs_list = (get_vtk_tetra_kwargs(), get_ecoli_kwargs(), get_forcefree_kwargs(), main_kwargs,)
    for t_kwargs in kwargs_list:
        for key in t_kwargs:
            problem_kwargs[key] = t_kwargs[key]
    return problem_kwargs


def print_case_info(caseIntro='-->Ecoli in pipe case, force free case.',
                    **problem_kwargs):
    fileHandle = problem_kwargs['fileHandle']
    PETSc.Sys.Print(caseIntro)
    print_solver_info(**problem_kwargs)
    print_forcefree_info(**problem_kwargs)
    print_ecoli_info(fileHandle, **problem_kwargs)
    return True


def ecoli_restart(**main_kwargs):
    err_msg = 'keyword fileHandle is necessary. '
    assert 'fileHandle' in main_kwargs.keys(), err_msg
    new_kwargs = get_problem_kwargs(**main_kwargs)
    fileHandle = new_kwargs['fileHandle']
    t_name = check_file_extension(fileHandle, '_pick.bin')
    with open(t_name, 'rb') as myinput:
        unpick = pickle.Unpickler(myinput)
        problem = unpick.load()
    problem.unpick_myself()
    ecoli_comp = problem.get_obj_list()[0]

    old_kwargs = problem.get_kwargs()
    old_kwargs['matname'] = new_kwargs['matname']
    old_kwargs['bnodesHeadle'] = new_kwargs['bnodesHeadle']
    old_kwargs['belemsHeadle'] = new_kwargs['belemsHeadle']
    old_kwargs['ffweightx'] = new_kwargs['ffweightx']
    old_kwargs['ffweighty'] = new_kwargs['ffweighty']
    old_kwargs['ffweightz'] = new_kwargs['ffweightz']
    old_kwargs['ffweightT'] = new_kwargs['ffweightT']

    problem.set_kwargs(**old_kwargs)
    print_case_info(**old_kwargs)
    problem.print_info()
    problem.set_force_free()
    problem.solve()

    # post process
    head_U, tail_U = print_single_ecoli_forcefree_result(ecoli_comp, **old_kwargs)
    ecoli_U = ecoli_comp.get_ref_U()
    save_singleEcoli_vtk(problem, createHandle=createEcoliComp_tunnel)
    return head_U, tail_U, ecoli_U


def shear_sovle_velocity_1step(ecoli_comp: 'sf.ForceFreeComposite',
                               problem_ff: 'sf.ShearFlowForceFreeProblem',
                               problem: 'sf.ForceFreeIterateProblem',
                               v_ecoli_tor=1e-2, iter_tor=1e-3, **problem_kwargs):
    # single ecoli in shear flow, force and torque free, known speed ecoli_velocity,
    # sum(F)==sum(T)==0
    # norm(Urefx, Urefy, Urefz)==ecoli_velocity
    # Urel_head_x,y,z==0, Wrel_head_x,y,z==0
    # Urel_tail_x,y,z==0, Wrel_tail_x,y==0, Wrel_tail_z==W_motor
    # unknows: Urefx, Urefy, Urefz, Wrefx, Wrefy, Wrefz, Wrel_tail_z
    rel_Us = problem_kwargs['rel_Us']
    err_msg = 'current version must: rel_Us==np.zeros(6), so rel_Uh[3:]=w_motor.  '
    assert np.all(np.isclose(rel_Us, np.zeros(6))), err_msg

    ecoli_velocity = problem_kwargs['ecoli_velocity']
    planeShearRate = problem_kwargs['planeShearRate']
    problem_ff.create_matrix()
    problem.create_matrix()
    center = ecoli_comp.get_center()

    # 0) solve the velocity of a passive ecoli (w_motor==0)
    PETSc.Sys.Print('  0) solve the velocity of a passive ecoli (w_motor==0)')
    if np.all(np.isclose(planeShearRate, np.zeros(3))):
        U_pass = np.zeros(6)
    else:
        tmp_rel_U_list = ecoli_comp.get_rel_U_list()
        ecoli_comp.dbg_set_rel_U_list([np.zeros(6), np.zeros(6)])  # a passive ecoli
        problem_ff.create_F_U()
        problem_ff.set_force_free()
        problem_ff.solve()
        U_pass = ecoli_comp.get_ref_U()
        problem.create_F_U()
        U_pass, _, _ = problem.do_iterate(tolerate=iter_tor, ini_refU1=U_pass)
        print_single_ecoli_force_result(ecoli_comp, prefix='', part='full', **problem_kwargs)
        ecoli_comp.dbg_set_rel_U_list(tmp_rel_U_list)  # a active ecoli
        # problem_ff.create_F_U()    # command it to speedup since later the loop will down it first.
        # problem_ff.set_force_free()
        problem.create_F_U()

    # 1) find ini solution (uref, wref) that |uref|==|v_ecoli|
    PETSc.Sys.Print()
    PETSc.Sys.Print('  1). get the ini_guess of the ref_U')
    err_ref_u = np.inf
    tail_rel_U = ecoli_comp.get_rel_U_list()[1]
    tstp = 0
    while err_ref_u > v_ecoli_tor:
        tstp = tstp + 1
        problem_ff.create_F_U()
        problem_ff.set_force_free()
        problem_ff.solve()
        ref_U = ecoli_comp.get_ref_U()
        norm_ref_u = np.linalg.norm(ref_U[:3] - U_pass[:3])
        err_ref_u = np.abs((norm_ref_u - ecoli_velocity) / ecoli_velocity)
        tail_rel_U = tail_rel_U * ecoli_velocity / norm_ref_u
        tmp_rel_U_list = [rel_Us, tail_rel_U]
        ecoli_comp.dbg_set_rel_U_list(tmp_rel_U_list)
        PETSc.Sys.Print('    ---- STEP %03d ----' % tstp)
        PETSc.Sys.Print('    ref_U', ref_U)
        PETSc.Sys.Print('    norm_ref_U %f, err_ref_U %f' % (norm_ref_u, err_ref_u))
        PETSc.Sys.Print('    tail_rel_U', tail_rel_U)
    print_single_ecoli_force_result(ecoli_comp, prefix='', part='full', **problem_kwargs)

    # 2) using iterate method to optimize force and torque free
    PETSc.Sys.Print()
    PETSc.Sys.Print('  2). using iterate method to optimize force and torque free')
    # problem.print_info()
    ref_U, _, _ = problem.do_iterate(ini_refU1=ref_U)
    ecoli_comp.set_ref_U(ref_U)
    print_single_ecoli_force_result(ecoli_comp, prefix='', part='full', **problem_kwargs)
    PETSc.Sys.Print('    ref_U %s' % str(ref_U))
    PETSc.Sys.Print('    |ref_U| %s, %s' % (str(np.linalg.norm(ref_U[:3])), str(np.linalg.norm(ref_U[3:]))))
    PETSc.Sys.Print('    U_head %s' % str(ref_U + ecoli_comp.get_rel_U_list()[0]))
    PETSc.Sys.Print('    U_tail %s' % str(ref_U + ecoli_comp.get_rel_U_list()[1]))
    sumF = np.sum(np.vstack([tobj.get_total_force(center=center) for tobj in ecoli_comp.get_obj_list()]), axis=0)
    PETSc.Sys.Print('    check, sumF is %s' % str(sumF))
    PETSc.Sys.Print('    check, sumF/headF is %s' %
                    str(sumF / ecoli_comp.get_obj_list()[0].get_total_force(center=center)))

    # 3) iterate force and torque free and iterate velocity at the same time.
    PETSc.Sys.Print()
    PETSc.Sys.Print('  3). iterate force and torque free and iterate velocity at the same time. ')
    err_ref_u = np.inf
    tail_rel_U = ecoli_comp.get_rel_U_list()[1]
    tstp = 0
    while err_ref_u > v_ecoli_tor:
        tstp = tstp + 1
        problem.create_F_U()
        ref_U, _, _ = problem.do_iterate(tolerate=iter_tor, ini_refU1=ref_U)
        ecoli_comp.set_ref_U(ref_U)
        norm_ref_u = np.linalg.norm(ref_U[:3] - U_pass[:3])
        err_ref_u = np.abs((norm_ref_u - ecoli_velocity) / ecoli_velocity)
        tail_rel_U = tail_rel_U * ecoli_velocity / norm_ref_u
        tmp_rel_U_list = [rel_Us, tail_rel_U]
        ecoli_comp.dbg_set_rel_U_list(tmp_rel_U_list)
        PETSc.Sys.Print('    ---- STEP %03d ----' % tstp)
        PETSc.Sys.Print('    ref_U', ref_U)
        PETSc.Sys.Print('    norm_ref_U %f, err_ref_U %f' % (norm_ref_u, err_ref_u))
        PETSc.Sys.Print('    tail_rel_U', tail_rel_U)
    print_single_ecoli_force_result(ecoli_comp, prefix='', part='full', **problem_kwargs)
    return True

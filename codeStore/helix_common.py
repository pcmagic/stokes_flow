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

__all__ = ['get_problem_kwargs', 'print_case_info', 'AtBtCt']


def get_problem_kwargs(**main_kwargs):
    problem_kwargs = get_solver_kwargs()

    kwargs_list = (get_vtk_tetra_kwargs(), get_helix_kwargs(), get_forcefree_kwargs(), main_kwargs,)
    for t_kwargs in kwargs_list:
        for key in t_kwargs:
            problem_kwargs[key] = t_kwargs[key]
    return problem_kwargs


def print_case_info(caseIntro='-->(some introduce here)', **problem_kwargs):
    fileHandle = problem_kwargs['fileHandle']
    PETSc.Sys.Print(caseIntro)
    print_solver_info(**problem_kwargs)
    print_forcefree_info(**problem_kwargs)
    print_helix_info(fileHandle, **problem_kwargs)
    return True


def AtBtCt(problem: 'sf.StokesFlowProblem', pick_M=False):
    def _set_rigid_velocity(problem: 'sf.StokesFlowProblem', u, w):
        for obji in problem.get_obj_list():
            u_geo = obji.get_u_geo()
            tnorm = u_geo.get_geo_norm()
            U = np.hstack((tnorm * u, tnorm * w))
            u_geo.set_rigid_velocity(U)
        return True

    def _get_total_force(problem: 'sf.StokesFlowProblem'):
        f_info = []
        F_all = 0
        T_all = 0
        for obj in problem.get_obj_list():
            f_geo = obj.get_f_geo()
            tnorm = f_geo.get_geo_norm()
            center = f_geo.get_origin()
            r = f_geo.get_nodes() - center
            f = obj.get_force().reshape((-1, obj.get_n_unknown()))
            t = np.cross(r, f[:, :3])
            F_all += (f_geo.get_deltaLength() * f.T).sum(axis=-1) * tnorm
            T_all += (f_geo.get_deltaLength() * t.T).sum(axis=-1) * tnorm
            f_info.append((obj.get_f_nodes(), obj.get_force()))
        return F_all, T_all, f_info

    problem_kwargs = problem.get_kwargs()
    fileHandle = problem_kwargs['fileHandle']
    # translation
    u, w, t1 = 1, 0, 'tran'
    _set_rigid_velocity(problem, u, w)
    problem.create_F_U()
    problem.solve()
    if problem_kwargs['pickProblem']:
        problem.pickmyself('%s_%s' % (fileHandle, t1), pick_M=pick_M, mat_destroy=False)
    At, Bt1, ftr_info = _get_total_force(problem)
    At = np.linalg.norm(At)
    Bt1 = np.linalg.norm(Bt1)
    problem.vtk_self('%s_%s' % (fileHandle, t1))

    # rotation
    u, w, t1 = 0, 1, 'rot'
    _set_rigid_velocity(problem, u, w)
    problem.create_F_U()
    problem.solve()
    if problem_kwargs['pickProblem']:
        problem.pickmyself('%s_%s' % (fileHandle, t1), pick_M=pick_M, mat_destroy=False)
    Bt2, Ct, frt_info = _get_total_force(problem)
    Ct = np.linalg.norm(Ct)
    Bt2 = np.linalg.norm(Bt2)
    problem.vtk_self('%s_%s' % (fileHandle, t1))

    Bt = np.mean((Bt1, Bt2))
    PETSc.Sys.Print('-->>At=%f, Bt=%f, Ct=%f, rel_err of Bt is %e' % (At, Bt, Ct, (Bt1 - Bt2) / Bt))
    return At, Bt, Ct, ftr_info, frt_info

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
from src import geo

__all__ = ['get_problem_kwargs', 'print_case_info',
           'AtBtCt', 'AtBtCt_full', 'AtBtCt_full_dbg', 'AtBtCt_selfRotate', 'AtBtCt_multiObj']


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


def AtBtCt(problem: 'sf.StokesFlowProblem', pick_M=False, save_vtk=True,
           center=np.zeros(3), norm=None, print_each=False):
    def _set_rigid_velocity(problem: 'sf.StokesFlowProblem', u, w):
        U = np.hstack((norm * u, norm * w))
        for obji in problem.get_obj_list():
            u_geo = obji.get_u_geo()
            u_geo.set_rigid_velocity(U, center=center)
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
            tF = (f_geo.get_deltaLength() * f.T).sum(axis=-1)
            tT = (f_geo.get_deltaLength() * t.T).sum(axis=-1)
            # print(f_geo.get_deltaLength())
            F_all += tF * tnorm
            T_all += tT * tnorm
            f_info.append((obj.get_f_nodes(), obj.get_force()))
            if print_each:
                PETSc.Sys.Print('--->>%s, tF: %s, tT: %s' % (obj.get_name(), str(tF), str(tT)))
        return F_all, T_all, f_info

    if norm is None:
        tnorm = problem.get_obj_list()[0].get_u_geo().get_geo_norm()
        for obji in problem.get_obj_list():
            tnorm2 = obji.get_u_geo().get_geo_norm()
            err_msg = '%s, %s' % (tnorm, tnorm2)
            assert np.allclose(tnorm, tnorm2), err_msg
        norm = tnorm
    assert np.isclose(np.linalg.norm(norm), 1)

    problem_kwargs = problem.get_kwargs()
    fileHandle = problem_kwargs['fileHandle']
    # translation
    u, w, t1 = 1, 0, 'tran'
    _set_rigid_velocity(problem, u, w)
    # problem.show_f_u_nodes()
    # problem.show_velocity()
    # assert 1 == 2
    problem.create_F_U()
    problem.solve()
    if problem_kwargs['pickProblem']:
        problem.pickmyself('%s_%s' % (fileHandle, t1), pick_M=pick_M, mat_destroy=False)
    At, Bt2, ftr_info = _get_total_force(problem)
    At = np.linalg.norm(At)
    Bt2 = np.linalg.norm(Bt2)
    if save_vtk:
        problem.vtk_self('%s_%s' % (fileHandle, t1))

    # rotation
    u, w, t1 = 0, 1, 'rot'
    _set_rigid_velocity(problem, u, w)
    problem.create_F_U()
    problem.solve()
    if problem_kwargs['pickProblem']:
        problem.pickmyself('%s_%s' % (fileHandle, t1), pick_M=pick_M, mat_destroy=False)
    Bt1, Ct, frt_info = _get_total_force(problem)
    Ct = np.linalg.norm(Ct)
    Bt1 = np.linalg.norm(Bt1)
    if save_vtk:
        problem.vtk_self('%s_%s' % (fileHandle, t1))

    Bt = np.mean((Bt1, Bt2))
    err_Bt = (Bt1 - Bt2) / Bt
    PETSc.Sys.Print('-->>At=%f, Bt=%f, Ct=%f' % (At, Bt, Ct))
    PETSc.Sys.Print('-->>  Bt1=%f, Bt2=%f, rel_err of Bt is %e' % (Bt1, Bt2, err_Bt))
    # print('-->>At=%f, Bt=%f, Ct=%f' % (At, Bt, Ct))
    # print('-->>  Bt1=%f, Bt2=%f, rel_err of Bt is %e' % (Bt1, Bt2, err_Bt))
    return At, Bt, Ct, ftr_info, frt_info


def AtBtCt_SBT(problem: 'sf.StokesFlowProblem', pick_M=False, save_vtk=True,
               center=np.zeros(3), norm=None, print_each=False):
    def _set_rigid_velocity(problem: 'sf.StokesFlowProblem', u, w):
        U = np.hstack((norm * u, norm * w))
        for obji in problem.get_obj_list():
            u_geo = obji.get_u_geo()
            u_geo.set_rigid_velocity(U, center=center)
        return True

    def _get_total_force(problem: 'sf.StokesFlowProblem'):
        f_info = []
        F_all = 0
        T_all = 0
        for obj in problem.get_obj_list():
            f_geo = obj.get_f_geo()  # type: geo.slb_geo
            tnorm = f_geo.get_geo_norm()
            center = f_geo.get_origin()
            r = f_geo.get_nodes() - center
            f = obj.get_force().reshape((-1, obj.get_n_unknown()))[:, :3]
            # d = 0
            t = np.cross(r, f)
            tF = (f_geo.get_deltaLength() * f.T).sum(axis=-1)
            tT = (f_geo.get_deltaLength() * t.T).sum(axis=-1)
            F_all += tF * tnorm
            T_all += tT * tnorm
            f_info.append((obj.get_f_nodes(), obj.get_force()))
            if print_each:
                PETSc.Sys.Print('--->>%s, tF: %s, tT: %s' % (obj.get_name(), str(tF), str(tT)))
        return F_all, T_all, f_info

    if norm is None:
        tnorm = problem.get_obj_list()[0].get_u_geo().get_geo_norm()
        for obji in problem.get_obj_list():
            tnorm2 = obji.get_u_geo().get_geo_norm()
            err_msg = '%s, %s' % (tnorm, tnorm2)
            assert np.allclose(tnorm, tnorm2), err_msg
        norm = tnorm
    assert np.isclose(np.linalg.norm(norm), 1)

    problem_kwargs = problem.get_kwargs()
    fileHandle = problem_kwargs['fileHandle']
    # translation
    u, w, t1 = 1, 0, 'tran'
    _set_rigid_velocity(problem, u, w)
    # problem.show_f_u_nodes()
    # problem.show_velocity()
    # assert 1 == 2
    problem.create_F_U()
    problem.solve()
    if problem_kwargs['pickProblem']:
        problem.pickmyself('%s_%s' % (fileHandle, t1), pick_M=pick_M, mat_destroy=False)
    At, Bt2, ftr_info = _get_total_force(problem)
    At = np.linalg.norm(At)
    Bt2 = np.linalg.norm(Bt2)
    if save_vtk:
        problem.vtk_self('%s_%s' % (fileHandle, t1))

    # rotation
    u, w, t1 = 0, 1, 'rot'
    _set_rigid_velocity(problem, u, w)
    problem.create_F_U()
    problem.solve()
    if problem_kwargs['pickProblem']:
        problem.pickmyself('%s_%s' % (fileHandle, t1), pick_M=pick_M, mat_destroy=False)
    Bt1, Ct, frt_info = _get_total_force(problem)
    Ct = np.linalg.norm(Ct)
    Bt1 = np.linalg.norm(Bt1)
    if save_vtk:
        problem.vtk_self('%s_%s' % (fileHandle, t1))

    Bt = np.mean((Bt1, Bt2))
    err_Bt = (Bt1 - Bt2) / Bt
    PETSc.Sys.Print('-->>At=%f, Bt=%f, Ct=%f' % (At, Bt, Ct))
    PETSc.Sys.Print('-->>  Bt1=%f, Bt2=%f, rel_err of Bt is %e' % (Bt1, Bt2, err_Bt))
    # print('-->>At=%f, Bt=%f, Ct=%f' % (At, Bt, Ct))
    # print('-->>  Bt1=%f, Bt2=%f, rel_err of Bt is %e' % (Bt1, Bt2, err_Bt))
    return At, Bt, Ct, ftr_info, frt_info


def AtBtCt_full(problem: 'sf.StokesFlowProblem', pick_M=False, save_vtk=True,
                center=np.zeros(3), print_each=False, save_name=None,
                u_use=1, w_use=1, uNormFct=1, wNormFct=1, uwNormFct=1):
    def _set_rigid_velocity(problem: 'sf.StokesFlowProblem', u, w, center, norm):
        U = np.hstack((norm * u, norm * w))
        for obji in problem.get_obj_list():
            u_geo = obji.get_u_geo()
            u_geo.set_rigid_velocity(U, center=center)
        return True

    def _get_total_force(problem: 'sf.StokesFlowProblem'):
        f_info = []
        F_all = 0
        T_all = 0
        for obj in problem.get_obj_list():
            f_geo = obj.get_f_geo()
            r = f_geo.get_nodes() - center
            # PETSc.Sys.Print(r.min(), r.max())
            f = obj.get_force().reshape((-1, obj.get_n_unknown()))
            t = np.cross(r, f[:, :3])
            # tF = (f_geo.get_deltaLength() * f.T).sum(axis=-1)
            # tT = (f_geo.get_deltaLength() * t.T).sum(axis=-1)
            tF = f.sum(axis=0)
            tT = t.sum(axis=0)
            F_all += tF
            T_all += tT
            f_info.append(obj.get_force())
            if print_each:
                PETSc.Sys.Print('--->>%s, tF: %s, tT: %s' % (obj.get_name(), str(tF), str(tT)))
                # PETSc.Sys.Print(obj.get_force()[:10])
        return F_all, T_all, f_info

    def _do_solve_once(u, w, t1, norm):
        _set_rigid_velocity(problem, u, w, center, norm)
        problem.create_F_U()
        problem.solve()
        if problem_kwargs['pickProblem']:
            problem.pickmyself('%s_%s' % (fileHandle, t1), pick_M=pick_M, mat_destroy=False)
        F_all, T_all, f_info = _get_total_force(problem)
        if save_vtk:
            problem.vtk_self('%s_%s' % (fileHandle, t1))
        return F_all, T_all, f_info

    def _do_solve_case(norm, thandle, u_use=1, w_use=1):
        u, w, t1 = u_use, 0, 'tran_%s' % thandle
        F_all, T_all, f_info = _do_solve_once(u, w, t1, norm)
        At_list.append(F_all / u_use)
        Bt2_list.append(T_all / u_use)
        f_info_list.append(f_info)
        u, w, t1 = 0, w_use, 'rot_%s' % thandle
        F_all, T_all, f_info = _do_solve_once(u, w, t1, norm)
        Bt1_list.append(F_all / w_use)
        Ct_list.append(T_all / w_use)
        f_info_list.append(f_info)
        return True

    problem_kwargs = problem.get_kwargs()
    fileHandle = problem_kwargs['fileHandle']
    At_list = []
    Bt1_list = []
    Bt2_list = []
    Ct_list = []
    f_info_list = []

    _do_solve_case(norm=np.array((1, 0, 0)), thandle='100', u_use=u_use, w_use=w_use)
    _do_solve_case(norm=np.array((0, 1, 0)), thandle='010', u_use=u_use, w_use=w_use)
    _do_solve_case(norm=np.array((0, 0, 1)), thandle='001', u_use=u_use, w_use=w_use)
    At = np.vstack(At_list).T
    Bt1 = np.vstack(Bt1_list).T
    Bt2 = np.vstack(Bt2_list).T
    Ct = np.vstack(Ct_list).T

    PETSc.Sys.Print('-->>At / %f = ' % uNormFct)
    PETSc.Sys.Print(At / uNormFct)
    PETSc.Sys.Print('-->>Bt1 / %f = ' % uwNormFct)
    PETSc.Sys.Print(Bt1 / uwNormFct)
    PETSc.Sys.Print('-->>Bt2 / %f = ' % uwNormFct)
    PETSc.Sys.Print(Bt2 / uwNormFct)
    PETSc.Sys.Print('-->>Bt rel err')
    t1 = np.abs(Bt1 - Bt2.T) / np.linalg.norm(Bt1 + Bt2.T) * 2
    PETSc.Sys.Print(t1)
    PETSc.Sys.Print('-->>Ct / %f = ' % wNormFct)
    PETSc.Sys.Print(Ct / wNormFct)

    if save_name is not None:
        comm = PETSc.COMM_WORLD.tompi4py()
        rank = comm.Get_rank()
        save_name = check_file_extension(save_name, '.pickle')
        tpickle = [problem_kwargs, At, Bt1, Bt2, Ct, ]
        if rank == 0:
            with open(save_name, 'wb') as output:
                pickle.dump(tpickle, output, protocol=4)
            print('save AtBtCt data to %s' % save_name)
    return At, Bt1, Bt2, Ct


def AtBtCt_full_dbg(problem: 'sf.StokesFlowProblem', pick_M=False, save_vtk=True,
                    center=np.zeros(3), print_each=False, save_name=None,
                    u_use=1, w_use=1, uNormFct=1, wNormFct=1, uwNormFct=1):
    def _set_rigid_velocity(problem: 'sf.StokesFlowProblem', u, w, center, norm):
        U = np.hstack((norm * u, norm * w))
        for obji in problem.get_obj_list()[:-1]:
            u_geo = obji.get_u_geo()
            u_geo.set_rigid_velocity(U, center=center)
        return True

    def _get_total_force(problem: 'sf.StokesFlowProblem'):
        f_info = []
        F_all = 0
        T_all = 0
        for obj in problem.get_obj_list():
            f_geo = obj.get_f_geo()
            r = f_geo.get_nodes() - center
            # PETSc.Sys.Print(r.min(), r.max())
            f = obj.get_force().reshape((-1, obj.get_n_unknown()))
            t = np.cross(r, f[:, :3])
            # tF = (f_geo.get_deltaLength() * f.T).sum(axis=-1)
            # tT = (f_geo.get_deltaLength() * t.T).sum(axis=-1)
            tF = f.sum(axis=0)
            tT = t.sum(axis=0)
            F_all += tF
            T_all += tT
            f_info.append(obj.get_force())
            if print_each:
                PETSc.Sys.Print('--->>%s, tF: %s, tT: %s' % (obj.get_name(), str(tF), str(tT)))
                # PETSc.Sys.Print(obj.get_force()[:10])
        return F_all, T_all, f_info

    def _do_solve_once(u, w, t1, norm):
        _set_rigid_velocity(problem, u, w, center, norm)
        problem.create_F_U()
        problem.solve()
        if problem_kwargs['pickProblem']:
            problem.pickmyself('%s_%s' % (fileHandle, t1), pick_M=pick_M, mat_destroy=False)
        F_all, T_all, f_info = _get_total_force(problem)
        if save_vtk:
            problem.vtk_self('%s_%s' % (fileHandle, t1))
        return F_all, T_all, f_info

    def _do_solve_case(norm, thandle, u_use=1, w_use=1):
        u, w, t1 = u_use, 0, 'tran_%s' % thandle
        F_all, T_all, f_info = _do_solve_once(u, w, t1, norm)
        At_list.append(F_all / u_use)
        Bt2_list.append(T_all / u_use)
        f_info_list.append(f_info)
        u, w, t1 = 0, w_use, 'rot_%s' % thandle
        F_all, T_all, f_info = _do_solve_once(u, w, t1, norm)
        Bt1_list.append(F_all / w_use)
        Ct_list.append(T_all / w_use)
        f_info_list.append(f_info)
        return True

    problem_kwargs = problem.get_kwargs()
    fileHandle = problem_kwargs['fileHandle']
    At_list = []
    Bt1_list = []
    Bt2_list = []
    Ct_list = []
    f_info_list = []

    _do_solve_case(norm=np.array((1, 0, 0)), thandle='100', u_use=u_use, w_use=w_use)
    _do_solve_case(norm=np.array((0, 1, 0)), thandle='010', u_use=u_use, w_use=w_use)
    _do_solve_case(norm=np.array((0, 0, 1)), thandle='001', u_use=u_use, w_use=w_use)
    At = np.vstack(At_list).T
    Bt1 = np.vstack(Bt1_list).T
    Bt2 = np.vstack(Bt2_list).T
    Ct = np.vstack(Ct_list).T

    PETSc.Sys.Print('################## This is DBG code ##################')
    PETSc.Sys.Print('-->>At / %f = ' % uNormFct)
    PETSc.Sys.Print(At / uNormFct)
    PETSc.Sys.Print('-->>Bt1 / %f = ' % uwNormFct)
    PETSc.Sys.Print(Bt1 / uwNormFct)
    PETSc.Sys.Print('-->>Bt2 / %f = ' % uwNormFct)
    PETSc.Sys.Print(Bt2 / uwNormFct)
    PETSc.Sys.Print('-->>Bt rel err')
    t1 = np.abs(Bt1 - Bt2.T) / np.linalg.norm(Bt1 + Bt2.T) * 2
    PETSc.Sys.Print(t1)
    PETSc.Sys.Print('-->>Ct / %f = ' % wNormFct)
    PETSc.Sys.Print(Ct / wNormFct)

    if save_name is not None:
        comm = PETSc.COMM_WORLD.tompi4py()
        rank = comm.Get_rank()
        save_name = check_file_extension(save_name, '.pickle')
        tpickle = [problem_kwargs, At, Bt1, Bt2, Ct, ]
        if rank == 0:
            with open(save_name, 'wb') as output:
                pickle.dump(tpickle, output, protocol=4)
            print('save AtBtCt data to %s' % save_name)
    return At, Bt1, Bt2, Ct


def AtBtCt_multiObj(problem: 'sf.StokesFlowProblem', pick_M=False, save_vtk=True,
                    print_each=False, save_name=None,
                    u_use=1, w_use=1, uNormFct=1, wNormFct=1, uwNormFct=1):
    def _do_solve_once(obj, u, w, t1, norm, tfct):
        for tobj in problem.get_obj_list():
            tobj.get_u_geo().set_rigid_velocity(np.zeros(6))
        U = np.hstack((norm * u, norm * w))
        u_geo = obj.get_u_geo()
        center = u_geo.get_center()
        u_geo.set_rigid_velocity(U, center=center)

        problem.create_F_U()
        problem.solve()
        if problem_kwargs['pickProblem']:
            problem.pickmyself('%s_%s' % (fileHandle, t1), pick_M=pick_M, mat_destroy=False)
        if save_vtk:
            problem.vtk_self('%s_%s' % (fileHandle, t1))

        F_list = []
        T_list = []
        for tobj in problem.get_obj_list():
            center = tobj.get_u_geo().get_center()
            tFT = tobj.get_total_force(center=center) / tfct
            tF = tFT[:3]
            tT = tFT[3:]
            F_list.append(tF)
            T_list.append(tT)
            if print_each:
                PETSc.Sys.Print('--->>%s, tF: %s' % (tobj.get_name(), str(tF)))
                PETSc.Sys.Print('--->>%s, tT: %s' % (tobj.get_name(), str(tT)))
        return F_list, T_list

    def _do_solve_case(obj, norm, thandle, u_use, w_use):
        u, w, t1 = u_use, 0, 'tran_%s' % thandle
        F_all, T_all = _do_solve_once(obj, u, w, t1, norm, u_use)
        for tlist, tF in zip(obj_At_list, F_all):
            tlist.append(tF)
        for tlist, tF in zip(obj_Bt2_list, T_all):
            tlist.append(tF)

        u, w, t1 = 0, w_use, 'rot_%s' % thandle
        F_all, T_all = _do_solve_once(obj, u, w, t1, norm, w_use)
        for tlist, tF in zip(obj_Bt1_list, F_all):
            tlist.append(tF)
        for tlist, tF in zip(obj_Ct_list, T_all):
            tlist.append(tF)
        return True

    problem_kwargs = problem.get_kwargs()
    fileHandle = problem_kwargs['fileHandle']
    At_list = []
    Bt1_list = []
    Bt2_list = []
    Ct_list = []

    for tobj0 in problem.get_obj_list():
        obj_At_list = [[] for _ in problem.get_obj_list()]
        obj_Bt1_list = [[] for _ in problem.get_obj_list()]
        obj_Bt2_list = [[] for _ in problem.get_obj_list()]
        obj_Ct_list = [[] for _ in problem.get_obj_list()]
        _do_solve_case(tobj0, norm=np.array((1, 0, 0)), thandle='100', u_use=u_use, w_use=w_use)
        _do_solve_case(tobj0, norm=np.array((0, 1, 0)), thandle='010', u_use=u_use, w_use=w_use)
        _do_solve_case(tobj0, norm=np.array((0, 0, 1)), thandle='001', u_use=u_use, w_use=w_use)

        obj_At_list2 = []
        obj_Bt1_list2 = []
        obj_Bt2_list2 = []
        obj_Ct_list2 = []
        for tAt, tBt1, tBt2, tCt in zip(obj_At_list, obj_Bt1_list, obj_Bt2_list, obj_Ct_list):
            obj_At_list2.append(np.vstack(tAt).T)
            obj_Bt1_list2.append(np.vstack(tBt1).T)
            obj_Bt2_list2.append(np.vstack(tBt2).T)
            obj_Ct_list2.append(np.vstack(tCt).T)

        for tobj1, tAt, tBt1, tBt2, tCt in zip(problem.get_obj_list(),
                                               obj_At_list2, obj_Bt1_list2,
                                               obj_Bt2_list2, obj_Ct_list2):
            PETSc.Sys.Print('%s - %s' % (str(tobj0), str(tobj1)))
            PETSc.Sys.Print('-->>At / %f = ' % uNormFct)
            PETSc.Sys.Print(np.vstack(tAt) / uNormFct)
            PETSc.Sys.Print('-->>Bt1 / %f = ' % uwNormFct)
            PETSc.Sys.Print(np.vstack(tBt1) / uwNormFct)
            PETSc.Sys.Print('-->>Bt2 / %f = ' % uwNormFct)
            PETSc.Sys.Print(np.vstack(tBt2) / uwNormFct)
            PETSc.Sys.Print('-->>Bt rel err')
            t1 = np.abs(tBt1 - tBt2.T) / np.linalg.norm(tBt1 + tBt2.T) * 2
            PETSc.Sys.Print(t1)
            PETSc.Sys.Print('-->>Ct / %f = ' % wNormFct)
            PETSc.Sys.Print(np.vstack(tCt) / wNormFct)

        At_list.append(obj_At_list2)
        Bt1_list.append(obj_Bt1_list2)
        Bt2_list.append(obj_Bt2_list2)
        Ct_list.append(obj_Ct_list2)

    if save_name is not None:
        comm = PETSc.COMM_WORLD.tompi4py()
        rank = comm.Get_rank()
        save_name = check_file_extension(save_name, '.pickle')
        tpickle = [problem_kwargs, At_list, Bt1_list, Bt2_list, Ct_list, ]
        if rank == 0:
            with open(save_name, 'wb') as output:
                pickle.dump(tpickle, output, protocol=4)
            print('save AtBtCt data to %s' % save_name)
    return At_list, Bt1_list, Bt2_list, Ct_list


def AtBtCt_selfRotate(problem: 'sf.SelfRotateProblem', pick_M=False, save_vtk=True,
                      center=np.zeros(3), print_each=False, save_name=None,
                      u_use=1, w_use=1):
    def _get_total_force(problem: 'sf.SelfRotateProblem'):
        f_info = []
        F_all = 0
        T_all = 0
        for obj in problem.get_obj_list():
            f_t = obj.get_total_force(center=center)
            tF = f_t[:3]
            tT = f_t[3:]
            F_all += tF
            T_all += tT
            f_info.append(obj.get_force())
            if print_each:
                PETSc.Sys.Print('--->>%s, tF: %s, tT: %s' % (obj.get_name(), str(tF), str(tT)))
                # PETSc.Sys.Print(obj.get_force()[:10])
        return F_all, T_all, f_info

    def _do_solve_once(u, w, t1):
        problem.set_rigid_velocity(u, w)
        problem.create_F_U()
        problem.solve()
        if problem_kwargs['pickProblem']:
            problem.pickmyself('%s_%s' % (fileHandle, t1), pick_M=pick_M, mat_destroy=False)
        F_all, T_all, f_info = _get_total_force(problem)
        if save_vtk:
            problem.vtk_self('%s_%s' % (fileHandle, t1))
        return F_all, T_all, f_info

    def _do_solve_case(thandle, u_use=1, w_use=1):
        u, w, t1 = u_use, 0, 'tran_%s' % thandle
        F_all, T_all, f_info = _do_solve_once(u, w, t1)
        At_list.append(F_all / u_use)
        Bt2_list.append(T_all / u_use)
        f_info_list.append(f_info)
        u, w, t1 = 0, w_use, 'rot_%s' % thandle
        F_all, T_all, f_info = _do_solve_once(u, w, t1)
        Bt1_list.append(F_all / w_use)
        Ct_list.append(T_all / w_use)
        f_info_list.append(f_info)
        return True

    problem_kwargs = problem.get_kwargs()
    fileHandle = problem_kwargs['fileHandle']
    At_list = []
    Bt1_list = []
    Bt2_list = []
    Ct_list = []
    f_info_list = []

    _do_solve_case(thandle='100', u_use=u_use, w_use=w_use)
    At = At_list[0]
    Bt1 = Bt1_list[0]
    Bt2 = Bt2_list[0]
    Ct = Ct_list[0]

    # t1 = np.abs(Bt1 - Bt2.T) / np.linalg.norm(Bt1 + Bt2.T) * 2
    # PETSc.Sys.Print('-->>At = %f, Bt1 = %f, Bt2 = %f, Ct = %f' % (At, Bt1, Bt2, Ct))
    # PETSc.Sys.Print('-->>Bt rel err = %f' % t1)
    PETSc.Sys.Print('-->>At')
    PETSc.Sys.Print(At)
    PETSc.Sys.Print('-->>Bt1')
    PETSc.Sys.Print(Bt1)
    PETSc.Sys.Print('-->>Bt2')
    PETSc.Sys.Print(Bt2)
    PETSc.Sys.Print('-->>Bt rel err')
    t1 = np.abs(Bt1 - Bt2.T) / np.linalg.norm(Bt1 + Bt2.T) * 2
    PETSc.Sys.Print(t1)
    PETSc.Sys.Print('-->>Ct')
    PETSc.Sys.Print(Ct)

    if save_name is not None:
        comm = PETSc.COMM_WORLD.tompi4py()
        rank = comm.Get_rank()
        save_name = check_file_extension(save_name, '.pickle')
        tpickle = [problem_kwargs, At, Bt1, Bt2, Ct, ]
        if rank == 0:
            with open(save_name, 'wb') as output:
                pickle.dump(tpickle, output, protocol=4)
            print('save AtBtCt data to %s' % save_name)
    return At, Bt1, Bt2, Ct

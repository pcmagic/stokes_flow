import sys

import petsc4py

petsc4py.init(sys.argv)

import numpy as np
import pickle
# from time import time
# from scipy.io import loadmat
# from src.stokes_flow import problem_dic, obj_dic
# from src.geo import *
from petsc4py import PETSc
from src import stokes_flow as sf
from src.myio import *
from src.objComposite import *
# from src.myvtk import *
from src.support_class import *
from codeStore import ecoli_common
from codeStore.helix_common import *


def get_problem_kwargs(**main_kwargs):
    problem_kwargs = ecoli_common.get_problem_kwargs()
    OptDB = PETSc.Options()
    fileHandle = OptDB.getString('f', 'ecoliNearSphere')
    problem_kwargs['fileHandle'] = fileHandle

    kwargs_list = (main_kwargs,)
    for t_kwargs in kwargs_list:
        for key in t_kwargs:
            problem_kwargs[key] = t_kwargs[key]

    ecoli_ini_rot_theta = OptDB.getReal('ecoli_ini_rot_theta', 0)
    ecoli_ini_rot_phi = OptDB.getReal('ecoli_ini_rot_phi', 0)
    ecoli_ini_rot_psi = OptDB.getReal('ecoli_ini_rot_psi', 0)
    problem_kwargs['ecoli_ini_rot_theta'] = ecoli_ini_rot_theta
    problem_kwargs['ecoli_ini_rot_phi'] = ecoli_ini_rot_phi
    problem_kwargs['ecoli_ini_rot_psi'] = ecoli_ini_rot_psi

    tail_ini_beta = OptDB.getReal('tail_ini_beta', 0)
    tail_ini_theta = OptDB.getReal('tail_ini_theta', 0)
    tail_ini_phi = OptDB.getReal('tail_ini_phi', 0)
    tail_ini_psi = OptDB.getReal('tail_ini_psi', 0)
    tail_rot_delta = OptDB.getReal('tail_rot_delta', 0)
    problem_kwargs['tail_ini_beta'] = tail_ini_beta
    problem_kwargs['tail_ini_theta'] = tail_ini_theta
    problem_kwargs['tail_ini_phi'] = tail_ini_phi
    problem_kwargs['tail_ini_psi'] = tail_ini_psi
    problem_kwargs['tail_rot_delta'] = tail_rot_delta
    return problem_kwargs


def print_case_info(**problem_kwargs):
    assert np.isclose(problem_kwargs['zoom_factor'], 1)
    assert np.isclose(problem_kwargs['rot_theta'], 0)

    PETSc.Sys.Print()
    PETSc.Sys.Print('Input information')
    caseIntro = 'ecoli with skew tail, force and toque free case. '
    t1 = ecoli_common.print_case_info(caseIntro=caseIntro, **problem_kwargs)

    ecoli_ini_rot_theta = problem_kwargs['ecoli_ini_rot_theta']
    ecoli_ini_rot_phi = problem_kwargs['ecoli_ini_rot_phi']
    ecoli_ini_rot_psi = problem_kwargs['ecoli_ini_rot_psi']
    tail_ini_beta = problem_kwargs['tail_ini_beta']
    tail_ini_theta = problem_kwargs['tail_ini_theta']
    tail_ini_phi = problem_kwargs['tail_ini_phi']
    tail_ini_psi = problem_kwargs['tail_ini_psi']
    tail_rot_delta = problem_kwargs['tail_rot_delta']
    PETSc.Sys.Print('initial position of microswimmer: ')
    PETSc.Sys.Print('  ecoli rotation, theta: %f, phi: %f, psi: %f. ' %
                    (ecoli_ini_rot_theta, ecoli_ini_rot_phi, ecoli_ini_rot_psi))
    PETSc.Sys.Print('  tail rotation, beta: %f, delta: %f' % (tail_ini_beta, tail_rot_delta))
    PETSc.Sys.Print('  tail rotation, theta: %f, phi: %f, psi: %f. ' %
                    (tail_ini_theta, tail_ini_phi, tail_ini_psi))
    return t1


def _check_geo_available(ecoli_comp, **problem_kwargs):
    # assert tail is outside the body.
    rs1 = problem_kwargs['rs1']
    rs2 = problem_kwargs['rs2']

    head_obj, tail_obj = ecoli_comp.get_obj_list()
    head_center = head_obj.get_u_geo().get_center()
    head_norm = head_obj.get_u_geo().get_geo_norm()
    tX = tail_obj.get_u_geo().get_nodes() - head_center
    tX_t_norm = np.einsum('ji, i', tX, head_norm) / np.linalg.norm(head_norm)
    tX_t_norm2 = tX_t_norm ** 2
    tX_n_norm2 = np.linalg.norm(tX, axis=-1) ** 2 - tX_t_norm2
    ds = tX_t_norm2 / (rs1 ** 2) + tX_n_norm2 / (rs2 ** 2)
    err_msg = 'wrong tail ini angles. '
    assert np.all(ds > 1), err_msg
    return True


def create_skew_tail_comp(**problem_kwargs):
    ph = problem_kwargs['ph']
    ch = problem_kwargs['ch']
    # rel_Us = problem_kwargs['rel_Us']
    rel_Uh = problem_kwargs['rel_Uh']
    ecoli_ini_rot_theta = problem_kwargs['ecoli_ini_rot_theta']
    ecoli_ini_rot_phi = problem_kwargs['ecoli_ini_rot_phi']
    ecoli_ini_rot_psi = problem_kwargs['ecoli_ini_rot_psi']
    tail_ini_beta = problem_kwargs['tail_ini_beta']
    tail_ini_theta = problem_kwargs['tail_ini_theta']
    tail_ini_phi = problem_kwargs['tail_ini_phi']
    tail_ini_psi = problem_kwargs['tail_ini_psi']
    tail_rot_delta = problem_kwargs['tail_rot_delta']
    beta_norm = np.array([0, 1, 0])

    # rotate the tail relative velocity.
    rotM0 = Rloc2glb(tail_ini_theta, tail_ini_phi, tail_ini_psi)
    rotM1 = rot_vec2rot_mtx(beta_norm * tail_ini_beta)
    rel_Uh0 = np.hstack((np.dot(rotM0, rel_Uh[:3]),
                         np.dot(rotM0, rel_Uh[3:]),))
    rel_Uh1 = np.hstack((np.dot(rotM1, rel_Uh0[:3]),
                         np.dot(rotM1, rel_Uh0[3:]),))
    problem_kwargs['rel_Uh'] = rel_Uh1

    # generate ecoli composite
    ecoli_comp = create_ecoli_2part(**problem_kwargs)
    head_obj, tail_obj = ecoli_comp.get_obj_list()
    # rotate the tail object.
    head_center = head_obj.get_u_geo().get_center()
    tail_center = tail_obj.get_u_geo().get_center()
    tail_norm = tail_obj.get_u_geo().get_geo_norm()
    tail_end0 = tail_center + ph * ch / 2 * tail_norm
    tail_obj.node_rotM(rotM0, rotation_origin=tail_end0)
    tail_obj.node_rotation(norm=beta_norm, theta=tail_ini_beta,
                           rotation_origin=head_center)

    ecoli_comp.set_norm(head_obj.get_u_geo().get_geo_norm())
    ecoli_comp.set_center(head_obj.get_u_geo().get_center())
    _check_geo_available(ecoli_comp, **problem_kwargs)
    return ecoli_comp, problem_kwargs


def create_skew_tail_comp_v2(**problem_kwargs):
    # err_msg = 'this is dbg code. '
    # assert 1 == 2, err_msg
    ph = problem_kwargs['ph']
    ch = problem_kwargs['ch']
    # rel_Us = problem_kwargs['rel_Us']
    rel_Uh = problem_kwargs['rel_Uh']
    ecoli_ini_rot_theta = problem_kwargs['ecoli_ini_rot_theta']
    ecoli_ini_rot_phi = problem_kwargs['ecoli_ini_rot_phi']
    ecoli_ini_rot_psi = problem_kwargs['ecoli_ini_rot_psi']
    tail_ini_beta = problem_kwargs['tail_ini_beta']
    tail_ini_theta = problem_kwargs['tail_ini_theta']
    tail_ini_phi = problem_kwargs['tail_ini_phi']
    tail_ini_psi = problem_kwargs['tail_ini_psi']
    tail_rot_delta = problem_kwargs['tail_rot_delta']
    beta_norm = np.array([0, 1, 0])
    rs1 = problem_kwargs['rs1']
    rs2 = problem_kwargs['rs2']

    # rotate the tail relative velocity.
    rotM0 = Rloc2glb(tail_ini_theta, tail_ini_phi, tail_ini_psi)
    rotM1 = rot_vec2rot_mtx(beta_norm * tail_ini_beta)
    rel_Uh0 = np.hstack((np.dot(rotM0, rel_Uh[:3]),
                         np.dot(rotM0, rel_Uh[3:]),))
    rel_Uh1 = np.hstack((np.dot(rotM1, rel_Uh0[:3]),
                         np.dot(rotM1, rel_Uh0[3:]),))
    problem_kwargs['rel_Uh'] = rel_Uh1

    # generate ecoli composite
    ecoli_comp = create_ecoli_2part(**problem_kwargs)
    head_obj, tail_obj = ecoli_comp.get_obj_list()

    # rotate the tail object.
    head_center = head_obj.get_u_geo().get_center()
    head_norm = head_obj.get_u_geo().get_geo_norm()
    trs = rs1 * rs2 / np.sqrt((rs1 * np.sin(tail_ini_beta)) ** 2 +
                              (rs2 * np.cos(tail_ini_beta)) ** 2)
    head_end0 = head_center - trs * head_norm
    # tail_center = tail_obj.get_u_geo().get_center()
    # tail_norm = tail_obj.get_u_geo().get_geo_norm()
    # tail_end0 = tail_center + ph * ch / 2 * tail_norm

    # # dbg
    # rotation_origin = tail_end0
    # PETSc.Sys.Print('dbg code 3', head_center)
    # PETSc.Sys.Print('dbg code 3', head_norm)
    # PETSc.Sys.Print('dbg code 3', tail_center)
    # PETSc.Sys.Print('dbg code 3', tail_norm)
    # PETSc.Sys.Print('dbg code 3', head_end0)
    # PETSc.Sys.Print('dbg code 3', tail_end0)
    # ecoli_comp.show_u_nodes(linestyle='')
    # PETSc.Sys.Print('dbg code 4', tail_obj.get_u_geo().get_nodes()[:3])
    rotation_origin = head_end0
    tail_obj.node_rotM(rotM0, rotation_origin=rotation_origin)
    tail_obj.node_rotation(norm=beta_norm, theta=tail_ini_beta,
                           rotation_origin=head_center)
    # PETSc.Sys.Print('dbg code 5', tail_obj.get_u_geo().get_nodes()[:3])
    # PETSc.Sys.Print('dbg code 6', head_obj.get_u_geo().get_nodes()[:3])

    ecoli_comp.set_norm(head_obj.get_u_geo().get_geo_norm())
    ecoli_comp.set_center(head_obj.get_u_geo().get_center())
    _check_geo_available(ecoli_comp, **problem_kwargs)
    return ecoli_comp, problem_kwargs


def fun_ecoli_center(ecoli_comp: 'sf.ForceFreeComposite'):
    # https://math.stackexchange.com/questions/270767/find-intersection-of-two-3d-lines/271366
    head_obj, tail_obj = ecoli_comp.get_obj_list()
    D = head_obj.get_u_geo().get_center()
    C = tail_obj.get_u_geo().get_center()
    f = head_obj.get_u_geo().get_geo_norm()
    e = tail_obj.get_u_geo().get_geo_norm()
    g = D - C
    g = g / np.linalg.norm(g)
    h = np.linalg.norm(np.cross(f, g))
    k = np.linalg.norm(np.cross(f, e))
    assert not np.isclose(h, 0)
    assert not np.isclose(k, 0)
    assert np.isclose(np.linalg.norm(np.cross(np.cross(f, g), np.cross(f, e))), 0)
    signM = np.sign(np.dot(h, k))
    M = C + signM * (np.linalg.norm(np.cross(f, D - C))
                     / np.linalg.norm(np.cross(f, e))) * e
    # print(C, signM, D - C, f, e)
    # print(np.cross(f, D - C), np.cross(f, e))
    # print(M)
    ecoli_comp.set_center(M)

    # # dbg
    # # ecoli_comp.show_f_u_nodes(linestyle='')
    #
    # from matplotlib import pyplot as plt
    # from codeStore.support_fun import set_axes_equal
    # figsize, dpi = np.array((16, 9)) * 0.5, 100
    # fig, axi = plt.subplots(nrows=1, ncols=1, figsize=figsize, dpi=dpi,
    #                         subplot_kw=dict(projection='3d'))
    # t1 = np.linspace(-np.linalg.norm(D - C), np.linalg.norm(D - C))
    # th = D + np.einsum('i, j', t1, f)
    # tt = C + np.einsum('i, j', t1, e)
    # axi.plot(*th.T, '-b')
    # axi.plot(*tt.T, '-r')
    # axi.scatter(*M, marker='s', c='k')
    # axi.scatter(*head_obj.get_u_geo().get_center(), marker='s', c='b')
    # axi.scatter(*tail_obj.get_u_geo().get_center(), marker='s', c='r')
    # set_axes_equal(axi)
    # plt.show()
    return True


def create_skew_head_comp(**problem_kwargs):
    rel_Us = problem_kwargs['rel_Us']
    # rel_Uh = problem_kwargs['rel_Uh']
    ecoli_ini_rot_theta = problem_kwargs['ecoli_ini_rot_theta']
    ecoli_ini_rot_phi = problem_kwargs['ecoli_ini_rot_phi']
    ecoli_ini_rot_psi = problem_kwargs['ecoli_ini_rot_psi']
    tail_ini_beta = problem_kwargs['tail_ini_beta']
    tail_ini_theta = problem_kwargs['tail_ini_theta']
    tail_ini_phi = problem_kwargs['tail_ini_phi']
    tail_ini_psi = problem_kwargs['tail_ini_psi']
    tail_rot_delta = problem_kwargs['tail_rot_delta']
    beta_norm = np.array([0, 1, 0])
    rs1 = problem_kwargs['rs1']
    rs2 = problem_kwargs['rs2']

    # # dbg
    # tail_ini_beta, tail_ini_theta, tail_ini_phi, tail_ini_psi = \
    #     np.random.sample(4) * (2, 1, 2, 2) * np.pi
    # tail_ini_phi, tail_ini_psi = 0, 0
    # print('dbg random skew', tail_ini_beta, tail_ini_theta, tail_ini_phi, tail_ini_psi)

    # rotate the head relative velocity.
    rotM0 = Rloc2glb(tail_ini_theta, tail_ini_phi, tail_ini_psi)
    rotM1 = rot_vec2rot_mtx(beta_norm * -tail_ini_beta)
    rel_Us0 = np.hstack((np.dot(rotM0.T, rel_Us[:3]),
                         np.dot(rotM0.T, rel_Us[3:]),))
    rel_Us1 = np.hstack((np.dot(rotM1, rel_Us0[:3]),
                         np.dot(rotM1, rel_Us0[3:]),))
    problem_kwargs['rel_Us'] = rel_Us1

    # generate ecoli composite
    ecoli_comp = create_ecoli_2part(**problem_kwargs)
    head_obj, tail_obj = ecoli_comp.get_obj_list()
    head_center = head_obj.get_u_geo().get_center()
    head_norm = head_obj.get_u_geo().get_geo_norm()
    trs = rs1 * rs2 / np.sqrt((rs1 * np.sin(tail_ini_beta)) ** 2 +
                              (rs2 * np.cos(tail_ini_beta)) ** 2)
    head_end0 = head_center - trs * head_norm - tail_rot_delta * head_norm

    # rotate the head object in opposite position.
    rotM0 = Rloc2glb(tail_ini_theta, tail_ini_phi, tail_ini_psi)
    head_obj.node_rotation(norm=beta_norm, theta=-1 * tail_ini_beta,
                           rotation_origin=head_center)
    head_obj.node_rotM(rotM0.T, rotation_origin=head_end0)

    # # dbg code, rotate the ecoli composite back
    # OptDB = PETSc.Options()
    # dbg_head_rot = OptDB.getBool('dbg_head_rot', False)
    # PETSc.Sys.Print('dbg_head_rot', dbg_head_rot)
    # if dbg_head_rot:
    #     ecoli_comp.node_rotM(rotM0, rotation_origin=head_end0)
    #     ecoli_comp.node_rotation(norm=beta_norm, theta=tail_ini_beta,
    #                              rotation_origin=head_center)
    #     problem_kwargs['rel_Uh'] = ecoli_comp.get_rel_U_list()[1]

    ecoli_comp.set_norm(head_obj.get_u_geo().get_geo_norm())
    # fun_ecoli_center(ecoli_comp)
    ecoli_comp.set_center(head_obj.get_u_geo().get_center())
    _check_geo_available(ecoli_comp, **problem_kwargs)

    # # # dbg code, the center of ecoli_comp, 20201216
    # from matplotlib import pyplot as plt
    # from codeStore.support_fun import set_axes_equal
    # figsize, dpi = np.array((16, 9)) * 0.5, 100
    # fig, axi = plt.subplots(nrows=1, ncols=1, figsize=figsize, dpi=dpi,
    #                         subplot_kw=dict(projection='3d'))
    # for tobj in ecoli_comp.get_obj_list():
    #     tgeo = tobj.get_u_geo()
    #     axi.plot(tgeo.get_nodes_x(), tgeo.get_nodes_y(), tgeo.get_nodes_z(), '.')
    # axi.scatter(*head_end0, marker='s', c='k')
    # set_axes_equal(axi)
    # plt.show()

    # # dbg code, the center of ecoli_comp, 20201216
    # t1 = head_obj.get_u_geo().get_center() + np.array((1, 1, 1)) * 0
    # ecoli_comp.set_center(t1)

    # # dbg code, the center of ecoli_comp, 20201212
    # # OptDB = PETSc.Options()
    # # dbg_center_fct = OptDB.getReal('dbg_center_fct', 1)
    # dbg_center_fct = np.random.sample(1)[0]
    # PETSc.Sys.Print('dbg_center_fct', dbg_center_fct)
    # head_center = head_obj.get_u_geo().get_center()
    # tail_center = tail_obj.get_u_geo().get_center()
    # tcenter = dbg_center_fct * head_center + (1 - dbg_center_fct) * tail_center
    # # PETSc.Sys.Print(head_center, tail_center, tcenter)
    # ecoli_comp.set_center(tcenter)
    return ecoli_comp, problem_kwargs


def pickle_data(ecoli_comp, **problem_kwargs):
    fileHandle = problem_kwargs['fileHandle']
    rel_Us = problem_kwargs['rel_Us']
    rel_Uh = problem_kwargs['rel_Uh']

    ecoli_U = ecoli_comp.get_ref_U()
    head_obj = ecoli_comp.get_obj_list()[0]
    tail_obj = ecoli_comp.get_obj_list()[1]
    head_center = head_obj.get_u_geo().get_center()
    tail_center = tail_obj.get_u_geo().get_center()
    ecoli_center = ecoli_comp.get_center()
    drbc = head_center - ecoli_center
    drtc = tail_center - ecoli_center
    head_U = ecoli_U + rel_Us + np.hstack((np.cross(ecoli_U[3:], drbc), (0, 0, 0)))
    tail_U = ecoli_U + rel_Uh + np.hstack((np.cross(ecoli_U[3:], drtc), (0, 0, 0)))
    head_F = ecoli_comp.get_obj_list()[0].get_total_force(center=ecoli_center)
    tail_F = np.sum([tobj.get_total_force(center=ecoli_center)
                     for tobj in ecoli_comp.get_obj_list()[1:]], axis=0)

    comm = PETSc.COMM_WORLD.tompi4py()
    rank = comm.Get_rank()
    save_name = '%s.pickle' % fileHandle
    tpickle = {'problem_kwargs': problem_kwargs,
               'head_U':         head_U,
               'tail_U':         tail_U,
               'head_F':         head_F,
               'tail_F':         tail_F,
               'ecoli_U':        ecoli_U,
               'ecoli_comp':     ecoli_comp, }
    if rank == 0:
        with open(save_name, 'wb') as output:
            pickle.dump(tpickle, output, protocol=4)
        print('save pickle data to %s. ' % save_name)
    return True


def fun_Mpart(use_obj, AtBtCt_center=None, **problem_kwargs):
    matrix_method = problem_kwargs['matrix_method']
    fileHandle = problem_kwargs['fileHandle']

    PETSc.Sys.Print('')
    PETSc.Sys.Print('Calculate the mobility matrix of %s' % str(use_obj))
    problem_part_mob = sf.problem_dic[matrix_method](**problem_kwargs)
    problem_part_mob.add_obj(use_obj)
    # problem_part_mob.print_info()
    problem_part_mob.create_matrix()

    if AtBtCt_center is None:
        AtBtCt_center = use_obj.get_u_geo().get_center()
    A, B1, B2, C = \
        AtBtCt_full(problem_part_mob, pick_M=False, save_vtk=False,
                    center=AtBtCt_center, print_each=False, save_name=fileHandle,
                    u_use=1, w_use=1, uNormFct=1, wNormFct=1, uwNormFct=1, )
    R = np.vstack((np.hstack((A, B2)),
                   np.hstack((B1, C)),))
    M = np.linalg.inv(R)
    # PETSc.Sys.Print(R)
    # PETSc.Sys.Print(M)
    problem_part_mob.destroy()
    return M


def Resolve_UW(use_obj0, **problem_kwargs):
    FTpart = use_obj0.get_total_force(center=use_obj0.get_u_geo().get_center())
    Mpart = fun_Mpart(use_obj0, **problem_kwargs)
    UWpart = np.dot(Mpart, FTpart)
    PETSc.Sys.Print('  Resolved velocity is %s' % str(UWpart))
    return FTpart, Mpart


def main_skew_tail(**main_kwargs):
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    print_case_info(**problem_kwargs)

    ecoli_comp, problem_kwargs = create_skew_tail_comp(**problem_kwargs)

    problem = sf.ForceFreeProblem(**problem_kwargs)
    problem.add_obj(ecoli_comp)
    problem.print_info()
    problem.create_matrix()
    problem.solve()

    print_single_ecoli_forcefree_result(ecoli_comp, **problem_kwargs)
    pickle_data(ecoli_comp, **problem_kwargs)
    return True


def main_skew_tail_v2(**main_kwargs):
    # err_msg = 'this is dbg code. '
    # assert 1 == 2, err_msg
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    print_case_info(**problem_kwargs)

    ecoli_comp, problem_kwargs = create_skew_tail_comp_v2(**problem_kwargs)

    problem = sf.ForceFreeProblem(**problem_kwargs)
    problem.add_obj(ecoli_comp)
    problem.print_info()
    problem.create_matrix()
    problem.solve()

    print_single_ecoli_forcefree_result(ecoli_comp, **problem_kwargs)
    pickle_data(ecoli_comp, **problem_kwargs)
    return True


def main_skew_head(**main_kwargs):
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    print_case_info(**problem_kwargs)

    ecoli_comp, problem_kwargs = create_skew_head_comp(**problem_kwargs)

    problem = sf.ForceFreeProblem(**problem_kwargs)
    problem.add_obj(ecoli_comp)
    problem.print_info()
    problem.create_matrix()
    # tdbg = problem.get_M()
    # print(tdbg)
    problem.solve()

    print_single_ecoli_forcefree_result(ecoli_comp, **problem_kwargs)
    pickle_data(ecoli_comp, **problem_kwargs)
    return True


def main_skew_head_vrf(**main_kwargs):
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    matrix_method = problem_kwargs['matrix_method']
    fileHandle = problem_kwargs['fileHandle']
    print_case_info(**problem_kwargs)

    ecoli_comp, problem_kwargs = create_skew_head_comp(**problem_kwargs)

    # calculate the force and the velocity of a skewed microswimmer
    problem = sf.ForceFreeProblem(**problem_kwargs)
    problem.add_obj(ecoli_comp)
    problem.print_info()
    problem.create_matrix()
    problem.solve()
    print_single_ecoli_forcefree_result(ecoli_comp, **problem_kwargs)

    # check the accuracy of the boundary velocity
    problem.mat_destroy()
    ref_U = ecoli_comp.get_ref_U()
    check_kwargs = problem_kwargs.copy()
    check_kwargs['nth'] = problem_kwargs['nth'] * 2 - 1
    check_kwargs['ds'] = problem_kwargs['ds'] * 0.3
    check_kwargs['hfct'] = 1
    check_kwargs['Tfct'] = 1
    ecoli_comp_check, _ = create_skew_head_comp(**check_kwargs)
    ecoli_comp_check.set_ref_U(ref_U)
    ecoli_comp_check.set_problem(problem)
    ecoli_comp_check.set_name('%s_check' % ecoli_comp.get_name())
    PETSc.Sys.Print('')
    PETSc.Sys.Print('Relative boundary velocity error: %s' % str(ecoli_comp_check))
    velocity_err_list = problem.vtk_check(fileHandle, ecoli_comp_check)
    for tobj, t0 in zip(ecoli_comp_check.get_obj_list(), velocity_err_list):
        PETSc.Sys.Print('  %s: %s' % (tobj.get_name(), str(t0)))

    # calculate the mobility matrix, whole microswimmer
    PETSc.Sys.Print('')
    PETSc.Sys.Print('Calculate the whole mobility matrix of %s' % str(ecoli_comp))
    problem_whole_mob = sf.problem_dic[matrix_method](**problem_kwargs)
    ecoli_comp_cp = ecoli_comp.copy()
    for tobj in ecoli_comp_cp.get_obj_list():
        problem_whole_mob.add_obj(tobj)
    # problem_whole_mob.print_info()
    problem_whole_mob.create_matrix()
    A_list, B1_list, B2_list, C_list = \
        AtBtCt_multiObj(problem_whole_mob, save_vtk=False, pick_M=False, save_name=fileHandle,
                        print_each=False, uNormFct=1, wNormFct=1, uwNormFct=1, )
    a00 = np.vstack(A_list[0][0])
    a10 = np.vstack(A_list[0][1])
    a01 = np.vstack(A_list[1][0])
    a11 = np.vstack(A_list[1][1])
    #
    b1_00 = np.vstack(B1_list[0][0])
    b1_10 = np.vstack(B1_list[0][1])
    b1_01 = np.vstack(B1_list[1][0])
    b1_11 = np.vstack(B1_list[1][1])
    #
    b2_00 = np.vstack(B2_list[0][0])
    b2_10 = np.vstack(B2_list[0][1])
    b2_01 = np.vstack(B2_list[1][0])
    b2_11 = np.vstack(B2_list[1][1])
    #
    c00 = np.vstack(C_list[0][0])
    c10 = np.vstack(C_list[0][1])
    c01 = np.vstack(C_list[1][0])
    c11 = np.vstack(C_list[1][1])
    Rwhole = np.vstack((np.hstack((a00, b1_00, a01, b1_01)),
                        np.hstack((b2_00, c00, b2_01, c01)),
                        np.hstack((a10, b1_10, a11, b1_11)),
                        np.hstack((b2_10, c10, b2_11, c11))))
    Mwhole = np.linalg.inv(Rwhole)
    FTwhole = np.hstack([use_obj0.get_total_force(center=use_obj0.get_u_geo().get_center())
                         for use_obj0 in ecoli_comp.get_obj_list()])
    tu = np.dot(Mwhole, FTwhole)
    PETSc.Sys.Print('----->>>>', tu)

    # calculate the mobility matrix, head and tail
    ecoli_comp_cp = ecoli_comp.copy()
    FThead, Mhead = Resolve_UW(ecoli_comp_cp.get_obj_list()[0], **problem_kwargs)
    FTtail, Mtail = Resolve_UW(ecoli_comp_cp.get_obj_list()[1], **problem_kwargs)

    # pickle problem
    vrf_dict = {'FThead':     FThead,
                'Mhead':      Mhead,
                'FTtail':     FTtail,
                'Mtail':      Mtail,
                'FTwhole':    FTwhole,
                'Mwhole':     Mwhole,
                'ecoli_comp': ecoli_comp}
    problem_kwargs['vrf_dict'] = vrf_dict
    pickle_data(ecoli_comp, **problem_kwargs)
    return True


def main_tail_ABC(**main_kwargs):
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    matrix_method = problem_kwargs['matrix_method']
    fileHandle = problem_kwargs['fileHandle']
    print_case_info(**problem_kwargs)

    ecoli_comp, problem_kwargs = create_skew_tail_comp(**problem_kwargs)
    tail_obj = ecoli_comp.get_obj_list()[1]
    tail_obj.show_u_nodes()
    tail_center = tail_obj.get_u_geo().get_center()
    PETSc.Sys.Print('  tail dof: %d nodes' % tail_obj.get_u_geo().get_n_nodes())
    PETSc.Sys.Print('  now use tail center=%s' % str(tail_center))
    Mtail0 = fun_Mpart(tail_obj, AtBtCt_center=tail_center, **problem_kwargs)
    PETSc.Sys.Print('-->>alpha_t')
    PETSc.Sys.Print(Mtail0[0:3, 0:3])
    PETSc.Sys.Print('-->>beta_t1')
    PETSc.Sys.Print(Mtail0[3:6, 0:3])
    PETSc.Sys.Print('-->>beta_t2')
    PETSc.Sys.Print(Mtail0[0:3, 3:6])
    PETSc.Sys.Print('-->>gamma_t')
    PETSc.Sys.Print(Mtail0[3:6, 3:6])

    # dbg, move center, simulation
    PETSc.Sys.Print('###############################################')
    ecoli_center = ecoli_comp.get_center()
    PETSc.Sys.Print('  now use ecoli center=%s' % str(ecoli_center))
    Mtail1 = fun_Mpart(tail_obj, AtBtCt_center=ecoli_center, **problem_kwargs)
    PETSc.Sys.Print('-->>alpha_t')
    PETSc.Sys.Print(Mtail1[0:3, 0:3])
    PETSc.Sys.Print('-->>beta_t1')
    PETSc.Sys.Print(Mtail1[3:6, 0:3])
    PETSc.Sys.Print('-->>beta_t2')
    PETSc.Sys.Print(Mtail1[0:3, 3:6])
    PETSc.Sys.Print('-->>gamma_t')
    PETSc.Sys.Print(Mtail1[3:6, 3:6])

    # # dbg, move center, theory
    # from codeStore import support_fun_mobility as spf_mob
    # PETSc.Sys.Print('')
    # PETSc.Sys.Print('###############################################')
    # alpha0 = Mtail0[0:3, 0:3]
    # beta10 = Mtail0[3:6, 0:3]
    # beta20 = Mtail0[0:3, 3:6]
    # gamma0 = Mtail0[3:6, 3:6]
    # alpha1 = Mtail1[0:3, 0:3]
    # beta11 = Mtail1[3:6, 0:3]
    # beta21 = Mtail1[0:3, 3:6]
    # gamma1 = Mtail1[3:6, 3:6]
    #
    # dtc = spf_mob.cross_matrix(tail_center - ecoli_center)
    # alpha = alpha1 - np.dot(dtc, np.dot(gamma1, dtc)) - np.dot(dtc, beta21) - np.dot(dtc, beta11)
    # beta1 = beta11 - np.dot(dtc, gamma1)
    # beta2 = beta21 + np.dot(dtc, gamma1)
    # gamma = gamma1
    # PETSc.Sys.Print('-->>alpha_t')
    # PETSc.Sys.Print(alpha)
    # PETSc.Sys.Print(np.linalg.norm(alpha - alpha0))
    # PETSc.Sys.Print('-->>beta_t1')
    # PETSc.Sys.Print(beta1)
    # PETSc.Sys.Print(np.linalg.norm(beta1 - beta10))
    # PETSc.Sys.Print('-->>beta_t2')
    # PETSc.Sys.Print(beta2)
    # PETSc.Sys.Print(np.linalg.norm(beta2 - beta20))
    # PETSc.Sys.Print('-->>gamma_t')
    # PETSc.Sys.Print(gamma)
    # PETSc.Sys.Print(np.linalg.norm(gamma - gamma0))
    #
    # PETSc.Sys.Print('')
    # PETSc.Sys.Print('###############################################')
    # PETSc.Sys.Print(dtc)
    # PETSc.Sys.Print(np.dot(dtc, gamma1))
    # PETSc.Sys.Print('')
    # PETSc.Sys.Print('')

    comm = PETSc.COMM_WORLD.tompi4py()
    rank = comm.Get_rank()
    if rank == 0:
        pickle_dict = {'Mtail0':       Mtail0,
                       'Mtail1':       Mtail1,
                       'tail_center':  tail_center,
                       'ecoli_center': ecoli_center, }
        save_name = '%s.pickle' % fileHandle
        with open(save_name, 'wb') as handle:
            pickle.dump(pickle_dict, handle, protocol=4)
        print('save pickle data to %s. ' % save_name)
    return True


def main_tail_B(**main_kwargs):
    def _fun_B1(U):
        PETSc.Sys.Print('###############################################')
        u_geo = tail_obj.get_u_geo()
        u_geo.set_rigid_velocity(U, center=tail_center)
        problem_part_mob.create_F_U()
        problem_part_mob.solve()
        PETSc.Sys.Print(U)
        FT = tail_obj.get_total_force(center=tail_center)
        PETSc.Sys.Print('total force:', FT[:3])
        PETSc.Sys.Print('total torque:', FT[3:])
        return True
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    matrix_method = problem_kwargs['matrix_method']
    fileHandle = problem_kwargs['fileHandle']
    print_case_info(**problem_kwargs)

    ecoli_comp, problem_kwargs = create_skew_tail_comp(**problem_kwargs)
    tail_obj = ecoli_comp.get_obj_list()[1]
    # tail_obj.show_u_nodes()
    tail_center = tail_obj.get_u_geo().get_center()
    tail_obj.move(-tail_center)
    tail_center = tail_obj.get_u_geo().get_center()
    problem_part_mob = sf.problem_dic[matrix_method](**problem_kwargs)
    problem_part_mob.add_obj(tail_obj)
    problem_part_mob.print_info()
    problem_part_mob.create_matrix()

    # A, B1, B2, C = \
    #     AtBtCt_full(problem_part_mob, pick_M=False, save_vtk=False,
    #                 center=tail_center, print_each=False, save_name=fileHandle,
    #                 u_use=1, w_use=1, uNormFct=1, wNormFct=1, uwNormFct=1, )

    # calculate B1 manually.
    for psi in np.linspace(0, 0.5, 6) * np.pi:
        rotM = Rloc2glb(0, 0, psi).T
        U = np.hstack((np.dot(rotM, (1, 0, 0)), (0, 0, 0)))
        _fun_B1(U)
    return True


def main_tail_B2(**main_kwargs):
    def _fun_B1(U):
        PETSc.Sys.Print('###############################################')
        u_geo = tail_obj.get_u_geo()
        u_geo.set_rigid_velocity(U, center=tail_center)
        problem_part_mob.create_F_U()
        problem_part_mob.solve()
        PETSc.Sys.Print(U)
        FT = tail_obj.get_total_force(center=tail_center)
        PETSc.Sys.Print('total force:', FT[:3])
        PETSc.Sys.Print('total torque:', FT[3:])
        return True

    problem_kwargs = get_problem_kwargs(**main_kwargs)
    matrix_method = problem_kwargs['matrix_method']
    fileHandle = problem_kwargs['fileHandle']
    print_case_info(**problem_kwargs)

    for chi in np.linspace(3, 2, 11):
        problem_kwargs['ch'] = chi
        ecoli_comp, problem_kwargs = create_skew_tail_comp(**problem_kwargs)
        tail_obj = ecoli_comp.get_obj_list()[1]
        tail_center = tail_obj.get_u_geo().get_center()
        tail_obj.move(-tail_center)
        tail_center = tail_obj.get_u_geo().get_center()
        problem_part_mob = sf.problem_dic[matrix_method](**problem_kwargs)
        problem_part_mob.add_obj(tail_obj)
        problem_part_mob.print_info()
        problem_part_mob.create_matrix()

        psi = 0
        rotM = Rloc2glb(0, 0, psi).T
        U = np.hstack((np.dot(rotM, (1, 0, 0)), (0, 0, 0)))
        _fun_B1(U)
        tail_obj.show_u_nodes()
        problem_part_mob.destroy()
    return True


def main_mobility_matrix(**main_kwargs):
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    matrix_method = problem_kwargs['matrix_method']
    fileHandle = problem_kwargs['fileHandle']
    print_case_info(**problem_kwargs)

    ecoli_comp, problem_kwargs = create_skew_head_comp(**problem_kwargs)

    # calculate the mobility matrix, head and tail
    problem = sf.ForceFreeProblem(**problem_kwargs)
    problem.add_obj(ecoli_comp)  # just for print information
    problem.print_info()
    _, Mhead = Resolve_UW(ecoli_comp.get_obj_list()[0], **problem_kwargs)
    PETSc.Sys.Print('-->>alpha_b')
    PETSc.Sys.Print(Mhead[0:3, 0:3])
    PETSc.Sys.Print('-->>beta_b1')
    PETSc.Sys.Print(Mhead[3:6, 0:3])
    PETSc.Sys.Print('-->>beta_b2')
    PETSc.Sys.Print(Mhead[0:3, 3:6])
    PETSc.Sys.Print('-->>gamma_b')
    PETSc.Sys.Print(Mhead[3:6, 3:6])

    _, Mtail = Resolve_UW(ecoli_comp.get_obj_list()[1], **problem_kwargs)
    PETSc.Sys.Print('-->>alpha_t')
    PETSc.Sys.Print(Mtail[0:3, 0:3])
    PETSc.Sys.Print('-->>beta_t1')
    PETSc.Sys.Print(Mtail[3:6, 0:3])
    PETSc.Sys.Print('-->>beta_t2')
    PETSc.Sys.Print(Mtail[0:3, 3:6])
    PETSc.Sys.Print('-->>gamma_t')
    PETSc.Sys.Print(Mtail[3:6, 3:6])

    # pickle problem
    comm = PETSc.COMM_WORLD.tompi4py()
    rank = comm.Get_rank()
    save_name = '%s.pickle' % fileHandle
    tpickle = {'problem_kwargs': problem_kwargs,
               'Mhead':          Mhead,
               'Mtail':          Mtail,
               'ecoli_comp':     ecoli_comp}
    if rank == 0:
        with open(save_name, 'wb') as output:
            pickle.dump(tpickle, output, protocol=4)
        print('save pickle data to %s. ' % save_name)
    return True


if __name__ == '__main__':
    # python ecoliSkewTail.py  -main_skew_tail 1 -sm lg_rs -legendre_m 3 -legendre_k 2 -epsilon 3.000000  -rh1 0.500000 -rh2 0.050000 -nth 9 -eh 0.000000 -ph 2.000000 -ch 2.000000 -n_tail 1  -hfct 1.000000 -with_cover 2 -with_T_geo 0 -left_hand 0  -rs1 2.000000 -rs2 1.000000 -ds 0.2 -es 0.000000 -dist_hs 0.500000 -rel_whz 1.000000  -centerx 0.000000 -centery 0.00000 -centerz 0.000000  -ksp_max_it 1000 -ksp_rtol 1.000000e-20 -ksp_atol 1.000000e-200 -f dbg
    OptDB = PETSc.Options()
    if OptDB.getBool('main_skew_tail', False):
        OptDB.setValue('main_fun', False)
        main_skew_tail()

    if OptDB.getBool('main_skew_tail_v2', False):
        OptDB.setValue('main_fun', False)
        main_skew_tail_v2()

    if OptDB.getBool('main_skew_head', False):
        OptDB.setValue('main_fun', False)
        main_skew_head()

    if OptDB.getBool('main_skew_head_vrf', False):
        OptDB.setValue('main_fun', False)
        main_skew_head_vrf()

    if OptDB.getBool('main_tail_ABC', False):
        OptDB.setValue('main_fun', False)
        main_tail_ABC()

    if OptDB.getBool('main_tail_B', False):
        OptDB.setValue('main_fun', False)
        main_tail_B()

    if OptDB.getBool('main_tail_B2', False):
        OptDB.setValue('main_fun', False)
        main_tail_B2()

    if OptDB.getBool('main_mobility_matrix', False):
        OptDB.setValue('main_fun', False)
        main_mobility_matrix()

    # if OptDB.getBool('main_fun', True):
    #     main_fun()

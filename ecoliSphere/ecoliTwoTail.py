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
    # tail_rot_delta = OptDB.getReal('tail_rot_delta', 0)
    dist_t1t2 = OptDB.getReal('dist_t1t2', 1)
    tail_ch2 = OptDB.getReal('tail_ch2', 1)
    problem_kwargs['tail_ini_beta'] = tail_ini_beta
    problem_kwargs['tail_ini_theta'] = tail_ini_theta
    problem_kwargs['tail_ini_phi'] = tail_ini_phi
    problem_kwargs['tail_ini_psi'] = tail_ini_psi
    # problem_kwargs['tail_rot_delta'] = tail_rot_delta
    problem_kwargs['dist_t1t2'] = dist_t1t2
    problem_kwargs['tail_ch2'] = tail_ch2
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
    # tail_rot_delta = problem_kwargs['tail_rot_delta']
    dist_t1t2 = problem_kwargs['dist_t1t2']
    tail_ch2 = problem_kwargs['tail_ch2']
    PETSc.Sys.Print('ADDITIONAL PARAMETERS of the MICROSWIMMER: ')
    PETSc.Sys.Print('  ecoli rotation, theta: %f, phi: %f, psi: %f. ' %
                    (ecoli_ini_rot_theta, ecoli_ini_rot_phi, ecoli_ini_rot_psi))
    PETSc.Sys.Print('  tail ch2: %f, distance from tail 1 to tail 2: %f' % (tail_ch2, dist_t1t2))
    PETSc.Sys.Print('  tail rotation, beta: %f' % (tail_ini_beta,))
    PETSc.Sys.Print('  tail rotation, theta: %f, phi: %f, psi: %f. ' %
                    (tail_ini_theta, tail_ini_phi, tail_ini_psi))
    return t1


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
    # tail_rot_delta = problem_kwargs['tail_rot_delta']
    beta_norm = np.array([0, 1, 0])
    rs1 = problem_kwargs['rs1']
    rs2 = problem_kwargs['rs2']
    ch = problem_kwargs['ch']
    ph = problem_kwargs['ph']
    dist_hs = problem_kwargs['dist_hs']
    dist_t1t2 = problem_kwargs['dist_t1t2']
    tail_ch2 = problem_kwargs['tail_ch2']
    wbc = problem_kwargs['rel_Us'][-1]
    wtc = problem_kwargs['rel_Uh'][-1]

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
    dmv1 = rs1 + dist_hs + ch * ph + dist_t1t2
    head_end0 = head_center - dmv1 * head_norm


    # rotate the first ecoli object in opposite position.
    rotM0 = Rloc2glb(tail_ini_theta, tail_ini_phi, tail_ini_psi)
    ecoli_comp.node_rotation(norm=beta_norm, theta=-1 * tail_ini_beta,
                             rotation_origin=head_center)
    ecoli_comp.node_rotM(rotM0.T, rotation_origin=head_end0)

    # generate ecoli with two tails.
    head_obj, tail_obj = ecoli_comp.get_obj_list()
    tail_kwargs = problem_kwargs.copy()
    tail_kwargs['ch'] = tail_ch2
    tail2_obj = create_ecoli_tail(np.zeros(3), **tail_kwargs)[0]
    head_center = head_obj.get_u_geo().get_center()
    head_norm = head_obj.get_u_geo().get_geo_norm()
    dmv2 = -dmv1 * head_norm - tail_ch2 * ph / 2 * np.array((0, 0, 1)) + head_center
    # dmv2 = np.zeros(3)
    tail2_obj.move(dmv2)
    ecoli_comp2 = sf.ForceFreeComposite(center=head_center, norm=head_norm, name='ecoli_0')
    for tobj, tw in zip((head_obj, tail_obj, tail2_obj), (wbc, wtc, wtc)):
        tnorm = tobj.get_u_geo().get_geo_norm()
        rel_U = np.hstack((0, 0, 0, tw * tnorm))
        ecoli_comp2.add_obj(obj=tobj, rel_U=rel_U)

    ecoli_comp2.set_norm(head_obj.get_u_geo().get_geo_norm())
    ecoli_comp2.set_center(head_obj.get_u_geo().get_center())
    return ecoli_comp2, problem_kwargs


def main_fun(**main_kwargs):
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    print_case_info(**problem_kwargs)
    ecoli_comp, problem_kwargs = create_skew_head_comp(**problem_kwargs)
    problem = sf.ForceFreeProblem(**problem_kwargs)
    problem.add_obj(ecoli_comp)
    problem.print_info()
    problem.create_matrix()
    problem.solve()
    print_single_ecoli_forcefree_result(ecoli_comp, **problem_kwargs)
    return True


if __name__ == '__main__':
    OptDB = PETSc.Options()
    # if OptDB.getBool('main_mobility_matrix', False):
    #     OptDB.setValue('main_fun', False)
    #     main_mobility_matrix()

    if OptDB.getBool('main_fun', True):
        main_fun()

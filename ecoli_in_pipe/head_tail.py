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
from src.support_class import *
from src.objComposite import createEcoliComp_tunnel
from src.myvtk import *
from src.objComposite import *


def get_problem_kwargs(**main_kwargs):
    problem_kwargs = get_solver_kwargs()
    OptDB = PETSc.Options()
    fileHandle = OptDB.getString('f', '...')
    err_msg = 'specify the fileHandle. '
    assert fileHandle != '...', err_msg
    problem_kwargs['fileHandle'] = fileHandle

    import os
    t_name = os.path.basename(__file__)
    need_args = ['head_U', 'tail_U', ]
    for key in need_args:
        if key not in main_kwargs:
            err_msg = 'information about ' + key + ' is necessary for %s . ' % t_name
            raise ValueError(err_msg)

    kwargs_list = (main_kwargs,)
    for t_kwargs in kwargs_list:
        for key in t_kwargs:
            problem_kwargs[key] = t_kwargs[key]
    return problem_kwargs


def print_case_info(**problem_kwargs):
    fileHandle = problem_kwargs['fileHandle']
    PETSc.Sys.Print('-->Ecoli in pipe case, given velocity case.')
    # print_solver_info(**problem_kwargs)
    print_ecoli_info(fileHandle, **problem_kwargs)

    ecoli_U = problem_kwargs['ecoli_U']
    rel_Us = problem_kwargs['rel_Us']
    rel_Uh = problem_kwargs['rel_Uh']
    head_U = ecoli_U + rel_Us
    tail_U = ecoli_U + rel_Uh
    PETSc.Sys.Print('given velocity of head is', head_U * [0, 0, 1, 0, 0, 1])
    PETSc.Sys.Print('given velocity of tail is', tail_U * [0, 0, 1, 0, 0, 1])
    return True


def given_velocity_case(problem, old_obj_list, U_list, **kwargs):
    old_obj_list = list(tube_flatten((old_obj_list,)))
    U_list = list(tube_flatten((U_list,)))
    err_msg = 'length of obj_list and U_list must be same. '
    assert len(old_obj_list) == len(U_list), err_msg

    newProb = sf.StokesFlowProblem(**kwargs)
    new_obj_list = [t_obj.copy() for t_obj in old_obj_list]
    for new_obj, obj_U in zip(new_obj_list, U_list):
        new_obj.set_rigid_velocity(obj_U)
        newProb.add_obj(new_obj)
    newProb.create_F_U()
    new_M = newProb.create_empty_M()
    for uobj_old, uobj_new in zip(old_obj_list, new_obj_list):
        for fobj_old, fobj_new in zip(old_obj_list, new_obj_list):
            problem.create_part_matrix(uobj_old, fobj_old, uobj_new, fobj_new, new_M)
    newProb.solve()
    return newProb


def given_velocity_ecoli(problem, head_U, tail_U, prefix, **kwargs):
    # head_obj_list = [problem.get_obj_list()[0].get_obj_list()[0], ]
    # tail_obj_list = problem.get_obj_list()[0].get_obj_list()[1:]
    obj_list = problem.get_obj_list()[0].get_obj_list()[:]
    head_U_list = [head_U, ]
    tail_U_list = [tail_U, ] * (len(obj_list) - 1)
    U_list = head_U_list + tail_U_list
    fileHandle = problem.get_kwargs()['fileHandle'] + '_full_' + prefix
    kwargs['fileHandle'] = fileHandle
    kwargs['rel_Us'] = head_U
    kwargs['rel_Uh'] = tail_U

    PETSc.Sys.Print('')
    PETSc.Sys.Print('%s case results: ' % fileHandle)
    newProb = given_velocity_case(problem, obj_list, U_list, **kwargs)
    save_singleEcoli_U_vtk(newProb, createHandle=createEcoli_tunnel, part='full')
    prefix = 'ecoli full ' + prefix
    print_single_ecoli_force_result(newProb, prefix=prefix)
    newProb.destroy()
    return True


def given_velocity_ecoli_NoTgeo(problem, head_U, tail_U, prefix, **kwargs):
    obj_list = problem.get_obj_list()[0].get_obj_list()[0:3]
    head_U_list = [head_U, ]
    tail_U_list = [tail_U, ] * 2
    U_list = head_U_list + tail_U_list
    fileHandle = problem.get_kwargs()['fileHandle'] + '_NoTgeo_' + prefix
    kwargs['fileHandle'] = fileHandle
    kwargs['rel_Us'] = head_U
    kwargs['rel_Uh'] = tail_U

    PETSc.Sys.Print('')
    PETSc.Sys.Print('%s case results: ' % fileHandle)
    newProb = given_velocity_case(problem, obj_list, U_list, **kwargs)
    save_singleEcoli_U_vtk(newProb, createHandle=createEcoli_tunnel, part='full')
    prefix = 'ecoli NoTgeo ' + prefix
    print_single_ecoli_force_result(newProb, prefix=prefix)
    newProb.destroy()
    return True


def given_velocity_ecoli_4part(problem, U_list, prefix, **kwargs):
    # consider the ecoli constituted by four separate part: head, helix0, helix1, and Tgeo.
    obj_list = problem.get_obj_list()[0].get_obj_list()[:]
    fileHandle = problem.get_kwargs()['fileHandle'] + '_separ_' + prefix
    kwargs['fileHandle'] = fileHandle

    PETSc.Sys.Print('')
    PETSc.Sys.Print('%s case results: ' % fileHandle)
    newProb = given_velocity_case(problem, obj_list, U_list, **kwargs)
    save_singleEcoli_U_4part_vtk(newProb, U_list, createHandle=createEcoli_tunnel)
    print_single_ecoli_force_result(newProb, prefix=prefix, **kwargs)
    newProb.destroy()
    return True


def main_fun_bck(**main_kwargs):
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    fileHandle = problem_kwargs['fileHandle']
    head_U = problem_kwargs['head_U']
    tail_U = problem_kwargs['tail_U']
    # OptDB = PETSc.Options()
    # OptDB.setValue('save_singleEcoli_vtk', True)

    with open(fileHandle + '_pick.bin', 'rb') as myinput:
        unpick = pickle.Unpickler(myinput)
        problem = unpick.load()
        problem.unpickmyself()
    kwargs = problem.get_kwargs()
    # assume ecoli_U==0, rel_Us and rel_Uh are true (physical) velocity of head and tail respectively.
    kwargs['ecoli_U'] = np.zeros(6)
    kwargs['rel_Us'] = head_U
    kwargs['rel_Uh'] = tail_U
    print_case_info(**kwargs)
    problem.print_info()

    # # separate each part of objects from the base problem.
    # head_obj_list = [problem.get_obj_list()[0].get_obj_list()[0], ]
    # tail_obj_list = problem.get_obj_list()[0].get_obj_list()[1:]
    # zero_U = np.zeros(6)
    # head_U_move = head_U * [0, 0, 1, 0, 0, 1]
    # head_U_tran = head_U * [0, 0, 1, 0, 0, 0]
    # head_U_rota = head_U * [0, 0, 0, 0, 0, 1]
    # head_U_zero_list = [zero_U, ]
    # head_U_move_list = [head_U_move, ]
    # head_U_tran_list = [head_U_tran, ]
    # head_U_rota_list = [head_U_rota, ]
    # tail_U_move = tail_U * [0, 0, 1, 0, 0, 1]
    # tail_U_tran = tail_U * [0, 0, 1, 0, 0, 0]
    # tail_U_rota = tail_U * [0, 0, 0, 0, 0, 1]
    # tail_U_zero_list = [zero_U, ] * len(tail_obj_list)
    # tail_U_move_list = [tail_U_move, ] * len(tail_obj_list)
    # tail_U_tran_list = [tail_U_tran, ] * len(tail_obj_list)
    # tail_U_rota_list = [tail_U_rota, ] * len(tail_obj_list)

    # # translation velocity, solve head separately.
    # kwargs['fileHandle'] = fileHandle + '_head_tran'
    # kwargs['rel_Us'] = head_U_tran
    # kwargs['rel_Uh'] = np.zeros(6)
    # newProb = given_velocity_case(problem, head_obj_list, head_U_tran_list, **kwargs)
    # t_force = newProb.get_total_force()
    # save_singleEcoli_U_vtk(newProb, createHandle=createEcoli_tunnel, part='head')
    # newProb.destroy()
    # PETSc.Sys.Print('tran head resultant is', t_force)
    #
    # # rotation velocity, solve head separately.
    # kwargs['fileHandle'] = fileHandle + '_head_rota'
    # kwargs['rel_Us'] = head_U_rota
    # kwargs['rel_Uh'] = np.zeros(6)
    # newProb = given_velocity_case(problem, head_obj_list, head_U_rota_list, **kwargs)
    # t_force = newProb.get_total_force()
    # save_singleEcoli_U_vtk(newProb, createHandle=createEcoli_tunnel, part='head')
    # newProb.destroy()
    # PETSc.Sys.Print('rota head resultant is', t_force)
    #
    # # trans&rotat velocity, solve head separately.
    # kwargs['fileHandle'] = fileHandle + '_head_move'
    # kwargs['rel_Us'] = head_U_move
    # kwargs['rel_Uh'] = np.zeros(6)
    # newProb = given_velocity_case(problem, head_obj_list, head_U_move_list, **kwargs)
    # t_force = newProb.get_total_force()
    # save_singleEcoli_U_vtk(newProb, createHandle=createEcoli_tunnel, part='head')
    # newProb.destroy()
    # PETSc.Sys.Print('move head resultant is', t_force)
    #
    # # translation velocity, solve tail separately.
    # kwargs['fileHandle'] = fileHandle + '_tail_tran'
    # kwargs['rel_Us'] = np.zeros(6)
    # kwargs['rel_Uh'] = tail_U_tran
    # newProb = given_velocity_case(problem, tail_obj_list, tail_U_tran_list, **kwargs)
    # t_force = newProb.get_total_force()
    # save_singleEcoli_U_vtk(newProb, createHandle=createEcoli_tunnel, part='tail')
    # newProb.destroy()
    # PETSc.Sys.Print('tran tail resultant is', t_force)
    #
    # # rotation velocity, solve tail separately.
    # kwargs['fileHandle'] = fileHandle + '_tail_rota'
    # kwargs['rel_Us'] = np.zeros(6)
    # kwargs['rel_Uh'] = tail_U_rota
    # newProb = given_velocity_case(problem, tail_obj_list, tail_U_rota_list, **kwargs)
    # t_force = newProb.get_total_force()
    # save_singleEcoli_U_vtk(newProb, createHandle=createEcoli_tunnel, part='tail')
    # newProb.destroy()
    # PETSc.Sys.Print('rota tail resultant is', t_force)
    #
    # # trans&rotat velocity, solve tail separately.
    # kwargs['fileHandle'] = fileHandle + '_tail_move'
    # kwargs['rel_Us'] = np.zeros(6)
    # kwargs['rel_Uh'] = tail_U_move
    # newProb = given_velocity_case(problem, tail_obj_list, tail_U_move_list, **kwargs)
    # t_force = newProb.get_total_force()
    # save_singleEcoli_U_vtk(newProb, createHandle=createEcoli_tunnel, part='tail')
    # newProb.destroy()
    # PETSc.Sys.Print('move tail resultant is', t_force)

    # # given velocity, solve total ecoli.
    # obj_list = head_obj_list + tail_obj_list
    # U_list = head_U_move_list + tail_U_move_list
    # kwargs['fileHandle'] = fileHandle + '_full'
    # # # dbg
    # # for obj in obj_list:
    # #     filename = kwargs['fileHandle'] + '_' + str(obj)
    # #     obj.get_u_geo().save_nodes(filename + '_U')
    # #     obj.get_f_geo().save_nodes(filename + '_f')
    # newProb = given_velocity_case(problem, obj_list, U_list, **kwargs)
    # # newProb.saveM_mat(kwargs['fileHandle'])
    # save_singleEcoli_U_vtk(newProb, createHandle=createEcoli_tunnel, part='full')
    # newProb.destroy()
    # total_force = print_single_ecoli_force_result(newProb)


def main_head(problem):
    # solve head only.
    kwargs = problem.get_kwargs().copy()
    fileHandle = kwargs['fileHandle']
    head_U = kwargs['rel_Us']

    # separate each part of objects from the base problem.
    head_obj_list = [problem.get_obj_list()[0].get_obj_list()[0], ]
    head_U_move = head_U * [0, 0, 1, 0, 0, 1]
    head_U_tran = head_U * [0, 0, 1, 0, 0, 0]
    head_U_rota = head_U * [0, 0, 0, 0, 0, 1]
    head_U_move_list = [head_U_move, ]
    head_U_tran_list = [head_U_tran, ]
    head_U_rota_list = [head_U_rota, ]

    # translation velocity, solve head separately.
    kwargs['fileHandle'] = fileHandle + '_head_tran'
    kwargs['rel_Us'] = head_U_tran
    kwargs['rel_Uh'] = np.zeros(6)
    PETSc.Sys.Print()
    newProb = given_velocity_case(problem, head_obj_list, head_U_tran_list, **kwargs)
    t_force = newProb.get_total_force()
    save_singleEcoli_U_vtk(newProb, createHandle=createEcoli_tunnel, part='head')
    newProb.destroy()
    PETSc.Sys.Print('tran head resultant is', t_force)

    # rotation velocity, solve head separately.
    kwargs['fileHandle'] = fileHandle + '_head_rota'
    kwargs['rel_Us'] = head_U_rota
    kwargs['rel_Uh'] = np.zeros(6)
    PETSc.Sys.Print()
    newProb = given_velocity_case(problem, head_obj_list, head_U_rota_list, **kwargs)
    t_force = newProb.get_total_force()
    save_singleEcoli_U_vtk(newProb, createHandle=createEcoli_tunnel, part='head')
    newProb.destroy()
    PETSc.Sys.Print('rota head resultant is', t_force)

    # trans&rotat velocity, solve head separately.
    kwargs['fileHandle'] = fileHandle + '_head_move'
    kwargs['rel_Us'] = head_U_move
    kwargs['rel_Uh'] = np.zeros(6)
    PETSc.Sys.Print()
    newProb = given_velocity_case(problem, head_obj_list, head_U_move_list, **kwargs)
    t_force = newProb.get_total_force()
    save_singleEcoli_U_vtk(newProb, createHandle=createEcoli_tunnel, part='head')
    newProb.destroy()
    PETSc.Sys.Print('move head resultant is', t_force)
    return True


def main_helix_tail(problem):
    # only solve tail constituted of two helix, ignore Tgeo.
    kwargs = problem.get_kwargs().copy()
    fileHandle = kwargs['fileHandle']
    tail_U = kwargs['rel_Uh']

    # separate each part of objects from the base problem.
    tail_obj_list = problem.get_obj_list()[0].get_obj_list()[1:3]
    tail_U_tran = tail_U * [0, 0, 1, 0, 0, 0]
    tail_U_rota = tail_U * [0, 0, 0, 0, 0, 1]
    tail_U_move = tail_U * [0, 0, 1, 0, 0, 1]
    tail_U_tran_list = [tail_U_tran, tail_U_tran]
    tail_U_rota_list = [tail_U_rota, tail_U_rota]
    tail_U_move_list = [tail_U_move, tail_U_move]

    # translation velocity, solve tail separately.
    kwargs['fileHandle'] = fileHandle + '_tail_tran'
    kwargs['rel_Us'] = np.zeros(6)
    kwargs['rel_Uh'] = tail_U_tran
    PETSc.Sys.Print()
    newProb = given_velocity_case(problem, tail_obj_list, tail_U_tran_list, **kwargs)
    t_force = newProb.get_total_force()
    save_singleEcoli_U_vtk(newProb, createHandle=createEcoli_tunnel, part='tail')
    newProb.destroy()
    PETSc.Sys.Print('tran tail resultant is', t_force)

    # rotation velocity, solve tail separately.
    kwargs['fileHandle'] = fileHandle + '_tail_rota'
    kwargs['rel_Us'] = np.zeros(6)
    kwargs['rel_Uh'] = tail_U_rota
    PETSc.Sys.Print()
    newProb = given_velocity_case(problem, tail_obj_list, tail_U_rota_list, **kwargs)
    t_force = newProb.get_total_force()
    save_singleEcoli_U_vtk(newProb, createHandle=createEcoli_tunnel, part='tail')
    newProb.destroy()
    PETSc.Sys.Print('rota tail resultant is', t_force)

    # trans&rotat velocity, solve tail separately.
    kwargs['fileHandle'] = fileHandle + '_tail_move'
    kwargs['rel_Us'] = np.zeros(6)
    kwargs['rel_Uh'] = tail_U_move
    PETSc.Sys.Print()
    newProb = given_velocity_case(problem, tail_obj_list, tail_U_move_list, **kwargs)
    t_force = newProb.get_total_force()
    save_singleEcoli_U_vtk(newProb, createHandle=createEcoli_tunnel, part='tail')
    newProb.destroy()
    PETSc.Sys.Print('move tail resultant is', t_force)
    return True


def main_ecoli_NoTgeo(problem, funHeadle=given_velocity_ecoli):
    kwargs = problem.get_kwargs()

    head_U = kwargs['rel_Us']
    tail_U = kwargs['rel_Uh']
    zero_U = np.zeros(6)
    head_U_tran = head_U * [0, 0, 1, 0, 0, 0]
    head_U_rota = head_U * [0, 0, 0, 0, 0, 1]
    head_U_move = head_U * [0, 0, 1, 0, 0, 1]
    tail_U_tran = tail_U * [0, 0, 1, 0, 0, 0]
    tail_U_rota = tail_U * [0, 0, 0, 0, 0, 1]
    tail_U_move = tail_U * [0, 0, 1, 0, 0, 1]

    prefix = 'head_tran'
    funHeadle(problem, head_U_tran, zero_U, prefix, **kwargs)

    prefix = 'head_rota'
    funHeadle(problem, head_U_rota, zero_U, prefix, **kwargs)

    prefix = 'head_move'
    funHeadle(problem, head_U_move, zero_U, prefix, **kwargs)

    prefix = 'tail_tran'
    funHeadle(problem, zero_U, tail_U_tran, prefix, **kwargs)

    prefix = 'tail_rota'
    funHeadle(problem, zero_U, tail_U_rota, prefix, **kwargs)

    prefix = 'tail_move'
    funHeadle(problem, zero_U, tail_U_move, prefix, **kwargs)

    prefix = 'ecol_tran'
    funHeadle(problem, head_U_tran, tail_U_tran, prefix, **kwargs)

    prefix = 'ecol_rota'
    funHeadle(problem, head_U_rota, tail_U_rota, prefix, **kwargs)

    prefix = 'head_tran'
    funHeadle(problem, head_U_move, tail_U_move, prefix, **kwargs)

    return True


def main_4part(**main_kwargs):
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    fileHandle = problem_kwargs['fileHandle']
    head_U = problem_kwargs['head_U']
    tail_U = problem_kwargs['tail_U']

    with open(fileHandle + '_pick.bin', 'rb') as myinput:
        unpick = pickle.Unpickler(myinput)
        problem = unpick.load()
        problem.unpickmyself()
    kwargs = problem.get_kwargs()
    # given velocity part by part
    kwargs['ecoli_U'] = np.zeros(6)
    kwargs['rel_Us'] = np.zeros(6)
    kwargs['rel_Uh'] = np.zeros(6)
    print_case_info(**kwargs)
    problem.print_info()

    # separate each part of objects from the base problem.
    zero_U = np.zeros(6)
    head_U_tran = head_U * [0, 0, 1, 0, 0, 0]
    head_U_rota = head_U * [0, 0, 0, 0, 0, 1]
    head_U_move = head_U * [0, 0, 1, 0, 0, 1]
    tail_U_tran = tail_U * [0, 0, 1, 0, 0, 0]
    tail_U_rota = tail_U * [0, 0, 0, 0, 0, 1]
    tail_U_move = tail_U * [0, 0, 1, 0, 0, 1]

    U_list = (zero_U, zero_U, zero_U, tail_U_rota)
    prefix = 'Tgeo_tran'
    given_velocity_ecoli_4part(problem, U_list, prefix, **kwargs)


def main_fun():
    head_U = np.array([0, 0, 1, 0, 0, 1])
    tail_U = np.array([0, 0, 1, 0, 0, 1])
    main_kwargs = {'head_U': head_U,
                   'tail_U': tail_U, }
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    fileHandle = problem_kwargs['fileHandle']
    with open(fileHandle + '_pick.bin', 'rb') as myinput:
        unpick = pickle.Unpickler(myinput)
        problem = unpick.load()
        problem.unpickmyself()
    kwargs = problem.get_kwargs()
    # assume ecoli_U==0, rel_Us and rel_Uh are true (physical) velocity of head and tail respectively.
    kwargs['ecoli_U'] = np.zeros(6)
    kwargs['rel_Us'] = head_U
    kwargs['rel_Uh'] = tail_U
    kwargs['with_T_geo'] = 0
    if 'ffweight' in kwargs.keys():
        kwargs['ffweightx'] = kwargs['ffweight']
        kwargs['ffweighty'] = kwargs['ffweight']
        kwargs['ffweightz'] = kwargs['ffweight']
        kwargs['ffweightT'] = kwargs['ffweight']
    if 'factor' in kwargs.keys():
        kwargs['stokesletsInPipe_pipeFactor'] = kwargs['factor']
    problem.set_kwargs(**kwargs)
    print_case_info(**kwargs)
    problem.print_info()

    main_head(problem)
    main_helix_tail(problem)
    main_ecoli_NoTgeo(problem, given_velocity_ecoli_NoTgeo)
    return True

if __name__ == '__main__':
    main_fun()

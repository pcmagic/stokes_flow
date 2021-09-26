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
# from src.support_class import *
from codeStore import ecoli_common


def get_problem_kwargs(**main_kwargs):
    problem_kwargs = ecoli_common.get_problem_kwargs()
    OptDB = PETSc.Options()
    fileHandle = OptDB.getString('f', 'ecoliNearSphere')
    problem_kwargs['fileHandle'] = fileHandle

    kwargs_list = (get_one_ellipse_kwargs_v2(), main_kwargs)
    for t_kwargs in kwargs_list:
        for key in t_kwargs:
            problem_kwargs[key] = t_kwargs[key]

    ecoli_ini_rot_theta = OptDB.getReal('ecoli_ini_rot_theta', 0)
    ecoli_ini_rot_phi = OptDB.getReal('ecoli_ini_rot_phi', 0)
    ecoli_ini_rot_psi = OptDB.getReal('ecoli_ini_rot_psi', 0)
    problem_kwargs['ecoli_ini_rot_theta'] = ecoli_ini_rot_theta
    problem_kwargs['ecoli_ini_rot_phi'] = ecoli_ini_rot_phi
    problem_kwargs['ecoli_ini_rot_psi'] = ecoli_ini_rot_psi
    return problem_kwargs


def print_case_info(**problem_kwargs):
    assert np.isclose(problem_kwargs['zoom_factor'], 1)
    assert np.isclose(problem_kwargs['rot_theta'], 0)

    fileHandle = problem_kwargs['fileHandle']
    t1 = ecoli_common.print_case_info(**problem_kwargs)
    print_one_ellipse_info_v2(fileHandle, **problem_kwargs)
    ecoli_ini_rot_theta = problem_kwargs['ecoli_ini_rot_theta']
    ecoli_ini_rot_phi = problem_kwargs['ecoli_ini_rot_phi']
    ecoli_ini_rot_psi = problem_kwargs['ecoli_ini_rot_psi']
    PETSc.Sys.Print('ecoli rotation, theta: %f, phi: %f, psi: %f. ' %
                    (ecoli_ini_rot_theta, ecoli_ini_rot_phi, ecoli_ini_rot_psi))
    return t1


def create_comp(**problem_kwargs):
    ecoli_comp = create_ecoli_2part(**problem_kwargs)
    head_center = ecoli_comp.get_obj_list()[0].get_u_geo().get_center()
    ecoli_center = ecoli_comp.get_center()
    tmove = (ecoli_center - head_center) * np.array((0, 0, 1))
    ecoli_comp.move(tmove)
    sphere_obj = create_one_ellipse_v2(**problem_kwargs)
    sphere_comp = sf.ForceFreeComposite(center=np.zeros(3), name='sphere_comp',
                                        norm=sphere_obj.get_u_geo().get_geo_norm())
    sphere_comp.add_obj(obj=sphere_obj, rel_U=np.zeros(6))
    return ecoli_comp, sphere_comp


def pickle_data(ecoli_comp, sphere_comp, **problem_kwargs):
    fileHandle = problem_kwargs['fileHandle']
    rel_Us = problem_kwargs['rel_Us']
    rel_Uh = problem_kwargs['rel_Uh']

    head_U = ecoli_comp.get_ref_U() + rel_Us
    tail_U = ecoli_comp.get_ref_U() + rel_Uh
    sphere_U = sphere_comp.get_ref_U()
    ecoli_center = ecoli_comp.get_center()
    head_F = ecoli_comp.get_obj_list()[0].get_total_force(center=ecoli_center)
    tail_F = np.sum([tobj.get_total_force(center=ecoli_center)
                     for tobj in ecoli_comp.get_obj_list()[1:]], axis=0)
    sphere_center = sphere_comp.get_center()
    sphere_F = sphere_comp.get_obj_list()[0].get_total_force(center=sphere_center)

    comm = PETSc.COMM_WORLD.tompi4py()
    rank = comm.Get_rank()
    save_name = '%s.pickle' % fileHandle
    tpickle = {'problem_kwargs': problem_kwargs,
               'head_U':         head_U,
               'tail_U':         tail_U,
               'sphere_U':       sphere_U,
               'head_F':         head_F,
               'tail_F':         tail_F,
               'sphere_F':       sphere_F, }
    if rank == 0:
        with open(save_name, 'wb') as output:
            pickle.dump(tpickle, output, protocol=4)
        print('save pickle data to %s. ' % save_name)

    # # dbg
    # PETSc.Sys.Print('#################### DBG code2')
    # PETSc.Sys.Print(head_U)
    # PETSc.Sys.Print(tail_U)
    # PETSc.Sys.Print(ecoli_U)
    # PETSc.Sys.Print(sphere_obj.get_u_geo().get_velocity()[:10])
    return True


def main_fix_sphere(**main_kwargs):
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    print_case_info(**problem_kwargs)

    ecoli_comp, sphere_comp = create_comp(**problem_kwargs)
    sphere_obj = sphere_comp.get_obj_list()[0]

    problem = sf.ForceFreeProblem(**problem_kwargs)
    problem.add_obj(ecoli_comp)
    problem.add_obj(sphere_obj)
    problem.print_info()
    problem.create_matrix()
    problem.solve()

    print_single_ecoli_forcefree_result(ecoli_comp, **problem_kwargs)
    pickle_data(ecoli_comp, sphere_comp, **problem_kwargs)
    return True


def main_fix_sphere_inter(**main_kwargs):
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    print_case_info(**problem_kwargs)

    ecoli_comp, sphere_comp = create_comp(**problem_kwargs)
    sphere_obj = sphere_comp.get_obj_list()[0]

    # prepare step
    PETSc.Sys.Print('########## prepare step start ##########')
    problem = sf.ForceFreeProblem(**problem_kwargs)
    problem.add_obj(ecoli_comp)
    problem.add_obj(sphere_obj)
    problem.print_info()
    problem.create_matrix()
    problem.solve()
    print_single_ecoli_forcefree_result(ecoli_comp, **problem_kwargs)
    ecoli_U = ecoli_comp.get_ref_U()
    problem.destroy()
    PETSc.Sys.Print('########## prepare step finish ##########')

    # solve step
    problem = sf.ForceFreeIterateProblem(**problem_kwargs)
    problem.add_obj(ecoli_comp)
    problem.add_obj(sphere_obj)
    problem.set_iterate_comp(ecoli_comp)
    problem.create_matrix()
    problem.do_iterate3(ini_refU1=ecoli_U, rtol=1e-10, atol=1e-20)

    print_single_ecoli_forcefree_result(ecoli_comp, **problem_kwargs)
    pickle_data(ecoli_comp, sphere_comp, **problem_kwargs)
    return True


def main_flex_sphere(**main_kwargs):
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    print_case_info(**problem_kwargs)

    ecoli_comp, sphere_comp = create_comp(**problem_kwargs)

    problem = sf.ForceFreeProblem(**problem_kwargs)
    problem.add_obj(ecoli_comp)
    problem.add_obj(sphere_comp)
    problem.print_info()
    problem.create_matrix()
    problem.solve()

    print_single_ecoli_forcefree_result(ecoli_comp, **problem_kwargs)
    pickle_data(ecoli_comp, sphere_comp, **problem_kwargs)
    return True


def main_no_sphere(**main_kwargs):
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    print_case_info(**problem_kwargs)

    ecoli_comp, _ = create_comp(**problem_kwargs)

    problem = sf.ForceFreeProblem(**problem_kwargs)
    problem.add_obj(ecoli_comp)
    problem.print_info()
    problem.create_matrix()
    problem.solve()

    print_single_ecoli_forcefree_result(ecoli_comp, **problem_kwargs)
    return True


if __name__ == '__main__':
    OptDB = PETSc.Options()
    if OptDB.getBool('main_fix_sphere', False):
        OptDB.setValue('main_fun', False)
        main_fix_sphere()

    if OptDB.getBool('main_fix_sphere_inter', False):
        OptDB.setValue('main_fun', False)
        main_fix_sphere_inter()

    if OptDB.getBool('main_flex_sphere', False):
        OptDB.setValue('main_fun', False)
        main_flex_sphere()

    if OptDB.getBool('main_no_sphere', False):
        OptDB.setValue('main_fun', False)
        main_no_sphere()

    # if OptDB.getBool('main_fun', True):
    #     main_fun()

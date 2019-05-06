# coding=utf-8

import sys
import petsc4py

petsc4py.init(sys.argv)

import numpy as np
from time import time
from scipy.io import savemat
# from src.stokes_flow import problem_dic, obj_dic
from src.geo import *
from petsc4py import PETSc
from src import stokes_flow as sf
from src.myio import *
from src.StokesFlowMethod import light_stokeslets_matrix_3d
from src.support_class import *
from src.objComposite import createEcoliComp_ellipse
# from src.myvtk import save_singleEcoli_vtk
import ecoli_in_pipe.ecoli_common as ec
import os


# import import_my_lib

# Todo: rewrite input and print process.
def get_problem_kwargs(**main_kwargs):
    OptDB = PETSc.Options()
    fileHandle = OptDB.getString('f', 'motion_ecoli')
    OptDB.setValue('f', fileHandle)
    problem_kwargs = ec.get_problem_kwargs()
    problem_kwargs['fileHandle'] = fileHandle

    kwargs_list = (get_shearFlow_kwargs(), get_update_kwargs(), main_kwargs,)
    for t_kwargs in kwargs_list:
        for key in t_kwargs:
            problem_kwargs[key] = t_kwargs[key]

    # vtk_matname = OptDB.getString('vtk_matname', 'pipe_dbg')
    # t_path = os.path.dirname(os.path.abspath(__file__))
    # vtk_matname = os.path.normpath(os.path.join(t_path, vtk_matname))
    # problem_kwargs['vtk_matname'] = vtk_matname
    return problem_kwargs


def print_case_info(**problem_kwargs):
    caseIntro = '-->Ecoli in infinite shear flow case, force free case. '
    ec.print_case_info(caseIntro, **problem_kwargs)
    print_update_info(**problem_kwargs)
    print_shearFlow_info(**problem_kwargs)
    return True


# @profile
def main_fun(**main_kwargs):
    comm = PETSc.COMM_WORLD.tompi4py()
    rank = comm.Get_rank()
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    print_case_info(**problem_kwargs)
    fileHandle = problem_kwargs['fileHandle']
    max_iter = problem_kwargs['max_iter']
    update_order = problem_kwargs['update_order']
    eval_dt = problem_kwargs['eval_dt']
    rs1 = problem_kwargs['rs1']

    if not problem_kwargs['restart']:
        # create obj
        ecoli_comp = createEcoliComp_ellipse(name='ecoli0', **problem_kwargs)
        ecoli_comp.set_update_para(fix_x=False, fix_y=False, fix_z=False,
                                   update_order=update_order, update_fun=Adams_Moulton_Methods)
        OptDB = PETSc.Options()
        iterateTolerate = OptDB.getReal('iterateTolerate', 1e-4)
        problem = sf.ShearFlowForceFreeIterateProblem(**problem_kwargs)
        problem.add_obj(ecoli_comp)
        problem.set_iterate_comp(ecoli_comp)
        problem.print_info()
        problem.create_matrix()
        refU, Ftol, Ttol = problem.do_iterate(tolerate=iterateTolerate)
        ecoli_comp.set_ref_U(refU)
        print_single_ecoli_force_result(ecoli_comp, prefix='', part='full', **problem_kwargs)
        PETSc.Sys.Print('---->>>reference velocity is', refU)
        dbg_u1 = problem.dbg_get_U()
        problem.destroy()

        # dbg, check if force and torque free
        ecoli_comp1 = ecoli_comp.copy()
        ref_U = ecoli_comp.get_ref_U()
        # ref_U = np.array([-0.01706285, -0.01050803, 0.00706543, 0.00976113, 0.74549499, -0.43994034])
        center = ecoli_comp.get_center()
        problem = sf.ShearFlowProblem(**problem_kwargs)
        for obj, rel_U in zip(ecoli_comp.get_obj_list(), ecoli_comp.get_rel_U_list()):
            u_geo = obj.get_u_geo()
            u_geo.set_rigid_velocity(ref_U + rel_U, center=center)
            problem.add_obj(obj)
        # problem.print_info()
        problem.create_matrix()
        dbg_u2 = problem.dbg_get_U()
        if rank == 0:
            savemat('dbg',
                    {'headU1': ecoli_comp1.get_obj_list()[0].get_u_geo().get_velocity(),
                     'headU2': ecoli_comp.get_obj_list()[0].get_u_geo().get_velocity(),
                     'dbg_u2': dbg_u2,
                     'dbg_u1': dbg_u1}, oned_as='column')
        problem.solve()
        print_single_ecoli_force_result(ecoli_comp, prefix='', part='full', **problem_kwargs)
        PETSc.Sys.Print('ref_U %s' % str(ref_U))
        PETSc.Sys.Print('|ref_U| %s, %s' % (str(np.linalg.norm(ref_U[:3])), str(np.linalg.norm(ref_U[3:]))))
        PETSc.Sys.Print('U_head %s' % str(ref_U + ecoli_comp.get_rel_U_list()[0]))
        PETSc.Sys.Print('U_tail %s' % str(ref_U + ecoli_comp.get_rel_U_list()[1]))
        sumF = np.sum(np.vstack([tobj.get_total_force(center=center) for tobj in ecoli_comp.get_obj_list()]), axis=0)
        PETSc.Sys.Print('check, sumF is %s' % str(sumF))
        PETSc.Sys.Print('check, sumF/headF is %s' %
                        str(sumF / ecoli_comp.get_obj_list()[0].get_total_force(center=center)))
    else:
        pass
    return True


if __name__ == '__main__':
    main_fun()

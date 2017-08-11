# coding=utf-8
# 1. generate velocity and force nodes of sphere using MATLAB,
# 2. for each force node, get b, solve surrounding velocity boundary condition (pipe and cover, named boundary velocity) using formula from Liron's paper, save .mat file
# 3. read .mat file, for each boundary velocity, solve associated boundary force.
# 4. solve sphere M matrix using boundary force.
# 5. solve problem and check.

import sys

import petsc4py

petsc4py.init(sys.argv)
from os import path as ospath

t_path = sys.path[0]
t_path = ospath.dirname(t_path)
if ospath.isdir(t_path):
    sys.path = [t_path] + sys.path
else:
    err_msg = "can not add path father path"
    raise ValueError(err_msg)

import numpy as np
import pickle
from time import time
from scipy.io import loadmat
from petsc4py import PETSc
from src import stokes_flow as sf
from src.stokes_flow import problem_dic, obj_dic
from src.geo import *
from src.myio import *
from src.objComposite import createEcoli_tunnel

def save_vtk(problem: sf.stokesFlowProblem):
    t0 = time( )
    problem_kwargs = problem.get_kwargs( )
    fileHeadle = problem_kwargs['fileHeadle']
    matrix_method = problem_kwargs['matrix_method']
    with_T_geo = problem_kwargs['with_T_geo']

    problem.vtk_obj(fileHeadle)

    # bgeo = geo()
    # bnodesHeadle = problem_kwargs['bnodesHeadle']
    # matname = problem_kwargs['matname']
    # bgeo.mat_nodes(filename=matname, mat_handle=bnodesHeadle)
    # belemsHeadle = problem_kwargs['belemsHeadle']
    # bgeo.mat_elmes(filename=matname, mat_handle=belemsHeadle, elemtype='tetra')
    # problem.vtk_tetra(fileHeadle + '_Velocity', bgeo)

    # create check obj
    check_kwargs = problem_kwargs.copy( )
    check_kwargs['nth'] = problem_kwargs['nth'] - 2 if problem_kwargs['nth'] >= 6 else problem_kwargs['nth'] + 1
    check_kwargs['ds'] = problem_kwargs['ds'] * 1.2
    check_kwargs['hfct'] = 1
    objtype = obj_dic[matrix_method]
    ecoli_comp = problem.get_obj_list( )[0]
    ecoli_comp_check = createEcoli_tunnel(objtype, **check_kwargs)
    ecoli_comp_check.set_ref_U(ecoli_comp.get_ref_U( ))
    if with_T_geo:
        velocity_err_sphere, velocity_err_helix0, velocity_err_helix1, velocity_err_Tgeo= \
            problem.vtk_check(fileHeadle, ecoli_comp_check)
        PETSc.Sys.Print('velocity error of sphere (total, x, y, z): ', velocity_err_sphere)
        PETSc.Sys.Print('velocity error of helix0 (total, x, y, z): ', velocity_err_helix0)
        PETSc.Sys.Print('velocity error of helix1 (total, x, y, z): ', velocity_err_helix1)
        PETSc.Sys.Print('velocity error of Tgeo (total, x, y, z): ', velocity_err_Tgeo)
    else:
        velocity_err_sphere, velocity_err_helix0, velocity_err_helix1= \
            problem.vtk_check(fileHeadle, ecoli_comp_check)
        PETSc.Sys.Print('velocity error of sphere (total, x, y, z): ', velocity_err_sphere)
        PETSc.Sys.Print('velocity error of helix0 (total, x, y, z): ', velocity_err_helix0)
        PETSc.Sys.Print('velocity error of helix1 (total, x, y, z): ', velocity_err_helix1)

    t1 = time( )
    PETSc.Sys.Print('%s: write vtk files use: %fs' % (str(problem), (t1 - t0)))

    return True


def get_problem_kwargs(**main_kwargs):
    problem_kwargs = get_solver_kwargs(**main_kwargs)
    OptDB = PETSc.Options( )
    fileHeadle = OptDB.getString('f', 'singleEcoliPro')
    problem_kwargs['fileHeadle'] = fileHeadle

    kwargs_list = (main_kwargs, get_vtk_tetra_kwargs( ), get_ecoli_kwargs( ), get_forceFree_kwargs( ))
    for t_kwargs in kwargs_list:
        for key in t_kwargs:
            problem_kwargs[key] = t_kwargs[key]
    return problem_kwargs


def print_case_info(**problem_kwargs):
    fileHeadle = problem_kwargs['fileHeadle']
    print_solver_info_forceFree(**problem_kwargs)
    print_forceFree_info(**problem_kwargs)
    print_ecoli_info(fileHeadle, **problem_kwargs)
    return True


# @profile
def main_fun(**main_kwargs):
    comm = PETSc.COMM_WORLD.tompi4py( )
    rank = comm.Get_rank( )
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    fileHeadle = problem_kwargs['fileHeadle']

    if not problem_kwargs['restart']:
        print_case_info(**problem_kwargs)
        matrix_method = problem_kwargs['matrix_method']
        rh1 = problem_kwargs['rh1']
        zoom_factor = problem_kwargs['zoom_factor']
        with_T_geo = problem_kwargs['with_T_geo']

        # create ecoli
        ecoli_comp = createEcoli_tunnel(name='ecoli_0', **problem_kwargs)

        problem = sf.forceFreeProblem(**problem_kwargs)
        if problem_kwargs['pickProblem']:
            problem.pickmyself(fileHeadle, check=True)
        problem.add_obj(ecoli_comp)
        problem.print_info( )
        if problem_kwargs['plot_geo']:
            # vsobj.show_f_nodes(' ')
            # vsobj.show_u_nodes(' ')
            # vhobj0.show_f_nodes(' ')
            # vhobj0.show_u_nodes(' ')
            # vhobj1.show_f_nodes(' ')
            # vhobj1.show_u_nodes(' ')
            # ecoli_comp.show_f_nodes(' ')
            # ecoli_comp.show_u_nodes(' ')
            # vsobj.show_f_u_nodes(' ')
            # vhobj0.show_f_u_nodes(' ')
            ecoli_comp.show_f_u_nodes('-')

        problem.create_matrix( )
        problem.solve( )
        # # debug
        # problem.saveM_ASCII('%s_M.txt' % fileHeadle)
        # problem.saveF_ASCII('%s_F.txt' % fileHeadle)
        # problem.saveV_ASCII('%s_V.txt' % fileHeadle)

        print_single_ecoli_forceFree_result(ecoli_comp, **problem_kwargs)

        if problem_kwargs['pickProblem']:
            problem.pickmyself(fileHeadle)
        save_vtk(problem)
    else:
        pass
        # with open(fileHeadle + '_pick.bin', 'rb') as input:
        #     unpick = pickle.Unpickler(input)
        #     problem = unpick.load()
        #     problem.unpickmyself()
        #     residualNorm = problem.get_residualNorm()
        #     PETSc.Sys.Print('---->>>unpick the problem from file %s_pick.bin' % (fileHeadle))
        #
        #     problem_kwargs1 = get_problem_kwargs(**main_kwargs)
        #     problem_kwargs = problem.get_kwargs()
        #     problem_kwargs['matname'] = problem_kwargs1['matname']
        #     problem_kwargs['bnodesHeadle'] = problem_kwargs1['bnodesHeadle']
        #     problem_kwargs['belemsHeadle'] = problem_kwargs1['belemsHeadle']
        #     problem.set_kwargs(**problem_kwargs)
        #     print_case_info(**problem_kwargs)
        #     problem.print_info()
        #     # problem.create_matrix()
        #     save_vtk(problem)

    return True


if __name__ == '__main__':
    main_fun( )

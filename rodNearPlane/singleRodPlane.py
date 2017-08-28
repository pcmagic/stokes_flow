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
# sys.path = ['/home/zhangji/stokes_flow-master'] + sys.path

import numpy as np
# import pickle
# from time import time
# from scipy.io import loadmat
# from src.stokes_flow import problem_dic, obj_dic
from src.geo import *
from petsc4py import PETSc
from src import stokes_flow as sf
from src.stokes_flow import problem_dic, obj_dic
from src.myio import *


# from src.objComposite import createEcoliComp_tunnel
# from src.myvtk import save_singleEcoli_vtk


def get_problem_kwargs(**main_kwargs):
    problem_kwargs = get_solver_kwargs()
    OptDB = PETSc.Options()
    fileHeadle = OptDB.getString('f', 'singleRodPlane')
    problem_kwargs['fileHeadle'] = fileHeadle

    kwargs_list = (main_kwargs, get_rod_kwargs(), get_givenForce_kwargs())
    for t_kwargs in kwargs_list:
        for key in t_kwargs:
            problem_kwargs[key] = t_kwargs[key]
    return problem_kwargs


def print_case_info(**problem_kwargs):
    fileHeadle = problem_kwargs['fileHeadle']
    print_solver_info_forceFree(**problem_kwargs)
    print_givenForce_info(**problem_kwargs)
    return True


# @profile
def main_fun(**main_kwargs):
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    fileHeadle = problem_kwargs['fileHeadle']
    matrix_method = problem_kwargs['matrix_method']

    if not problem_kwargs['restart']:
        rRod = problem_kwargs['rRod']
        lRod = problem_kwargs['lRod']
        ntRod = problem_kwargs['ntRod']
        eRod = problem_kwargs['eRod']
        Rodfct = problem_kwargs['Rodfct']
        RodThe = problem_kwargs['RodThe']
        rel_URod = problem_kwargs['rel_URod']
        center = problem_kwargs['center']
        zoom_factor = problem_kwargs['zoom_factor']
        givenF = problem_kwargs['givenF']
        dth = 2 * np.pi / ntRod
        rod_geo = tunnel_geo()
        rod_geo.create_deltatheta(dth=dth, radius=rRod, length=lRod, epsilon=eRod,
                                  with_cover=True, factor=Rodfct, left_hand=False)
        rod_geo.move(displacement=center)
        rod_geo.node_zoom(factor=zoom_factor, zoom_origin=center)
        norm = np.array((0, 1, 0))
        theta = -(np.pi / 2 + RodThe)
        rod_geo.node_rotation(norm=norm, theta=theta, rotation_origin=center)
        rod_obj = sf.obj_dic[matrix_method]()
        rod_obj.set_data(f_geo=rod_geo, u_geo=rod_geo, name='rod_obj_0')
        rod_comp = sf.givenForceComposite(center=center, name='rod_comp_0', givenF=givenF)
        rod_comp.add_obj(obj=rod_obj, rel_U=rel_URod)
        problem_kwargs['delta'] = eRod * dth * rRod

        print_case_info(**problem_kwargs)
        problem = sf.givenForceProblem(**problem_kwargs)
        problem.add_obj(rod_comp)

        if problem_kwargs['pickProblem']:
            problem.pickmyself(fileHeadle, check=True)
        problem.print_info()
        # problem.create_matrix()
        # problem.solve()
        # # debug
        # # problem.saveM_ASCII('%s_M.txt' % fileHeadle)
        #
        # PETSc.Sys.Print('---->>>Min node z is', rod_geo.get_nodes_z().min())
        # PETSc.Sys.Print('---->>>center is', center)
        # rod_F = rod_comp.get_total_force()
        # PETSc.Sys.Print('---->>>Resultant err is', np.sqrt(np.sum((rod_F - givenF) ** 2)))
        # rod_U = rod_comp.get_ref_U() + rel_URod
        # PETSc.Sys.Print('---->>>Rod velocity is', rod_U)
        #
        # if problem_kwargs['pickProblem']:
        #     problem.pickmyself(fileHeadle, pick_M=True)
        # problem.destroy()
        rod_U = 0

    else:
        rod_U = 0
        pass

    return rod_U

def job_script():
    lRod = 5
    OptDB = PETSc.Options()
    OptDB.setValue('sm', 'rs_plane')
    OptDB.setValue('ksp_max_it', 1000)
    OptDB.setValue('lRod', lRod)
    OptDB.setValue('eRod', 0.5)
    OptDB.setValue('ntRod', 50)
    OptDB.setValue('givenTz', 1)
    OptDB.setValue('centerz', 2.75)
    # OptDB.setValue('plot_geo', 1)

    RodThe = np.pi * np.linspace(0.2, 0, 10)
    rod_U = []
    for t_RodThe in RodThe:
        OptDB.setValue('RodThe', t_RodThe)
        rod_U.append(main_fun())
    u = np.vstack(rod_U)[:, 1]
    w = np.vstack(rod_U)[:, 5]
    print(u / (w * lRod))
    print(w)

if __name__ == '__main__':
    # main_fun()
    job_script()


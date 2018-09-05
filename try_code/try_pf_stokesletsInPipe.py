import sys
import petsc4py

petsc4py.init(sys.argv)

# from scipy.io import savemat, loadmat
# from src.ref_solution import *
# import warnings
# from memory_profiler import profile
# from time import time
import pickle
import numpy as np
from src import stokes_flow as sf
from src.stokes_flow import problem_dic, obj_dic, stokesFlowObj
from petsc4py import PETSc
from src.geo import *
from src.myio import *
from src.objComposite import *
from src.myvtk import *
from src.StokesFlowMethod import *
from src.stokesletsInPipe import *


def print_case_info(**problem_kwargs):
    fileHeadle = problem_kwargs['fileHeadle']
    print_solver_info(**problem_kwargs)
    print_sphere_info(fileHeadle, **problem_kwargs)
    return True


def get_problem_kwargs(**main_kwargs):
    problem_kwargs = get_solver_kwargs()
    OptDB = PETSc.Options()
    fileHeadle = OptDB.getString('f', 'try_dual')
    OptDB.setValue('f', fileHeadle)
    problem_kwargs['fileHeadle'] = fileHeadle

    kwargs_list = (main_kwargs, get_vtk_tetra_kwargs(), get_sphere_kwargs(),)
    for t_kwargs in kwargs_list:
        for key in t_kwargs:
            problem_kwargs[key] = t_kwargs[key]
    return problem_kwargs


def main_fun(**main_kwargs):
    OptDB = PETSc.Options()
    # main_kwargs['matrix_method'] = 'pf_dualPotential'
    main_kwargs['matrix_method'] = 'pf'
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    matrix_method = problem_kwargs['matrix_method']
    fileHeadle = problem_kwargs['fileHeadle']
    print_case_info(**problem_kwargs)
    GreenFunThreshold = 50

    # place a force in the tunnel to solve boundary condition
    b = OptDB.getReal('b', 0.1)
    stokeslets_post = np.array((b, 0, 0)).reshape(1, 3)
    stokeslets_f = np.array((1, 0, 0))
    stokeslets_f_petsc = PETSc.Vec().create(comm=PETSc.COMM_WORLD)
    stokeslets_f_petsc.setSizes(3)
    stokeslets_f_petsc.setFromOptions()
    stokeslets_f_petsc.setUp()
    stokeslets_f_petsc[:] = stokeslets_f[:]
    stokeslets_f_petsc.assemble()
    img_stokeslets_post = np.array((2 - b, 0, 0)).reshape(1, 3)
    img_stokeslets_f = np.array((0, 0, 0))
    img_stokeslets_f_petsc = PETSc.Vec().create(comm=PETSc.COMM_WORLD)
    img_stokeslets_f_petsc.setSizes(3)
    img_stokeslets_f_petsc.setFromOptions()
    img_stokeslets_f_petsc.setUp()
    img_stokeslets_f_petsc[:] = img_stokeslets_f[:]
    img_stokeslets_f_petsc.assemble()

    # Tunnel geo
    nt = OptDB.getReal('nt', 3)
    tfct = OptDB.getReal('tfct', 1)
    dth = 2 * np.pi / nt
    tunnel_length = 2
    tunnel_u_geo = tunnel_geo()  # pf, force geo
    if 'dualPotential' in matrix_method:
        tunnel_u_geo.set_dof(4)
    epsilon = OptDB.getReal('et', 1)
    tunnel_f_geo = tunnel_u_geo.create_deltatheta(dth=dth, radius=1, length=tunnel_length, epsilon=epsilon,
                                                  with_cover=2, factor=tfct)
    # 1). offset stokeslets velocity
    m0_petsc = light_stokeslets_matrix_3d_petsc(tunnel_u_geo.get_nodes(), stokeslets_post)
    u0_petsc = m0_petsc.createVecLeft()
    m0_petsc.mult(stokeslets_f_petsc, u0_petsc)
    scatter, temp = PETSc.Scatter().toAll(u0_petsc)
    scatter.scatterBegin(u0_petsc, temp, False, PETSc.Scatter.Mode.FORWARD)
    scatter.scatterEnd(u0_petsc, temp, False, PETSc.Scatter.Mode.FORWARD)
    u0 = temp.getArray()
    u0_petsc.destroy()
    img_m0_petsc = light_stokeslets_matrix_3d_petsc(tunnel_u_geo.get_nodes(), img_stokeslets_post)
    img_u0_petsc = img_m0_petsc.createVecLeft()
    img_m0_petsc.mult(img_stokeslets_f_petsc, img_u0_petsc)
    scatter, temp = PETSc.Scatter().toAll(img_u0_petsc)
    scatter.scatterBegin(img_u0_petsc, temp, False, PETSc.Scatter.Mode.FORWARD)
    scatter.scatterEnd(img_u0_petsc, temp, False, PETSc.Scatter.Mode.FORWARD)
    img_u0 = temp.getArray()
    img_u0_petsc.destroy()
    # 2). stokeslets in pipe velocity
    greenFun = detail(threshold=GreenFunThreshold, b=np.sqrt(np.sum(stokeslets_post ** 2)))
    greenFun.solve_prepare()
    u1, u2, u3 = greenFun.solve_uxyz(tunnel_u_geo.get_nodes())
    tunnel_u_geo.set_velocity((u1 * stokeslets_f[0] + u2 * stokeslets_f[1] + u3 * stokeslets_f[2]).flatten() - u0 - img_u0)
    # tunnel_u_geo.set_velocity(u0 + img_u0)
    obj_tunnel = obj_dic[matrix_method]()
    obj_tunnel.set_data(tunnel_f_geo, tunnel_u_geo, name='tunnel')

    problem = problem_dic[matrix_method](**problem_kwargs)
    problem.add_obj(obj_tunnel)
    problem.show_velocity(length_factor=1, show_nodes=False)
    problem.print_info()
    problem.create_matrix()
    problem.solve()

    tunnel_geo_check = tunnel_geo()  # pf, force geo
    dth = 2 * np.pi / 30
    tunnel_geo_check.create_deltatheta(dth=dth, radius=1, length=tunnel_length, epsilon=0, with_cover=1)
    # 1). offset stokeslets velocity
    m0_petsc = light_stokeslets_matrix_3d_petsc(tunnel_geo_check.get_nodes(), stokeslets_post)
    u0_petsc = m0_petsc.createVecLeft()
    m0_petsc.mult(stokeslets_f_petsc, u0_petsc)
    scatter, temp = PETSc.Scatter().toAll(u0_petsc)
    scatter.scatterBegin(u0_petsc, temp, False, PETSc.Scatter.Mode.FORWARD)
    scatter.scatterEnd(u0_petsc, temp, False, PETSc.Scatter.Mode.FORWARD)
    u0 = temp.getArray()
    u0_petsc.destroy()
    img_m0_petsc = light_stokeslets_matrix_3d_petsc(tunnel_geo_check.get_nodes(), img_stokeslets_post)
    img_u0_petsc = img_m0_petsc.createVecLeft()
    img_m0_petsc.mult(img_stokeslets_f_petsc, img_u0_petsc)
    scatter, temp = PETSc.Scatter().toAll(img_u0_petsc)
    scatter.scatterBegin(img_u0_petsc, temp, False, PETSc.Scatter.Mode.FORWARD)
    scatter.scatterEnd(img_u0_petsc, temp, False, PETSc.Scatter.Mode.FORWARD)
    img_u0 = temp.getArray()
    img_u0_petsc.destroy()
    # 2). stokeslets in pipe velocity
    greenFun = detail(threshold=GreenFunThreshold, b=np.sqrt(np.sum(stokeslets_post ** 2)))
    greenFun.solve_prepare()
    u1, u2, u3 = greenFun.solve_uxyz(tunnel_geo_check.get_nodes())
    tunnel_geo_check.set_velocity((u1 * stokeslets_f[0] + u2 * stokeslets_f[1] + u3 * stokeslets_f[2]).flatten() - u0 - img_u0)
    obj_check = obj_dic[matrix_method]()
    obj_check.set_data(tunnel_geo_check, tunnel_geo_check, name='full')
    tunnel_err = problem.vtk_check(fileHeadle + '_Check_tunnel', obj_check)
    PETSc.Sys.Print('velocity error of tunnel (total, x, y, z): ', next(tunnel_err))
    return True


if __name__ == '__main__':
    main_fun()

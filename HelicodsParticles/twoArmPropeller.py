import sys

import petsc4py

petsc4py.init(sys.argv)
# from scipy.io import savemat, loadmat
# from src.ref_solution import *
# import warnings
# from memory_profiler import profile
# from time import time
from src.myio import *
from src.objComposite import *
from src.StokesFlowMethod import *
from src.geo import *
from src import stokes_flow as sf
from src import slender_body as slb
from codeStore.helix_common import AtBtCt, AtBtCt_full


def get_problem_kwargs(**main_kwargs):
    problem_kwargs = get_solver_kwargs()
    OptDB = PETSc.Options()
    fileHandle = OptDB.getString('f', 'twoArmPropeller')
    OptDB.setValue('f', fileHandle)
    problem_kwargs['fileHandle'] = fileHandle
    dumb_d = OptDB.getReal('dumb_d', 5)
    problem_kwargs['dumb_d'] = dumb_d
    dumb_theta = OptDB.getReal('dumb_theta', np.pi / 3)
    problem_kwargs['dumb_theta'] = dumb_theta

    kwargs_list = (get_obj_helicoid_kwargs(), get_sphere_kwargs(),
                   get_forcefree_kwargs(), main_kwargs,)
    for t_kwargs in kwargs_list:
        for key in t_kwargs:
            problem_kwargs[key] = t_kwargs[key]

    dumb_rs2_fct = OptDB.getReal('dumb_rs2_fct', 1)
    problem_kwargs['dumb_rs2_fct'] = dumb_rs2_fct
    dumb_ds2_fct = OptDB.getReal('dumb_ds2_fct', 1)
    problem_kwargs['dumb_ds2_fct'] = dumb_ds2_fct
    return problem_kwargs


def print_case_info(**problem_kwargs):
    fileHandle = problem_kwargs['fileHandle']
    print_solver_info(**problem_kwargs)
    print_forcefree_info(**problem_kwargs)
    print_sphere_info(fileHandle, **problem_kwargs)
    dumb_d = problem_kwargs['dumb_d']
    dumb_theta = problem_kwargs['dumb_theta']
    dumb_rs2_fct = problem_kwargs['dumb_rs2_fct']
    dumb_ds2_fct = problem_kwargs['dumb_ds2_fct']
    PETSc.Sys.Print('  dumb_d: %f, dumb_theta: %f' % (dumb_d, dumb_theta))
    PETSc.Sys.Print('  dumb_rs2_fct: %f, dumb_ds2_fct: %f' % (dumb_rs2_fct, dumb_ds2_fct))
    print_obj_helicoid_info(**problem_kwargs)
    return True


def main_resistanceMatrix(**main_kwargs):
    # OptDB = PETSc.Options()
    main_kwargs['zoom_factor'] = 1
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    matrix_method = problem_kwargs['matrix_method']
    fileHandle = problem_kwargs['fileHandle']
    # pickProblem = problem_kwargs['pickProblem']
    print_case_info(**problem_kwargs)

    dumb_d = problem_kwargs['dumb_d']
    dumb_theta = problem_kwargs['dumb_theta']
    ds = problem_kwargs['ds']
    rs = problem_kwargs['rs']
    dumb_rs2_fct = problem_kwargs['dumb_rs2_fct']
    dumb_ds2_fct = problem_kwargs['dumb_ds2_fct']
    sphere_geo0 = sphere_geo()
    sphere_geo0.create_delta(ds, rs)
    # sphere_geo1 = sphere_geo0.copy()
    sphere_geo1 = sphere_geo()
    sphere_geo1.create_delta(ds * dumb_ds2_fct, rs * dumb_rs2_fct)
    sphere_geo0.move(np.array((0, 0, dumb_d / 2)))
    sphere_geo1.move(np.array((0, 0, -dumb_d / 2)))
    dumb_geo = base_geo()
    dumb_geo.combine([sphere_geo0, sphere_geo1], origin=np.zeros(3), geo_norm=np.array((0, 0, 1)))
    dumb_geo.node_rotation(norm=np.array((1, 0, 0)), theta=dumb_theta)
    dumb_obj1 = sf.StokesFlowObj()
    dumb_obj1.set_data(dumb_geo, dumb_geo, 'dumb')

    helicoid_r = problem_kwargs['helicoid_r']
    dumb_obj1.move(np.array((helicoid_r, 0, 0)))
    dumb_obj2 = dumb_obj1.copy()
    dumb_obj2.node_rotation(norm=np.array((0, 0, 1)), theta=np.pi, rotation_origin=np.zeros(3))

    problem = sf.problem_dic[matrix_method](**problem_kwargs)
    problem.add_obj(dumb_obj1)
    problem.add_obj(dumb_obj2)
    problem.print_info()
    problem.create_matrix()
    # PETSc.Sys.Print(problem.get_obj_list()[0].get_u_nodes()[:10])
    # PETSc.Sys.Print(problem.get_M()[:5, :5])
    # PETSc.Sys.Print(helicoid_center)
    At, Bt1, Bt2, Ct = AtBtCt_full(problem, save_vtk=False, pick_M=False, print_each=False,
                                   center=np.zeros(3), save_name=fileHandle)
    PETSc.Sys.Print(At)
    PETSc.Sys.Print(At[2, 2] - At[0, 0])
    PETSc.Sys.Print(np.trace(Bt1), np.trace(Bt2))
    return True


if __name__ == '__main__':
    # code resluts are wrong.
    OptDB = PETSc.Options()
    # pythonmpi helicoid.py -sm lg_rs -legendre_m 3 -legendre_k 2 -epsilon 3 -ffweight 2 -main_fun_noIter 1 -vortexStrength 1 -helicoid_r1 1 -helicoid_r2 0.3 -helicoid_ds 0.03
    # if OptDB.getBool('main_fun_noIter', False):
    #     OptDB.setValue('main_fun', False)
    #     main_fun_noIter()

    if OptDB.getBool('main_resistanceMatrix', False):
        OptDB.setValue('main_fun', False)
        main_resistanceMatrix()

    # if OptDB.getBool('main_fun', True):
    #     main_fun()

import sys
import petsc4py

petsc4py.init(sys.argv)

import numpy as np
from petsc4py import PETSc
import logging
import time
# from src import stokes_flow as sf
# from src.support_class import *
from src import jeffery_model as jm


def get_my_logger(**problem_kwargs):
    fileHandle = problem_kwargs['fileHandle']

    my_logger = logging.getLogger('my_logger')
    my_logger.setLevel(logging.DEBUG)
    txt_name = '%s.txt' % fileHandle
    output_file_handler = logging.FileHandler(txt_name)
    stdout_handler = logging.StreamHandler(sys.stdout)
    my_logger.addHandler(output_file_handler)
    my_logger.addHandler(stdout_handler)
    return my_logger


def get_shearFlow_kwargs():
    OptDB = PETSc.Options()
    flow_strength = OptDB.getReal('flow_strength', 0)
    kwargs = {'planeShearRate': np.array((1, 0, 0)) * flow_strength, }
    return kwargs


def print_shearFlow_kwargs(rank=0, **problem_kwargs):
    my_logger = problem_kwargs['my_logger']
    problemHandle = problem_kwargs['problemHandle']
    planeShearRate = problem_kwargs['planeShearRate']

    if rank == 0:
        my_logger.info('Given background flow: shear flow')
        my_logger.info('  problemHandle: %s' % (problemHandle,))
        my_logger.info('  planeShearRate: %s' % planeShearRate)
        my_logger.info('')
    return True


def get_ABCFlow_kwargs():
    OptDB = PETSc.Options()
    ABC_info = OptDB.getString('ABC_info', 'userinput')
    if ABC_info.lower() == 'homogeneous':
        ABC_A, ABC_B, ABC_C = np.ones(3) * 1
        ABC_D, ABC_E, ABC_F = np.ones(3) * 1
        ABC_G, ABC_H, ABC_I = np.ones(3) * 0
    elif ABC_info.lower() == 'userinput':
        ABC_A = OptDB.getReal('ABC_A', 1)
        ABC_B = OptDB.getReal('ABC_B', 1)
        ABC_C = OptDB.getReal('ABC_C', 1)
        ABC_D = OptDB.getReal('ABC_D', 1)
        ABC_E = OptDB.getReal('ABC_E', 1)
        ABC_F = OptDB.getReal('ABC_F', 1)
        ABC_G = OptDB.getReal('ABC_G', 0)
        ABC_H = OptDB.getReal('ABC_H', 0)
        ABC_I = OptDB.getReal('ABC_I', 0)
    elif ABC_info.lower().endswith('.txt'):
        t1 = np.loadtxt(ABC_info).flatten()
        assert t1.size == 9
        ABC_A, ABC_B, ABC_C, ABC_D, ABC_E, ABC_F, ABC_G, ABC_H, ABC_I = t1
    else:
        raise Exception('  Wrong ABC_info parameter. ')
    problem_kwargs = {'ABC_A': ABC_A,
                      'ABC_B': ABC_B,
                      'ABC_C': ABC_C,
                      'ABC_D': ABC_D,
                      'ABC_E': ABC_E,
                      'ABC_F': ABC_F,
                      'ABC_G': ABC_G,
                      'ABC_H': ABC_H,
                      'ABC_I': ABC_I, }
    return problem_kwargs


def print_ABCFlow_kwargs(rank=0, **problem_kwargs):
    my_logger = problem_kwargs['my_logger']
    problemHandle = problem_kwargs['problemHandle']
    ABC_A = problem_kwargs['ABC_A']
    ABC_B = problem_kwargs['ABC_B']
    ABC_C = problem_kwargs['ABC_C']
    ABC_D = problem_kwargs['ABC_D']
    ABC_E = problem_kwargs['ABC_E']
    ABC_F = problem_kwargs['ABC_F']
    ABC_G = problem_kwargs['ABC_G']
    ABC_H = problem_kwargs['ABC_H']
    ABC_I = problem_kwargs['ABC_I']

    if rank == 0:
        my_logger.info('Given background flow: ABC flow')
        my_logger.info('  problemHandle: %s' % (problemHandle,))
        my_logger.info('  A: %f, B: %f, C: %f ' % (ABC_A, ABC_B, ABC_C))
        my_logger.info('  D: %e, E: %e, F: %e ' % (ABC_D, ABC_E, ABC_F))
        my_logger.info('  G: %f, H: %f, I: %f ' % (ABC_G, ABC_H, ABC_I))
        my_logger.info('')
    return True


def get_baseflow_kwargs():
    problem_dict = {'abc':   jm.ABCFlowProblem,
                    'shear': jm.ShearJefferyProblem}

    get_kwargs_dict = {'abc':   get_ABCFlow_kwargs,
                       'shear': get_shearFlow_kwargs}

    OptDB = PETSc.Options()
    external_flow = OptDB.getString('external_flow', 'shear').lower()
    ini_x = OptDB.getReal('ini_x', 0)
    ini_y = OptDB.getReal('ini_y', 0)
    ini_z = OptDB.getReal('ini_z', 0)

    ini_theta = OptDB.getReal('ini_theta', 0)
    ini_phi = OptDB.getReal('ini_phi', 0)
    ini_psi = OptDB.getReal('ini_psi', 0)
    ini_center = np.array((ini_x, ini_y, ini_z))
    ini_t = OptDB.getReal('ini_t', 0)
    max_t = OptDB.getReal('max_t', 1)
    update_fun = OptDB.getString('update_fun', '5bs')
    problemHandle = problem_dict[external_flow]
    save_every = OptDB.getInt('save_every', 1)
    rtol = OptDB.getReal('rtol', 1e-3)
    atol = OptDB.getReal('atol', 1e-6)
    eval_dt = OptDB.getReal('eval_dt', 0.001)
    table_name = OptDB.getString('table_name', 'ellipsoidB05_baseFlow_theo')
    fileHandle = OptDB.getString('f', '')
    omega_tail = OptDB.getReal('omega_tail', 0)

    problem_kwargs = {'ini_theta':     ini_theta,
                      'ini_phi':       ini_phi,
                      'ini_psi':       ini_psi,
                      'ini_center':    ini_center,
                      'ini_t':         ini_t,
                      'max_t':         max_t,
                      'update_fun':    update_fun,
                      'problemHandle': problemHandle,
                      'save_every':    save_every,
                      'rtol':          rtol,
                      'atol':          atol,
                      'eval_dt':       eval_dt,
                      'table_name':    table_name,
                      'fileHandle':    fileHandle,
                      'omega_tail':    omega_tail,
                      'external_flow': external_flow, }

    flow_kwargs = get_kwargs_dict[external_flow]()
    for key in flow_kwargs:
        problem_kwargs[key] = flow_kwargs[key]

    problem_kwargs['my_logger'] = get_my_logger(**problem_kwargs)

    comm = PETSc.COMM_WORLD.tompi4py()
    rank = comm.Get_rank()
    problem_kwargs['rank'] = rank
    return problem_kwargs


def print_baseflow_kwargs(**problem_kwargs):
    print_kwargs_dict = {'abc':   print_ABCFlow_kwargs,
                         'shear': print_shearFlow_kwargs}

    ini_theta = problem_kwargs['ini_theta']
    ini_phi = problem_kwargs['ini_phi']
    ini_psi = problem_kwargs['ini_psi']
    ini_center = problem_kwargs['ini_center']
    ini_t = problem_kwargs['ini_t']
    max_t = problem_kwargs['max_t']
    update_fun = problem_kwargs['update_fun']
    save_every = problem_kwargs['save_every']
    rtol = problem_kwargs['rtol']
    atol = problem_kwargs['atol']
    eval_dt = problem_kwargs['eval_dt']
    table_name = problem_kwargs['table_name']
    fileHandle = problem_kwargs['fileHandle']
    omega_tail = problem_kwargs['omega_tail']
    external_flow = problem_kwargs['external_flow']
    my_logger = problem_kwargs['my_logger']
    rank = problem_kwargs['rank']

    if rank == 0:
        my_logger.info('')
        my_logger.info('#' * 60)
        my_logger.info('Start, %s' % time.ctime())
        my_logger.info('')
        my_logger.info('Problem information: ')
        my_logger.info('  fileHandle: %s' % (fileHandle,))
        my_logger.info(
                '  ini_theta: %f, ini_phi: %f, ini_psi_t: %f ' % (ini_theta, ini_phi, ini_psi))
        my_logger.info('  ini_center: %s ' % str(ini_center))
        my_logger.info('  ini_t: %f, max_t: %f' % (ini_t, max_t))
        my_logger.info('  updata_fun: %s, rtol: %e, atol: %e' % (update_fun, rtol, atol))
        my_logger.info('  save_every: %d, eval_dt: %e, ' % (save_every, eval_dt))
        my_logger.info('  table_name: %s, omega_tail: %f' % (table_name, omega_tail))
        my_logger.info('')

    print_kwargs_dict[external_flow](**problem_kwargs)
    return True


def print_finish_solve(t0, data, **problem_kwargs):
    my_logger = problem_kwargs['my_logger']
    rank = problem_kwargs['rank']

    base_t, base_dt, base_X, base_thphps, base_U, base_W, base_psi_t = data
    t1 = time.time()
    if rank == 0:
        my_logger.info('Simulation use %fs and %d steps. ' % ((t1 - t0), base_psi_t.size))
        my_logger.info('  lst_theta: %f, lst_phi: %f, lst_psi_t: %f ' %
                       (base_thphps[-1, 0], base_thphps[-1, 1], base_psi_t[-1]))
        my_logger.info('  lst_center: %s ' % str(base_X[-1]))
        my_logger.info('Finish, %s' % time.ctime())
        my_logger.info('#' * 60)
        my_logger.info('')
    return True

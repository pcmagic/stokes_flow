import sys
import matplotlib
import petsc4py

matplotlib.use('agg')
petsc4py.init(sys.argv)

import numpy as np
import pickle
from time import time
from codeStore import support_fun_table as spf_tb
from petsc4py import PETSc
from datetime import datetime
from tqdm import tqdm


# get kewrgs
def get_problem_kwargs(**main_kwargs):
    calculate_fun_dict = {'do_calculate_helix_Petsc4n':
                              spf_tb.do_calculate_helix_Petsc4n,
                          'do_calculate_helix_AvrPetsc4n':
                              spf_tb.do_calculate_helix_AvrPetsc4n,
                          'do_calculate_ellipse_Petsc4n':
                              spf_tb.do_calculate_ellipse_Petsc4n,
                          'do_calculate_ellipse_AvrPetsc4n':
                              spf_tb.do_calculate_ellipse_AvrPetsc4n,
                          'do_calculate_ecoli_Petsc4n':
                              spf_tb.do_calculate_ecoli_Petsc4n,
                          'do_calculate_ecoli_Petsc4nPsi':
                              spf_tb.do_calculate_ecoli_Petsc4nPsi,
                          'do_ShearFlowPetsc4nPsiObj':
                              spf_tb.do_ShearFlowPetsc4nPsiObj,
                          'do_ShearFlowPetsc4nPsiObj_dbg':
                              spf_tb.do_ShearFlowPetsc4nPsiObj_dbg,
                          'do_calculate_ecoli_AvrPetsc4n':
                              spf_tb.do_calculate_ecoli_AvrPetsc4n,
                          'do_calculate_ecoli_passive_Petsc4n':
                              spf_tb.do_calculate_ecoli_passive_Petsc4n,
                          'do_calculate_ecoli_passive_AvrPetsc4n':
                              spf_tb.do_calculate_ecoli_passive_AvrPetsc4n, }
    OptDB = PETSc.Options()
    ini_theta = OptDB.getReal('ini_theta', 0)
    ini_phi = OptDB.getReal('ini_phi', 0)
    ini_psi = OptDB.getReal('ini_psi', 0)
    ini_t = OptDB.getReal('ini_t', 0)
    max_t = OptDB.getReal('max_t', 1)
    rtol = OptDB.getReal('rtol', 1e-3)
    atol = OptDB.getReal('atol', 1e-6)
    eval_dt = OptDB.getReal('eval_dt', 0.01)
    calculate_fun = OptDB.getString('calculate_fun', 'do_calculate_helix_Petsc4n')
    update_fun = OptDB.getString('update_fun', '5bs')
    table_name = OptDB.getString('table_name', 'hlxB01_tau1a')
    fileHandle = OptDB.getString('f', '')
    omega_tail = OptDB.getReal('omega_tail', 0)
    flow_strength = OptDB.getReal('flow_strength', 0)

    problem_kwargs = {'ini_theta':     ini_theta,
                      'ini_phi':       ini_phi,
                      'ini_psi':       ini_psi,
                      'ini_t':         ini_t,
                      'max_t':         max_t,
                      'update_fun':    update_fun,
                      'calculate_fun': calculate_fun_dict[calculate_fun],
                      'rtol':          rtol,
                      'atol':          atol,
                      'eval_dt':       eval_dt,
                      'table_name':    table_name,
                      'fileHandle':    fileHandle,
                      'omega_tail':    omega_tail,
                      'flow_strength': flow_strength, }

    kwargs_list = (main_kwargs,)
    for t_kwargs in kwargs_list:
        for key in t_kwargs:
            problem_kwargs[key] = t_kwargs[key]
    return problem_kwargs


def do_pickel(Table_t, Table_dt, Table_X, Table_P, Table_P2, Table_theta, Table_phi, Table_psi,
              Table_eta, simulate_t, **problem_kwargs):
    ini_theta = problem_kwargs['ini_theta']
    ini_phi = problem_kwargs['ini_phi']
    ini_psi = problem_kwargs['ini_psi']
    ini_t = problem_kwargs['ini_t']
    max_t = problem_kwargs['max_t']
    update_fun = problem_kwargs['update_fun']
    calculate_fun = problem_kwargs['calculate_fun']
    rtol = problem_kwargs['rtol']
    atol = problem_kwargs['atol']
    eval_dt = problem_kwargs['eval_dt']
    table_name = problem_kwargs['table_name']
    omega_tail = problem_kwargs['omega_tail']
    flow_strength = problem_kwargs['flow_strength']
    save_every = problem_kwargs['save_every']
    t_name = problem_kwargs['t_name']

    expt_str = ''
    expt_str = expt_str + 'table_name: %s \n' % table_name
    expt_str = expt_str + 'omega_tail: %f \n' % omega_tail
    expt_str = expt_str + 'flow_strength: %f \n' % flow_strength
    expt_str = expt_str + 'init normal angle: %f, %f, %f \n' % \
               (ini_theta, ini_phi, ini_psi)
    expt_str = expt_str + 'last normal angle: %f, %f, %f \n' % \
               (Table_theta[-1], Table_phi[-1], Table_psi[-1])
    expt_str = expt_str + '%s: ini_t=%f, max_t=%f, n_t=%d \n' % \
               (calculate_fun, ini_t, max_t, Table_t.size)
    expt_str = expt_str + '%s rt%.0e, at%.0e, %.1fs \n' % \
               (update_fun, rtol, atol, simulate_t)

    save_list = ('ini_theta', 'ini_phi', 'ini_psi', 'ini_t', 'max_t', 'eval_dt', 'update_fun',
                 'rtol', 'atol', 'table_name', 'omega_tail', 'flow_strength', 't_name',
                 'save_every', 'simulate_t', 'Table_t', 'Table_dt',
                 'Table_X', 'Table_P', 'Table_P2',
                 'Table_theta', 'Table_phi', 'Table_psi', 'Table_eta')
    t_pick = {}
    for var_name in save_list:
        t_pick[var_name] = locals()[var_name]
    t_pick['problem_kwargs'] = problem_kwargs
    with open('%s.pickle' % t_name, 'wb') as handle:
        pickle.dump(t_pick, handle, protocol=pickle.HIGHEST_PROTOCOL)
    expt_str = expt_str + 'save to %s \n' % t_name

    # spf_tb.save_table_result('%s.jpg' % t_name, Table_t, Table_dt, Table_X, Table_P, Table_P2,
    #                          Table_theta, Table_phi, Table_psi, Table_eta, save_every)
    spf_tb.save_table_result_v2('%s.jpg' % t_name, Table_t, Table_dt, Table_X, Table_P, Table_P2,
                                Table_theta, Table_phi, Table_psi, Table_eta, save_every, dpi=200)
    return expt_str


def main_fun(**main_kwargs):
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    ini_theta = problem_kwargs['ini_theta']
    ini_phi = problem_kwargs['ini_phi']
    ini_psi = problem_kwargs['ini_psi']
    ini_t = problem_kwargs['ini_t']
    max_t = problem_kwargs['max_t']
    update_fun = problem_kwargs['update_fun']
    calculate_fun = problem_kwargs['calculate_fun']
    rtol = problem_kwargs['rtol']
    atol = problem_kwargs['atol']
    eval_dt = problem_kwargs['eval_dt']
    table_name = problem_kwargs['table_name']
    omega_tail = problem_kwargs['omega_tail']
    problem_kwargs['save_every'] = 1
    save_every = problem_kwargs['save_every']
    idx_time = datetime.today().strftime('D%Y%m%d_T%H%M%S')
    fileHandle = problem_kwargs['fileHandle']
    fileHandle = fileHandle + '_' if len(fileHandle) > 0 else ''
    t_name = '%sth%5.3f_ph%5.3f_ps%5.3f_%s' % (fileHandle, ini_theta, ini_phi, ini_psi, idx_time)
    problem_kwargs['t_name'] = t_name

    t0 = time()
    tnorm = np.array((np.sin(ini_theta) * np.cos(ini_phi), np.sin(ini_theta) * np.sin(ini_phi),
                      np.cos(ini_theta)))
    Table_t, Table_dt, Table_X, Table_P, Table_P2, Table_theta, Table_phi, Table_psi, Table_eta = \
        calculate_fun(tnorm, ini_psi, max_t, update_fun=update_fun, rtol=rtol, atol=atol,
                      eval_dt=eval_dt, ini_t=ini_t, table_name=table_name, save_every=save_every,
                      tqdm_fun=tqdm, omega_tail=omega_tail)
    t1 = time()
    simulate_t = t1 - t0
    expt_str = do_pickel(Table_t, Table_dt, Table_X, Table_P, Table_P2, Table_theta, Table_phi,
                         Table_psi, Table_eta, simulate_t, **problem_kwargs)

    print(expt_str)
    with open('%s.txt' % t_name, 'w') as text_file:
        text_file.write(expt_str)
    return True


def main_fun_base_flow(**main_kwargs):
    OptDB = PETSc.Options()
    assert OptDB.getString('calculate_fun') in ('do_ShearFlowPetsc4nPsiObj',
                                                'do_ShearFlowPetsc4nPsiObj_dbg',)

    problem_kwargs = get_problem_kwargs(**main_kwargs)
    ini_theta = problem_kwargs['ini_theta']
    ini_phi = problem_kwargs['ini_phi']
    ini_psi = problem_kwargs['ini_psi']
    ini_t = problem_kwargs['ini_t']
    max_t = problem_kwargs['max_t']
    update_fun = problem_kwargs['update_fun']
    rtol = problem_kwargs['rtol']
    atol = problem_kwargs['atol']
    eval_dt = problem_kwargs['eval_dt']
    table_name = problem_kwargs['table_name']
    omega_tail = problem_kwargs['omega_tail']
    flow_strength = problem_kwargs['flow_strength']
    problem_kwargs['save_every'] = 1
    save_every = problem_kwargs['save_every']
    idx_time = datetime.today().strftime('D%Y%m%d_T%H%M%S')
    fileHandle = problem_kwargs['fileHandle']
    fileHandle = fileHandle + '_' if len(fileHandle) > 0 else ''
    t_name = '%sth%5.3f_ph%5.3f_ps%5.3f_%s' % (fileHandle, ini_theta, ini_phi, ini_psi, idx_time)
    problem_kwargs['t_name'] = t_name
    calculate_fun = problem_kwargs['calculate_fun']

    t0 = time()
    tnorm = np.array((np.sin(ini_theta) * np.cos(ini_phi), np.sin(ini_theta) * np.sin(ini_phi),
                      np.cos(ini_theta)))
    Table_t, Table_dt, Table_X, Table_P, Table_P2, Table_theta, Table_phi, Table_psi, Table_eta = \
        calculate_fun(tnorm, ini_psi, max_t, update_fun=update_fun, rtol=rtol, atol=atol,
                      eval_dt=eval_dt, ini_t=ini_t, table_name=table_name, save_every=save_every,
                      tqdm_fun=tqdm, omega_tail=omega_tail, flow_strength=flow_strength,
                      return_psi_body=False)
    t1 = time()
    simulate_t = t1 - t0
    expt_str = do_pickel(Table_t, Table_dt, Table_X, Table_P, Table_P2, Table_theta, Table_phi,
                         Table_psi, Table_eta, simulate_t, **problem_kwargs)

    print(expt_str)
    with open('%s.txt' % t_name, 'w') as text_file:
        text_file.write(expt_str)
    return True


if __name__ == '__main__':
    OptDB = PETSc.Options()
    if OptDB.getString('calculate_fun') in ('do_ShearFlowPetsc4nPsiObj',
                                            'do_ShearFlowPetsc4nPsiObj_dbg',):
        OptDB.setValue('main_fun', False)
        main_fun_base_flow()

    if OptDB.getBool('main_fun', True):
        main_fun()

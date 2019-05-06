import numpy as np
import pickle
from time import time
from codeStore import support_fun_table as spf_tb
from petsc4py import PETSc

# get kewrgs
OptDB = PETSc.Options()


# active ecoli petsc family method 
t0 = time()
t_theta, t_phi, t_psi = 0, 0, 0
max_t = 100
update_fun = '3'
rtol = 1e-6
atol = 1e-9
eval_dt = 0.00001
save_every = np.ceil(1 / eval_dt) / 100

tnorm = np.array((np.sin(t_theta) * np.cos(t_phi), np.sin(t_theta) * np.sin(t_phi), np.cos(t_theta)))
Table_t, Table_X, Table_P, Table_P2, Table_theta, Table_phi, Table_psi, Table_eta = \
    spf_tb.do_calculate_ecoli_Petsc(tnorm, t_psi, max_t, update_fun=update_fun,
                                    rtol=rtol, atol=atol, eval_dt=eval_dt,
                                    save_every=save_every)
t1 = time()
print('ini angles: ', t_theta, t_phi, t_psi)
print('lst angles: ', Table_theta[-1], ',', Table_phi[-1], ',', Table_psi[-1])
print('%s: run %d loops/times using %fs' % ('do_calculate_ecoli_Petsc', max_t, (t1 - t0)))
print('%s_%s rt%.0e, at%.0e, dt%.0e %.1fs' %
      ('PETSC RK', update_fun, rtol, atol, eval_dt, (t1 - t0)))

t_pick = (t_theta, t_phi, t_psi, max_t, update_fun, rtol, atol, eval_dt,
          Table_t, Table_X, Table_P, Table_P2, Table_theta, Table_phi, Table_psi, Table_eta)
idx = np.load('../motion_ecoliB01_table/idx.npy')
t_name = 'idx%03d_th%5.3f_ph%5.3f_ps%5.3f.pickle' % (idx, t_theta, t_phi, t_psi)
np.save('../motion_ecoliB01_table/idx.npy', (idx + 1))
with open('../motion_ecoliB01_table/%s' % t_name, 'wb') as handle:
    pickle.dump(t_pick, handle, protocol=pickle.HIGHEST_PROTOCOL)
print('save to %s' % t_name)
spf_tb.save_table_result(t_name, Table_t,
                         Table_theta, Table_phi, Table_psi, Table_eta,
                         Table_X, save_every)

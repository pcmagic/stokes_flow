import numpy as np
from codeStore import support_fun_table as spf_tb
from codeStore import support_fun_baseflow as spf_bf
from src import jeffery_model as jm
from tqdm import tqdm


# t_theta, t_phi, t_psi = 0, 0, 0
# max_t = 10
# update_fun = '5bs'
# rtol = 1e-6
# atol = 1e-9
# eval_dt = 1
# save_every = 1
# table_name = 'ecoC01B05_tao0_wm1'
# omega_tail = 1
#
# norm = np.array((np.sin(t_theta) * np.cos(t_phi), np.sin(t_theta) * np.sin(t_phi), np.cos(t_theta)))
# ini_psi = np.ones(1) * t_psi
# Table_t, Table_dt, Table_X, Table_P, Table_P2, Table_theta, Table_phi, Table_psi, Table_eta \
#     = spf_tb.do_calculate_ecoli_Petsc4nPsi(norm, ini_psi, max_t, update_fun=update_fun,
#                                            rtol=rtol, atol=atol, tqdm_fun=tqdm,
#                                            eval_dt=eval_dt, save_every=save_every,
#                                            table_name=table_name, omega_tail=omega_tail)
# print(1)


# t_theta, t_phi, t_psi = np.zeros(3)
t_theta, t_phi, t_psi = np.random.sample(3) * np.pi * (1, 2, 2)
max_t = 10
update_fun = '5bs'
rtol = 1e-6
atol = 1e-9
eval_dt = 0.0001
save_every = 1
omega_tail = 0
flow_strength = 1
# table_name2 = 'ecoC01B05_baseFlow'
table_name2 = 'ellipsoidB05_baseFlow'

figsize = np.array((16, 9)) * 2
dpi = 100
kwargs = {'ABC_A': 1,
          'ABC_B': 1,
          'ABC_C': 1,
          'name':  'GivenFlowProblem', }
ini_center = np.zeros(3)
problemHandle = jm.ABCFlowProblem

# print(t_theta, t_phi, t_psi)
d2 = spf_bf.do_GivenFlowObj(t_theta, t_phi, t_psi, max_t, table_name=table_name2,
                            update_fun=update_fun, rtol=rtol, atol=atol, eval_dt=eval_dt,
                            save_every=save_every, tqdm_fun=tqdm, return_psi_body=False,
                            omega_tail=omega_tail, ini_center=ini_center,
                            problemHandle=problemHandle, **kwargs)
base_t, base_dt, base_X, base_thphps, base_U, base_W, base_psi_t = d2

print('init \\theta=%.3f, \\phi=%.3f, \\psi=%.3f, ' % (t_theta, t_phi, t_psi))
print('last \\theta=%.3f, \\phi=%.3f, \\psi=%.3f, '
      % (base_thphps[-1, 0], base_thphps[-1, 1], base_thphps[-1, 2]))

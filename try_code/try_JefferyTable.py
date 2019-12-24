import numpy as np
from codeStore import support_fun_table as spf_tb

t_theta, t_phi, t_psi = 0, 0, 0
max_t = 10
update_fun = '5bs'
rtol = 1e-6
atol = 1e-9
eval_dt = 1
save_every = 1
table_name = 'ecoC01B05_tao0_wm1'
omega_tail = 1

norm = np.array((np.sin(t_theta) * np.cos(t_phi), np.sin(t_theta) * np.sin(t_phi), np.cos(t_theta)))
ini_psi = np.ones(1) * t_psi
Table_t, Table_dt, Table_X, Table_P, Table_P2, Table_theta, Table_phi, Table_psi, Table_eta \
    = spf_tb.do_calculate_ecoli_Petsc4nPsi(norm, ini_psi, max_t, update_fun=update_fun,
                                           rtol=rtol, atol=atol,
                                           eval_dt=eval_dt, save_every=save_every,
                                           table_name=table_name, omega_tail=omega_tail)
print(1)

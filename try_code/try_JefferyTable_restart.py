import numpy as np
from codeStore import support_fun_table as spf_tb
from src import jeffery_model as jm
from tqdm import tqdm

max_t = 1
update_fun = '1fe'
update_fun = '5bs'
rtol = 1e-6
atol = 1e-9
save_every = 1
norm = np.array((0, 0, 1))
ini_psi = 0

table_name = 'hlxB01_tau1a'
P0, P20, tcenter, problem = spf_tb.do_calculate_prepare(norm)
helix_kwargs = spf_tb.do_helix_kwargs(tcenter, P0, P20, ini_psi, table_name=table_name)
helix_obj = jm.TablePetsc4nEcoli(**helix_kwargs)
obj = helix_obj
obj.set_update_para(fix_x=False, fix_y=False, fix_z=False, update_fun=update_fun,
                    rtol=rtol, atol=atol, save_every=save_every, tqdm_fun=tqdm)
problem.add_obj(obj)

# update 1
t0 = 0
t1 = 10
eval_dt = 2
Table_t, Table_dt, Table_X, Table_P, Table_P2 = obj.update_self(t0=t0, t1=t1, eval_dt=eval_dt)
print(Table_t)

# update 2
t0 = 10
t1 = 20
eval_dt = 2
Table_t, Table_dt, Table_X, Table_P, Table_P2 = obj.update_self(t0=t0, t1=t1, eval_dt=eval_dt)
print(Table_t)

# update 3
t0 = 20
t1 = 30
eval_dt = 2
Table_t, Table_dt, Table_X, Table_P, Table_P2 = obj.update_self(t0=t0, t1=t1, eval_dt=eval_dt)
print(Table_t)

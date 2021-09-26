import sys
import petsc4py

petsc4py.init(sys.argv)

from petsc4py import PETSc
from tqdm import tqdm
from codeStore import support_fun_baseflow as spf_bf


def new_dir_fun_se(**case_kwargs):
    t_dir = case_kwargs['t_dir']
    save_every = case_kwargs['save_every']
    new_dir = '%s_se%d' % (t_dir, save_every)
    return new_dir


case_para = [{'t_dir':       'ellipsoidB05/A1.00_B1.00_C1.00_rand_a',
              'save_every':  1000,
              'new_dir_fun': new_dir_fun_se,
              't_headle':    '(.*?).pickle',
              'n_load':      None,
              'rand_mode':   False,
              'tqdm_fun':    tqdm}, ]

if __name__ == '__main__':
    OptDB = PETSc.Options()
    case_idx = OptDB.getInt('case_idx', 0)
    n_load = OptDB.getInt('n_load', -1)
    n_load = None if n_load < 0 else n_load
    rand_mode = OptDB.getBool('rand_mode', False)
    save_every = OptDB.getInt('save_every', 1000)
    t_dir = OptDB.getString('t_dir', '')

    case_kwargs = case_para[case_idx]
    case_kwargs['n_load'] = n_load
    case_kwargs['rand_mode'] = rand_mode
    case_kwargs['t_dir'] = t_dir
    case_kwargs['save_every'] = save_every
    case_kwargs['new_dir'] = case_kwargs['new_dir_fun'](**case_kwargs)
    case_kwargs['t_headle'] = '(.*?).pickle'
    PETSc.Sys.Print(case_kwargs)
    PETSc.Sys.Print()
    spf_bf.every_pickle_dir(**case_kwargs)

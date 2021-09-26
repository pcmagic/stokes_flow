import sys
import matplotlib
import petsc4py

# matplotlib.use('agg')
petsc4py.init(sys.argv)

import numpy as np
import pickle
import time
# from codeStore import support_fun_table as spf_tb
from codeStore import support_fun_baseflow as spf_bf
from petsc4py import PETSc
from datetime import datetime
from tqdm import tqdm
from src.baseflow_io import *
from src import jeffery_model as jm


# get kwargs
def get_problem_kwargs(**main_kwargs):
    problem_kwargs = get_baseflow_kwargs()

    kwargs_list = (main_kwargs,)
    for t_kwargs in kwargs_list:
        for key in t_kwargs:
            problem_kwargs[key] = t_kwargs[key]
    return problem_kwargs

def print_case_info(**problem_kwargs):
    print_baseflow_kwargs(**problem_kwargs)
    return True

def main_fun(**main_kwargs):
    t0 = time.time()
    OptDB = PETSc.Options()
    fileHandle = OptDB.getString('f', 'do_baseflow')
    OptDB.setValue('f', fileHandle)
    main_kwargs['fileHandle'] = fileHandle
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    problem_kwargs['tqdm_fun'] = tqdm
    print_case_info(**problem_kwargs)

    data = spf_bf.do_GivenFlowObj(**problem_kwargs)
    # base_t, base_dt, base_X, base_thphps, base_U, base_W, base_psi_t = data
    spf_bf.pick_problem(data, **problem_kwargs)
    # print(matplotlib.get_backend())
    # matplotlib.use('pdf')
    spf_bf.save_fig_result_v2(data, **problem_kwargs)
    print_finish_solve(t0, data, **problem_kwargs)


    return True

if __name__ == '__main__':
    main_fun()

import sys

import petsc4py

petsc4py.init(sys.argv)
import numpy as np
# from time import time
# from scipy.io import loadmat
# from src.stokes_flow import problem_dic, obj_dic
from src.geo import *
from petsc4py import PETSc
from src import stokes_flow as sf
from codeStore.ecoli_common import *
from ecoli_in_pipe import do_slenderbodytheory as do_SLB

OptDB = PETSc.Options()
ph = 10
ch = 6
rt1 = 1
rt2 = 0.1
n_segment_fct = OptDB.getInt('n_segment_fct', 10)
n_segment = ch * n_segment_fct
slb_epsabs = 1e-200
slb_epsrel = 1e-10
slb_limit = 100
n_hlx = 2
neighbor_range = OptDB.getInt('neighbor_range', 1)

# At, Bt, Ct, ftr_info, frt_info = do_SLB.do_Lightill_nhelix(ph, rt1, rt2, ch, n_segment)
At, Bt, Ct, ftr_info, frt_info = \
    do_SLB.do_mod_KRJ_nhelix(ph, rt1, rt2, ch, n_segment, n_hlx=n_hlx, slb_epsabs=slb_epsabs,
                             slb_epsrel=slb_epsrel, slb_limit=slb_limit,
                             neighbor_range=neighbor_range, dbg_Lsbt=False)
# At, Bt, Ct, ftr_info, frt_info = \
#     do_SLB.do_KRJ_nhelix(ph, rt1, rt2, ch, n_segment, n_hlx=n_hlx, slb_epsabs=slb_epsabs,
#                          slb_epsrel=slb_epsrel, slb_limit=slb_limit, dbg_Lsbt=True)
# print(At, Bt, Ct)

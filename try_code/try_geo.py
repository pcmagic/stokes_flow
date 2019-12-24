# coding=utf-8
import sys
import petsc4py

petsc4py.init(sys.argv)
from petsc4py import PETSc
from src.objComposite import *
import ecoli_in_pipe.ecoli_common as ec
from src.geo import *
import numpy as np
from src import stokes_flow as sf

# OptDB = PETSc.Options()
# R1 = OptDB.getReal('R1', 1)
# R2 = OptDB.getReal('R2', 1)
# ph = OptDB.getReal('ph', 1)
# B = ph / (2 * np.pi)
# vhgeo = FatHelix()
# t1 = np.random.sample(4) * np.random.randint(0, 100, 4)
# vhgeo.dbg_frame_left_hand(*t1)
# dth = 2 * np.pi / 6
# fhgeo = vhgeo.create_deltatheta(dth=dth, radius=0.1, R1=R1, R2=R2, B=B, n_c=3, epsilon=0,
#                                 with_cover=1, factor=0.1)
# vhgeo.show_nodes()

# problem_kwargs = ec.get_problem_kwargs()
# head_obj, tail_obj_list1, tail_obj_list2 = createEcoli_2tails(name='ecoli0', **problem_kwargs)
# head_obj.set_name('head_obj')
# tail_obj = sf.StokesFlowObj()
# tail_obj.set_name('tail_obj')
# tail_obj.combine((tail_obj_list1, tail_obj_list2))
# head_geo = head_obj.get_u_geo()
# ecoli_comp = sf.ForceFreeComposite(center=np.zeros(3), norm=head_geo.get_geo_norm(),
#                                    name='ecoli_0')
# ecoli_comp.add_obj(obj=head_obj, rel_U=np.zeros(6))
# ecoli_comp.add_obj(obj=tail_obj, rel_U=np.zeros(6))
# ecoli_comp.show_u_nodes()

OptDB = PETSc.Options()
theta = OptDB.getReal('theta', 0)
phi = OptDB.getReal('phi', 0)
psi1 = OptDB.getReal('psi1', 0)
psi2 = OptDB.getReal('psi2', 0)
problem_kwargs = ec.get_problem_kwargs()
ecoli_comp = create_ecoli_dualTail_at(theta, phi, psi1, psi2, **problem_kwargs)
ecoli_comp.show_u_nodes()

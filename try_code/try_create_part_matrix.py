import sys

import petsc4py

petsc4py.init(sys.argv)

import numpy as np
import pickle
# from time import time
# from scipy.io import loadmat
# from src.stokes_flow import problem_dic, obj_dic
# from src.geo import *
from petsc4py import PETSc
from src import stokes_flow as sf
from src.myio import *
from src.objComposite import createEcoliComp_tunnel
from src.myvtk import save_singleEcoli_vtk

fileHeadle = 'singleEcoliPro'
rigid_velocity = [2.88870196e-08, 1.25399086e-08, -5.91414513e-03, 7.34158974e-09, 2.73837520e-08, 4.32433478e-01]
with open(fileHeadle + '_pick.bin', 'rb') as input:
    unpick = pickle.Unpickler(input)
    problem = unpick.load()
    problem.unpickmyself()
kwargs = problem.get_kwargs()

newProb = sf.stokesFlowProblem(**kwargs)
t_obj_list = problem.get_obj_list()[0].get_obj_list()[1:]
new_obj_list = [t_obj.copy() for t_obj in t_obj_list]
for new_obj in new_obj_list:
    new_obj.set_rigid_velocity(rigid_velocity)
    newProb.add_obj(new_obj)
newProb.show_velocity(length_factor=0.01)
newProb.create_F_U()
new_M = newProb.create_empty_M()
for t_obj, new_obj in zip(t_obj_list, new_obj_list):
    problem.create_part_matrix(t_obj, t_obj, new_obj, new_obj, new_M)
newProb.solve()
t_force = newProb.get_total_force()
PETSc.Sys.Print('---->>>tail resultant is', t_force / 6 / np.pi)

rigid_velocity = [2.88870196e-08, 1.25399086e-08, -5.91414513e-03, 7.34158974e-09, 2.73837520e-08, -5.67566522e-01]
newProb = sf.stokesFlowProblem(**kwargs)
t_obj_list = [problem.get_obj_list()[0].get_obj_list()[0], ]
new_obj_list = [t_obj.copy() for t_obj in t_obj_list]
for new_obj in new_obj_list:
    new_obj.set_rigid_velocity(rigid_velocity)
    newProb.add_obj(new_obj)
newProb.create_F_U()
new_M = newProb.create_empty_M()
for t_obj, new_obj in zip(t_obj_list, new_obj_list):
    problem.create_part_matrix(t_obj, t_obj, new_obj, new_obj, new_M)
newProb.solve()
newProb.show_velocity(length_factor=0.01)
t_force = newProb.get_total_force()
PETSc.Sys.Print('---->>>head resultant is', t_force / 6 / np.pi)

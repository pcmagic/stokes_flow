"""
the figure of speedup ratio of r_t/R=0.8 case for eq02 geometry using pure numerical result and numerical+theoretical methods is different.
this code load M matrix for this two methods, and check the difference.
"""
import sys

import petsc4py

petsc4py.init(sys.argv)

from petsc4py import PETSc
import pickle
from src import stokes_flow as sf
from src.myio import *
from src.objComposite import createEcoliComp_tunnel

def main_fun(**main_kwargs):
    def save_mat_light(fileHeadle):
        with open(fileHeadle + '_pick.bin', 'rb') as input_bin:
            unpick = pickle.Unpickler(input_bin)
            problem = unpick.load()
            problem.unpickmyself()
            problem.saveM_mat(fileHeadle + '_M')
            problem.destroy()
            for obj in problem.get_obj_list():
                obj.save_mat()

    OptDB = PETSc.Options()
    pure_num_headle = OptDB.getString('pure_num_headle', 'eq02_016_0800')
    num_the_headle = OptDB.getString('num_the_headle', 'eq02_num_016_0800')

    save_mat_light(pure_num_headle)
    save_mat_light(num_the_headle)


if __name__ == '__main__':
    main_fun()
from tqdm.notebook import tqdm as tqdm_notebook
import os
import glob
import pickle
import numpy as np


# load the resistance matrix form dir, standard version
def load_ABC_list(job_dir):
    t_dir = os.path.join(job_dir, '*.pickle')
    pickle_names = glob.glob(t_dir)
    problem_kwarg_list = []
    A_list = []
    B1_list = []
    B2_list = []
    C_list = []

    for pickle_name in pickle_names:
        with open(pickle_name, 'rb') as myinput:
            problem_kwargs, A, B1, B2, C, = pickle.load(myinput)[:5]
        problem_kwarg_list.append(problem_kwargs)
        A_list.append(A)
        B1_list.append(B1)
        B2_list.append(B2)
        C_list.append(C)

    A_list = np.array(A_list)
    B1_list = np.array(B1_list)
    B2_list = np.array(B2_list)
    C_list = np.array(C_list)
    problem_kwarg_list = np.array(problem_kwarg_list)
    return problem_kwarg_list, A_list, B1_list, B2_list, C_list
#
#
# # load (u_i^{Ej}, \omega_i^{Ej}) and (u_i^a, \omega_i^a), standard version.
# #   see the method of base flow for detail
# def load_MBF(pickle_name):)
#
# # load (u_i^{Ej}, \omega_i^{Ej}) and (u_i^a, \omega_i^a) from dir, standard version.
# #   see the method of base flow for detail
# def load_MBF_list(job_dir):
#     t_dir = os.path.join(job_dir, '*.pickle')
#     pickle_names = glob.glob(t_dir)
#     A_list = []
#     B1_list = []
#     B2_list = []
#     C_list = []
#
#     for pickle_name in pickle_names:
#         with open(pickle_name, 'rb') as myinput:
#             problem_kwargs, A, B1, B2, C, = pickle.load(myinput)
#         problem_kwarg_list.append(problem_kwargs)
#         A_list.append(A)
#         B1_list.append(B1)
#         B2_list.append(B2)
#         C_list.append(C)

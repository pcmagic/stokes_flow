import sys
from os import path as ospath
t_path = sys.path[0]
# t_path = ospath.join(ospath.dirname(t_path), 'src')
t_path = ospath.dirname(t_path)
if ospath.isdir(t_path):
    sys.path = [t_path] + sys.path
else:
    err_msg = "can not add path father path"
    raise ValueError(err_msg)

from src.stokes_flow import problem_dic


def main_fun(**main_kwargs):
    for mypath in sys.path:
        # PETSc.Sys.Print(mypath)
        if 'stokes' in mypath:
            PETSc.Sys.Print(mypath)

if __name__ == '__main__':
    main_fun()

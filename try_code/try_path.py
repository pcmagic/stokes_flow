import sys

from src.stokes_flow import problem_dic


def main_fun(**main_kwargs):
    for mypath in sys.path:
        # PETSc.Sys.Print(mypath)
        if 'stokes' in mypath:
            PETSc.Sys.Print(mypath)

if __name__ == '__main__':
    main_fun()

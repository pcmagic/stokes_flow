from time import time
from petsc4py import PETSc
from src import stokes_flow as sf
from src.objComposite import createEcoli_tunnel
from src.stokes_flow import obj_dic

__all__ = ['save_singleEcoli_vtk', ]


def save_singleEcoli_vtk(problem: sf.stokesFlowProblem):
    t0 = time( )
    problem_kwargs = problem.get_kwargs( )
    fileHeadle = problem_kwargs['fileHeadle']
    matrix_method = problem_kwargs['matrix_method']
    with_T_geo = problem_kwargs['with_T_geo']

    problem.vtk_obj(fileHeadle)

    # bgeo = geo()
    # bnodesHeadle = problem_kwargs['bnodesHeadle']
    # matname = problem_kwargs['matname']
    # bgeo.mat_nodes(filename=matname, mat_handle=bnodesHeadle)
    # belemsHeadle = problem_kwargs['belemsHeadle']
    # bgeo.mat_elmes(filename=matname, mat_handle=belemsHeadle, elemtype='tetra')
    # problem.vtk_tetra(fileHeadle + '_Velocity', bgeo)

    # create check obj
    check_kwargs = problem_kwargs.copy( )
    check_kwargs['nth'] = problem_kwargs['nth'] - 2 if problem_kwargs['nth'] >= 6 else problem_kwargs['nth'] + 1
    check_kwargs['ds'] = problem_kwargs['ds'] * 1.2
    check_kwargs['hfct'] = 1
    check_kwargs['Tfct'] = 1
    objtype = obj_dic[matrix_method]
    ecoli_comp = problem.get_obj_list( )[0]
    ecoli_comp_check = createEcoli_tunnel(objtype, **check_kwargs)
    ecoli_comp_check.set_ref_U(ecoli_comp.get_ref_U( ))
    if with_T_geo:
        velocity_err_sphere, velocity_err_helix0, velocity_err_helix1, velocity_err_Tgeo = \
            problem.vtk_check(fileHeadle, ecoli_comp_check)
        PETSc.Sys.Print('velocity error of sphere (total, x, y, z): ', velocity_err_sphere)
        PETSc.Sys.Print('velocity error of helix0 (total, x, y, z): ', velocity_err_helix0)
        PETSc.Sys.Print('velocity error of helix1 (total, x, y, z): ', velocity_err_helix1)
        PETSc.Sys.Print('velocity error of Tgeo (total, x, y, z): ', velocity_err_Tgeo)
    else:
        velocity_err_sphere, velocity_err_helix0, velocity_err_helix1 = \
            problem.vtk_check(fileHeadle, ecoli_comp_check)
        PETSc.Sys.Print('velocity error of sphere (total, x, y, z): ', velocity_err_sphere)
        PETSc.Sys.Print('velocity error of helix0 (total, x, y, z): ', velocity_err_helix0)
        PETSc.Sys.Print('velocity error of helix1 (total, x, y, z): ', velocity_err_helix1)

    t1 = time( )
    PETSc.Sys.Print('%s: write vtk files use: %fs' % (str(problem), (t1 - t0)))
    return True

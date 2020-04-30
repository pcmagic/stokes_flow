import sys
import petsc4py

petsc4py.init(sys.argv)
import numpy as np
# from time import time
# from scipy.io import loadmat
# from src.stokes_flow import problem_dic, obj_dic
from petsc4py import PETSc
from src.geo import *
from src import stokes_flow as sf
from src import slender_body as slb
# from src.support_class import *
# from codeStore.helix_common import *
from src.myio import *
from codeStore.helix_common import *
import pickle


def get_problem_kwargs(ph, ch, rt1, rt2, n_segment, matrix_method, fileHandle='', slb_epsabs=1e-200,
                       slb_epsrel=1e-08, slb_limit=10000, neighbor_range=1, n_hlx=1,
                       slb_geo_fun=slb_helix, **kwargs):
    OptDB = PETSc.Options()
    OptDB.setValue('ksp_max_it', 100)
    comm = PETSc.COMM_WORLD.tompi4py()
    MPISIZE = comm.Get_size()

    problem_kwargs = {'rh1':                   rt1,
                      'rh11':                  rt1,
                      'rh12':                  rt1,
                      'rh2':                   rt2,
                      'ph':                    ph,
                      'ch':                    ch,
                      'n_tail':                n_hlx,
                      'n_segment':             n_segment,
                      'slb_geo_fun':           slb_geo_fun,
                      'nth':                   0,
                      'eh':                    0,
                      'repeat_n':              0,
                      'hfct':                  0,
                      'with_cover':            0,
                      'left_hand':             0,
                      'rel_Uh':                0,
                      'zoom_factor':           0,
                      'matrix_method':         matrix_method,
                      'fileHandle':            fileHandle,
                      'slb_epsabs':            slb_epsabs,
                      'slb_epsrel':            slb_epsrel,
                      'slb_limit':             slb_limit,
                      'neighbor_range':        neighbor_range,
                      'solve_method':          'gmres',
                      'getConvergenceHistory': False,
                      'plot_geo':              False,
                      'precondition_method':   'none',
                      'MPISIZE':               MPISIZE,
                      'pickProblem':           False, }
    for key in kwargs:
        problem_kwargs[key] = kwargs[key]
    return problem_kwargs


def print_case_info(caseIntro='-->(some introduce here)', **problem_kwargs):
    fileHandle = problem_kwargs['fileHandle']
    PETSc.Sys.Print(caseIntro)
    print_solver_info(**problem_kwargs)
    print_helix_info(fileHandle, **problem_kwargs)
    n_segment = problem_kwargs['n_segment']
    slb_geo_fun = problem_kwargs['slb_geo_fun']
    slb_epsabs = problem_kwargs['slb_epsabs']
    slb_epsrel = problem_kwargs['slb_epsrel']
    slb_limit = problem_kwargs['slb_limit']
    PETSc.Sys.Print('  %s, n_segment %d' %
                    (slb_geo_fun, n_segment))
    PETSc.Sys.Print('  slb_epsabs %e, n_segment %e, n_segment %d' %
                    (slb_epsabs, slb_epsrel, slb_limit))
    return True


def do_KRJ_1helix(ph, rt1, rt2, ch, n_segment, slb_epsabs=1e-200, slb_epsrel=1e-08,
                  slb_limit=10000, ):
    matrix_method = 'KRJ_slb'
    fileHandle = 'KRJ_1helix'
    # rt2_fct, slb_geo_fun = 1, slb_helix
    rt2_fct, slb_geo_fun = 1, Johnson_helix
    # rt2_fct, slb_geo_fun = 4 / np.pi, slb_helix  # This factor following Rodenborn2013
    # rt2_fct, slb_geo_fun = 4 / np.pi, Johnson_helix  # This factor following Rodenborn2013
    # rt2_fct, slb_geo_fun = 1, expJohnson_helix
    rt2 = rt2 * rt2_fct
    problem_kwargs = get_problem_kwargs(ph=ph, ch=ch, rt1=rt1, rt2=rt2, n_segment=n_segment,
                                        matrix_method=matrix_method, fileHandle=fileHandle,
                                        slb_epsabs=slb_epsabs, slb_epsrel=slb_epsrel,
                                        slb_limit=slb_limit, slb_geo_fun=slb_geo_fun, )
    hlx_geo = slb_geo_fun(ph, ch, rt1, rt2)
    hlx_geo.create_nSegment(n_segment, check_nth=problem_kwargs['matrix_method'] == 'lightill_slb')
    hlx_obj = sf.StokesFlowObj()
    hlx_obj.set_data(hlx_geo, hlx_geo, name='helix1')
    problem = slb.problem_dic[matrix_method](**problem_kwargs)
    problem.add_obj(hlx_obj)
    problem.print_info()
    problem.create_matrix()
    At, Bt, Ct, ftr_info, frt_info = AtBtCt(problem)
    # print(At, Bt, Ct)
    return At, Bt, Ct, ftr_info, frt_info


def do_KRJ_nhelix(ph, rt1, rt2, ch, n_segment, n_hlx=1, slb_epsabs=1e-200, slb_epsrel=1e-08,
                  slb_limit=10000, fileHandle='KRJ_slb', slb_geo_fun=slb_helix, **kwargs):
    matrix_method = 'KRJ_slb'
    # # rt2_fct, slb_geo_fun = 1, slb_helix
    # rt2_fct, slb_geo_fun = 1, Johnson_helix
    # # rt2_fct, slb_geo_fun = 4 / np.pi, slb_helix  # This factor following Rodenborn2013
    # # rt2_fct, slb_geo_fun = 4 / np.pi, Johnson_helix  # This factor following Rodenborn2013
    # # rt2_fct, slb_geo_fun = 1, expJohnson_helix
    # rt2 = rt2 * rt2_fct
    problem_kwargs = get_problem_kwargs(ph=ph, ch=ch, rt1=rt1, rt2=rt2, n_segment=n_segment,
                                        matrix_method=matrix_method, fileHandle=fileHandle,
                                        slb_epsabs=slb_epsabs, slb_epsrel=slb_epsrel,
                                        slb_limit=slb_limit, n_hlx=n_hlx, slb_geo_fun=slb_geo_fun,
                                        **kwargs)
    print_case_info('do_KRJ_nhelix', **problem_kwargs)
    problem = slb.problem_dic[matrix_method](**problem_kwargs)
    check_nth = matrix_method == 'lightill_slb'
    for i0, theta0 in enumerate(np.linspace(0, 2 * np.pi, n_hlx, endpoint=False)):
        hlx1_geo = slb_geo_fun(ph, ch, rt1, rt2, theta0=theta0)
        hlx1_geo.create_nSegment(n_segment, check_nth=check_nth)
        hlx1_obj = sf.StokesFlowObj()
        obj_name = 'helix%d' % i0
        hlx1_obj.set_data(hlx1_geo, hlx1_geo, name=obj_name)
        problem.add_obj(hlx1_obj)
    problem.print_info()
    problem.create_matrix()
    At, Bt, Ct, ftr_info, frt_info = AtBtCt(problem)
    # print(At, Bt, Ct)
    return At, Bt, Ct, ftr_info, frt_info


def do_mod_KRJ_nhelix(ph, rt1, rt2, ch, n_segment, n_hlx=1, slb_epsabs=1e-200, slb_epsrel=1e-08,
                      slb_limit=10000, neighbor_range=0, **kwargs):
    matrix_method = 'mod_KRJ_slb'
    fileHandle = 'mod_KRJ_%dhelix' % n_hlx
    # rt2_fct, slb_geo_fun = 1, slb_helix
    rt2_fct, slb_geo_fun = 1, Johnson_helix
    # rt2_fct, slb_geo_fun = 4 / np.pi, slb_helix  # This factor following Rodenborn2013
    # rt2_fct, slb_geo_fun = 4 / np.pi, Johnson_helix  # This factor following Rodenborn2013
    # rt2_fct, slb_geo_fun = 1, expJohnson_helix
    rt2 = rt2 * rt2_fct
    problem_kwargs = get_problem_kwargs(ph=ph, ch=ch, rt1=rt1, rt2=rt2, n_segment=n_segment,
                                        matrix_method=matrix_method, fileHandle=fileHandle,
                                        slb_epsabs=slb_epsabs, slb_epsrel=slb_epsrel,
                                        slb_limit=slb_limit, neighbor_range=neighbor_range,
                                        slb_geo_fun=slb_geo_fun, **kwargs)
    problem = slb.problem_dic[matrix_method](**problem_kwargs)
    check_nth = matrix_method == 'lightill_slb'
    for i0, theta0 in enumerate(np.linspace(0, 2 * np.pi, n_hlx, endpoint=False)):
        hlx1_geo = slb_geo_fun(ph, ch, rt1, rt2, theta0=theta0)
        hlx1_geo.create_nSegment(n_segment, check_nth=check_nth)
        hlx1_obj = sf.StokesFlowObj()
        obj_name = 'helix%d' % i0
        hlx1_obj.set_data(hlx1_geo, hlx1_geo, name=obj_name)
        problem.add_obj(hlx1_obj)
    problem.print_info()
    problem.create_matrix()
    At, Bt, Ct, ftr_info, frt_info = AtBtCt(problem)
    # print(At, Bt, Ct)
    return At, Bt, Ct, ftr_info, frt_info


def do_Lightill_nhelix(ph, rt1, rt2, ch, n_segment, n_hlx=1, slb_epsabs=1e-200, slb_epsrel=1e-08,
                       slb_limit=10000, ):
    matrix_method = 'lightill_slb'
    fileHandle = 'Lightill_%dhelix' % n_hlx
    rt2_fct, slb_geo_fun = 1, slb_helix
    rt2 = rt2 * rt2_fct
    problem_kwargs = get_problem_kwargs(ph=ph, ch=ch, rt1=rt1, rt2=rt2, n_segment=n_segment,
                                        matrix_method=matrix_method, fileHandle=fileHandle,
                                        slb_epsabs=slb_epsabs, slb_epsrel=slb_epsrel,
                                        slb_limit=slb_limit, slb_geo_fun=slb_geo_fun, )
    problem = slb.problem_dic[matrix_method](**problem_kwargs)
    check_nth = matrix_method == 'lightill_slb'
    for i0, theta0 in enumerate(np.linspace(0, 2 * np.pi, n_hlx, endpoint=False)):
        hlx1_geo = slb_geo_fun(ph, ch, rt1, rt2, theta0=theta0)
        hlx1_geo.create_nSegment(n_segment, check_nth=check_nth)
        hlx1_obj = sf.StokesFlowObj()
        obj_name = 'helix%d' % i0
        hlx1_obj.set_data(hlx1_geo, hlx1_geo, name=obj_name)
        problem.add_obj(hlx1_obj)
    problem.print_info()
    problem.create_matrix()
    At, Bt, Ct, ftr_info, frt_info = AtBtCt(problem)
    # print(At, Bt, Ct)
    return At, Bt, Ct, ftr_info, frt_info


if __name__ == '__main__':
    OptDB = PETSc.Options()
    fileHandle = OptDB.getString('f', 'do_slenderbodytheory')
    problem_kwargs = get_helix_kwargs()
    problem_kwargs['fileHandle'] = fileHandle
    ph = problem_kwargs['ph']
    ch = problem_kwargs['ch']
    rt1 = problem_kwargs['rh11']
    rt2 = problem_kwargs['rh2']
    n_segment = OptDB.getInt('n_segment', 10)
    n_hlx = problem_kwargs['n_tail']
    slb_epsabs = OptDB.getReal('slb_epsabs', 1e-200)
    slb_epsrel = OptDB.getReal('slb_epsrel', 1e-8)
    slb_limit = OptDB.getReal('slb_limit', 10000)

    matrix_method = OptDB.getString('sm', 'rs_stokeslets')
    assert matrix_method in ('do_KRJ_nhelix', 'do_mod_KRJ_nhelix', 'do_Lightill_nhelix')
    if matrix_method == 'do_KRJ_nhelix':
        slb_fun = do_KRJ_nhelix
    slb_geo_fun = OptDB.getString('slb_geo_fun', 'slb_helix')
    assert slb_geo_fun in ('slb_helix', 'Johnson_helix')
    if slb_geo_fun == 'slb_helix':
        slb_geo_fun = slb_helix
    elif slb_geo_fun == 'Johnson_helix':
        slb_geo_fun = Johnson_helix
    At, Bt, Ct, ftr_info, frt_info = slb_fun(ph=ph, rt1=rt1, rt2=rt2, ch=ch, n_segment=n_segment,
                                             n_hlx=n_hlx, slb_geo_fun=slb_geo_fun,
                                             slb_epsabs=slb_epsabs, slb_epsrel=slb_epsrel,
                                             slb_limit=slb_limit, fileHandle=fileHandle)
    tname = '%s.pickle' % fileHandle
    with open(tname, 'wb') as handle:
        pickle.dump((At, Bt, Ct, ftr_info, frt_info), handle, protocol=pickle.HIGHEST_PROTOCOL)

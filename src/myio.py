import sys
import petsc4py

petsc4py.init(sys.argv)

import numpy as np
from petsc4py import PETSc
from src import stokes_flow as sf

__all__ = ['print_ecoli_info',
           'print_solver_info_forceFree', 'print_forceFree_info', 'print_givenForce_info',
           'print_single_ecoli_forceFree_result',
           'get_ecoli_kwargs', 'get_rod_kwargs',
           'get_vtk_tetra_kwargs',
           'get_solver_kwargs', 'get_forceFree_kwargs', 'get_givenForce_kwargs']


def print_ecoli_info(ecoName, **problem_kwargs):
    nth = problem_kwargs['nth']
    hfct = problem_kwargs['hfct']
    eh = problem_kwargs['eh']
    ch = problem_kwargs['ch']
    rh1 = problem_kwargs['rh1']
    rh2 = problem_kwargs['rh2']
    ph = problem_kwargs['ph']
    ds = problem_kwargs['ds']
    rs1 = problem_kwargs['rs1']
    rs2 = problem_kwargs['rs2']
    ls = problem_kwargs['ls']
    es = problem_kwargs['es']
    center = problem_kwargs['center']
    rel_Us = problem_kwargs['rel_Us']
    rel_Uh = problem_kwargs['rel_Uh']
    dist_hs = problem_kwargs['dist_hs']
    rT1 = problem_kwargs['rT1']
    rT2 = problem_kwargs['rT2']
    ntT = problem_kwargs['nth']
    eT = problem_kwargs['eT']
    Tfct = problem_kwargs['Tfct']
    zoom_factor = problem_kwargs['zoom_factor']

    PETSc.Sys.Print(ecoName, 'geo information: ')
    PETSc.Sys.Print('  helix radius: %f and %f, helix pitch: %f, helix cycle: %f' % (rh1, rh2, ph, ch))
    PETSc.Sys.Print('    nth, hfct and epsilon of helix are %d, %f and %f, ' % (nth, hfct, eh))
    PETSc.Sys.Print('  head radius: %f and %f, length: %f, delta length: %f, epsilon: %f' % (rs1, rs2, ls, ds, es))
    PETSc.Sys.Print('  Tgeo radius: %f and %f' % (rT1, rT2))
    PETSc.Sys.Print('    ntT, eT and Tfct of Tgeo are: %d, %f and %f' % (ntT, eT, Tfct))
    PETSc.Sys.Print('  ecoli center: %s, distance from head to tail is %f' % (str(center), dist_hs))
    PETSc.Sys.Print('  relative velocity of head and tail are %s and %s' % (str(rel_Us), str(rel_Uh)))
    PETSc.Sys.Print('  geometry zoom factor is %f' % zoom_factor)
    return True

# def print_Rod_info(RodName, **problem_kwargs):

def print_forceFree_info(**problem_kwargs):
    ffweightx = problem_kwargs['ffweightx']
    ffweighty = problem_kwargs['ffweighty']
    ffweightz = problem_kwargs['ffweightz']
    ffweightT = problem_kwargs['ffweightT']
    PETSc.Sys.Print('  force free weight of Fx, Fy, Fz, and (Tx, Ty, Tz) are %f, %f, %f, %f' %
                    (ffweightx, ffweighty, ffweightz, ffweightT))
    return True

def print_givenForce_info(**problem_kwargs):
    print_forceFree_info(**problem_kwargs)
    givenF = problem_kwargs['givenF']
    PETSc.Sys.Print('  given Force:', givenF)
    return True

def print_solver_info_forceFree(**problem_kwargs):
    comm = PETSc.COMM_WORLD.tompi4py()
    size = comm.Get_size()

    fileHeadle = problem_kwargs['fileHeadle']
    matrix_method = problem_kwargs['matrix_method']
    solve_method = problem_kwargs['solve_method']
    precondition_method = problem_kwargs['precondition_method']

    err_msg = "Only 'pf', 'pf_stokesletsInPipe', 'pf_stokesletsTwoPlate'" \
              " and 'rs' methods are accept for this main code. "
    acceptType = ('rs', 'rs_plane', 'pf', 'pf_stokesletsInPipe', 'pf_stokesletsTwoPlate')
    assert matrix_method in acceptType, err_msg
    PETSc.Sys.Print('output file headle: ' + fileHeadle)
    PETSc.Sys.Print('  create matrix method: %s, ' % matrix_method)
    if matrix_method in ('rs', 'pf', 'rs_plane'):
        pass
    elif matrix_method in ('pf_stokesletsInPipe',):
        forcepipe = problem_kwargs['forcepipe']
        PETSc.Sys.Print('  read force of pipe from: ' + forcepipe)
    elif matrix_method in ('pf_stokesletsTwoPlate',):
        # raise Exception('set how to print matrix method please. ')
        pass
    else:
        raise Exception('set how to print matrix method please. ')

    PETSc.Sys.Print('  solve method: %s, precondition method: %s'
                    % (solve_method, precondition_method))
    PETSc.Sys.Print('  output file headle: ' + fileHeadle)
    PETSc.Sys.Print('  MPI size: %d' % size)


def print_single_ecoli_forceFree_result(ecoli_comp: sf.forceFreeComposite, **kwargs):
    rh1 = kwargs['rh1']
    zoom_factor = kwargs['zoom_factor']
    rel_Us = kwargs['rel_Us']
    rel_Uh = kwargs['rel_Uh']

    with_T_geo = len(ecoli_comp.get_obj_list()) == 4
    if with_T_geo:
        vsobj, vhobj0, vhobj1, vTobj = ecoli_comp.get_obj_list()
        temp_f = 0.5 * (np.abs(vsobj.get_force().reshape((-1, 3)).sum(axis=0)) +
                        np.abs(vhobj0.get_force().reshape((-1, 3)).sum(axis=0) +
                               vhobj1.get_force().reshape((-1, 3)).sum(axis=0) +
                               vTobj.get_force().reshape((-1, 3)).sum(axis=0)))
    else:
        vsobj, vhobj0, vhobj1 = ecoli_comp.get_obj_list()
        temp_f = 0.5 * (np.abs(vsobj.get_force().reshape((-1, 3)).sum(axis=0)) +
                        np.abs(vhobj0.get_force().reshape((-1, 3)).sum(axis=0) +
                               vhobj1.get_force().reshape((-1, 3)).sum(axis=0)))
    temp_F = np.hstack((temp_f, temp_f * zoom_factor))
    non_dim_F = ecoli_comp.get_total_force() / temp_F
    t_nondim = np.sqrt(np.sum((rel_Uh[-3:] + rel_Us[-3:]) ** 2))
    non_dim_U = ecoli_comp.get_ref_U() / np.array(
            (zoom_factor * rh1, zoom_factor * rh1, zoom_factor * rh1, 1, 1, 1)) / t_nondim
    PETSc.Sys.Print('non_dim_U', non_dim_U)
    PETSc.Sys.Print('non_dim_F', non_dim_F)
    PETSc.Sys.Print('velocity_sphere', rel_Us + ecoli_comp.get_ref_U())
    PETSc.Sys.Print('velocity_helix', rel_Uh + ecoli_comp.get_ref_U())


def get_ecoli_kwargs():
    OptDB = PETSc.Options()
    rh1 = OptDB.getReal('rh1', 0.2)  # radius of helix
    rh2 = OptDB.getReal('rh2', 0.05)  # radius of helix
    nth = OptDB.getInt('nth', 2)  # amount of nodes on each cycle of helix
    eh = OptDB.getReal('eh', -0.1)  # epsilon of helix
    ch = OptDB.getReal('ch', 0.1)  # cycles of helix
    ph = OptDB.getReal('ph', 3)  # helix pitch
    hfct = OptDB.getReal('hfct', 1)  # helix axis line factor, put more nodes near both tops
    with_cover = OptDB.getBool('with_cover', True)
    left_hand = OptDB.getBool('left_hand', False)
    rs = OptDB.getReal('rs', 0.5)  # radius of head
    rs1 = OptDB.getReal('rs1', rs * 2)  # radius of head
    rs2 = OptDB.getReal('rs2', rs)  # radius of head
    ls = OptDB.getReal('ls', rs1 * 2)  # length of head
    ds = OptDB.getReal('ds', 1)  # delta length of sphere
    es = OptDB.getReal('es', -0.1)  # epsilon of sphere
    rT1 = OptDB.getReal('rT1', rh1)  # radius of Tgeo
    rT2 = OptDB.getReal('rT2', rh2)  # radius of Tgeo
    ntT = OptDB.getReal('ntT', nth)  # amount of nodes on each cycle of Tgeo
    eT = OptDB.getReal('eT', eh)  # epsilon of Tgeo
    Tfct = OptDB.getReal('Tfct', 1)  # Tgeo axis line factor, put more nodes near both tops
    with_T_geo = OptDB.getBool('with_T_geo', True)

    rel_Usx = OptDB.getReal('rel_Usx', 0)
    rel_Usy = OptDB.getReal('rel_Usy', 0)
    rel_Usz = OptDB.getReal('rel_Usz', 0)
    rel_Uhx = OptDB.getReal('rel_Uhx', 0)
    rel_Uhy = OptDB.getReal('rel_Uhy', 0)
    rel_Uhz = OptDB.getReal('rel_Uhz', 1)
    rel_Us = np.array((0, 0, 0, rel_Usx, rel_Usy, rel_Usz))  # relative omega of sphere
    rel_Uh = np.array((0, 0, 0, rel_Uhx, rel_Uhy, rel_Uhz))  # relative omega of helix
    dist_hs = OptDB.getReal('dist_hs', 2)  # distance between head and tail
    centerx = OptDB.getReal('centerx', 0)
    centery = OptDB.getReal('centery', 0)
    centerz = OptDB.getReal('centerz', 0)
    center = np.array((centerx, centery, centerz))  # center of ecoli
    zoom_factor = OptDB.getReal('zoom_factor', 1)

    ecoli_kwargs = {
        'rh1':         rh1,
        'rh2':         rh2,
        'nth':         nth,
        'eh':          eh,
        'ch':          ch,
        'ph':          ph,
        'hfct':        hfct,
        'with_cover':  with_cover,
        'left_hand':   left_hand,
        'rs1':         rs1,
        'rs2':         rs2,
        'ls':          ls,
        'ds':          ds,
        'es':          es,
        'rT1':         rT1,
        'rT2':         rT2,
        'ntT':         ntT,
        'eT':          eT,
        'Tfct':        Tfct,
        'with_T_geo':  with_T_geo,
        'rel_Us':      rel_Us,
        'rel_Uh':      rel_Uh,
        'dist_hs':     dist_hs,
        'center':      center,
        'zoom_factor': zoom_factor,
    }
    return ecoli_kwargs


def get_vtk_tetra_kwargs():
    OptDB = PETSc.Options()
    matname = OptDB.getString('bmat', 'body1')
    bnodesHeadle = OptDB.getString('bnodes', 'bnodes')  # body nodes, for vtu output
    belemsHeadle = OptDB.getString('belems', 'belems')  # body tetrahedron mesh, for vtu output
    vtk_tetra_kwargs = {
        'matname':      matname,
        'bnodesHeadle': bnodesHeadle,
        'belemsHeadle': belemsHeadle,
    }
    return vtk_tetra_kwargs


def get_solver_kwargs():
    OptDB = PETSc.Options()
    solve_method = OptDB.getString('s', 'gmres')
    precondition_method = OptDB.getString('g', 'none')
    matrix_method = OptDB.getString('sm', 'pf')
    restart = OptDB.getBool('restart', False)
    n_node_threshold = OptDB.getInt('n_threshold', 10000)
    getConvergenceHistory = OptDB.getBool('getConvergenceHistory', False)
    pickProblem = OptDB.getBool('pickProblem', False)
    plot_geo = OptDB.getBool('plot_geo', False)

    problem_kwargs = {
        'matrix_method':         matrix_method,
        'solve_method':          solve_method,
        'precondition_method':   precondition_method,
        'restart':               restart,
        'n_node_threshold':      n_node_threshold,
        'getConvergenceHistory': getConvergenceHistory,
        'pickProblem':           pickProblem,
        'plot_geo':              plot_geo,
    }

    if matrix_method in ('pf_stokesletsInPipe',):
        forcepipe = OptDB.getString('forcepipe', 'dbg')
        t_headle = '_force_pipe.mat'
        forcepipe = forcepipe if forcepipe[-len(t_headle):] == t_headle else forcepipe + t_headle
        problem_kwargs['forcepipe'] = forcepipe
    return problem_kwargs


def get_forceFree_kwargs():
    OptDB = PETSc.Options()
    ffweight = OptDB.getReal('ffweight', 1)  # force free condition weight
    ffweightx = OptDB.getReal('ffweightx', ffweight)  # weight of sum(Fx) = 0
    ffweighty = OptDB.getReal('ffweighty', ffweight)  # weight of sum(Fy) = 0
    ffweightz = OptDB.getReal('ffweightz', ffweight)  # weight of sum(Fz) = 0
    ffweightT = OptDB.getReal('ffweightT', ffweight)  # weight of sum(Tx) = sum(Ty) = sum(Tz) = 0
    problem_kwargs = {
        'ffweightx': ffweightx,
        'ffweighty': ffweighty,
        'ffweightz': ffweightz,
        'ffweightT': ffweightT,
    }
    return problem_kwargs


def get_givenForce_kwargs():
    problem_kwargs = get_forceFree_kwargs()
    OptDB = PETSc.Options()
    givenf = OptDB.getReal('givenf', 0)
    givenfx = OptDB.getReal('givenfx', givenf)
    givenfy = OptDB.getReal('givenfy', givenf)
    givenfz = OptDB.getReal('givenfz', givenf)
    givent = OptDB.getReal('givent', 0)
    giventx = OptDB.getReal('giventx', givent)
    giventy = OptDB.getReal('giventy', givent)
    giventz = OptDB.getReal('giventz', givent)
    givenF = np.array((givenfx, givenfy, givenfz, giventx, giventy, giventz))
    problem_kwargs['givenF'] = givenF
    return problem_kwargs


def get_rod_kwargs():
    OptDB = PETSc.Options()
    rRod = OptDB.getReal('rRod', 1)  # radius of Rod
    lRod = OptDB.getReal('lRod', 5)  # length of Rod
    ntRod = OptDB.getReal('ntRod', 3)  # amount of nodes on each cycle of Rod
    eRod = OptDB.getReal('eRod', -0.1)  # epsilon of Rod
    Rodfct = OptDB.getReal('Rodfct', 1)  # Rod axis line factor, put more nodes near both tops
    RodThe = OptDB.getReal('RodThe', 0) # Angle between the rod and XY plane.
    rel_uRodx = OptDB.getReal('rel_uRodx', 0)
    rel_uRody = OptDB.getReal('rel_uRody', 0)
    rel_uRodz = OptDB.getReal('rel_uRodz', 0)
    rel_wRodx = OptDB.getReal('rel_wRodx', 0)
    rel_wRody = OptDB.getReal('rel_wRody', 0)
    rel_wRodz = OptDB.getReal('rel_wRodz', 0)
    rel_URod = np.array((rel_uRodx, rel_uRody, rel_uRodz, rel_wRodx, rel_wRody, rel_wRodz))  # relative velocity of Rod
    centerx = OptDB.getReal('centerx', 0)
    centery = OptDB.getReal('centery', 0)
    centerz = OptDB.getReal('centerz', 2)
    center = np.array((centerx, centery, centerz))  # center of Rod
    zoom_factor = OptDB.getReal('zoom_factor', 1)
    rod_kwargs = {
        'rRod':        rRod,
        'lRod':        lRod,
        'ntRod':       ntRod,
        'eRod':        eRod,
        'Rodfct':      Rodfct,
        'RodThe':      RodThe,
        'rel_URod':    rel_URod,
        'center':      center,
        'zoom_factor': zoom_factor,
    }
    return rod_kwargs

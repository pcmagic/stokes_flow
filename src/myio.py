import sys

import petsc4py

petsc4py.init(sys.argv)

from petsc4py import PETSc

__all__ = ['print_case_info',
           'get_ecoli_kwargs', 'get_vtk_tetra_kwargs', 'get_problem_kwargs_base']


def print_case_info(ecoName, **problem_kwargs):
    comm = PETSc.COMM_WORLD.tompi4py( )
    rank = comm.Get_rank( )
    size = comm.Get_size( )

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
    zoom_factor = problem_kwargs['zoom_factor']
    ffweight = problem_kwargs['ffweight']

    PETSc.Sys.Print(ecoName, ' information: ')
    PETSc.Sys.Print('  helix radius: %f and %f, helix pitch: %f, helix cycle: %f' % (rh1, rh2, ph, ch))
    PETSc.Sys.Print('  nth, hfct and epsilon of helix are %d, %f and %f, ' % (nth, hfct, eh))
    PETSc.Sys.Print('  head radius: %f and %f, head length: %f, delta length: %f, epsilon: %f' % (rs1, rs2, ls, ds, es))
    PETSc.Sys.Print('  Tgeo radius: %f and %f, ntT and eT of Tgeo are: %f and %f' % (rT1, rT2, ntT, eT))
    PETSc.Sys.Print('  ecoli center: %s, distance from head to tail is %f' % (str(center), dist_hs))
    PETSc.Sys.Print('  relative velocity of head and tail are %s and %s' % (str(rel_Us), str(rel_Uh)))
    PETSc.Sys.Print('  geometry zoom factor is %f, force free weight mode is %s' % (zoom_factor, ffweight))

    err_msg = "Only 'rs', 'tp_rs', 'lg_rs', and 'pf' methods are accept for this main code. "
    acceptType = ('rs', 'tp_rs', 'lg_rs', 'pf')
    assert matrix_method in acceptType, err_msg
    if matrix_method in 'rs':
        PETSc.Sys.Print('  create matrix method: %s, ' % matrix_method)
    elif matrix_method in 'tp_rs':
        twoPara_n = problem_kwargs['twoPara_n']
        PETSc.Sys.Print('  create matrix method: %s, order: %d'
                        % (matrix_method, twoPara_n))
    elif matrix_method in 'lg_rs':
        legendre_m = problem_kwargs['legendre_m']
        legendre_k = problem_kwargs['legendre_k']
        PETSc.Sys.Print('  create matrix method: %s, m: %d, k: %d, p: %d'
                        % (matrix_method, legendre_m, legendre_k, (legendre_m + 2 * legendre_k + 1)))
    elif matrix_method in 'pf':
        PETSc.Sys.Print('  create matrix method: %s ' % matrix_method)
    else:
        raise Exception('set how to print matrix method please. ')

    solve_method = problem_kwargs['solve_method']
    precondition_method = problem_kwargs['precondition_method']
    PETSc.Sys.Print('  solve method: %s, precondition method: %s'
                    % (solve_method, precondition_method))
    PETSc.Sys.Print('  output file headle: ' + fileHeadle)
    PETSc.Sys.Print('MPI size: %d' % size)
    return True


def get_ecoli_kwargs( ):
    OptDB = PETSc.Options( )
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

    # rel_Usx = OptDB.getReal('rel_Usx', 0)
    # rel_Usy = OptDB.getReal('rel_Usy', 0)
    rel_Usz = OptDB.getReal('rel_Usz', 0)
    # rel_Uhx = OptDB.getReal('rel_Uhx', 0)
    # rel_Uhy = OptDB.getReal('rel_Uhy', 0)
    rel_Uhz = OptDB.getReal('rel_Uhz', 1)
    rel_Us = np.array((0, 0, 0, 0, 0, rel_Usz))  # relative omega of sphere
    rel_Uh = np.array((0, 0, 0, 0, 0, rel_Uhz))  # relative omega of helix
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
        'rel_Us':      rel_Us,
        'rel_Uh':      rel_Uh,
        'dist_hs':     dist_hs,
        'center':      center,
        'zoom_factor': zoom_factor,
    }
    return ecoli_kwargs


def get_vtk_tetra_kwargs( ):
    OptDB = PETSc.Options( )
    matname = OptDB.getString('bmat', 'body1')
    bnodesHeadle = OptDB.getString('bnodes', 'bnodes')  # body nodes, for vtu output
    belemsHeadle = OptDB.getString('belems', 'belems')  # body tetrahedron mesh, for vtu output
    vtk_tetra_kwargs = {
        'matname':      matname,
        'bnodesHeadle': bnodesHeadle,
        'belemsHeadle': belemsHeadle,
    }
    return vtk_tetra_kwargs


def get_problem_kwargs_base(**main_kwargs):
    OptDB = PETSc.Options( )
    fileHeadle = OptDB.getString('f', 'singleEcoliPro')
    solve_method = OptDB.getString('s', 'gmres')
    precondition_method = OptDB.getString('g', 'none')
    matrix_method = OptDB.getString('sm', 'pf')
    restart = OptDB.getBool('restart', False)
    n_node_threshold = OptDB.getInt('n_threshold', 10000)
    getConvergenceHistory = OptDB.getBool('getConvergenceHistory', False)
    pickProblem = OptDB.getBool('pickProblem', False)
    plot_geo = OptDB.getBool('plot_geo', False)
    prb_index = OptDB.getInt('prb_index', -1)

    problem_kwargs = {
        'matrix_method':         matrix_method,
        'prb_index':             prb_index,
        'solve_method':          solve_method,
        'precondition_method':   precondition_method,
        'fileHeadle':            fileHeadle,
        'restart':               restart,
        'n_node_threshold':      n_node_threshold,
        'getConvergenceHistory': getConvergenceHistory,
        'pickProblem':           pickProblem,
        'plot_geo':              plot_geo,
    }
    for key in main_kwargs:
        problem_kwargs[key] = main_kwargs[key]
    return problem_kwargs

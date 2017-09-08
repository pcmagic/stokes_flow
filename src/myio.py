import sys
import petsc4py

petsc4py.init(sys.argv)

import numpy as np
from petsc4py import PETSc
from src import stokes_flow as sf

__all__ = ['get_solver_kwargs', 'get_forceFree_kwargs', 'get_givenForce_kwargs', 'get_vtk_tetra_kwargs',
           'print_solver_info', 'print_forceFree_info', 'print_givenForce_info',
           'get_shearFlow_kwargs', 'print_shearFlow_info',
           'get_ecoli_kwargs', 'print_ecoli_info', 'print_ecoli_U_info', 'print_single_ecoli_forceFree_result',
           'get_rod_kwargs', 'print_Rod_info',
           'get_sphere_kwargs', 'print_sphere_info', ]


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
    head_U = rel_Us + ecoli_comp.get_ref_U()
    tail_U = rel_Uh + ecoli_comp.get_ref_U()
    PETSc.Sys.Print('velocity_sphere', head_U)
    PETSc.Sys.Print('velocity_helix', tail_U)
    return head_U, tail_U


def print_ecoli_U_info(ecoName, **problem_kwargs):
    rel_Us = problem_kwargs['rel_Us']
    rel_Uh = problem_kwargs['rel_Uh']
    ecoli_U = problem_kwargs['ecoli_U']
    PETSc.Sys.Print(ecoName, 'given velocity information: ')
    PETSc.Sys.Print('  reference velocity of ecoli is %s' % str(ecoli_U))
    PETSc.Sys.Print('  global velocity of head is %s' % str(rel_Us + ecoli_U))
    PETSc.Sys.Print('  global velocity of tail is %s' % str(rel_Uh + ecoli_U))
    return True


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

    # rotate the ecoli, original it is along z axis.
    rot_theta = OptDB.getReal('rot_theta', 0)
    rot_norm = np.array((1, 0, 0))  # currently is x axis.
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
        'rot_theta':   rot_theta,
        'rot_norm':    rot_norm,
    }
    return ecoli_kwargs


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
    rot_norm = problem_kwargs['rot_norm']
    rot_theta = problem_kwargs['rot_theta']

    PETSc.Sys.Print(ecoName, 'geo information: ')
    PETSc.Sys.Print('  helix radius: %f and %f, helix pitch: %f, helix cycle: %f' % (rh1, rh2, ph, ch))
    PETSc.Sys.Print('    nth, hfct and epsilon of helix are %d, %f and %f, ' % (nth, hfct, eh))
    PETSc.Sys.Print('  head radius: %f and %f, length: %f, delta length: %f, epsilon: %f' % (rs1, rs2, ls, ds, es))
    PETSc.Sys.Print('  Tgeo radius: %f and %f' % (rT1, rT2))
    PETSc.Sys.Print('    ntT, eT and Tfct of Tgeo are: %d, %f and %f' % (ntT, eT, Tfct))
    PETSc.Sys.Print('  ecoli center: %s, distance from head to tail is %f' % (str(center), dist_hs))
    PETSc.Sys.Print('  relative velocity of head and tail are %s and %s' % (str(rel_Us), str(rel_Uh)))
    PETSc.Sys.Print('  rot_norm is %s, rot_theta is %f*pi' % (str(rot_norm), rot_theta))
    PETSc.Sys.Print('  geometry zoom factor is %f' % zoom_factor)
    return True


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
    comm = PETSc.COMM_WORLD.tompi4py()
    MPISIZE = comm.Get_size()

    problem_kwargs = {
        'matrix_method':         matrix_method,
        'solve_method':          solve_method,
        'precondition_method':   precondition_method,
        'restart':               restart,
        'n_node_threshold':      n_node_threshold,
        'getConvergenceHistory': getConvergenceHistory,
        'pickProblem':           pickProblem,
        'plot_geo':              plot_geo,
        'MPISIZE':               MPISIZE,
    }

    if matrix_method in ('pf_stokesletsInPipe',):
        forcepipe = OptDB.getString('forcepipe', 'dbg')
        t_headle = '_force_pipe.mat'
        forcepipe = forcepipe if forcepipe[-len(t_headle):] == t_headle else forcepipe + t_headle
        problem_kwargs['forcepipe'] = forcepipe
    elif matrix_method in ('pf_stokesletsTwoPlane',):
        twoPlateHeight = OptDB.getReal('twoPlateHeight', 1)  # twoPlateHeight
        problem_kwargs['twoPlateHeight'] = twoPlateHeight

    return problem_kwargs


def print_solver_info(**problem_kwargs):
    comm = PETSc.COMM_WORLD.tompi4py()
    size = comm.Get_size()

    fileHeadle = problem_kwargs['fileHeadle']
    matrix_method = problem_kwargs['matrix_method']
    solve_method = problem_kwargs['solve_method']
    precondition_method = problem_kwargs['precondition_method']

    err_msg = "Only 'pf', 'pf_stokesletsInPipe', 'pf_stokesletsTwoPlane'" \
              " and 'rs' methods are accept for this main code. "
    acceptType = ('rs', 'rs_plane', 'pf', 'pf_stokesletsInPipe', 'pf_stokesletsTwoPlane', )
    assert matrix_method in acceptType, err_msg
    PETSc.Sys.Print('output file headle: ' + fileHeadle)
    PETSc.Sys.Print('  create matrix method: %s, ' % matrix_method)
    if matrix_method in ('rs', 'pf', 'rs_plane'):
        pass
    elif matrix_method in ('pf_stokesletsInPipe',):
        forcepipe = problem_kwargs['forcepipe']
        PETSc.Sys.Print('  read force of pipe from: ' + forcepipe)
    elif matrix_method in ('pf_stokesletsTwoPlane',):
        twoPlateHeight = problem_kwargs['twoPlateHeight']
        PETSc.Sys.Print('Height of upper plane is %f ' % twoPlateHeight)
    else:
        raise Exception('set how to print matrix method please. ')

    PETSc.Sys.Print('  solve method: %s, precondition method: %s'
                    % (solve_method, precondition_method))
    PETSc.Sys.Print('  output file headle: ' + fileHeadle)
    PETSc.Sys.Print('  MPI size: %d' % size)


def get_shearFlow_kwargs():
    OptDB = PETSc.Options()
    planeShearRatex = OptDB.getReal('planeShearRatex', 0)  #
    planeShearRatey = OptDB.getReal('planeShearRatey', 0)  #
    planeShearRate = np.array((planeShearRatex, planeShearRatey, 0)).reshape((1, 3))
    problem_kwargs = {'planeShearRate': planeShearRate}
    return problem_kwargs


def print_shearFlow_info(**problem_kwargs):
    planeShearRate = problem_kwargs['planeShearRate']
    PETSc.Sys.Print('Given background flow: shear flow, rate: %s ' % str(planeShearRate.flatten()))
    return True


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


def print_forceFree_info(**problem_kwargs):
    ffweightx = problem_kwargs['ffweightx']
    ffweighty = problem_kwargs['ffweighty']
    ffweightz = problem_kwargs['ffweightz']
    ffweightT = problem_kwargs['ffweightT']
    PETSc.Sys.Print('  force free weight of Fx, Fy, Fz, and (Tx, Ty, Tz) are %f, %f, %f, %f' %
                    (ffweightx, ffweighty, ffweightz, ffweightT))
    return True


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


def print_givenForce_info(**problem_kwargs):
    print_forceFree_info(**problem_kwargs)
    givenF = problem_kwargs['givenF']
    PETSc.Sys.Print('  given Force:', givenF)
    return True


def get_rod_kwargs():
    OptDB = PETSc.Options()
    rRod = OptDB.getReal('rRod', 1)  # radius of Rod
    lRod = OptDB.getReal('lRod', 5)  # length of Rod
    ntRod = OptDB.getReal('ntRod', 3)  # amount of nodes on each cycle of Rod
    eRod = OptDB.getReal('eRod', 0.1)  # epsilon of Rod
    Rodfct = OptDB.getReal('Rodfct', 1)  # Rod axis line factor, put more nodes near both tops
    RodThe = OptDB.getReal('RodThe', 0)  # Angle between the rod and XY plane.
    RodPhi = OptDB.getReal('RodPhi', 0)  # Angle between the rod and XY plane.
    rel_uRodx = OptDB.getReal('rel_uRodx', 0)
    rel_uRody = OptDB.getReal('rel_uRody', 0)
    rel_uRodz = OptDB.getReal('rel_uRodz', 0)
    rel_wRodx = OptDB.getReal('rel_wRodx', 0)
    rel_wRody = OptDB.getReal('rel_wRody', 0)
    rel_wRodz = OptDB.getReal('rel_wRodz', 0)
    rel_URod = np.array((rel_uRodx, rel_uRody, rel_uRodz, rel_wRodx, rel_wRody, rel_wRodz))  # relative velocity of Rod
    RodCenterx = OptDB.getReal('RodCenterx', 0)
    RodCentery = OptDB.getReal('RodCentery', 0)
    RodCenterz = OptDB.getReal('RodCenterz', 2)
    RodCenter = np.array((RodCenterx, RodCentery, RodCenterz))  # center of Rod
    zoom_factor = OptDB.getReal('zoom_factor', 1)
    rod_kwargs = {
        'rRod':        rRod,
        'lRod':        lRod,
        'ntRod':       ntRod,
        'eRod':        eRod,
        'Rodfct':      Rodfct,
        'RodThe':      RodThe,
        'RodPhi':      RodPhi,
        'rel_URod':    rel_URod,
        'RodCenter':   RodCenter,
        'zoom_factor': zoom_factor,
    }
    return rod_kwargs


def print_Rod_info(RodName, **problem_kwargs):
    rRod = problem_kwargs['rRod']
    lRod = problem_kwargs['lRod']
    ntRod = problem_kwargs['ntRod']
    eRod = problem_kwargs['eRod']
    Rodfct = problem_kwargs['Rodfct']
    RodThe = problem_kwargs['RodThe']
    RodPhi = problem_kwargs['RodPhi']
    rel_URod = problem_kwargs['rel_URod']
    RodCenter = problem_kwargs['RodCenter']
    zoom_factor = problem_kwargs['zoom_factor']

    PETSc.Sys.Print(RodName, 'geo information: ')
    PETSc.Sys.Print('  rod radius: %f, length: %f, RodThe: %f, RodPhi %f' %
                    (rRod, lRod, RodThe, RodPhi))
    PETSc.Sys.Print('    ntRod, Rodfct and ntRod are %d, %f and %f, ' % (ntRod, Rodfct, eRod))
    PETSc.Sys.Print('    RodCenter is %s ' % str(RodCenter))
    PETSc.Sys.Print('  relative velocity of rod is %s' % str(rel_URod))
    PETSc.Sys.Print('  geometry zoom factor is %f' % zoom_factor)


# def print_singleRod_givenforce_result(ecoli_comp: sf.forceFreeComposite, **kwargs):
#     pass


def get_sphere_kwargs():
    """
    u: translating velocity of sphere.
    w: rotation velocity of sphere.
    ds: distance between two point on sphere.
    es: distance between tow point on force sphere.
    """
    OptDB = PETSc.Options()
    rs = OptDB.getReal('rs', 0.5)
    ds = OptDB.getReal('ds', 0.1)
    es = OptDB.getReal('es', -0.1)
    ux = OptDB.getReal('ux', 0)
    uy = OptDB.getReal('uy', 0)
    uz = OptDB.getReal('uz', 0)
    wx = OptDB.getReal('wx', 0)
    wy = OptDB.getReal('wy', 0)
    wz = OptDB.getReal('wz', 0)

    '''Generate a grid of sphere'''
    n_obj = OptDB.getInt('n', 1)
    n_obj_x = OptDB.getInt('nx', n_obj)
    n_obj_y = OptDB.getInt('ny', n_obj)
    n_obj_z = OptDB.getInt('nz', n_obj)
    distance = OptDB.getReal('dist', 3)
    distance_x = OptDB.getReal('distx', distance)
    distance_y = OptDB.getReal('disty', distance)
    distance_z = OptDB.getReal('distz', distance)
    t_x_coord = distance_x * np.arange(n_obj_x)
    t_y_coord = distance_y * np.arange(n_obj_y)
    t_z_coord = distance_z * np.arange(n_obj_z)
    x_coord, y_coord, z_coord = np.meshgrid(t_x_coord, t_y_coord, t_z_coord)
    sphere_coord = np.vstack((x_coord.flatten(), y_coord.flatten(), z_coord.flatten())).T

    random_velocity = OptDB.getBool('random_velocity', False)
    t_velocity = np.array((ux, uy, uz, wx, wy, wz))
    if random_velocity:
        sphere_velocity = np.random.sample((x_coord.size, 6)) * t_velocity
    else:
        sphere_velocity = np.ones((x_coord.size, 6)) * t_velocity

    sphere_kwargs = {
        'rs':              rs,
        'sphere_velocity': sphere_velocity,
        'ds':              ds,
        'es':              es,
        'sphere_coord':    sphere_coord, }
    return sphere_kwargs


def print_sphere_info(sphereName, **problem_kwargs):
    rs = problem_kwargs['rs']
    sphere_velocity = problem_kwargs['sphere_velocity']
    ds = problem_kwargs['ds']
    es = problem_kwargs['es']
    sphere_coord = problem_kwargs['sphere_coord']

    PETSc.Sys.Print(sphereName, 'geo information: ')
    PETSc.Sys.Print('  radius deltalength and epsilon of sphere: {rs}, {ds}, {es}'.format(rs=rs, ds=ds, es=es))
    PETSc.Sys.Print('  center coordinates and rigid body velocity are:')
    for t_coord, t_velocity in zip(sphere_coord, sphere_velocity):
        PETSc.Sys.Print(' ', t_coord, '&', t_velocity)
    return True

import numpy as np
from petsc4py import PETSc
from src.geo import *
from src import stokes_flow as sf

__all__ = ['createEcoliComp_ellipse', 'createEcoliComp_tunnel', 'createEcoli_tunnel',
           'create_capsule',
           'create_rod',
           'create_sphere', 'create_move_single_sphere']


def createEcoliComp_ellipse(name='...', **kwargs):
    nth = kwargs['nth']
    hfct = kwargs['hfct']
    eh = kwargs['eh']
    ch = kwargs['ch']
    rh1 = kwargs['rh1']
    rh2 = kwargs['rh2']
    ph = kwargs['ph']
    with_cover = kwargs['with_cover']
    ds = kwargs['ds']
    rs1 = kwargs['rs1']
    rs2 = kwargs['rs2']
    es = kwargs['es']
    left_hand = kwargs['left_hand']
    # sphere_rotation = kwargs['sphere_rotation'] if 'sphere_rotation' in kwargs.keys() else 0
    zoom_factor = kwargs['zoom_factor'] if 'zoom_factor' in kwargs.keys() else 1
    dist_hs = kwargs['dist_hs']
    center = kwargs['center']
    matrix_method = kwargs['matrix_method']
    lh = ph * ch  # length of helix
    movesz = 0.5 * (dist_hs - 2 * rs1 + lh) + rs1
    movehz = 0.5 * (dist_hs + 2 * rs1 - lh) + lh / 2
    moves = np.array((0, 0, movesz)) + center  # move distance of sphere
    moveh = np.array((0, 0, -movehz)) + center  # move distance of helix
    objtype = sf.obj_dic[matrix_method]

    # create helix
    B = ph / (2 * np.pi)
    vhgeo0 = supHelix()  # velocity node geo of helix
    dth = 2 * np.pi / nth
    fhgeo0 = vhgeo0.create_deltatheta(dth=dth, radius=rh2, R=rh1, B=B, n_c=ch, epsilon=eh, with_cover=with_cover,
                                      factor=hfct, left_hand=left_hand)
    vhobj0 = objtype()
    vhobj0.set_data(fhgeo0, vhgeo0, name='helix_0')
    vhobj0.zoom(zoom_factor)
    vhobj0.move(moveh * zoom_factor)
    vhobj1 = vhobj0.copy()
    vhobj1.node_rotation(norm=(0, 0, 1), theta=np.pi, rotation_origin=(0, 0, 0))
    vhobj1.set_name('helix_1')

    # create sphere
    vsgeo = ellipse_geo()  # velocity node geo of sphere
    vsgeo.create_delta(ds, rs1, rs2)
    vsgeo.node_rotation(norm=np.array((0, 1, 0)), theta=np.pi / 2)
    fsgeo = vsgeo.copy()  # force node geo of sphere
    fsgeo.node_zoom(1 + ds / (0.5 * (rs1 + rs2)) * es)
    vsobj = objtype()
    vsobj.set_data(fsgeo, vsgeo, name='sphere_0')
    vsobj.zoom(zoom_factor)
    vsobj.move(moves * zoom_factor)

    rel_Us = kwargs['rel_Us']
    rel_Uh = kwargs['rel_Uh']
    ecoli_comp = sf.forceFreeComposite(center, name)
    ecoli_comp.add_obj(vsobj, rel_U=rel_Us)
    ecoli_comp.add_obj(vhobj0, rel_U=rel_Uh)
    ecoli_comp.add_obj(vhobj1, rel_U=rel_Uh)

    rot_norm = kwargs['rot_norm']
    rot_theta = kwargs['rot_theta'] * np.pi
    ecoli_comp.node_rotation(norm=rot_norm, theta=rot_theta, rotation_origin=center)
    return ecoli_comp


def create_capsule(rs1, rs2, ls, ds):
    lvs3 = ls - 2 * rs2
    dth = ds / rs2
    err_msg = 'geo parameter of create_capsule head is wrong. '
    assert lvs3 >= 0, err_msg
    vsgeo = geo()

    vsgeo1 = ellipse_geo()  # velocity node geo of head
    vsgeo1.create_half_delta(ds, rs1, rs2)
    vsgeo2 = vsgeo1.copy()
    vsgeo1.node_rotation(norm=np.array((0, 1, 0)), theta=-np.pi / 2)
    vsgeo1.node_rotation(norm=np.array((0, 0, 1)), theta=-np.pi / 2)
    vsgeo1.move((0, 0, -lvs3 / 2))
    vsgeo2.node_rotation(norm=np.array((0, 1, 0)), theta=+np.pi / 2)
    vsgeo2.node_rotation(norm=np.array((0, 0, 1)), theta=+np.pi / 2 - dth)
    vsgeo2.move((0, 0, +lvs3 / 2))
    vsgeo2.set_nodes(np.flipud(vsgeo2.get_nodes()), deltalength=vsgeo2.get_deltaLength())
    if lvs3 > ds:
        vsgeo3 = tunnel_geo()
        vsgeo3.create_deltatheta(dth=dth, radius=rs2, length=lvs3)
        vsgeo.combine([vsgeo1, vsgeo3, vsgeo2])
    else:
        vsgeo.combine([vsgeo1, vsgeo2])
    return vsgeo


def createEcoli_tunnel(**kwargs):
    nth = kwargs['nth']
    hfct = kwargs['hfct']
    eh = kwargs['eh']
    ch = kwargs['ch']
    rh1 = kwargs['rh1']
    rh2 = kwargs['rh2']
    ph = kwargs['ph']
    with_cover = kwargs['with_cover']
    ds = kwargs['ds']
    rs1 = kwargs['rs1']
    rs2 = kwargs['rs2']
    ls = kwargs['ls']
    es = kwargs['es']
    left_hand = kwargs['left_hand']
    # sphere_rotation = kwargs['sphere_rotation'] if 'sphere_rotation' in kwargs.keys() else 0
    zoom_factor = kwargs['zoom_factor']
    dist_hs = kwargs['dist_hs']
    center = kwargs['center']
    rT1 = kwargs['rT1']
    rT2 = kwargs['rT2']
    ntT = kwargs['ntT']
    eT = kwargs['eT']
    Tfct = kwargs['Tfct']
    matrix_method = kwargs['matrix_method']
    lh = ph * ch  # length of helix
    movesz = 0.5 * (dist_hs - ls + lh) + ls / 2
    movehz = 0.5 * (dist_hs + ls - lh) + lh / 2
    # movesz = (ls + dist_hs) / 2
    # movehz = (lh + dist_hs) / 2
    moves = np.array((0, 0, movesz)) + center  # move distance of sphere
    moveh = np.array((rT1 - rh1, 0, -movehz)) + center  # move distance of helix
    lT = (rT1 + rh2) * 2
    objtype = sf.obj_dic[matrix_method]

    # create helix
    B = ph / (2 * np.pi)
    vhgeo0 = supHelix()  # velocity node geo of helix
    dth = 2 * np.pi / nth
    fhgeo0 = vhgeo0.create_deltatheta(dth=dth, radius=rh2, R=rh1, B=B, n_c=ch, epsilon=eh, with_cover=with_cover,
                                      factor=hfct, left_hand=left_hand)
    vhobj0 = objtype()
    vhobj0.set_data(fhgeo0, vhgeo0, name='helix_0')
    vhobj0.zoom(zoom_factor)
    # # dbg
    # OptDB = PETSc.Options( )
    # factor = OptDB.getReal('dbg_theta_factor', 0.5)
    # PETSc.Sys.Print('--------------------> DBG: dbg_theta_factor = %f' % factor)
    # theta = np.pi * ch + (rT2 + rh2 * factor) / (rh1 + rh2)
    theta = np.pi * ch + (rT2 + rh2 * 0.5) / (rh1 + rh2)
    vhobj0.node_rotation(norm=np.array((0, 0, 1)), theta=theta)
    vhobj0.move(moveh * zoom_factor)
    vhobj1 = vhobj0.copy()
    vhobj1.node_rotation(norm=(0, 0, 1), theta=np.pi, rotation_origin=(0, 0, 0))
    vhobj1.set_name('helix_1')

    # create head
    vsgeo = create_capsule(rs1, rs2, ls, ds)
    fsgeo = vsgeo.copy()  # force node geo of sphere
    fsgeo.node_zoom(1 + ds / (0.5 * (rs1 + rs2)) * es)
    fsgeo.node_zoom_z(1 - ds / (0.5 * (rs1 + rs2)) * es)
    vsobj = objtype()
    vsobj.set_data(fsgeo, vsgeo, name='sphere_0')
    vsobj.zoom(zoom_factor)
    vsobj.move(moves * zoom_factor)

    # create T shape
    dtT = 2 * np.pi / ntT
    # # dbg
    # OptDB = PETSc.Options( )
    # factor = OptDB.getReal('dbg_move_factor', 1)
    # PETSc.Sys.Print('--------------------> DBG: dbg_move_factor = %f' % factor)
    # moveT = np.array((0, 0, moveh[-1] + lh / 2 + rh2 * factor))
    moveT = np.array((0, 0, moveh[-1] + lh / 2))
    vTgeo = tunnel_geo()
    fTgeo = vTgeo.create_deltatheta(dth=dtT, radius=rT2, factor=Tfct, length=lT, epsilon=eT, with_cover=True)
    vTobj = objtype()
    vTobj.set_data(fTgeo, vTgeo, name='T_shape_0')
    theta = -np.pi / 2
    vTobj.node_rotation(norm=np.array((0, 1, 0)), theta=theta)
    vTobj.zoom(zoom_factor)
    vTobj.move(moveT * zoom_factor)

    vsobj.node_rotation(norm=np.array((0, 0, 1)), theta=np.pi / 4)
    vhobj0.node_rotation(norm=np.array((0, 0, 1)), theta=np.pi / 4)
    vhobj1.node_rotation(norm=np.array((0, 0, 1)), theta=np.pi / 4)
    vTobj.node_rotation(norm=np.array((0, 0, 1)), theta=np.pi / 4)
    return vsobj, vhobj0, vhobj1, vTobj


def createEcoliComp_tunnel(name='...', **kwargs):
    vsobj, vhobj0, vhobj1, vTobj = createEcoli_tunnel(**kwargs)
    with_T_geo = kwargs['with_T_geo']
    center = kwargs['center']
    rel_Us = kwargs['rel_Us']
    rel_Uh = kwargs['rel_Uh']
    ecoli_comp = sf.forceFreeComposite(center, name)
    ecoli_comp.add_obj(vsobj, rel_U=rel_Us)
    ecoli_comp.add_obj(vhobj0, rel_U=rel_Uh)
    ecoli_comp.add_obj(vhobj1, rel_U=rel_Uh)
    if with_T_geo:
        ecoli_comp.add_obj(vTobj, rel_U=rel_Uh)
    return ecoli_comp


def create_sphere(namehandle='sphereObj', **kwargs):
    matrix_method = kwargs['matrix_method']
    rs = kwargs['rs']
    sphere_velocity = kwargs['sphere_velocity']
    ds = kwargs['ds']
    es = kwargs['es']
    sphere_coord = kwargs['sphere_coord']
    objtype = sf.obj_dic[matrix_method]

    sphere_geo0 = sphere_geo()  # force geo
    sphere_geo0.create_delta(ds, rs)
    sphere_geo0.set_rigid_velocity([0, 0, 0, 0, 0, 0])
    sphere_geo1 = sphere_geo0.copy()
    if matrix_method in ('pf',):
        sphere_geo1.node_zoom((rs + ds * es) / rs)
    obj_sphere = objtype()
    obj_sphere.set_data(sphere_geo1, sphere_geo0)

    obj_list = []
    for i0, (t_coord, t_velocity) in enumerate(zip(sphere_coord, sphere_velocity)):
        obj2 = obj_sphere.copy()
        obj2.set_name('%s_%d' % (namehandle, i0))
        obj2.move(t_coord)
        obj2.get_u_geo().set_rigid_velocity(t_velocity)
        obj_list.append(obj2)
    return obj_list


def create_move_single_sphere(namehandle='sphereObj', **kwargs):
    movez = kwargs['movez']
    obj_sphere = create_sphere(namehandle, **kwargs)[0]
    displacement = np.array((0, 0, movez))
    obj_sphere.move(displacement)
    obj_list = (obj_sphere, )
    return obj_list


def create_rod(namehandle='rod_obj', **problem_kwargs):
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
    givenF = problem_kwargs['givenF']
    matrix_method = problem_kwargs['matrix_method']

    dth = 2 * np.pi / ntRod
    rod_geo = tunnel_geo()
    rod_geo.create_deltatheta(dth=dth, radius=rRod, length=lRod, epsilon=eRod,
                              with_cover=True, factor=Rodfct, left_hand=False)
    # first displace the rod above the surface, rotate to horizon.
    rod_geo.move(displacement=RodCenter)
    rod_geo.node_zoom(factor=zoom_factor, zoom_origin=RodCenter)
    norm = np.array((0, 1, 0))
    theta = -np.pi / 2
    rod_geo.node_rotation(norm=norm, theta=theta, rotation_origin=RodCenter)
    # then, the rod is rotate in a specified plane, which is parabled to XY plane (the wall) first, then
    #   rotated angle theta, of an angle phi.
    norm = np.array((0, np.sin(RodPhi), np.cos(RodPhi)))
    rod_geo.node_rotation(norm=norm, theta=-RodThe, rotation_origin=RodCenter)

    rod_obj = sf.obj_dic[matrix_method]()
    name = namehandle + '_obj_0'
    rod_obj.set_data(f_geo=rod_geo, u_geo=rod_geo, name=name)
    name = namehandle + '_0'
    rod_comp = sf.givenForceComposite(center=RodCenter, name=name, givenF=givenF)
    rod_comp.add_obj(obj=rod_obj, rel_U=rel_URod)

    rod_list = (rod_comp,)
    return rod_list

import numpy as np
from petsc4py import PETSc
from src.geo import *
from src import stokes_flow as sf

__all__ = ['createEcoli_ellipse', 'createEcoli_tunnel', 'capsule']


def createEcoli_ellipse(name='...', **kwargs):
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
    zoom_factor = kwargs['zoom_factor'] if 'zoom_factor' in kwargs.keys( ) else 1
    dist_hs = kwargs['dist_hs']
    matrix_method = kwargs['matrix_method']
    lh = ph * ch  # length of helix
    movesz = 0.5 * (dist_hs - 2 * rs1 + lh) + rs1
    movehz = 0.5 * (dist_hs + 2 * rs1 - lh) + lh / 2
    moves = np.array((0, 0, movesz))  # move distance of sphere
    moveh = np.array((0, 0, -movehz))  # move distance of helix
    objtype = sf.obj_dic[matrix_method]

    # create helix
    B = ph / (2 * np.pi)
    vhgeo0 = supHelix( )  # velocity node geo of helix
    dth = 2 * np.pi / nth
    fhgeo0 = vhgeo0.create_deltatheta(dth=dth, radius=rh2, R=rh1, B=B, n_c=ch, epsilon=eh, with_cover=with_cover,
                                      factor=hfct, left_hand=left_hand)
    vhobj0 = objtype( )
    vhobj0.set_data(fhgeo0, vhgeo0, name='helix_0')
    vhobj0.zoom(zoom_factor)
    vhobj0.move(moveh * zoom_factor)
    vhobj1 = vhobj0.copy( )
    vhobj1.node_rotation(norm=(0, 0, 1), theta=np.pi, rotation_origin=(0, 0, 0))
    vhobj1.set_name('helix_1')

    # create sphere
    vsgeo = ellipse_geo( )  # velocity node geo of sphere
    vsgeo.create_delta(ds, rs1, rs2)
    vsgeo.node_rotation(norm=np.array((0, 1, 0)), theta=np.pi / 2)
    fsgeo = vsgeo.copy( )  # force node geo of sphere
    fsgeo.node_zoom(1 + ds / (0.5 * (rs1 + rs2)) * es)
    vsobj = objtype( )
    vsobj.set_data(fsgeo, vsgeo, name='sphere_0')
    vsobj.zoom(zoom_factor)
    vsobj.move(moves * zoom_factor)

    center = kwargs['center']
    rel_Us = kwargs['rel_Us']
    rel_Uh = kwargs['rel_Uh']
    ecoli_comp = sf.forceFreeComposite(center, name)
    ecoli_comp.add_obj(vsobj, rel_U=rel_Us)
    ecoli_comp.add_obj(vhobj0, rel_U=rel_Uh)
    ecoli_comp.add_obj(vhobj1, rel_U=rel_Uh)
    return ecoli_comp

def capsule(rs1, rs2, ls, ds):
    lvs3 = ls - 2 * rs2
    dth = ds / rs2
    err_msg = 'geo parameter of capsule head is wrong. '
    assert lvs3 >= 0, err_msg

    vsgeo1 = ellipse_geo( )  # velocity node geo of head
    vsgeo1.create_half_delta(ds, rs1, rs2)
    vsgeo2 = vsgeo1.copy( )
    vsgeo1.node_rotation(norm=np.array((0, 1, 0)), theta=-np.pi / 2)
    vsgeo1.node_rotation(norm=np.array((0, 0, 1)), theta=-np.pi / 2)
    vsgeo1.move((0, 0, -lvs3 / 2))
    vsgeo2.node_rotation(norm=np.array((0, 1, 0)), theta=+np.pi / 2)
    vsgeo2.node_rotation(norm=np.array((0, 0, 1)), theta=+np.pi / 2 - dth)
    vsgeo2.move((0, 0, +lvs3 / 2))
    vsgeo2.set_nodes(np.flipud(vsgeo2.get_nodes( )), deltalength=vsgeo2.get_deltaLength( ))
    vsgeo3 = tunnel_geo( )
    vsgeo3.create_deltatheta(dth=dth, radius=rs2, length=lvs3)
    vsgeo = geo( )
    vsgeo.combine([vsgeo1, vsgeo3, vsgeo2])
    return vsgeo

def createEcoli_tunnel(name='...', **kwargs):
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
    rT1 = kwargs['rT1']
    rT2 = kwargs['rT2']
    ntT = kwargs['ntT']
    eT = kwargs['eT']
    matrix_method = kwargs['matrix_method']
    with_T_geo = kwargs['with_T_geo']
    lh = ph * ch  # length of helix
    movesz = 0.5 * (dist_hs - ls + lh) + ls / 2
    movehz = 0.5 * (dist_hs + ls - lh) + lh / 2
    moves = np.array((0, 0, movesz))  # move distance of sphere
    moveh = np.array((rT1 - rh1, 0, -movehz))  # move distance of helix
    lT = (rT1 + rh2) * 2
    objtype = sf.obj_dic[matrix_method]

    # create helix
    B = ph / (2 * np.pi)
    vhgeo0 = supHelix( )  # velocity node geo of helix
    dth = 2 * np.pi / nth
    fhgeo0 = vhgeo0.create_deltatheta(dth=dth, radius=rh2, R=rh1, B=B, n_c=ch, epsilon=eh, with_cover=with_cover,
                                      factor=hfct, left_hand=left_hand)
    vhobj0 = objtype( )
    vhobj0.set_data(fhgeo0, vhgeo0, name='helix_0')
    vhobj0.zoom(zoom_factor)
    # dbg
    OptDB = PETSc.Options( )
    factor = OptDB.getReal('dbg_theta_factor', 0.5)
    PETSc.Sys.Print('--------------------> DBG: theta factor = %f' % factor)
    theta = np.pi * ch + (rT2 + rh2 * factor) / (rh1 + rh2)
    vhobj0.node_rotation(norm=np.array((0, 0, 1)), theta=theta)
    vhobj0.move(moveh * zoom_factor)
    vhobj1 = vhobj0.copy( )
    vhobj1.node_rotation(norm=(0, 0, 1), theta=np.pi, rotation_origin=(0, 0, 0))
    vhobj1.set_name('helix_1')

    # create head
    vsgeo = capsule(rs1, rs2, ls, ds)
    fsgeo = vsgeo.copy( )  # force node geo of sphere
    fsgeo.node_zoom(1 + ds / (0.5 * (rs1 + rs2)) * es)
    fsgeo.node_zoom_z(1 - ds / (0.5 * (rs1 + rs2)) * es)
    vsobj = objtype( )
    vsobj.set_data(fsgeo, vsgeo, name='sphere_0')
    vsobj.zoom(zoom_factor)
    vsobj.move(moves * zoom_factor)

    # create T shape
    dtT = 2 * np.pi / ntT
    # dbg
    OptDB = PETSc.Options( )
    moveT = np.array((0, 0, moveh[-1] + lh / 2))
    vTgeo = tunnel_geo( )
    fTgeo = vTgeo.create_deltatheta(dth=dtT, radius=rT2, length=lT, epsilon=eT, with_cover=True)
    vTobj = objtype( )
    vTobj.set_data(fTgeo, vTgeo, name='T_shape_0')
    theta = -np.pi / 2
    vTobj.node_rotation(norm=np.array((0, 1, 0)), theta=theta)
    vTobj.zoom(zoom_factor)
    vTobj.move(moveT * zoom_factor)

    vsobj.node_rotation(norm=np.array((0, 0, 1)), theta=np.pi / 4)
    vhobj0.node_rotation(norm=np.array((0, 0, 1)), theta=np.pi / 4)
    vhobj1.node_rotation(norm=np.array((0, 0, 1)), theta=np.pi / 4)
    vTobj.node_rotation(norm=np.array((0, 0, 1)), theta=np.pi / 4)

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

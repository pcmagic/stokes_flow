import numpy as np
from petsc4py import PETSc
from src.geo import *
from src import stokes_flow as sf
from src.support_class import *
from src.StokesFlowMethod import *

__all__ = ['createEcoli_ellipse', 'createEcoliComp_ellipse', 'createEcoli_2tails',
           'createEcoliComp_tunnel', 'createEcoli_tunnel', 'create_ecoli_dualTail',
           'create_ecoli_2part', 'create_ecoli_tail', 'create_ecoli_tail_at',
           'create_rotlets_tail_2part', 'create_selfRepeat_tail',
           'create_ecoli_2part_at', 'create_ecoli_dualTail_at',
           'get_tail_nodes_split_at', 'get_ecoli_nodes_split_at',
           'create_capsule',
           'create_rod',
           'create_infHelix',
           'create_helicoid_list', 'create_helicoid_comp',
           'obj2helicoid_list', 'obj2helicoid_comp',
           'create_sphere', 'create_move_single_sphere', 'create_one_ellipse']


def create_capsule(rs1, rs2, ls, ds, node_dof=3):
    lvs3 = ls - 2 * rs2
    dth = ds / rs2
    err_msg = 'geo parameter of create_capsule head is wrong. '
    assert lvs3 >= 0, err_msg
    vsgeo = base_geo()
    vsgeo.set_dof(node_dof)

    vsgeo1 = ellipse_base_geo()  # velocity node geo of head
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


def create_ecoli_tail(moveh, **kwargs):
    nth = kwargs['nth']
    hfct = kwargs['hfct']
    eh = kwargs['eh']
    ch = kwargs['ch']
    rh11 = kwargs['rh11']
    rh12 = kwargs['rh12']
    rh2 = kwargs['rh2']
    ph = kwargs['ph']
    n_tail = kwargs['n_tail']
    with_cover = kwargs['with_cover']
    with_T_geo = kwargs['with_T_geo'] if 'with_T_geo' in kwargs.keys() else 0
    left_hand = kwargs['left_hand']
    rT2 = kwargs['rT2']
    center = kwargs['center']
    matrix_method = kwargs['matrix_method']
    zoom_factor = kwargs['zoom_factor']
    obj_type = sf.obj_dic[matrix_method]

    # create helix
    vhobj0 = obj_type()
    node_dof = vhobj0.get_n_unknown()
    B = ph / (2 * np.pi)
    vhgeo0 = FatHelix()  # velocity node geo of helix
    if 'dualPotential' in matrix_method:
        vhgeo0.set_check_epsilon(False)
    vhgeo0.set_dof(node_dof)
    dth = 2 * np.pi / nth
    fhgeo0 = vhgeo0.create_deltatheta(dth=dth, radius=rh2, R1=rh11, R2=rh12, B=B, n_c=ch,
                                      epsilon=eh, with_cover=with_cover, factor=hfct,
                                      left_hand=left_hand)
    vhobj0.set_data(fhgeo0, vhgeo0, name='helix_0')
    vhobj0.zoom(zoom_factor)
    if with_T_geo:
        # dbg
        OptDB = PETSc.Options()
        factor = OptDB.getReal('dbg_theta_factor', 1.5)
        PETSc.Sys.Print('--------------------> DBG: dbg_theta_factor = %f' % factor)
        theta = np.pi * ch + (rT2 + rh2 * factor) / (rh11 + rh2)
        vhobj0.node_rotation(norm=np.array((0, 0, 1)), theta=theta)
    vhobj0.move(moveh * zoom_factor)

    tail_list = uniqueList()
    for i0 in range(n_tail):
        theta = 2 * np.pi / n_tail * i0
        vhobj1 = vhobj0.copy()
        vhobj1.node_rotation(norm=(0, 0, 1), theta=theta, rotation_origin=center.copy())
        vhobj1.set_name('helix_%d' % i0)
        tail_list.append(vhobj1)

    return tail_list


def create_ecoli_tail_bck(moveh, **kwargs):
    nth = kwargs['nth']
    hfct = kwargs['hfct']
    eh = kwargs['eh']
    ch = kwargs['ch']
    rh11 = kwargs['rh11']
    rh12 = kwargs['rh12']
    rh2 = kwargs['rh2']
    ph = kwargs['ph']
    n_tail = kwargs['n_tail']
    with_cover = kwargs['with_cover']
    left_hand = kwargs['left_hand']
    rT2 = kwargs['rT2']
    center = kwargs['center']
    matrix_method = kwargs['matrix_method']
    zoom_factor = kwargs['zoom_factor']
    obj_type = sf.obj_dic[matrix_method]

    # create helix
    vhobj0 = obj_type()
    node_dof = vhobj0.get_n_unknown()
    B = ph / (2 * np.pi)
    vhgeo0 = FatHelix()  # velocity node geo of helix
    if 'dualPotential' in matrix_method:
        vhgeo0.set_check_epsilon(False)
    vhgeo0.set_dof(node_dof)
    dth = 2 * np.pi / nth
    fhgeo0 = vhgeo0.create_deltatheta(dth=dth, radius=rh2, R1=rh11, R2=rh12, B=B, n_c=ch,
                                      epsilon=eh, with_cover=with_cover, factor=hfct,
                                      left_hand=left_hand)
    vhobj0.set_data(fhgeo0, vhgeo0, name='helix_0')
    vhobj0.zoom(zoom_factor)
    # dbg
    OptDB = PETSc.Options()
    factor = OptDB.getReal('dbg_theta_factor', 1.5)
    PETSc.Sys.Print('--------------------> DBG: dbg_theta_factor = %f' % factor)
    theta = np.pi * ch + (rT2 + rh2 * factor) / (rh11 + rh2)
    vhobj0.node_rotation(norm=np.array((0, 0, 1)), theta=theta)
    vhobj0.move(moveh * zoom_factor)

    tail_list = uniqueList()
    for i0 in range(n_tail):
        theta = 2 * np.pi / n_tail * i0
        vhobj1 = vhobj0.copy()
        vhobj1.node_rotation(norm=(0, 0, 1), theta=theta, rotation_origin=center.copy())
        vhobj1.set_name('helix_%d' % i0)
        tail_list.append(vhobj1)

    return tail_list


def create_selfRepeat_tail(moveh, **kwargs):
    nth = kwargs['nth']
    hfct = kwargs['hfct']
    eh = kwargs['eh']
    ch = kwargs['ch']
    rh11 = kwargs['rh11']
    rh12 = kwargs['rh12']
    rh2 = kwargs['rh2']
    ph = kwargs['ph']
    n_tail = kwargs['n_tail']
    with_cover = kwargs['with_cover']
    with_T_geo = kwargs['with_T_geo'] if 'with_T_geo' in kwargs.keys() else 0
    left_hand = kwargs['left_hand']
    rT2 = kwargs['rT2']
    repeat_n = kwargs['repeat_n']
    center = kwargs['center']
    matrix_method = kwargs['matrix_method']
    zoom_factor = kwargs['zoom_factor']
    obj_type = sf.obj_dic[matrix_method]

    # create helix
    vhobj0 = obj_type()  # type: sf.StokesFlowObj
    node_dof = vhobj0.get_n_unknown()
    B = ph / (2 * np.pi)
    vhgeo0 = SelfRepeat_FatHelix(repeat_n)  # velocity node geo of helix
    if 'dualPotential' in matrix_method:
        vhgeo0.set_check_epsilon(False)
    vhgeo0.set_dof(node_dof)
    dth = 2 * np.pi / nth
    fhgeo0 = vhgeo0.create_deltatheta(dth=dth, radius=rh2, R1=rh11, R2=rh12, B=B, n_c=ch,
                                      epsilon=eh, with_cover=with_cover, factor=hfct,
                                      left_hand=left_hand)  # type: SelfRepeat_FatHelix
    vhobj0.set_data(fhgeo0, vhgeo0, name='helix_0')
    vhobj0.zoom(zoom_factor)
    if with_T_geo:
        # dbg
        OptDB = PETSc.Options()
        factor = OptDB.getReal('dbg_theta_factor', 1.5)
        PETSc.Sys.Print('--------------------> DBG: dbg_theta_factor = %f' % factor)
        theta = np.pi * ch + (rT2 + rh2 * factor) / (rh11 + rh2)
        vhobj0.node_rotation(norm=np.array((0, 0, 1)), theta=theta)
    vhobj0.move(moveh * zoom_factor)

    tail_list = uniqueList()
    for i0 in range(n_tail):
        theta = 2 * np.pi / n_tail * i0
        vhobj1 = vhobj0.copy()
        vhobj1.node_rotation(norm=(0, 0, 1), theta=theta, rotation_origin=center.copy())
        vhobj1.set_name('helix_%d' % i0)
        tail_list.append(vhobj1)

    tail_start_list = []
    tail_body0_list = []
    tail_end_list = []
    for tobj in tail_list:
        vhgeo0 = tobj.get_u_geo()
        fhgeo0 = tobj.get_f_geo()
        #
        part_obj = obj_type()
        part_ugeo = vhgeo0.get_start_geo()
        part_fgeo = fhgeo0.get_start_geo()
        part_obj.set_data(part_fgeo, part_ugeo, name='helix_0_start')
        tail_start_list.append(part_obj)
        #
        part_obj = sf.SelfRepeatObj()
        part_ugeo = vhgeo0.get_body_mid_geo()
        part_fgeo = fhgeo0.get_body_mid_geo()
        part_obj.set_data(part_fgeo, part_ugeo, name='helix_0_body0')
        tail_body0_list.append(part_obj)
        #
        part_obj = obj_type()
        part_ugeo = vhgeo0.get_end_geo()
        part_fgeo = fhgeo0.get_end_geo()
        part_obj.set_data(part_fgeo, part_ugeo, name='helix_0_end')
        tail_end_list.append(part_obj)

    return tail_list, tail_start_list, tail_body0_list, tail_end_list


def create_ecoli_tail_at(theta, phi, psi_tail, now_center=np.zeros(3), **problem_kwargs):
    tail_list = create_ecoli_tail(np.zeros(3), **problem_kwargs)
    tail_obj = sf.StokesFlowObj()
    tail_obj.set_name('tail_obj')
    tail_obj.combine(tail_list)
    tail_obj.node_rotation(np.array((0, 1, 0)), theta)
    tail_obj.node_rotation(np.array((0, 0, 1)), phi)
    tail_obj.node_rotation(tail_obj.get_u_geo().get_geo_norm(), psi_tail)
    tail_obj.move(now_center)
    return tail_obj


def get_tail_nodes_split_at(theta, phi, psi_tail, now_center=np.zeros(3), **problem_kwargs):
    tail_list = create_ecoli_tail(np.zeros(3), **problem_kwargs)
    tail_obj = sf.StokesFlowObj()
    tail_obj.set_name('tail_obj')
    tail_obj.combine(tail_list)
    tail_obj.node_rotation(np.array((0, 1, 0)), theta)
    tail_obj.node_rotation(np.array((0, 0, 1)), phi)
    tail_obj.node_rotation(tail_obj.get_u_geo().get_geo_norm(), psi_tail)
    tail_obj.move(now_center)

    n_tail = problem_kwargs['n_tail']
    t0 = np.split(tail_obj.get_u_nodes(), 2 * n_tail)
    t1 = np.vstack(t0[1::2])
    t2 = np.vstack(t0[0::2])
    return t1, t2


def createEcoli_ellipse(name='...', **kwargs):
    ch = kwargs['ch']
    ph = kwargs['ph']
    ds = kwargs['ds']
    rs1 = kwargs['rs1']
    rs2 = kwargs['rs2']
    es = kwargs['es']
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

    # create tail
    tail_list = create_ecoli_tail(moveh, **kwargs)

    # create head
    vsgeo = ellipse_base_geo()  # velocity node geo of sphere
    vsgeo.create_delta(ds, rs1, rs2)
    vsgeo.node_rotation(norm=np.array((0, 1, 0)), theta=-np.pi / 2)
    fsgeo = vsgeo.copy()  # force node geo of sphere
    fsgeo.node_zoom(1 + ds / (0.5 * (rs1 + rs2)) * es)
    vsobj = objtype()
    vsobj.set_data(fsgeo, vsgeo, name='sphere_0')
    vsobj.zoom(zoom_factor)
    vsobj.move(moves * zoom_factor)
    return vsobj, tail_list


def createEcoli_2tails(name='...', **kwargs):
    ch = kwargs['ch']
    ph = kwargs['ph']
    ds = kwargs['ds']
    rs1 = kwargs['rs1']
    rs2 = kwargs['rs2']
    es = kwargs['es']
    # sphere_rotation = kwargs['sphere_rotation'] if 'sphere_rotation' in kwargs.keys() else 0
    zoom_factor = kwargs['zoom_factor'] if 'zoom_factor' in kwargs.keys() else 1
    dist_hs = kwargs['dist_hs']
    center = kwargs['center']
    matrix_method = kwargs['matrix_method']
    lh = ph * ch  # length of helix
    objtype = sf.obj_dic[matrix_method]

    # create tail
    movez = np.array((0, 0, rs1 + dist_hs + lh / 2))
    tkwargs = kwargs.copy()
    tkwargs['left_hand'] = False
    tail_list1 = create_ecoli_tail(-movez, **tkwargs)
    tkwargs['left_hand'] = True
    tail_list2 = create_ecoli_tail(movez, **tkwargs)

    # create head
    vsgeo = ellipse_base_geo()  # velocity node geo of sphere
    vsgeo.create_delta(ds, rs1, rs2)
    vsgeo.node_rotation(norm=np.array((0, 1, 0)), theta=-np.pi / 2)
    fsgeo = vsgeo.copy()  # force node geo of sphere
    fsgeo.node_zoom(1 + ds / (0.5 * (rs1 + rs2)) * es)
    vsobj = objtype()
    vsobj.set_data(fsgeo, vsgeo, name='sphere_0')
    vsobj.zoom(zoom_factor)
    return vsobj, tail_list1, tail_list2


def createEcoliComp_ellipse(name='...', **kwargs):
    vsobj, tail_list = createEcoli_ellipse(name=name, **kwargs)
    vsgeo = vsobj.get_u_geo()
    center = kwargs['center']
    rel_Us = kwargs['rel_Us']
    rel_Uh = kwargs['rel_Uh']

    ecoli_comp = sf.ForceFreeComposite(center=center.copy(), norm=vsgeo.get_geo_norm().copy(),
                                       name=name)
    ecoli_comp.add_obj(vsobj, rel_U=rel_Us)
    for ti in tail_list:
        ecoli_comp.add_obj(ti, rel_U=rel_Uh)

    rot_norm = kwargs['rot_norm']
    rot_theta = kwargs['rot_theta'] * np.pi
    ecoli_comp.node_rotation(norm=rot_norm.copy(), theta=rot_theta, rotation_origin=center.copy())
    return ecoli_comp


def createEcoli_tunnel(**kwargs):
    ch = kwargs['ch']
    rh1 = kwargs['rh1']
    rh2 = kwargs['rh2']
    ph = kwargs['ph']
    ds = kwargs['ds']
    rs1 = kwargs['rs1']
    rs2 = kwargs['rs2']
    ls = kwargs['ls']
    es = kwargs['es']
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
    movehz = -1 * (0.5 * (dist_hs + ls - lh) + lh / 2)
    # movesz = (ls + dist_hs) / 2
    # movehz = (lh + dist_hs) / 2
    moves = np.array((0, 0, movesz)) + center  # move distance of sphere
    moveh = np.array((rT1 - rh1, 0, movehz)) + center  # move distance of helix
    lT = (rT1 + rh2) * 2
    objtype = sf.obj_dic[matrix_method]

    # create helix
    tail_list = create_ecoli_tail(moveh, **kwargs)

    # create head
    vsobj = objtype()
    node_dof = vsobj.get_n_unknown()
    vsgeo = create_capsule(rs1, rs2, ls, ds, node_dof)
    fsgeo = vsgeo.copy()  # force node geo of sphere
    fsgeo.node_zoom(1 + ds / (0.5 * (rs1 + rs2)) * es)
    fsgeo.node_zoom_z(1 - ds / (0.5 * (rs1 + rs2)) * es)
    vsobj.set_data(fsgeo, vsgeo, name='sphere_0')
    vsobj.zoom(zoom_factor)
    vsobj.move(moves * zoom_factor)

    # create T shape
    dtT = 2 * np.pi / ntT
    vTobj = objtype()
    node_dof = vTobj.get_n_unknown()
    # # dbg
    # OptDB = PETSc.Options( )
    # factor = OptDB.getReal('dbg_move_factor', 1)
    # PETSc.Sys.Print('--------------------> DBG: dbg_move_factor = %f' % factor)
    # moveT = np.array((0, 0, moveh[-1] + lh / 2 + rh2 * factor))
    moveT = np.array((0, 0, movehz + lh / 2)) + center
    vTgeo = tunnel_geo()
    if 'dualPotential' in matrix_method:
        vTgeo.set_check_epsilon(False)
    vTgeo.set_dof(node_dof)
    fTgeo = vTgeo.create_deltatheta(dth=dtT, radius=rT2, factor=Tfct, length=lT, epsilon=eT,
                                    with_cover=1)
    vTobj.set_data(fTgeo, vTgeo, name='T_shape_0')
    theta = -np.pi / 2
    vTobj.node_rotation(norm=np.array((0, 1, 0)), theta=theta)
    vTobj.zoom(zoom_factor)
    vTobj.move(moveT * zoom_factor)

    theta = np.pi / 4 - ch * np.pi
    vsobj.node_rotation(norm=np.array((0, 0, 1)), theta=theta, rotation_origin=center)
    for ti in tail_list:
        ti.node_rotation(norm=np.array((0, 0, 1)), theta=theta, rotation_origin=center)
    vTobj.node_rotation(norm=np.array((0, 0, 1)), theta=theta, rotation_origin=center)
    return vsobj, tail_list, vTobj


def createEcoliComp_tunnel(name='...', **kwargs):
    with_T_geo = kwargs['with_T_geo'] if 'with_T_geo' in kwargs.keys() else 0
    center = kwargs['center']
    rel_Us = kwargs['rel_Us']
    rel_Uh = kwargs['rel_Uh']

    if not with_T_geo:
        kwargs['rT1'] = kwargs['rh1']
    vsobj, tail_list, vTobj = createEcoli_tunnel(**kwargs)
    ecoli_comp = sf.ForceFreeComposite(center, norm=vsobj.get_u_geo().get_geo_norm(), name=name)
    ecoli_comp.add_obj(vsobj, rel_U=rel_Us)
    for ti in tail_list:
        ecoli_comp.add_obj(ti, rel_U=rel_Uh)
    if with_T_geo:
        ecoli_comp.add_obj(vTobj, rel_U=rel_Uh)
    return ecoli_comp


def create_ecoli_2part(**problem_kwargs):
    # create a ecoli contain two parts, one is head and one is tail.
    rel_Us = problem_kwargs['rel_Us']
    rel_Uh = problem_kwargs['rel_Uh']
    update_order = problem_kwargs['update_order'] if 'update_order' in problem_kwargs.keys() else 1
    update_fun = problem_kwargs[
        'update_fun'] if 'update_fun' in problem_kwargs.keys() else Adams_Bashforth_Methods
    with_T_geo = problem_kwargs['with_T_geo']
    err_msg = 'currently, do not support with_T_geo for this kind of ecoli. '
    assert not with_T_geo, err_msg

    head_obj, tail_obj_list = createEcoli_ellipse(name='ecoli0', **problem_kwargs)
    head_obj.set_name('head_obj')
    tail_obj = sf.StokesFlowObj()
    tail_obj.set_name('tail_obj')
    tail_obj.combine(tail_obj_list)
    head_geo = head_obj.get_u_geo()
    # ecoli_comp = sf.ForceFreeComposite(center=head_geo.get_center(), norm=head_geo.get_geo_norm(), name='ecoli_0')
    ecoli_comp = sf.ForceFreeComposite(center=np.zeros(3), norm=head_geo.get_geo_norm(),
                                       name='ecoli_0')
    ecoli_comp.add_obj(obj=head_obj, rel_U=rel_Us)
    ecoli_comp.add_obj(obj=tail_obj, rel_U=rel_Uh)
    ecoli_comp.set_update_para(fix_x=False, fix_y=False, fix_z=False,
                               update_fun=update_fun, update_order=update_order)
    return ecoli_comp


def create_rotlets_tail_2part(rotlet_strength=0, **problem_kwargs):
    # create a swimmer with a infinite small head (the limit is a rotlet) and tail(s).
    ch = problem_kwargs['ch']
    ph = problem_kwargs['ph']
    dist_hs = problem_kwargs['dist_hs']
    lh = ph * ch  # length of helix
    with_T_geo = problem_kwargs['with_T_geo']
    err_msg = 'currently, do not support with_T_geo for this kind of ecoli. '
    assert not with_T_geo, err_msg

    tail_list = create_ecoli_tail(np.zeros(3), **problem_kwargs)
    tail_obj0 = sf.StokesFlowObj()
    tail_obj0.combine(tail_list)
    tail_obj = sf.FundSoltObj()
    tail_obj.set_data(tail_obj0.get_u_geo(), tail_obj0.get_f_geo(), name='rotlets_tail_obj')
    location = np.array((0, 0, lh / 2 + dist_hs))
    tnorm = tail_obj0.get_u_geo().get_geo_norm()
    torque = tnorm * rotlet_strength
    tail_obj.add_point_force(location=location, force=torque,
                             StokesletsHandle=light_rotlets_matrix_3d)
    givenT = np.hstack((np.zeros(3), -1 * torque))

    ecoli_comp = sf.GivenForceComposite(center=np.zeros(3), norm=tnorm,
                                        name='rotlets_tail_comp', givenF=givenT)
    ecoli_comp.add_obj(obj=tail_obj, rel_U=np.zeros(6))
    update_order = problem_kwargs['update_order'] \
        if 'update_order' in problem_kwargs.keys() \
        else 1
    update_fun = problem_kwargs['update_fun'] \
        if 'update_fun' in problem_kwargs.keys() \
        else Adams_Bashforth_Methods
    ecoli_comp.set_update_para(fix_x=False, fix_y=False, fix_z=False,
                               update_fun=update_fun, update_order=update_order)
    return ecoli_comp


def create_ecoli_2part_at(theta, phi, psi_tail, now_center=np.zeros(3), **problem_kwargs):
    ecoli_comp = create_ecoli_2part(**problem_kwargs)
    ecoli_comp.node_rotation(np.array((0, 1, 0)), theta)
    ecoli_comp.node_rotation(np.array((0, 0, 1)), phi)
    tail_obj = ecoli_comp.get_obj_list()[1]
    tail_obj.node_rotation(tail_obj.get_u_geo().get_geo_norm(), psi_tail)
    ecoli_comp.move(now_center)
    return ecoli_comp


def get_ecoli_nodes_split_at(theta, phi, psi_tail, now_center=np.zeros(3), **problem_kwargs):
    ecoli_comp = create_ecoli_2part(**problem_kwargs)
    ecoli_comp.node_rotation(np.array((0, 1, 0)), theta)
    ecoli_comp.node_rotation(np.array((0, 0, 1)), phi)
    tail_obj = ecoli_comp.get_obj_list()[1]
    tail_obj.node_rotation(tail_obj.get_u_geo().get_geo_norm(), psi_tail)
    ecoli_comp.move(now_center)

    n_tail = problem_kwargs['n_tail']
    t0 = np.split(tail_obj.get_u_nodes(), 2 * n_tail)
    t1 = np.vstack(t0[1::2])
    t1 = np.vstack((ecoli_comp.get_obj_list()[0].get_u_nodes(), t1))
    t2 = np.vstack(t0[0::2])
    return t1, t2


def create_ecoli_dualTail(**problem_kwargs):
    # create a swimmer with two tails in the ends. one is left hand and one is right hand.
    #   the swimmer contain three parts, i.e. head, upper tail and down tail.
    rel_Us = problem_kwargs['rel_Us']
    rel_Uh = problem_kwargs['rel_Uh']
    update_order = problem_kwargs['update_order'] if 'update_order' in problem_kwargs.keys() else 1
    update_fun = problem_kwargs['update_fun'] if 'update_fun' in problem_kwargs.keys() \
        else Adams_Bashforth_Methods
    with_T_geo = problem_kwargs['with_T_geo']
    err_msg = 'currently, do not support with_T_geo for this kind of ecoli. '
    assert not with_T_geo, err_msg

    head_obj, tail_obj_l1, tail_obj_l2 = createEcoli_2tails(name='ecoli0', **problem_kwargs)
    head_obj.set_name('head_obj')
    tail_obj1 = sf.StokesFlowObj()
    tail_obj1.set_name('tail_obj1')
    tail_obj1.combine(tail_obj_l1)
    tail_obj2 = sf.StokesFlowObj()
    tail_obj2.set_name('tail_obj2')
    tail_obj2.combine(tail_obj_l2)
    head_geo = head_obj.get_u_geo()
    tnorm = head_geo.get_geo_norm()
    ecoli_comp = sf.ForceFreeComposite(center=np.zeros(3), norm=tnorm, name='ecoli_0')
    ecoli_comp.add_obj(obj=head_obj, rel_U=rel_Us)
    ecoli_comp.add_obj(obj=tail_obj1, rel_U=rel_Uh)
    ecoli_comp.add_obj(obj=tail_obj2, rel_U=-rel_Uh)
    ecoli_comp.set_update_para(fix_x=False, fix_y=False, fix_z=False,
                               update_fun=update_fun, update_order=update_order)
    return ecoli_comp


def create_ecoli_dualTail_at(theta, phi, psi_tail1, psi_tail2, center=np.zeros(3),
                             **problem_kwargs):
    assert 1 == 2
    ecoli_comp = create_ecoli_dualTail(**problem_kwargs)
    # ecoli_comp.node_rotation(np.array((0, 1, 0)), theta)
    # ecoli_comp.node_rotation(np.array((0, 0, 1)), phi)
    # tail_obj1 = ecoli_comp.get_obj_list()[1]
    # tail_obj1.node_rotation(tail_obj1.get_u_geo().get_geo_norm(), psi_tail1)
    # tail_obj2 = ecoli_comp.get_obj_list()[2]
    # tail_obj2.node_rotation(tail_obj2.get_u_geo().get_geo_norm(), psi_tail2)
    return ecoli_comp


def create_sphere(namehandle='sphereObj', **kwargs):
    matrix_method = kwargs['matrix_method']
    rs = kwargs['rs']
    sphere_velocity = kwargs['sphere_velocity']
    ds = kwargs['ds']
    es = kwargs['es']
    sphere_coord = kwargs['sphere_coord']
    objtype = sf.obj_dic[matrix_method]

    obj_sphere = objtype()
    sphere_geo0 = sphere_geo()  # force geo
    sphere_geo0.set_dof(obj_sphere.get_n_unknown())
    sphere_geo0.create_delta(ds, rs)
    sphere_geo0.set_rigid_velocity([0, 0, 0, 0, 0, 0])
    sphere_geo1 = sphere_geo0.copy()
    if 'pf' in matrix_method:
        sphere_geo1.node_zoom((rs + ds * es) / rs)
    obj_sphere.set_data(sphere_geo1, sphere_geo0)

    obj_list = []
    for i0, (t_coord, t_velocity) in enumerate(zip(sphere_coord, sphere_velocity)):
        obj2 = obj_sphere.copy()
        obj2.set_name('%s_%d' % (namehandle, i0))
        obj2.move(t_coord)
        obj2.get_u_geo().set_rigid_velocity(t_velocity)
        obj_list.append(obj2)
    return obj_list


def create_one_ellipse(namehandle='sphereObj', **kwargs):
    matrix_method = kwargs['matrix_method']
    rs1 = kwargs['rs1']
    rs2 = kwargs['rs2']
    sphere_velocity = kwargs['sphere_velocity']
    ds = kwargs['ds']
    es = kwargs['es']
    sphere_coord = kwargs['sphere_coord']
    objtype = sf.obj_dic[matrix_method]

    obj_sphere = objtype()  # type: sf.StokesFlowObj
    sphere_geo0 = ellipse_base_geo()  # force geo
    sphere_geo0.set_dof(obj_sphere.get_n_unknown())
    sphere_geo0.create_delta(ds, rs1, rs2)
    sphere_geo0.set_rigid_velocity(sphere_velocity)
    sphere_geo1 = sphere_geo0.copy()
    if 'pf' in matrix_method:
        sphere_geo1.node_zoom(1 + ds / (0.5 * (rs1 + rs2)) * es)
    obj_sphere.set_data(sphere_geo1, sphere_geo0, name=namehandle)
    obj_sphere.move(sphere_coord)
    return obj_sphere


def create_move_single_sphere(namehandle='sphereObj', **kwargs):
    movez = kwargs['movez']
    obj_sphere = create_sphere(namehandle, **kwargs)[0]
    displacement = np.array((0, 0, movez))
    obj_sphere.move(displacement)
    obj_list = (obj_sphere,)
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
                              with_cover=1, factor=Rodfct, left_hand=False)
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
    rod_comp = sf.GivenForceComposite(center=RodCenter, name=name, givenF=givenF.copy())
    rod_comp.add_obj(obj=rod_obj, rel_U=rel_URod)

    rod_list = (rod_comp,)
    return rod_list


def create_infHelix(namehandle='infhelix', normalize=False, **problem_kwargs):
    n_tail = problem_kwargs['n_tail']
    eh = problem_kwargs['eh']
    ch = problem_kwargs['ch']
    rh1 = problem_kwargs['rh1']
    rh2 = problem_kwargs['rh2']
    ph = problem_kwargs['ph']
    nth = problem_kwargs['nth']
    zoom_factor = problem_kwargs['zoom_factor']

    if normalize:
        rh2 = rh2 * zoom_factor
        ph = ph * zoom_factor
        rh1 = rh1 * zoom_factor

    helix_list = []
    for i0, theta0 in enumerate(np.linspace(0, 2 * np.pi, n_tail, endpoint=False)):
        infhelix_ugeo = infHelix()
        infhelix_ugeo.create_n(rh1, rh2, ph, ch, nth, theta0=theta0)
        infhelix_fgeo = infhelix_ugeo.create_fgeo(epsilon=eh)
        infhelix_obj = sf.StokesFlowObj()
        infhelix_obj.set_data(f_geo=infhelix_fgeo, u_geo=infhelix_ugeo,
                              name=namehandle + '%02d' % i0)
        helix_list.append(infhelix_obj)
    return helix_list


def create_helicoid_list(namehandle='helicoid', **problem_kwargs):
    r1 = problem_kwargs['helicoid_r1']
    r2 = problem_kwargs['helicoid_r2']
    ds = problem_kwargs['helicoid_ds']
    th_loc = problem_kwargs['helicoid_th_loc']
    ndsk_each = problem_kwargs['helicoid_ndsk_each']
    matrix_method = problem_kwargs['matrix_method']
    assert matrix_method in ('rs', 'lg_rs')

    tgeo = regularizeDisk()
    tgeo.create_ds(ds, r2)
    tgeo.node_rotation(norm=np.array([1, 0, 0]), theta=th_loc)
    tgeo.move(np.array((r1, 0, 0)))
    # tgeo.show_nodes()

    tgeo_list = []
    rot_dth = 2 * np.pi / ndsk_each
    for i0 in range(ndsk_each):
        rot_th = i0 * rot_dth + rot_dth / 2
        tgeo21 = tgeo.copy()
        tgeo21.node_rotation(norm=np.array([0, 0, 1]), theta=rot_th, rotation_origin=np.zeros(3))
        tgeo22 = tgeo21.copy()
        tgeo_list.append(tgeo21)
        tgeo22.node_rotation(norm=np.array([1, 0, 0]), theta=np.pi / 2, rotation_origin=np.zeros(3))
        tgeo23 = tgeo21.copy()
        tgeo_list.append(tgeo22)
        tgeo23.node_rotation(norm=np.array([0, 1, 0]), theta=np.pi / 2, rotation_origin=np.zeros(3))
        tgeo_list.append(tgeo23)
    # tgeo3 = base_geo()
    # tgeo3.combine(tgeo_list)
    # tgeo3.show_nodes(linestyle='')

    tobj_list = []
    for i0, tgeo in enumerate(tgeo_list):
        tobj = sf.StokesFlowObj()
        tobj.set_matrix_method(matrix_method)  # the geo is regularizeDisk
        tobj.set_data(f_geo=tgeo, u_geo=tgeo, name=namehandle + '%02d' % i0)
        tobj_list.append(tobj)
    return tobj_list


def create_helicoid_comp(*args, **kwargs):
    update_order = kwargs['update_order'] if 'update_order' in kwargs.keys() else 1
    update_fun = kwargs['update_fun'] if 'update_fun' in kwargs.keys() else Adams_Bashforth_Methods
    helicoid_list = create_helicoid_list(*args, **kwargs)
    helicoid_comp = sf.ForceFreeComposite(center=np.zeros(3), norm=np.array((1, 0, 0)),
                                          name='helicoid_comp')
    for tobj in helicoid_list:
        # print(tobj)
        helicoid_comp.add_obj(obj=tobj, rel_U=np.zeros(6))
    helicoid_comp.set_update_para(fix_x=False, fix_y=False, fix_z=False,
                                  update_fun=update_fun, update_order=update_order)
    return helicoid_comp


def obj2helicoid_list(tobj0, **problem_kwargs):
    helicoid_r = problem_kwargs['helicoid_r']
    ndsk_each = problem_kwargs['helicoid_ndsk_each']

    tobj = tobj0.copy()
    tobj.move(np.array((helicoid_r, 0, 0)))

    tobj_list = []
    rot_dth = 2 * np.pi / ndsk_each
    namehandle = tobj.get_name()
    for i0 in range(ndsk_each):
        rot_th = i0 * rot_dth + rot_dth / 2
        tobj21 = tobj.copy()
        tobj21.set_name('%s_%02d_%01d' % (namehandle, i0, 1))
        tobj21.node_rotation(norm=np.array([0, 0, 1]), theta=rot_th, rotation_origin=np.zeros(3))
        tobj_list.append(tobj21)
        tobj22 = tobj21.copy()
        tobj21.set_name('%s_%02d_%01d' % (namehandle, i0, 2))
        tobj22.node_rotation(norm=np.array([1, 0, 0]), theta=np.pi / 2, rotation_origin=np.zeros(3))
        tobj_list.append(tobj22)
        tobj23 = tobj21.copy()
        tobj21.set_name('%s_%02d_%01d' % (namehandle, i0, 3))
        tobj23.node_rotation(norm=np.array([0, 1, 0]), theta=np.pi / 2, rotation_origin=np.zeros(3))
        tobj_list.append(tobj23)
    return tobj_list


def obj2helicoid_comp(tobj0, *args, **kwargs):
    update_order = kwargs['update_order'] if 'update_order' in kwargs.keys() else 1
    update_fun = kwargs['update_fun'] if 'update_fun' in kwargs.keys() else Adams_Bashforth_Methods
    helicoid_list = obj2helicoid_list(tobj0, *args, **kwargs)
    helicoid_comp = sf.ForceFreeComposite(center=np.zeros(3), norm=np.array((1, 0, 0)),
                                          name='helicoid_comp')
    for tobj in helicoid_list:
        helicoid_comp.add_obj(obj=tobj, rel_U=np.zeros(6))
    helicoid_comp.set_update_para(fix_x=False, fix_y=False, fix_z=False,
                                  update_fun=update_fun, update_order=update_order)
    return helicoid_comp

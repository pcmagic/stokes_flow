def get_ecoli_table(tnorm, lateral_norm, tcenter, max_iter, eval_dt=0.007, update_order=1,
                    planeShearRate=np.array((1, 0, 0))):
    from time import time
    ellipse_kwargs = {'name':         'ecoli_torque',
                      'center':       tcenter,
                      'norm':         tnorm / np.linalg.norm(tnorm),
                      'lateral_norm': lateral_norm / np.linalg.norm(lateral_norm),
                      'speed':        0,
                      'lbd':          np.nan,
                      'omega_tail':   193.66659814,
                      'table_name':   'planeShearRatex_1c', }
    fileHandle = 'ShearTableProblem'
    ellipse_obj = jm.TableEcoli(**ellipse_kwargs)
    ellipse_obj.set_update_para(fix_x=False, fix_y=False, fix_z=False, update_order=update_order)
    problem = jm.ShearTableProblem(name=fileHandle, planeShearRate=planeShearRate)
    problem.add_obj(ellipse_obj)
    t0 = time()
    for idx in range(1, max_iter + 1):
        problem.update_location(eval_dt, print_handle='%d / %d' % (idx, max_iter))
    t1 = time()
    Table_X = np.vstack(ellipse_obj.center_hist)
    Table_U = np.vstack(ellipse_obj.U_hist)
    Table_P = np.vstack(ellipse_obj.norm_hist)
    Table_t = np.arange(max_iter) * eval_dt + eval_dt
    Table_theta, Table_phi, Table_psi = ellipse_obj.theta_phi_psi
    t1U = np.array([np.dot(t1, t2) for t1, t2 in zip(Table_U[:, :3], Table_P)]).reshape((-1, 1))
    t1W = np.array([np.dot(t1, t2) for t1, t2 in zip(Table_U[:, 3:], Table_P)]).reshape((-1, 1))
    Table_U_horizon = np.hstack((Table_P * t1U, Table_P * t1W))
    Table_U_vertical = Table_U - Table_U_horizon
    omega = Table_U[:, 3:]
    dP = np.vstack([np.cross(t1, t2) for t1, t2 in zip(omega, Table_P)])
    Table_dtheta = -dP[:, 2] / np.sin(np.abs(Table_theta))
    Table_dphi = (dP[:, 1] * np.cos(Table_phi) - dP[:, 0] * np.sin(Table_phi)) / np.sin(Table_theta)
    Table_eta = np.arccos(np.sin(Table_theta) * np.sin(Table_phi))
    print('%s: run %d loops using %f' % (fileHandle, max_iter, (t1 - t0)))
    return Table_t, Table_theta, Table_phi, Table_psi, Table_eta, Table_dtheta, Table_dphi, \
           Table_X, Table_U, Table_P

import numpy as np
from scipy.special import erfinv
from petsc4py import PETSc


# (1) -------------------------------------------------------------------------------------------------------------------------------------------------------------
def cal_dist_periodic(X, xi, length, width):
    
    dxi = X - xi
    dxi = dxi[:, :2]
    dxi = boundary_cond(dxi, length, width)
    dr2 = np.sum(dxi ** 2, axis=1)
    dr1 = np.sqrt(dr2)
    
    return dxi, dr1


def boundary_cond(dxi, length, width):
    
    dxi[:, 0] = np.where(dxi[:, 0] < -length / 2, dxi[:, 0] + length, np.where(dxi[:, 0] > length / 2, dxi[:, 0] - length, dxi[:, 0]))
    dxi[:, 1] = np.where(dxi[:, 1] < -width / 2, dxi[:, 1] + width, np.where(dxi[:, 1] > width / 2, dxi[:, 1] - width, dxi[:, 1]))
    
    return dxi


# (2) -------------------------------------------------------------------------------------------------------------------------------------------------------------
def MMD_lub(R, rs2):
    
    lamb_inter_list = []
    ptc_lub_list = []
    for i0, ri in enumerate(R):
        
        # lamb / optimizing intermediate variables.
        lamb = R / ri
        lamb_pow_2 = lamb ** 2
        lamb_plus_1 = 1 + lamb
        lamb_plus_1_pow_2 = (lamb_plus_1) ** 2
        lamb_plus_1_pow_neg3 = (lamb_plus_1) ** (- 3)
        lamb_plus_1_pow_neg4 = (lamb_plus_1) ** (- 4)
        
        # rs1
        rs1 = 2.0 + np.minimum(lamb, 1 / lamb)
        
        # gX_ / gY_
        gXA_0 = 2 * lamb_pow_2 * lamb_plus_1_pow_neg3
        gXA_1 = 1 / 5 * lamb * (1 + 7 * lamb + lamb_pow_2) * lamb_plus_1_pow_neg3
        gYA_1 = 4 / 15 * lamb * (2 + lamb + 2 * lamb_pow_2) * lamb_plus_1_pow_neg3
        gYB_1 = - 1 / 5 * lamb * (4 + lamb) * (lamb_plus_1) ** (- 2)
        gYC_1 = 2 / 5 * lamb * (lamb_plus_1) ** (- 1)
        gYC_3 = 4 / 5 * lamb_pow_2 * lamb_plus_1_pow_neg4
        
        # X_F0 / Y_F0 --> X_F / Y_F
        rs2_pow_2 = rs2 ** 2
        rs2_pow_3 = rs2 ** 3
        XaF0 = (3 / 2 / rs2 - (2 + 2 * lamb_pow_2) / lamb_plus_1_pow_2 / rs2_pow_3)
        YaF0 = (3 / 4 / rs2 + (1 + lamb_pow_2) / lamb_plus_1_pow_2 / rs2_pow_3)
        YbF0 = - 1 / 2 / rs2_pow_2
        XcF0 = 1 / rs2_pow_3
        YcF0 = - 1 / 2 / rs2_pow_3
        
        # X_110 / Y_110 / X_0 / Y_0 --> X_11 / Y_11 / X_ / Y_
        XA110 = (gXA_0 / (rs1 - 2) + gXA_1 * np.log(1 / (rs1 - 2)))
        YA110 = (gYA_1 * np.log(1 / (rs1 - 2)))
        YB110 = (gYB_1 * np.log(1 / (rs1 - 2)))
        YC110 = (gYC_1 * np.log(1 / (rs1 - 2)))
        
        XA0 = - 2 / (lamb_plus_1) * (gXA_0 / (rs1 - 2) + gXA_1 * np.log(1 / (rs1 - 2)))
        YA0 = - 2 / (lamb_plus_1) * (gYA_1 * np.log(1 / (rs1 - 2)))
        YB0 = - 4 / lamb_plus_1_pow_2 * (gYB_1 * np.log(1 / (rs1 - 2)))
        YC0 = (gYC_3 * np.log(1 / (rs1 - 2)))
        
        lamb_inter_list.append((lamb, lamb_pow_2, lamb_plus_1, lamb_plus_1_pow_2, lamb_plus_1_pow_neg3, lamb_plus_1_pow_neg4, rs1))
        ptc_lub_list.append((gXA_0, gXA_1, gYA_1, gYB_1, gYC_1, gYC_3, XaF0, YaF0, YbF0, XcF0, YcF0, XA110, YA110, YB110, YC110, XA0, YA0, YB0, YC0))
    
    return lamb_inter_list, ptc_lub_list


def MMD_lub_wrapper(obj1, obj2, rs2):
    err_msg = 'current version, only support a single object during a simulation. '
    assert obj1 == obj2, err_msg
    R = obj1.get_u_geo().get_sphere_R()
    return MMD_lub(R, rs2)


def MMD_lub_petsc(obj1, obj2, rs2):
    err_msg = 'current version, only support a single object during a simulation. '
    assert obj1 == obj2, err_msg
    u_dmda = obj1.get_u_geo().get_dmda()
    R = obj1.get_u_geo().get_sphere_R()
    
    lamb_inter_list = []
    ptc_lub_list = []
    for i0 in range(u_dmda.getRanges()[0][0], u_dmda.getRanges()[0][1]):
        ri = R[i0]
        
        # lamb / optimizing intermediate variables.
        lamb = R / ri
        lamb_pow_2 = lamb ** 2
        lamb_plus_1 = 1 + lamb
        lamb_plus_1_pow_2 = (lamb_plus_1) ** 2
        lamb_plus_1_pow_neg3 = (lamb_plus_1) ** (- 3)
        lamb_plus_1_pow_neg4 = (lamb_plus_1) ** (- 4)
        
        # rs1
        rs1 = 2.0 + np.minimum(lamb, 1 / lamb)
        
        # gX_ / gY_
        gXA_0 = 2 * lamb_pow_2 * lamb_plus_1_pow_neg3
        gXA_1 = 1 / 5 * lamb * (1 + 7 * lamb + lamb_pow_2) * lamb_plus_1_pow_neg3
        gYA_1 = 4 / 15 * lamb * (2 + lamb + 2 * lamb_pow_2) * lamb_plus_1_pow_neg3
        gYB_1 = - 1 / 5 * lamb * (4 + lamb) * (lamb_plus_1) ** (- 2)
        gYC_1 = 2 / 5 * lamb * (lamb_plus_1) ** (- 1)
        gYC_3 = 4 / 5 * lamb_pow_2 * lamb_plus_1_pow_neg4
        
        # X_F0 / Y_F0 --> X_F / Y_F
        rs2_pow_2 = rs2 ** 2
        rs2_pow_3 = rs2 ** 3
        XaF0 = (3 / 2 / rs2 - (2 + 2 * lamb_pow_2) / lamb_plus_1_pow_2 / rs2_pow_3)
        YaF0 = (3 / 4 / rs2 + (1 + lamb_pow_2) / lamb_plus_1_pow_2 / rs2_pow_3)
        YbF0 = - 1 / 2 / rs2_pow_2
        XcF0 = 1 / rs2_pow_3
        YcF0 = - 1 / 2 / rs2_pow_3
        
        # X_110 / Y_110 / X_0 / Y_0 --> X_11 / Y_11 / X_ / Y_
        XA110 = (gXA_0 / (rs1 - 2) + gXA_1 * np.log(1 / (rs1 - 2)))
        YA110 = (gYA_1 * np.log(1 / (rs1 - 2)))
        YB110 = (gYB_1 * np.log(1 / (rs1 - 2)))
        YC110 = (gYC_1 * np.log(1 / (rs1 - 2)))
        
        XA0 = - 2 / (lamb_plus_1) * (gXA_0 / (rs1 - 2) + gXA_1 * np.log(1 / (rs1 - 2)))
        YA0 = - 2 / (lamb_plus_1) * (gYA_1 * np.log(1 / (rs1 - 2)))
        YB0 = - 4 / lamb_plus_1_pow_2 * (gYB_1 * np.log(1 / (rs1 - 2)))
        YC0 = (gYC_3 * np.log(1 / (rs1 - 2)))
        
        lamb_inter_list.append((lamb, lamb_pow_2, lamb_plus_1, lamb_plus_1_pow_2, lamb_plus_1_pow_neg3, lamb_plus_1_pow_neg4, rs1))
        ptc_lub_list.append((gXA_0, gXA_1, gYA_1, gYB_1, gYC_1, gYC_3, XaF0, YaF0, YbF0, XcF0, YcF0, XA110, YA110, YB110, YC110, XA0, YA0, YB0, YC0))
    return lamb_inter_list, ptc_lub_list


# (3) -------------------------------------------------------------------------------------------------------------------------------------------------------------
def M_R_fun(R, X, rs2, sdis, length, width, ptc_lub_list, lamb_inter_list, mu=1e-6):
    # R: Radius of each spheres. (每个球的半径)
    # X: Position information for all spheres. (所有球体的位置信息)
    # rs2: Maximum separation distance. (最大分离距离)
    # sdis: Minimum surface distance. (最小表面间距)
    # length / width: Length and width of calculating range. (计算范围的长度和宽度)
    # mu: Fluid viscosity. (血浆的粘度)(g/um/s)
    # frac: prepositioning factor. (前置系数)
    
    NS = R.size  # NS: Total number of spheres. (总的球体数)
    diag_err = 1e-16  # Avoiding errors introduced by nan values. (避免nan值引入的误差)
    frac = 1 / (np.pi * mu)  # 前置系数
    
    M_RPY = np.zeros((3 * NS, 3 * NS))
    R_lub = np.zeros((3 * NS, 3 * NS))
    
    # loop alone u_nodes.
    for i0, (xi, ri, ptc_lubi, lamb_inter) in enumerate(zip(X, R, ptc_lub_list, lamb_inter_list)):
        
        gXA_0, gXA_1, gYA_1, gYB_1, gYC_1, gYC_3, XaF0, YaF0, YbF0, XcF0, YcF0, XA110, YA110, YB110, YC110, XA0, YA0, YB0, YC0 = ptc_lubi
        lamb, lamb_pow_2, lamb_plus_1, lamb_plus_1_pow_2, lamb_plus_1_pow_neg3, lamb_plus_1_pow_neg4, rs1 = lamb_inter
        
        dxi, dr1 = cal_dist_periodic(X, xi, length, width)
        # dr1 = dr1 + diag_err
        
        # eij
        dr1[i0] = diag_err
        eij_0 = dxi[:, 0] / dr1
        eij_1 = dxi[:, 1] / dr1
        eij_0[i0] = 0
        eij_1[i0] = 0
        eij_00 = eij_0 * eij_0
        eij_01 = eij_0 * eij_1
        eij_10 = eij_1 * eij_0
        eij_11 = eij_1 * eij_1
        
        # rss / xi / optimizing intermediate variables.
        rss = (dr1 / R.T).T
        xi = rss - 2.0
        mask_rss_xi = rss < 2.0 + sdis
        # rss = np.where(mask_rss_xi, 2.0 + sdis, rss)
        # xi  = np.where(mask_rss_xi, sdis, xi)
        rss = np.where(mask_rss_xi, 2.0 + sdis, rss + diag_err)
        xi = np.where(mask_rss_xi, sdis, xi + diag_err)
        rss_pow_2 = rss ** 2
        rss_pow_3 = rss ** 3
        
        mask_XY_F_rss = rss <= rs2
        XaF = np.where(mask_XY_F_rss, 3 / 2 / rss - (2 + 2 * lamb_pow_2) / lamb_plus_1_pow_2 / rss_pow_3 - XaF0, 0)
        XaF[i0] = 0
        YaF = np.where(mask_XY_F_rss, 3 / 4 / rss + (1 + lamb_pow_2) / lamb_plus_1_pow_2 / rss_pow_3 - YaF0, 0)
        YaF[i0] = 0
        YbF = np.where(mask_XY_F_rss, - 1 / 2 / rss_pow_2 - YbF0, 0)
        YbF[i0] = 0
        XcF = np.where(mask_XY_F_rss, 1 / rss_pow_3 - XcF0, 0)
        XcF[i0] = 0
        YcF = np.where(mask_XY_F_rss, - 1 / 2 / rss_pow_3 - YcF0, 0)
        YcF[i0] = 0
        
        mask_XY_rss = rss <= rs1
        XA11 = np.where(mask_XY_rss, gXA_0 / xi + gXA_1 * np.log(1 / xi) - XA110, 0)
        XA11[i0] = 0
        YA11 = np.where(mask_XY_rss, gYA_1 * np.log(1 / xi) - YA110, 0)
        YA11[i0] = 0
        YB11 = np.where(mask_XY_rss, gYB_1 * np.log(1 / xi) - YB110, 0)
        YB11[i0] = 0
        YC11 = np.where(mask_XY_rss, gYC_1 * np.log(1 / xi) - YC110, 0)
        YC11[i0] = 0
        
        XA = np.where(mask_XY_rss, - 2 / (lamb_plus_1) * (gXA_0 / xi + gXA_1 * np.log(1 / xi)) - XA0, 0)
        XA[i0] = 0
        YA = np.where(mask_XY_rss, - 2 / (lamb_plus_1) * (gYA_1 * np.log(1 / xi)) - YA0, 0)
        YA[i0] = 0
        YB = np.where(mask_XY_rss, - (4 / lamb_plus_1_pow_2 * (gYB_1 * np.log(1 / xi)) - YB0), 0)
        YB[i0] = 0
        YC = np.where(mask_XY_rss, gYC_3 * np.log(1 / xi) - YC0, 0)
        YC[i0] = 0
        
        XaF, XcF, YaF, YbF, YcF, XA11, YA11, YB11, YC11, XA, YA, YB, YC = XaF.T, XcF.T, YaF.T, YbF.T, YcF.T, XA11.T, YA11.T, YB11.T, YC11.T, XA.T, YA.T, YB.T, YC.T
        # M_RPY
        m_mtt = 1 / (3 * np.pi * mu * (ri + R))
        m_mrt = 1 / (np.pi * mu * (ri + R) ** 2)
        m_mrr = 1 / (np.pi * mu * (ri + R) ** 3)
        
        ## MTT
        M_RPY[2 * i0 + 0, 0: (2 * NS): 2] = m_mtt[i0] * (XaF * eij_00 + YaF * (1 - eij_00))
        M_RPY[2 * i0 + 0, 1: (2 * NS): 2] = m_mtt[i0] * (XaF * eij_01 + YaF * (0 - eij_01))
        M_RPY[2 * i0 + 1, 0: (2 * NS): 2] = m_mtt[i0] * (XaF * eij_10 + YaF * (0 - eij_10))
        M_RPY[2 * i0 + 1, 1: (2 * NS): 2] = m_mtt[i0] * (XaF * eij_11 + YaF * (1 - eij_11))
        M_RPY[2 * i0 + 0, 2 * i0 + 0] = m_mtt[i0]
        M_RPY[2 * i0 + 1, 2 * i0 + 1] = m_mtt[i0]
        ## MRT
        M_RPY[i0 + ((2 * NS)), 0: (2 * NS): 2] = m_mrt[i0] * YbF * eij_1
        M_RPY[i0 + ((2 * NS)), 1: (2 * NS): 2] = m_mrt[i0] * YbF * -eij_0
        M_RPY[i0 + ((2 * NS)), 2 * i0] = 0
        ## MTR
        M_RPY[0: 2 * NS, 0 + 2 * NS: NS + 2 * NS] = M_RPY[0 + 2 * NS: NS + 2 * NS, 0: 2 * NS].T
        ## MTT
        M_RPY[i0 + ((2 * NS)), 2 * NS: NS + ((2 * NS)): 1] = m_mrr[i0] * YcF
        M_RPY[i0 + ((2 * NS)), i0 + ((2 * NS))] = m_mrr[i0]
        
        # R_lub
        r_rtt = 1 / m_mtt
        r_rrt = 1 / m_mrt
        r_rrr = 1 / m_mrr
        
        ## RTT
        R_lub[2 * i0 + 0, 0: (2 * NS): 2] = r_rtt[i0] * (XA * eij_00 + YA * (1 - eij_00))
        R_lub[2 * i0 + 0, 1: (2 * NS): 2] = r_rtt[i0] * (XA * eij_01 + YA * (0 - eij_01))
        R_lub[2 * i0 + 1, 0: (2 * NS): 2] = r_rtt[i0] * (XA * eij_10 + YA * (0 - eij_10))
        R_lub[2 * i0 + 1, 1: (2 * NS): 2] = r_rtt[i0] * (XA * eij_11 + YA * (1 - eij_11))
        R_lub[2 * i0 + 0, 2 * i0 + 0] = np.sum(r_rtt[i0] * (XA11 * eij_00 + YA11 * (1 - eij_00)))
        R_lub[2 * i0 + 0, 2 * i0 + 1] = np.sum(r_rtt[i0] * (XA11 * eij_01 + YA11 * (0 - eij_01)))
        R_lub[2 * i0 + 1, 2 * i0 + 0] = np.sum(r_rtt[i0] * (XA11 * eij_10 + YA11 * (0 - eij_10)))
        R_lub[2 * i0 + 1, 2 * i0 + 1] = np.sum(r_rtt[i0] * (XA11 * eij_11 + YA11 * (1 - eij_11)))
        ## RRT
        R_lub[i0 + ((2 * NS)), 0: (2 * NS): 2] = r_rrt[i0] * YB * eij_1
        R_lub[i0 + ((2 * NS)), 1: (2 * NS): 2] = r_rrt[i0] * YB * -eij_0
        R_lub[i0 + ((2 * NS)), 2 * i0 + 0] = np.sum(r_rrt[i0] * YB11 * eij_1)
        R_lub[i0 + ((2 * NS)), 2 * i0 + 1] = np.sum(r_rrt[i0] * YB11 * -eij_0)
        ## RTR
        R_lub[0: 2 * NS, 0 + 2 * NS: NS + 2 * NS] = R_lub[0 + 2 * NS: NS + 2 * NS, 0: 2 * NS].T
        ## RRR
        R_lub[i0 + ((2 * NS)), 2 * NS: NS + ((2 * NS)): 1] = r_rrr[i0] * YC
        R_lub[i0 + ((2 * NS)), i0 + ((2 * NS))] = np.sum(r_rrr[i0] * YC11)
    
    M_RPY = frac * M_RPY  # 远场的迁移率矩阵：M_inf
    R_lub = R_lub / frac  # 近场阻力矩阵：R_lub
    Rtol = M_RPY @ R_lub + np.eye(3 * NS, 3 * NS)
    # Minv_RPY = np.linalg.inv(M_RPY)                                              # 远场迁移率矩阵的逆矩阵：(M_inf)^(-1)：厄米矩阵
    # Rtol = R_lub + Minv_RPY                                                      # 全域阻力矩阵
    # Mtol = np.linalg.inv(Rtol)
    # 全域迁移率矩阵
    pass
    
    return M_RPY, R_lub, Rtol


def M_R_petsc(Minf_petsc, Rlub_petsc, Rtol_petsc, u_dmda,
              R, X, rs2, sdis, length, width, ptc_lub_list, lamb_inter_list,
              mu=1e-6, diag_err=1e-16):
    # R: Radius of each spheres. (每个球的半径)
    # X: Position information for all spheres. (所有球体的位置信息)
    # rs2: Maximum separation distance. (最大分离距离)
    # sdis: Minimum surface distance. (最小表面间距)
    # length / width: Length and width of calculating range. (计算范围的长度和宽度)
    # mu: Fluid viscosity. (血浆的粘度)(g/um/s)
    # frac: prepositioning factor. (前置系数)
    # diag_err: Avoiding errors introduced by nan values. (避免nan值引入的误差)
    
    NS = R.size  # NS: Total number of spheres. (总的球体数)
    frac = 1 / (np.pi * mu)  # 前置系数
    
    # loop alone u_nodes.
    uidx0, uidx1 = u_dmda.getRanges()[0]
    for i0 in range(uidx0, uidx1):
        xi, ri, ptc_lubi, lamb_inter = X[i0], R[i0], ptc_lub_list[i0 - uidx0], lamb_inter_list[i0 - uidx0]
        gXA_0, gXA_1, gYA_1, gYB_1, gYC_1, gYC_3, XaF0, YaF0, YbF0, XcF0, YcF0, XA110, YA110, YB110, YC110, XA0, YA0, YB0, YC0 = ptc_lubi
        lamb, lamb_pow_2, lamb_plus_1, lamb_plus_1_pow_2, lamb_plus_1_pow_neg3, lamb_plus_1_pow_neg4, rs1 = lamb_inter
        
        dxi, dr1 = cal_dist_periodic(X, xi, length, width)
        dr1[i0] = diag_err
        eij_0 = dxi[:, 0] / dr1
        eij_1 = dxi[:, 1] / dr1
        eij_0[i0] = 0
        eij_1[i0] = 0
        eij_00 = eij_0 * eij_0
        eij_01 = eij_0 * eij_1
        eij_10 = eij_1 * eij_0
        eij_11 = eij_1 * eij_1
        
        # rss / xi / optimizing intermediate variables.
        rss = (dr1 / R.T).T
        xi = rss - 2.0
        mask_rss_xi = rss < 2.0 + sdis
        # rss = np.where(mask_rss_xi, 2.0 + sdis, rss)
        # xi  = np.where(mask_rss_xi, sdis, xi)
        rss = np.where(mask_rss_xi, 2.0 + sdis, rss + diag_err)
        xi = np.where(mask_rss_xi, sdis, xi + diag_err)
        rss_pow_2 = rss ** 2
        rss_pow_3 = rss ** 3
        
        mask_XY_F_rss = rss <= rs2
        XaF = np.where(mask_XY_F_rss, 3 / 2 / rss - (2 + 2 * lamb_pow_2) / lamb_plus_1_pow_2 / rss_pow_3 - XaF0, 0)
        XaF[i0] = 0
        YaF = np.where(mask_XY_F_rss, 3 / 4 / rss + (1 + lamb_pow_2) / lamb_plus_1_pow_2 / rss_pow_3 - YaF0, 0)
        YaF[i0] = 0
        YbF = np.where(mask_XY_F_rss, - 1 / 2 / rss_pow_2 - YbF0, 0)
        YbF[i0] = 0
        XcF = np.where(mask_XY_F_rss, 1 / rss_pow_3 - XcF0, 0)
        XcF[i0] = 0
        YcF = np.where(mask_XY_F_rss, - 1 / 2 / rss_pow_3 - YcF0, 0)
        YcF[i0] = 0
        
        mask_XY_rss = rss <= rs1
        XA11 = np.where(mask_XY_rss, gXA_0 / xi + gXA_1 * np.log(1 / xi) - XA110, 0)
        XA11[i0] = 0
        YA11 = np.where(mask_XY_rss, gYA_1 * np.log(1 / xi) - YA110, 0)
        YA11[i0] = 0
        YB11 = np.where(mask_XY_rss, gYB_1 * np.log(1 / xi) - YB110, 0)
        YB11[i0] = 0
        YC11 = np.where(mask_XY_rss, gYC_1 * np.log(1 / xi) - YC110, 0)
        YC11[i0] = 0
        
        XA = np.where(mask_XY_rss, - 2 / (lamb_plus_1) * (gXA_0 / xi + gXA_1 * np.log(1 / xi)) - XA0, 0)
        XA[i0] = 0
        YA = np.where(mask_XY_rss, - 2 / (lamb_plus_1) * (gYA_1 * np.log(1 / xi)) - YA0, 0)
        YA[i0] = 0
        YB = np.where(mask_XY_rss, - (4 / lamb_plus_1_pow_2 * (gYB_1 * np.log(1 / xi)) - YB0), 0)
        YB[i0] = 0
        YC = np.where(mask_XY_rss, gYC_3 * np.log(1 / xi) - YC0, 0)
        YC[i0] = 0
        
        # XaF, XcF, YaF, YbF, YcF, XA11, YA11, YB11, YC11, XA, YA, YB, YC = XaF.T, XcF.T, YaF.T, YbF.T, YcF.T, XA11.T, YA11.T, YB11.T, YC11.T, XA.T, YA.T, YB.T, YC.T
        # M_RPY
        m_mtt = 1 / (3 * np.pi * mu * (ri + R))
        m_mrt = 1 / (np.pi * mu * (ri + R) ** 2)
        m_mrr = 1 / (np.pi * mu * (ri + R) ** 3)
        
        ## MTT
        Minf_petsc[2 * i0 + 0, 0: (2 * NS): 2] = m_mtt[i0] * (XaF * eij_00 + YaF * (1 - eij_00))
        Minf_petsc[2 * i0 + 0, 1: (2 * NS): 2] = m_mtt[i0] * (XaF * eij_01 + YaF * (0 - eij_01))
        Minf_petsc[2 * i0 + 1, 0: (2 * NS): 2] = m_mtt[i0] * (XaF * eij_10 + YaF * (0 - eij_10))
        Minf_petsc[2 * i0 + 1, 1: (2 * NS): 2] = m_mtt[i0] * (XaF * eij_11 + YaF * (1 - eij_11))
        Minf_petsc[2 * i0 + 0, 2 * i0 + 0] = m_mtt[i0]
        Minf_petsc[2 * i0 + 1, 2 * i0 + 1] = m_mtt[i0]
        ## MRT & MTR
        t1 = m_mrt[i0] * YbF * eij_1
        t2 = m_mrt[i0] * YbF * -eij_0
        t3 = 0
        Minf_petsc[i0 + (2 * NS), 0: (2 * NS): 2] = t1
        Minf_petsc[i0 + (2 * NS), 1: (2 * NS): 2] = t2
        Minf_petsc[i0 + (2 * NS), 2 * i0] = t3
        Minf_petsc[0: (2 * NS): 2, i0 + (2 * NS)] = t1
        Minf_petsc[1: (2 * NS): 2, i0 + (2 * NS)] = t2
        Minf_petsc[2 * i0, i0 + (2 * NS)] = t3
        ## MTT
        Minf_petsc[i0 + (2 * NS), 2 * NS: NS + (2 * NS): 1] = m_mrr[i0] * YcF
        Minf_petsc[i0 + (2 * NS), i0 + (2 * NS)] = m_mrr[i0]
        
        # R_lub
        r_rtt = 1 / m_mtt
        r_rrt = 1 / m_mrt
        r_rrr = 1 / m_mrr
        
        ## RTT
        Rlub_petsc[2 * i0 + 0, 0: (2 * NS): 2] = r_rtt[i0] * (XA * eij_00 + YA * (1 - eij_00))
        Rlub_petsc[2 * i0 + 0, 1: (2 * NS): 2] = r_rtt[i0] * (XA * eij_01 + YA * (0 - eij_01))
        Rlub_petsc[2 * i0 + 1, 0: (2 * NS): 2] = r_rtt[i0] * (XA * eij_10 + YA * (0 - eij_10))
        Rlub_petsc[2 * i0 + 1, 1: (2 * NS): 2] = r_rtt[i0] * (XA * eij_11 + YA * (1 - eij_11))
        Rlub_petsc[2 * i0 + 0, 2 * i0 + 0] = np.sum(r_rtt[i0] * (XA11 * eij_00 + YA11 * (1 - eij_00)))
        Rlub_petsc[2 * i0 + 0, 2 * i0 + 1] = np.sum(r_rtt[i0] * (XA11 * eij_01 + YA11 * (0 - eij_01)))
        Rlub_petsc[2 * i0 + 1, 2 * i0 + 0] = np.sum(r_rtt[i0] * (XA11 * eij_10 + YA11 * (0 - eij_10)))
        Rlub_petsc[2 * i0 + 1, 2 * i0 + 1] = np.sum(r_rtt[i0] * (XA11 * eij_11 + YA11 * (1 - eij_11)))
        ## RRT & RTR
        t1 = r_rrt[i0] * YB * eij_1
        t2 = r_rrt[i0] * YB * -eij_0
        t3 = np.sum(r_rrt[i0] * YB11 * eij_1)
        t4 = np.sum(r_rrt[i0] * YB11 * -eij_0)
        Rlub_petsc[i0 + (2 * NS), 0: (2 * NS): 2] = t1
        Rlub_petsc[i0 + (2 * NS), 1: (2 * NS): 2] = t2
        Rlub_petsc[i0 + (2 * NS), 2 * i0 + 0] = t3
        Rlub_petsc[i0 + (2 * NS), 2 * i0 + 1] = t4
        Rlub_petsc[0: (2 * NS): 2, i0 + (2 * NS)] = t1
        Rlub_petsc[1: (2 * NS): 2, i0 + (2 * NS)] = t2
        Rlub_petsc[2 * i0 + 0, i0 + (2 * NS)] = t3
        Rlub_petsc[2 * i0 + 1, i0 + (2 * NS)] = t4
        ## RRR
        Rlub_petsc[i0 + (2 * NS), 2 * NS: NS + (2 * NS): 1] = r_rrr[i0] * YC
        Rlub_petsc[i0 + (2 * NS), i0 + (2 * NS)] = np.sum(r_rrr[i0] * YC11)
    Minf_petsc.assemble()
    Rlub_petsc.assemble()
    Minf_petsc.scale(frac)  # 远场的迁移率矩阵：M_inf
    Rlub_petsc.scale(1 / frac)  # 近场阻力矩阵：R_lub
    #
    # tidx0, tidx1 = Rtol_petsc.getSize()
    # print(uidx0)
    # print(uidx1)
    comm = PETSc.COMM_WORLD.tompi4py()
    rank = comm.Get_rank()
    # print(rank, 'getOwnershipRange', Rtol_petsc.getOwnershipRange())
    # print(rank, 'getOwnershipRangeColumn', Rtol_petsc.getOwnershipRangeColumn())
    # print(rank, 'getOwnershipRanges', Rtol_petsc.getOwnershipRanges())
    # print(rank, 'getOwnershipRangesColumn', Rtol_petsc.getOwnershipRangesColumn())
    # print(rank, 'getSize', Rtol_petsc.getSize())
    # print(rank, 'getSizes', Rtol_petsc.getSizes())
    m_start, m_end = Rtol_petsc.getOwnershipRange()
    Minf_petsc.matMult(Rlub_petsc, result=Rtol_petsc)
    Rtol_petsc.setValues(range(m_start, m_end), range(m_start, m_end),
                         np.eye(m_end - m_start, m_end - m_start), addv=True)
    Rtol_petsc.assemble()
    return True


# 白噪声
# 给定力矩，计算平移速度和旋转速度
def F_fun(NS, For, Tor, phi):
    # add some command here
    F1 = np.zeros((3 * NS))
    F1[0: 2 * NS: 2] = For * np.cos(phi)
    F1[1: 2 * NS: 2] = For * np.sin(phi)  # 推进力和白噪声
    F1[2 * NS:] = Tor  # 推进力矩和白噪声
    
    return F1

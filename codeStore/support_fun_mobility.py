# from tqdm.notebook import tqdm as tqdm_notebook
# import os
# import glob
import pickle
import numpy as np
from src.support_class import *
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
from scipy import linalg
from codeStore import support_fun_bck as spf

colors11 = plt.get_cmap('Blues')
colors12 = plt.get_cmap('Reds')
colors1 = np.vstack((colors11(np.linspace(1, 0.2, 256)), colors12(np.linspace(0.4, 1, 256))))
cmpBR = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors1)


# generate the mobility matrix of the microswimmer from pickle file,
#  with force and torque free conditions,
#  ignore head tail interaction.


def fun_m_rot(mbase, R):
    ab = mbase[0:3, 0:3]
    bb1 = mbase[3:6, 0:3]
    bb2 = mbase[0:3, 3:6]
    cb = mbase[3:6, 3:6]
    m2 = np.zeros_like(mbase)
    m2[0:3, 0:3] = np.dot(R, np.dot(ab, R.T))
    m2[3:6, 0:3] = np.dot(R, np.dot(bb1, R.T)) * np.linalg.det(R)
    m2[0:3, 3:6] = np.dot(R, np.dot(bb2, R.T)) * np.linalg.det(R)
    m2[3:6, 3:6] = np.dot(R, np.dot(cb, R.T))
    return m2


def cross_matrix(v):
    assert v.shape == (3,)
    m = np.zeros((3, 3))
    m[0, 1] = -v[2]
    m[0, 2] = v[1]
    m[1, 0] = v[2]
    m[1, 2] = -v[0]
    m[2, 0] = -v[1]
    m[2, 1] = v[0]
    return m


def fun_rbc_rtc(rb1, rb2, ch, ph, dist_hs, tail_ini_beta, rotM):
    trs = rb1 * rb2 / np.sqrt((rb1 * np.sin(tail_ini_beta)) ** 2 +
                              (rb2 * np.cos(tail_ini_beta)) ** 2)
    tl = 2 * rb1 + ch * ph + dist_hs
    rbc_base = np.array((0, 0, tl / 2 - rb1))
    rtc = rbc_base - np.array((0, 0, rb1 + dist_hs + ch * ph / 2))

    head_end0 = rbc_base - np.array((0, 0, trs))
    rbc = np.dot(rotM.T, (rbc_base - head_end0)) + head_end0
    return rbc, rtc


def fun_mfull_ufull_core(mhead_base, mtail, dist_hs, beta, rotM, wbc, wtc,
                         rb1, rb2, ch, ph, body_size_fct=1, tail_size_fct=1, ):
    beta_norm = np.array([0, 1, 0])
    rotM_beta = get_rot_matrix(norm=beta_norm, theta=-beta)
    mhead = fun_m_rot(fun_m_rot(mhead_base, rotM_beta), rotM.T)
    rbc, rtc = fun_rbc_rtc(rb1, rb2, ch, ph, dist_hs, beta, rotM)
    rc = rbc  # current version the center of microswimmer is at the center of head.
    drbc = rbc - rc
    drtc = rtc - rc

    mhead[0:3, 0:3] = mhead[0:3, 0:3] * body_size_fct ** 1
    mhead[0:3, 3:6] = mhead[0:3, 3:6] * body_size_fct ** 2
    mhead[3:6, 0:3] = mhead[3:6, 0:3] * body_size_fct ** 2
    mhead[3:6, 3:6] = mhead[3:6, 3:6] * body_size_fct ** 3
    mtail[0:3, 0:3] = mtail[0:3, 0:3] * tail_size_fct ** 1
    mtail[0:3, 3:6] = mtail[0:3, 3:6] * tail_size_fct ** 2
    mtail[3:6, 0:3] = mtail[3:6, 0:3] * tail_size_fct ** 2
    mtail[3:6, 3:6] = mtail[3:6, 3:6] * tail_size_fct ** 3

    # generate M matrix with the force- and torque-free conditions.
    mfull = np.zeros((18, 18))
    mfull[0: 6, 0: 6] = mhead
    mfull[6:12, 6:12] = mtail
    mfull[0: 3, 12:15] = -np.eye(3)
    mfull[0: 3, 15:18] = cross_matrix(drbc)
    mfull[3: 6, 15:18] = -np.eye(3)
    mfull[6: 9, 12:15] = -np.eye(3)
    mfull[6: 9, 15:18] = cross_matrix(drtc)
    mfull[9:12, 15:18] = -np.eye(3)
    mfull[12:15, 0: 3] = -np.eye(3)
    mfull[12:15, 6: 9] = -np.eye(3)
    mfull[15:18, 0: 3] = -cross_matrix(drbc)
    mfull[15:18, 3: 6] = -np.eye(3)
    mfull[15:18, 6: 9] = -cross_matrix(drtc)
    mfull[15:18, 9:12] = -np.eye(3)

    # generate boundary conditions.
    norm_head = -np.dot(rotM.T, rotM_beta)[:, 2]
    norm_tail = np.array((0, 0, 1))
    ufull = np.zeros(18)
    ufull[0: 3] = 0
    ufull[3: 6] = wbc * norm_head
    ufull[6: 9] = 0
    ufull[9:12] = wtc * norm_tail

    mobility_kwargs = {'rbc':       rbc,
                       'rtc':       rtc,
                       'rc':        rc,
                       'norm_head': norm_head,
                       'norm_tail': norm_tail, }
    return mfull, ufull, mobility_kwargs


def fun_position_kwargs(case_kwargs):
    beta_norm = np.array([0, 1, 0])
    dist_hs = case_kwargs['dist_hs']
    beta = case_kwargs['tail_ini_beta']
    theta = case_kwargs['tail_ini_theta']
    phi = case_kwargs['tail_ini_phi']
    psi = case_kwargs['tail_ini_psi']
    rb1 = case_kwargs['rs1']
    rb2 = case_kwargs['rs2']
    ch = case_kwargs['ch']
    ph = case_kwargs['ph']

    rotM_beta = get_rot_matrix(norm=beta_norm, theta=-beta)
    rotM = Rloc2glb(theta, phi, psi)
    rbc, rtc = fun_rbc_rtc(rb1, rb2, ch, ph, dist_hs, beta, rotM)
    rc = rbc  # current version the center of microswimmer is at the center of head.
    norm_head = -np.dot(rotM.T, rotM_beta)[:, 2]
    norm_tail = np.array((0, 0, 1))

    position_kwargs = {'rbc':       rbc,
                       'rtc':       rtc,
                       'rc':        rc,
                       'norm_head': norm_head,
                       'norm_tail': norm_tail, }
    return position_kwargs


def fun_ut_un(u, w):
    ut = np.dot(u, w) * w / (np.linalg.norm(w) ** 2)
    un = u - ut
    return ut, un


def mobility_pickle(pickle_dir, beta, theta, phi, psi, dist_hs, wbc, wtc,
                    body_size_fct=1, tail_size_fct=1, ):
    with open(pickle_dir, 'rb') as handle:
        tpick = pickle.load(handle)
    problem_kwargs = tpick['problem_kwargs']
    rb1 = problem_kwargs['rs1']
    rb2 = problem_kwargs['rs2']
    ch = problem_kwargs['ch']
    ph = problem_kwargs['ph']
    mhead_base, mtail = tpick['Mhead'], tpick['Mtail']

    rotM = Rloc2glb(theta, phi, psi)
    mfull, ufull, mobility_kwargs = \
        fun_mfull_ufull_core(mhead_base, mtail, dist_hs, beta, rotM,
                             wbc, wtc, rb1, rb2, ch, ph,
                             body_size_fct=body_size_fct, tail_size_fct=tail_size_fct)
    mobility_kwargs['rb1'] = rb1
    mobility_kwargs['rb2'] = rb2
    mobility_kwargs['ch'] = ch
    mobility_kwargs['ph'] = ph
    return mfull, ufull, mobility_kwargs


def apx_resistance_pickle(pickle_dir, beta, theta, phi, psi, dist_hs, wbc, wtc):
    # decoupled method, resistance
    with open(pickle_dir, 'rb') as handle:
        tpick = pickle.load(handle)
    problem_kwargs = tpick['problem_kwargs']
    rb1 = problem_kwargs['rs1']
    rb2 = problem_kwargs['rs2']
    ch = problem_kwargs['ch']
    ph = problem_kwargs['ph']
    mhead_base, mtail = tpick['Mhead'], tpick['Mtail']
    #
    Rhead_base = np.linalg.inv(mhead_base)
    Rhead_base = np.diagflat(np.diag(Rhead_base))
    t1 = (Rhead_base[0, 0] + Rhead_base[1, 1]) / 2
    Rhead_base[0, 0] = t1
    Rhead_base[1, 1] = t1
    t1 = (Rhead_base[3, 3] + Rhead_base[4, 4]) / 2
    Rhead_base[3, 3] = t1
    Rhead_base[4, 4] = t1
    Rtail = np.linalg.inv(mtail)
    beta_norm = np.array([0, 1, 0])
    rotM_beta = get_rot_matrix(norm=beta_norm, theta=-beta)
    rotM = Rloc2glb(theta, phi, psi)
    Rhead = fun_m_rot(fun_m_rot(Rhead_base, rotM_beta), rotM.T)
    Ab_rt = Rhead[0:3, 0:3]
    Cb_rt = Rhead[3:6, 3:6]
    At = np.diagflat(np.diag(Rtail[0:3, 0:3]))
    t1 = (At[0, 0] + At[1, 1]) / 2
    At[0, 0] = t1
    At[1, 1] = t1
    Bt = np.diagflat(np.diag((Rtail[0:3, 3:6] + Rtail[3:6, 0:3]) / 2))
    t1 = (Bt[0, 0] + Bt[1, 1]) / 2 * 0
    Bt[0, 0] = t1
    Bt[1, 1] = t1
    Ct = np.diagflat(np.diag(Rtail[3:6, 3:6]))
    t1 = (Ct[0, 0] + Ct[1, 1]) / 2
    Ct[0, 0] = t1
    Ct[1, 1] = t1
    #
    rbc, rtc = fun_rbc_rtc(rb1, rb2, ch, ph, dist_hs, beta, rotM)
    rc = rbc  # current version the center of microswimmer is at the center of head.
    # drbc = rbc - rc
    drtc = rtc - rc
    dtc = cross_matrix(drtc)
    norm_head = -np.dot(rotM.T, rotM_beta)[:, 2]
    norm_tail = np.array((0, 0, 1))
    #
    Rfull = np.zeros((6, 6))
    Rfull[0:3, 0:3] = Ab_rt + At
    Rfull[0:3, 3:6] = - np.dot(At, dtc)
    Rfull[3:6, 0:3] = + np.dot(dtc, At)
    Rfull[3:6, 3:6] = Cb_rt + Ct + np.dot(dtc, Bt) - np.dot(Bt, dtc) - np.dot(dtc, np.dot(At, dtc))
    FFull = np.zeros(6)
    FFull[0:3] = -np.dot(Bt, wtc * norm_tail)
    FFull[3:6] = -np.dot(Cb_rt, wbc * norm_head) - \
                 np.dot(Ct, wtc * norm_tail) - np.dot(dtc, np.dot(Bt, wtc * norm_tail))

    resistance_kwargs = {'rbc':       rbc,
                         'rtc':       rtc,
                         'rc':        rc,
                         'norm_head': norm_head,
                         'norm_tail': norm_tail, }
    return Rfull, FFull, resistance_kwargs


def fun_alpha_bctc(model, wbc, wtc):
    mfull, ufull, mobility_kwargs = model.mobility_matrix(wbc, wtc)
    ffull = linalg.solve(mfull, ufull)
    pb, pt = mobility_kwargs['norm_head'], mobility_kwargs['norm_tail']
    Uc, Wc, Wbc = ffull[12:15], ffull[15:18], wbc * pb
    Wg = Wc + Wbc

    alpha_b = np.arccos(np.dot(pb, Wg) / np.linalg.norm(pb) / np.linalg.norm(Wg))
    alpha_b = np.pi - alpha_b if alpha_b > np.pi / 2 else alpha_b
    alpha_t = np.arccos(np.dot(pt, Wg) / np.linalg.norm(pt) / np.linalg.norm(Wg))
    alpha_t = np.pi - alpha_t if alpha_t > np.pi / 2 else alpha_t
    return alpha_b, alpha_t


def fun_kappa_alpha(model, wbc, wtc):
    alpha_b, alpha_t = fun_alpha_bctc(model, wbc, wtc)
    kappa_alpha = np.abs(alpha_b / alpha_t)
    return kappa_alpha


def fun_hook_torque(model, wbc, wtc):
    mfull, ufull, mobility_kwargs = model.mobility_matrix(wbc, wtc)
    ffull = linalg.solve(mfull, ufull)
    rb1 = mobility_kwargs['rb1']
    rbc = mobility_kwargs['rbc']
    pb = mobility_kwargs['norm_head']
    ds = rbc + rb1 * pb
    hookT = ffull[3:6] - np.cross(ds, ffull[0:3])
    return hookT


def plot_3D_Traj(axi, tplt, theta_list):
    axi.plot(np.zeros(1), np.zeros(1), np.zeros(1), ' ')
    axi.plot(tplt[:, 0], tplt[:, 1], tplt[:, 2], ' ')
    spf.set_axes_equal(axi)
    spf.colorline3d(tplt, theta_list / np.pi, ax0=axi, clb_title='$\\theta / \\pi$',
                    cmap=plt.get_cmap('viridis'))
    axi.scatter(axi.get_xlim()[0], np.zeros(1), np.zeros(1), marker='.', c='k')
    axi.scatter(np.zeros(1), axi.get_ylim()[1], np.zeros(1), marker='.', c='k')
    axi.scatter(np.zeros(1), np.zeros(1), axi.get_zlim()[0], marker='.', c='k')
    axi.plot(np.ones_like(theta_list) * axi.get_xlim()[0], tplt[:, 1], tplt[:, 2],
             '--', color='grey')
    axi.plot(tplt[:, 0], np.ones_like(theta_list) * axi.get_ylim()[1], tplt[:, 2],
             '--', color='grey')
    axi.plot(tplt[:, 0], tplt[:, 1], np.ones_like(theta_list) * axi.get_zlim()[0],
             '--', color='grey')
    axi.view_init(25, -60)
    axi.plot(np.zeros(1), np.zeros(1), np.zeros(1), marker='s', c='k')
    return True


def plot_color_line(axi, tx, ty, xlabel, ylabel, c, vmin, vmax,
                    cmap=cmpBR, xscale0='linear', yscale0='linear', s=4,
                    marker='o', label=''):
    axi.plot(tx, ty, linestyle='None')
    #     axi.relim()
    #     txlim0 = axi.get_xlim()
    #     tylim0 = axi.get_ylim()
    #     print(tylim0, ty.min())
    sc = axi.scatter(tx, ty, vmin=vmin, vmax=vmax, c=c, cmap=cmap, s=s,
                     marker=marker, label=label)
    axi.set_xlabel(xlabel)
    axi.set_ylabel(ylabel)
    axi.set_xscale(xscale0)
    axi.set_yscale(yscale0)
    #     axi.set_xlim(*txlim0)
    #     axi.set_ylim(*tylim0)
    return sc


def fun_cal_kwargs(Uc, Wc, wbc, pb, pt, kappa, mdf_alpha=True):
    Wbc = wbc * pb
    Wg = Wc + kappa * Wbc
    UcWg_t, UcWg_n = fun_ut_un(Uc, Wg)
    eta = np.arccos(np.dot(Uc, Wg) / np.linalg.norm(Uc) / np.linalg.norm(Wg))
    alpha_b = np.arccos(np.dot(pb, Wg) / np.linalg.norm(pb) / np.linalg.norm(Wg))
    alpha_t = np.arccos(np.dot(pt, Wg) / np.linalg.norm(pt) / np.linalg.norm(Wg))
    if mdf_alpha:
        alpha_b = np.pi - alpha_b if alpha_b > np.pi / 2 else alpha_b
        alpha_t = np.pi - alpha_t if alpha_t > np.pi / 2 else alpha_t
    R = np.linalg.norm(UcWg_n) / np.linalg.norm(Wg)
    uc_par = np.sign(np.dot(Uc, Wg)) * np.linalg.norm(UcWg_t)

    cal_kwargs = {'Wg':      Wg,
                  'eta':     eta,
                  'alpha_b': alpha_b,
                  'alpha_t': alpha_t,
                  'R':       R,
                  'uc_par':  uc_par,}
    return cal_kwargs


class DecouplingModel:
    def __init__(self, pickle_dir, beta_norm=np.array([0, 1, 0])):
        with open(pickle_dir, 'rb') as handle:
            tpick = pickle.load(handle)
        self._case_kwargs = tpick['problem_kwargs']
        self._rb1 = self._case_kwargs['rs1']
        self._rb2 = self._case_kwargs['rs2']
        self._ch = self._case_kwargs['ch']
        self._ph = self._case_kwargs['ph']
        self._mhead_base = tpick['Mhead']
        self._mtail_base = tpick['Mtail']
        self._beta_norm = beta_norm

        self._beta = 0
        self._theta = 0
        self._phi = 0
        self._psi = 0
        self._dist_hs = 0
        self._rotM_beta = np.eye(3)
        self._rotM = np.eye(3)

    @property
    def case_kwargs(self):
        return self._case_kwargs

    @property
    def rb1(self):
        return self._rb1

    @property
    def rb2(self):
        return self._rb2

    @property
    def ch(self):
        return self._ch

    @property
    def ph(self):
        return self._ph

    @property
    def mhead_base(self):
        return self._mhead_base

    @property
    def mtail_base(self):
        return self._mtail_base

    @property
    def beta_norm(self):
        return self._beta_norm

    @staticmethod
    def fun_ut_un(u, w):
        ut = np.dot(u, w) * w / (np.linalg.norm(w) ** 2)
        un = u - ut
        return ut, un

    @staticmethod
    def fun_MR_rot(mr_base, R):
        ab = mr_base[0:3, 0:3]
        bb1 = mr_base[3:6, 0:3]
        bb2 = mr_base[0:3, 3:6]
        cb = mr_base[3:6, 3:6]
        m2 = np.zeros_like(mr_base)
        m2[0:3, 0:3] = np.dot(R, np.dot(ab, R.T))
        m2[3:6, 0:3] = np.dot(R, np.dot(bb1, R.T)) * np.linalg.det(R)
        m2[0:3, 3:6] = np.dot(R, np.dot(bb2, R.T)) * np.linalg.det(R)
        m2[3:6, 3:6] = np.dot(R, np.dot(cb, R.T))
        return m2

    @staticmethod
    def cross_matrix(v):
        assert v.shape == (3,)
        m = np.zeros((3, 3))
        m[0, 1] = -v[2]
        m[0, 2] = v[1]
        m[1, 0] = v[2]
        m[1, 2] = -v[0]
        m[2, 0] = -v[1]
        m[2, 1] = v[0]
        return m

    @staticmethod
    def fun_rbc_rtc(rb1, rb2, ch, ph, dist_hs, tail_ini_beta, rotM):
        trs = rb1 * rb2 / np.sqrt((rb1 * np.sin(tail_ini_beta)) ** 2 +
                                  (rb2 * np.cos(tail_ini_beta)) ** 2)
        tl = 2 * rb1 + ch * ph + dist_hs
        rbc_base = np.array((0, 0, tl / 2 - rb1))
        rtc = rbc_base - np.array((0, 0, rb1 + dist_hs + ch * ph / 2))

        head_end0 = rbc_base - np.array((0, 0, trs))
        rbc = np.dot(rotM.T, (rbc_base - head_end0)) + head_end0
        return rbc, rtc

    def fun_position_kwargs(self):
        beta_norm = self.beta_norm
        rb1 = self.rb1
        rb2 = self.rb2
        ch = self.ch
        ph = self.ph
        beta = self._beta
        theta = self._theta
        phi = self._phi
        psi = self._psi
        dist_hs = self._dist_hs
        left_hand = self.case_kwargs['left_hand']

        rotM_beta = get_rot_matrix(norm=beta_norm, theta=-beta)
        rotM = Rloc2glb(theta, phi, psi)
        rbc, rtc = self.fun_rbc_rtc(rb1, rb2, ch, ph, dist_hs, beta, rotM)
        rc = rbc  # current version the center of microswimmer is at the center of head.
        if left_hand:
            norm_head = np.dot(rotM.T, rotM_beta)[:, 2]
            norm_tail = -np.array((0, 0, 1))
        else:
            norm_head = -np.dot(rotM.T, rotM_beta)[:, 2]
            norm_tail = np.array((0, 0, 1))

        position_kwargs = {'rbc':       rbc,
                           'rtc':       rtc,
                           'rc':        rc,
                           'norm_head': norm_head,
                           'norm_tail': norm_tail,
                           'rb1':       rb1,
                           'rb2':       rb2,
                           'ch':        ch,
                           'ph':        ph}
        return position_kwargs

    def fun_mfull_ufull_core(self, wbc, wtc, position_kwargs, body_size_fct=1, tail_size_fct=1):
        # current version, these factors are prohibited.
        assert body_size_fct == 1
        assert tail_size_fct == 1

        mhead_base = self.mhead_base
        mtail = self.mtail_base
        rotM = self._rotM
        rotM_beta = self._rotM_beta

        rbc = position_kwargs['rbc']
        rtc = position_kwargs['rtc']
        rc = position_kwargs['rc']
        norm_head = position_kwargs['norm_head']
        norm_tail = position_kwargs['norm_tail']
        mhead = self.fun_MR_rot(self.fun_MR_rot(mhead_base, rotM_beta), rotM.T)
        drbc = rbc - rc
        drtc = rtc - rc

        mhead[0:3, 0:3] = mhead[0:3, 0:3] * body_size_fct ** 1
        mhead[0:3, 3:6] = mhead[0:3, 3:6] * body_size_fct ** 2
        mhead[3:6, 0:3] = mhead[3:6, 0:3] * body_size_fct ** 2
        mhead[3:6, 3:6] = mhead[3:6, 3:6] * body_size_fct ** 3
        mtail[0:3, 0:3] = mtail[0:3, 0:3] * tail_size_fct ** 1
        mtail[0:3, 3:6] = mtail[0:3, 3:6] * tail_size_fct ** 2
        mtail[3:6, 0:3] = mtail[3:6, 0:3] * tail_size_fct ** 2
        mtail[3:6, 3:6] = mtail[3:6, 3:6] * tail_size_fct ** 3

        # generate M matrix with the force- and torque-free conditions.
        mfull = np.zeros((18, 18))
        mfull[0: 6, 0: 6] = mhead
        mfull[6:12, 6:12] = mtail
        mfull[0: 3, 12:15] = -np.eye(3)
        mfull[0: 3, 15:18] = self.cross_matrix(drbc)
        mfull[3: 6, 15:18] = -np.eye(3)
        mfull[6: 9, 12:15] = -np.eye(3)
        mfull[6: 9, 15:18] = self.cross_matrix(drtc)
        mfull[9:12, 15:18] = -np.eye(3)
        mfull[12:15, 0: 3] = -np.eye(3)
        mfull[12:15, 6: 9] = -np.eye(3)
        mfull[15:18, 0: 3] = -self.cross_matrix(drbc)
        mfull[15:18, 3: 6] = -np.eye(3)
        mfull[15:18, 6: 9] = -self.cross_matrix(drtc)
        mfull[15:18, 9:12] = -np.eye(3)

        # generate boundary conditions.
        ufull = np.zeros(18)
        ufull[0: 3] = 0
        ufull[3: 6] = wbc * norm_head
        ufull[6: 9] = 0
        ufull[9:12] = wtc * norm_tail
        return mfull, ufull

    def case_ini(self, beta, theta, phi, psi, dist_hs):
        beta_norm = self.beta_norm
        self._beta = beta
        self._theta = theta
        self._phi = phi
        self._psi = psi
        self._dist_hs = dist_hs
        self._rotM_beta = get_rot_matrix(norm=beta_norm, theta=-beta)
        self._rotM = Rloc2glb(theta, phi, psi)
        return True

    def mobility_matrix(self, wbc, wtc, body_size_fct=1, tail_size_fct=1, ):
        position_kwargs = self.fun_position_kwargs()
        mfull, ufull = self.fun_mfull_ufull_core(wbc, wtc, position_kwargs,
                                                 body_size_fct=body_size_fct,
                                                 tail_size_fct=tail_size_fct)
        return mfull, ufull, position_kwargs

    def fun_Rfull_Ffull_core(self, wbc, wtc, position_kwargs, body_size_fct=1, tail_size_fct=1):
        # current version, these factors are prohibited.
        assert body_size_fct == 1
        assert tail_size_fct == 1

        mhead_base = self.mhead_base
        mtail = self.mtail_base
        rotM = self._rotM
        rotM_beta = self._rotM_beta

        Rhead_base = np.linalg.inv(mhead_base)
        Rhead_base = np.diagflat(np.diag(Rhead_base))
        t1 = (Rhead_base[0, 0] + Rhead_base[1, 1]) / 2
        Rhead_base[0, 0] = t1
        Rhead_base[1, 1] = t1
        t1 = (Rhead_base[3, 3] + Rhead_base[4, 4]) / 2
        Rhead_base[3, 3] = t1
        Rhead_base[4, 4] = t1
        Rtail = np.linalg.inv(mtail)
        Rhead = fun_m_rot(fun_m_rot(Rhead_base, rotM_beta), rotM.T)
        Ab_rt = Rhead[0:3, 0:3]
        Cb_rt = Rhead[3:6, 3:6]
        At = np.diagflat(np.diag(Rtail[0:3, 0:3]))
        t1 = (At[0, 0] + At[1, 1]) / 2
        At[0, 0] = t1
        At[1, 1] = t1
        Bt = np.diagflat(np.diag((Rtail[0:3, 3:6] + Rtail[3:6, 0:3]) / 2))
        t1 = (Bt[0, 0] + Bt[1, 1]) / 2 * 0
        Bt[0, 0] = t1
        Bt[1, 1] = t1
        Ct = np.diagflat(np.diag(Rtail[3:6, 3:6]))
        t1 = (Ct[0, 0] + Ct[1, 1]) / 2
        Ct[0, 0] = t1
        Ct[1, 1] = t1

        # rbc = position_kwargs['rbc']
        rtc = position_kwargs['rtc']
        rc = position_kwargs['rc']
        norm_head = position_kwargs['norm_head']
        norm_tail = position_kwargs['norm_tail']
        # drbc = rbc - rc
        drtc = rtc - rc
        dtc = cross_matrix(drtc)

        Rfull = np.zeros((6, 6))
        Rfull[0:3, 0:3] = Ab_rt + At
        Rfull[0:3, 3:6] = - np.dot(At, dtc)
        Rfull[3:6, 0:3] = + np.dot(dtc, At)
        Rfull[3:6, 3:6] = Cb_rt + Ct + np.dot(dtc, Bt) - \
                          np.dot(Bt, dtc) - np.dot(dtc, np.dot(At, dtc))
        FFull = np.zeros(6)
        FFull[0:3] = -np.dot(Bt, wtc * norm_tail)
        FFull[3:6] = -np.dot(Cb_rt, wbc * norm_head) - \
                     np.dot(Ct, wtc * norm_tail) - np.dot(dtc, np.dot(Bt, wtc * norm_tail))
        print(np.dot(Cb_rt, wbc * norm_head))
        print(np.dot(Ct, wtc * norm_tail))
        print(np.dot(dtc, np.dot(Bt, wtc * norm_tail)))
        return Rfull, FFull

    def apx_resistance_matrix(self, wbc, wtc, body_size_fct=1, tail_size_fct=1, ):
        position_kwargs = self.fun_position_kwargs()
        Rfull, FFull = self.fun_Rfull_Ffull_core(wbc, wtc, position_kwargs,
                                                 body_size_fct=body_size_fct,
                                                 tail_size_fct=tail_size_fct)
        return Rfull, FFull, position_kwargs

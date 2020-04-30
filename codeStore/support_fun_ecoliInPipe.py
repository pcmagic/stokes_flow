def psi0_psi1(lhi, rh_rt, zft):
    zfh = rh_rt * zft
    if lhi in psi0.index.values:  # use calculated data.
        # psi0
        psi0i = psi0.loc[lhi] * rh_rt
        psi0i_use = interp_psi(psi0i, zfh, kind='quadratic')
        # psi1
        psi1i = psi1.loc[lhi] * rh_rt ** 3
        psi1i_use = interp_psi(psi1i, zfh, kind='quadratic')
    else:  # fit the trend of parameters.
        #         itp_kind = 'linear' if zfh < 0.05 else 'quadratic'
        # psi0
        int_a0 = interp1d(psi0.columns.values, fit_psi0[:, 0], kind='quadratic',
                          fill_value='extrapolate')
        int_a1 = interp1d(psi0.columns.values, fit_psi0[:, 1], kind='quadratic',
                          fill_value='extrapolate')
        psi0i_use = func_psi0(lhi, int_a0(zfh), int_a1(zfh)) * rh_rt
        # psi1
        int_a0 = interp1d(psi1.columns.values, fit_psi1[:, 0], kind='quadratic',
                          fill_value='extrapolate')
        int_a1 = interp1d(psi1.columns.values, fit_psi1[:, 1], kind='quadratic',
                          fill_value='extrapolate')
        psi1i_use = func_psi1(lhi, int_a0(zfh), int_a1(zfh)) * rh_rt ** 3
    #         if zft == 0:
    #             psi1i_use =  func_psi1(lhi, *fit_psi1[0, :]) * rh_rt ** 3
    #         else:
    #             t1 = 1 / zfh
    #             psi1i_use = 4 * np.pi * (t1**2) / (t1**2 - 1) * lhi * rh_rt ** 3
    return psi0i_use, psi1i_use


def intp_psi_tail(psi, phi, chi, zfi):
    tpsi = psi.loc[phi][zfi]
    tx = tpsi.index
    ty = tpsi.values
    idxi = np.isfinite(tx) & np.isfinite(ty)
    tint = interp1d(tx[idxi], ty[idxi], kind='quadratic', fill_value='extrapolate')
    return tint(chi)


def interp_psi(psi, zf, kind='quadratic'):
    def interp_once(psi, zf, kind='quadratic'):
        tx = psi.index
        ty = psi.values
        idxi = np.isfinite(tx) & np.isfinite(ty)
        int_psi = interp1d(tx[idxi], ty[idxi], kind=kind, fill_value='extrapolate')
        psi_zf = int_psi(zf)
        return psi_zf

    if isinstance(psi, pd.core.series.Series):
        psi_zf = interp_once(psi, zf, kind)
    elif isinstance(psi, pd.core.frame.DataFrame):
        psi_zf = psi.copy()
        for phi, chi in psi.index.values:
            t_psi = psi.loc[phi].loc[chi]
            t_psi_zf = interp_once(t_psi, zf, kind)
            psi_zf.loc[phi].loc[chi][:] = t_psi_zf
    return psi_zf


def fit_psi_tail(psi, phi, chi, zfi, chmin, chmax):
    x = psi.loc[phi][zfi].index
    y = psi.loc[phi][zfi].values
    idx = np.array(x > chmin) & np.array(x < chmax) & np.isfinite(x) & np.isfinite(y)
    popt_line, pcov = curve_fit(spf.func_line, x[idx], y[idx], maxfev=10000)
    return spf.func_line(chi, *popt_line)


def fit_psi20_psi60_old(psi, phi, chi, chmin=3, chmax=np.inf):
    def func_psi0(x, a0, a1):
        # y = a0 * x / np.log(x) + a1
        y = a0 * np.log(np.exp(x) - x) + a1
        return y

    # mathematically func_psi0 and func_psi0_2 are same, but numerically, func_psi0 is better for
    #    fitting process, while func_psi0_2 is better for calculate larger chi
    def func_psi0_2(x, a0, a1):
        y = a0 * x / np.log(x) + a1
        return y

    tpsi = psi.loc[phi][0]
    tx = tpsi.index
    ty = tpsi.values
    idxi = np.isfinite(tx) & np.isfinite(ty) & (tx > chmin) & (tx < chmax)
    fit_psi0_line = spf.fit_line(None, tx[idxi], ty[idxi], 3, np.inf, ifprint=False)
    fit_psi0, pcov = curve_fit(func_psi0, tx[idxi], ty[idxi], p0=fit_psi0_line,
                               bounds=((0, 0), (np.inf, np.inf)), maxfev=10000)
    return func_psi0_2(chi, *fit_psi0)


def fit_psi20_psi60(psi, phi, chi, chmin=3, chmax=np.inf):
    # try to fit x/ln(x) = b0*y + b1, so we have {a0 = 1/b0, a1 = -b1/b0}
    #   where y = a0 * x / np.log(x) + a1.
    tpsi = psi.loc[phi][0]
    tx = tpsi.index
    ty = tpsi.values
    idxi = np.isfinite(tx) & np.isfinite(ty) & (chmin <= tx) & (tx <= chmax)
    fit_x = ty[idxi]
    fit_y = tx[idxi] / np.log(tx[idxi])
    b0, b1 = spf.fit_line(None, fit_x, fit_y, 0, np.inf, ifprint=False)
    a0 = 1 / b0
    a1 = -b1 / b0
    return a0 * chi / np.log(chi) + a1


def psi20_psi30_psi60(phi, chi):
    def func_psi_tail_0(x, a0, a1):
        # fit function of psi2 and psi6 in bulk fluid
        # y = a0 + a1 * x / np.log(x)
        y = a0 + a1 * np.log(np.exp(x) - x)
        return y

    if psi2.index.levels[1].values.min() <= chi <= psi2.index.levels[1].values.max():
        # in the range, intepolate
        psi2i_0 = intp_psi_tail(psi2, phi, chi, 0)
        psi3i_0 = intp_psi_tail(psi3, phi, chi, 0)
        psi6i_0 = -intp_psi_tail(-psi6, phi, chi, 0)
    else:
        # out of the range, fit
        psi2i_0 = fit_psi20_psi60(psi2, phi, chi, 3, np.inf)
        psi3i_0 = fit_psi_tail(psi3, phi, chi, 0, 3, np.inf)
        psi6i_0 = -fit_psi20_psi60(-psi6, phi, chi, 3, np.inf)
    return psi2i_0, psi3i_0, psi6i_0


def psi2_psi3_psi6(phi, chi, zft):
    # intepolate or fit psi2, psi3, psi6
    if psi2.index.levels[1].values.min() <= chi <= psi2.index.levels[1].values.max():
        # in the range, intepolate
        psi2i_use = intp_psi_tail(psi2, phi, chi, zft)
        psi3i_use = intp_psi_tail(psi3, phi, chi, zft)
        psi6i_use = intp_psi_tail(psi6, phi, chi, zft)
    else:
        # out of the range, fit
        psi2i_use = fit_psi_tail(psi2, phi, chi, zft, 4, 8)
        psi3i_use = fit_psi_tail(psi3, phi, chi, zft, 4, 8)
        psi6i_use = fit_psi_tail(psi6, phi, chi, zft, 4, 8)
    #     else:
    #         # out of the range, fit
    #         psi2i_use = fit_psi_tail(psi2, phi, chi, zft, 1, 3)
    #         psi3i_use = fit_psi_tail(psi3, phi, chi, zft, 1, 3)
    #         psi6i_use = fit_psi_tail(psi6, phi, chi, zft, 1, 3)
    return psi2i_use, psi3i_use, psi6i_use

def _plot_fft(use_t, ty, f2, Pxx_den, figsize=np.array((16, 9)) * 0.5, dpi=100, ):
    fig, axs = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
    axi = axs[0]
    axi.plot(use_t, ty, '.-')
    axi = axs[1]
    axi.loglog(f2, Pxx_den, '.')
    tpk = signal.find_peaks(Pxx_den)[0]
    fft_abs_pk = Pxx_den[tpk]
    freq_pk = f2[tpk]
    tidx = np.argsort(fft_abs_pk)[-show_prim_freq:]
    axi.loglog(freq_pk[tidx], fft_abs_pk[tidx], '*')
    t1 = 'starred freq: \n' + '\n'.join(['$%.5f$' % freq_pk[ti] for ti in tidx])
    axi.text(axi.get_xlim()[0] * 1.1, axi.get_ylim()[0] * 2, t1)
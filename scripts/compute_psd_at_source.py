import mne

method = "eLORETA"
snr = 3.
lambda2 = 1. / snr ** 2


def compute(raw, fwd_model):
    freq_bands = dict(
        delta=(2, 4), theta=(5, 7), alpha=(8, 13), beta=(15, 29), gamma=(30, 45))
    topos = dict(vv=dict(), opm=dict())
    stcs_dict = dict(vv=dict(), opm=dict())

    noise_cov = mne.compute_raw_covariance(raw)
    inverse_operator = mne.minimum_norm.make_inverse_operator(
        raw.info, forward=fwd_model, noise_cov=noise_cov, verbose=True)

    stc_psd, sensor_psd = mne.minimum_norm.compute_source_psd(
        raw, inverse_operator, lambda2=lambda2, method=method,
        dB=False, return_sensor=True, verbose=True)

    topo_norm = sensor_psd.data.sum(axis=1, keepdims=True)
    stc_norm = stc_psd.sum()  # same operation on MNE object, sum across freqs
    # Normalize each source point by the total power across freqs

    for band, limits in freq_bands.items():
        data = sensor_psd.copy().crop(*limits).data.sum(axis=1, keepdims=True)
        topos[band] = mne.EvokedArray(
            100 * data / topo_norm, sensor_psd.info)
        stcs_dict[band] = \
            100 * stc_psd.copy().crop(*limits).sum() / stc_norm.data

    return topos, stcs_dict

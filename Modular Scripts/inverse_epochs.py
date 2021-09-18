import mne
from mne.minimum_norm import make_inverse_operator, apply_inverse, apply_inverse_epochs, apply_inverse_raw

method = "eLORETA"
snr = 3.
lambda2 = 1. / snr ** 2


def apply(data, inverse_operator, raw=True):
    if raw:
        stc_isc = apply_inverse_raw(
            data, inverse_operator, lambda2=lambda2, method=method)
    else:
        stc_isc = apply_inverse_epochs(
            data, inverse_operator, lambda2=lambda2, method=method)

    return stc_isc

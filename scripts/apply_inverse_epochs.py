import mne
from mne.minimum_norm import make_inverse_operator, apply_inverse, apply_inverse_epochs, apply_inverse_raw

method = "eLORETA"
snr = 3.
lambda2 = 1. / snr ** 2


def apply(raw,inverse_operator):
    stc_isc = apply_inverse_raw(raw, inverse_operator, lambda2,
                             method=method, pick_ori=None, verbose=True)
    return stc_isc
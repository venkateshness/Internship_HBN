import mne
from mne.minimum_norm.inverse import apply_inverse, apply_inverse_epochs


method = "eLORETA"
snr = 3.0
lambda2 = 1.0 / snr**2


def epochs(epochs, inverse_operator):
    stcs = apply_inverse_epochs(
        epochs, inverse_operator, lambda2=lambda2, method=method, verbose=True
    )

    return stcs


import mne
from mne.minimum_norm import compute_source_psd, source_band_induced_power
from mne.datasets import sample
from mne.minimum_norm.inverse import apply_inverse, apply_inverse_epochs
from mne.minimum_norm.time_frequency import compute_source_psd_epochs




method ='eLORETA'
snr = 3.
lambda2 = 1. / snr ** 2
def psd(epochs,inverse_operator):
    bands = dict(alpha=[8, 12])

    data_path = sample.data_path()
    label_name ='Aud-rh.label' # Have to use 2 labels at the same, but will deal with this later
    fname_label = data_path + '/MEG/sample/labels/%s' % label_name
    label_name2 = 'Aud-lh.label'
    fname_label2 = data_path + '/MEG/sample/labels/%s' % label_name2
    label = mne.read_label(fname_label)
    label2 = mne.read_label(fname_label2)
    bihemi = mne.BiHemiLabel(label,label2)

    stcs =  apply_inverse_epochs(epochs,inverse_operator, lambda2=lambda2,
                                    method=method,
                                    verbose=True)

    return stcs
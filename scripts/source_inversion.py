import mne
from mne.datasets import fetch_fsaverage
import os.path as op
from mne.minimum_norm import make_inverse_operator, apply_inverse, apply_inverse_epochs


fs_dir = fetch_fsaverage(verbose=True)
subjects_dir = op.dirname(fs_dir)


subject = 'fsaverage' # Subject ID for the MRI-head transformation
trans = 'fsaverage'  # MNE has a built-in fsaverage transformation
source_space = op.join(fs_dir, 'bem', 'fsaverage-ico-5-src.fif') 
bem = op.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')

method = "eLORETA"
snr = 3.
lambda2 = 1. / snr ** 2

def inversion(raw, epochs,fwd_model,tmax,tmin,method,cov):
    #fwd_model = mne.make_forward_solution(raw.info, trans=trans, src=source_space, bem=bem, eeg=True, mindist=5.0)
    
    #cov = mne.compute_covariance(epochs,tmax=tmax,method=method,tmin=tmin)
    
    inverse_operator = make_inverse_operator(raw.info, fwd_model, cov)

    return cov, inverse_operator

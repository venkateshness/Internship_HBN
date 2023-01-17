import mne
from mne.datasets import fetch_fsaverage
import os.path as op


fs_dir = fetch_fsaverage(verbose=True)
subjects_dir = op.dirname(fs_dir)


subject = "fsaverage"  # Subject ID for the MRI-head transformation
trans = "fsaverage"  # MNE has a built-in fsaverage transformation
source_space = op.join(fs_dir, "bem", "fsaverage-ico-5-src.fif")
bem = op.join(fs_dir, "bem", "fsaverage-5120-5120-5120-bem-sol.fif")


def fwd(raw):
    #    """Produces the fsaverage-based forward model with 20484 vertices across both hemispheres

    fwd_model = mne.make_forward_solution(
        raw.info, trans=trans, src=source_space, bem=bem, eeg=True, mindist=5.0
    )

    return fwd_model

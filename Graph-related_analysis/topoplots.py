#%%
import numpy as np
import scipy
import matplotlib.pyplot as plt
import scipy.stats
import matplotlib as mpl
from nilearn.regions import signals_to_img_labels

# load nilearn label masker for inverse transform
from nilearn.input_data import NiftiLabelsMasker, NiftiMasker
from nilearn.datasets import fetch_icbm152_2009
from nilearn import image, plotting
from nilearn import datasets
import nilearn
import os
from collections import defaultdict
import seaborn as sns

envelope_signal_bandpassed = np.load(
    "/users2/local/Venkatesh/Generated_Data/25_subjects_copy_FOR_TESTING/envelope_signal_bandpassed_with_beta_dichotomy.npz",
    mmap_mode="r",
)

# %%
def slicing(what_to_slice, where_to_slice, axis):
    """Temporal Slicing. Function to slice at the temporal axis.
    Args:
        what_to_slice (array): The array to do the temporal slicing for; the input dim = subject x entire video duration
        where_to_slice (array of "range"): The period(s) when to do the temporal slicing in; dim = 1D array
        axis (int): The axis of the array to do the temporal slicing on

    Returns:
        array: An array temporally sliced; dim = subject x `total_duration_in_samples` (375 = 3s)
    """
    array_to_append = list()
    if axis > 2:
        array_to_append.append(what_to_slice[:, :, where_to_slice])
    else:
        print("size for the what_to_slice:", np.shape(what_to_slice))
        array_to_append.append(what_to_slice[:, where_to_slice])
    return array_to_append


def stats_SEM(freqs):
    """SEM estimation -- Standard error of the Mean

    Args:
        freqs (dict): The grand-averaged graph smoothness to apply the SEM on

    Returns:
        array: SEMed graph smoothness
    """
    return scipy.stats.sem(freqs, axis=1)


trials = [8, 56, 68, 74, 86, 132, 162]
fs = 125
subjects = 25
baseline_duration_of_900ms_in_samples = (
    113  # baseline = -1000ms to -100ms, so 900ms; since fs = 125, 900 ms = 113 samples
)
total_duration_in_samples = 375
total_roi = 360
os.chdir("/homes/v20subra/S4B2/")
path_Glasser = "GSP/Glasser_masker.nii.gz"

mnitemp = fetch_icbm152_2009()
mask_mni = image.load_img(mnitemp["mask"])
glasser_atlas = image.load_img(path_Glasser)


def plot(band, threshold_pc):
    fig, ax = plt.subplots(2, 4, figsize=(25, 25))
    for i in range(len(trials)):  # looping over each trials
        indices = np.hstack([np.arange(trials[i] * fs - fs, trials[i] * fs + 2 * fs)])

        env = np.squeeze(
            slicing(envelope_signal_bandpassed[f"{band}"], indices, axis=3)
        )
        env_reordered = np.squeeze(
            np.moveaxis(env, [0, 1, 2], [0, 2, 1])
        )  # swapping axis for easier handling
        assert np.shape(env_reordered) == (
            subjects,
            total_duration_in_samples,
            total_roi,
        )

        env_reordered_baseline = env_reordered[
            :, :baseline_duration_of_900ms_in_samples, :
        ]
        env_reordered_baseline_averaged = np.expand_dims(
            np.mean(env_reordered_baseline, axis=1), 1
        )  # adding dim to match the dim for both subtracting elements
        baseline_done = (
            env_reordered - env_reordered_baseline_averaged
        ) / env_reordered_baseline_averaged

        assert np.shape(baseline_done) == (
            subjects,
            total_duration_in_samples,
            total_roi,
        )

        subjects_averaged = np.mean(baseline_done, axis=0)
        assert np.shape(subjects_averaged) == (total_duration_in_samples, total_roi)
        to_identify_peak = np.mean(subjects_averaged, axis=1)  # averaging thru ROI

        peak = np.where(to_identify_peak == np.max(to_identify_peak))[0]
        signal_at_peak = np.squeeze(subjects_averaged[peak])
        assert np.shape(signal_at_peak) == (total_roi,)
        signal = []
        U0_brain = []

        percentile = np.percentile(signal_at_peak, threshold_pc)
        zeroes = np.zeros(360)
        to_plot = np.squeeze((signal_at_peak > percentile))

        zeroes[to_plot] = signal_at_peak[to_plot]
        # print('sum of zeroes', np.sum(zeroes))

        signal = np.expand_dims(zeroes, axis=(0, 2))  # add dimension 1 to signal array

        U0_brain = signals_to_img_labels(signal, path_Glasser, mnitemp["mask"])

        plotting.plot_surf(
            U0_brain,
            title=f"Trial {i+1}",
            colorbar=True,
            plot_abs=False,
            display_mode="lzr",
            axes=ax.flatten()[i],
            cmap="hsv",
            figure=fig,
        )

        print("shape", np.shape(subjects_averaged[peak].T))

    fig.suptitle(
        f"Env. {band} on the brain, thresholded at {threshold_pc} %ile", fontsize=20
    )
    fig.savefig(
        f"/homes/v20subra/S4B2/3Source_Inversion_full_stack/Results/topoplots/{band}.jpg"
    )
    plt.show()


bands = ["theta", "alpha", "lower_beta", "higher_beta"]

for i in range(len(bands)):
    plot(bands[i], 90)


# %%
from nilearn import datasets

os.chdir("/homes/v20subra/S4B2/")
path_Glasser = "GSP/Glasser_masker.nii.gz"

mnitemp = fetch_icbm152_2009()
mask_mni = image.load_img(mnitemp["mask"])
glasser_atlas = image.load_img(path_Glasser)


fsaverage = datasets.fetch_surf_fsaverage()

plotting.plot_surf(
    surf_mesh=fsaverage["infl_left"],
    surf_map=np.ones(
        20484,
    ),
    hemi=["left", "right"],
)


# %%
a = nilearn.surface.load_surf_data(fsaverage["infl_left"])
b = nilearn.surface.load_surf_data(fsaverage["infl_right"])

# %%
d.keys()

# %%

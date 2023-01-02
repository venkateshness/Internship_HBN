#%%
from cProfile import label
import numpy as np
import importlib
import os
import matplotlib.pyplot as plt
from pkg_resources import to_filename
from scipy.fftpack import shift
from sklearn import cluster
from torch import eig

os.chdir("/homes/v20subra/S4B2/")
from scipy.stats import binom
from Modular_Scripts import graph_setup

importlib.reload(graph_setup)
from collections import defaultdict
import scipy.linalg as la

laplacian = graph_setup.NNgraph("FC")
[eigvals, eigevecs] = la.eigh(laplacian)

total_no_of_events = "30_events"
envelope_signal_bandpassed_bc_corrected = np.load(
    f"/users2/local/Venkatesh/Generated_Data/25_subjects_new/eloreta_cortical_signal_thresholded/bc_and_thresholded_signal/{total_no_of_events}/0_percentile_with_zscore.npz"
)

video_duration = 88
subjects = 25
regions = 360
number_of_clusters = 3
baseline_in_samples = 25
post_onset_in_samples = 63
n_surrogate = 19

gft_band_wise = defaultdict(dict)


def gft(signal):  # no change
    assert np.shape(signal) == (subjects, regions, video_duration)
    array_of_gft = list()

    for sub in range(subjects):
        signal_for_gft = signal[sub]
        transform = np.matmul(eigevecs.T, signal_for_gft)
        array_of_gft.append(transform)

    assert np.shape(array_of_gft) == (subjects, regions, video_duration)

    return array_of_gft


def signal_filtering(g_psd, low_freqs, high_freqs):  # updated
    assert np.shape(g_psd) == (subjects, regions, video_duration)

    lf_signal = list()
    hf_signal = list()

    for subs in range(subjects):
        g_psd_for_igft = g_psd[subs]
        assert np.shape(g_psd_for_igft) == (regions, video_duration)

        low_freq_signal = np.matmul(low_freqs, g_psd_for_igft)
        high_freqs_signal = np.matmul(high_freqs, g_psd_for_igft)

        lf_signal.append(low_freq_signal)
        hf_signal.append(high_freqs_signal)

    assert np.shape(lf_signal) == (subjects, regions, video_duration)
    assert np.shape(hf_signal) == (subjects, regions, video_duration)

    return lf_signal, hf_signal


def frobenius_norm(lf_signal, hf_signal, label):  # no change
    assert np.shape(lf_signal) == (subjects, regions, video_duration)
    assert np.shape(hf_signal) == (subjects, regions, video_duration)

    if label == "baseline":
        signal_lf = np.array(lf_signal)[:, :, :baseline_in_samples]
        signal_hf = np.array(hf_signal)[:, :, :baseline_in_samples]

    if label == "post_onset":
        signal_lf = np.array(lf_signal)[:, :, baseline_in_samples:]
        signal_hf = np.array(hf_signal)[:, :, baseline_in_samples:]

    normed_lf = np.linalg.norm(signal_lf, axis=-1)
    normed_hf = np.linalg.norm(signal_hf, axis=-1)

    assert np.shape(normed_lf) == (subjects, regions)
    assert np.shape(normed_hf) == (subjects, regions)

    return normed_lf, normed_hf


def SDIndex(signal_in_dict):  # no change
    lf_signal, hf_signal = signal_in_dict["lf"], signal_in_dict["hf"]
    index = hf_signal / lf_signal

    return index


def surrogacy(eigvector, signal):  # updated
    """Graph-informed Surrogacy control
    Args:
        eigvector (matrix): Eigenvector
        random_signs (matrix): Random sign change to flip the phase
        signal (array): Cortical brain/graph signal
    Returns:
        reconstructed_signal: IGFTed signal; recontructed, but with phase-flipped signal
    """
    surrogate_signal = list()
    for n in range(n_surrogate):

        np.random.seed(n)
        random_signs = np.round(
            np.random.rand(
                regions,
            )
        )
        random_signs[random_signs == 0] = -1
        random_signs = np.diag(random_signs)

        g_psd = np.matmul(eigevecs.T, signal)
        eigvector_manip = np.matmul(eigvector, random_signs)
        reconstructed_signal = np.matmul(eigvector_manip, g_psd)

        surrogate_signal.append(reconstructed_signal)

    assert np.shape(surrogate_signal) == (
        n_surrogate,
        subjects,
        regions,
        video_duration,
    )

    return surrogate_signal


def signal_to_SDI(lf_signal, hf_signal, normalization=False):  # no change
    # Norm
    normed_baseline_signal = defaultdict(dict)
    normed_post_onset_signal = defaultdict(dict)

    normed_baseline_signal["lf"], normed_baseline_signal["hf"] = frobenius_norm(
        lf_signal=lf_signal, hf_signal=hf_signal, label="baseline"
    )
    normed_post_onset_signal["lf"], normed_post_onset_signal["hf"] = frobenius_norm(
        lf_signal=lf_signal, hf_signal=hf_signal, label="post_onset"
    )

    SDIndex_baseline = SDIndex(normed_baseline_signal)
    SDIndex_post_onset = SDIndex(normed_post_onset_signal)

    assert np.shape(SDIndex_baseline) == (subjects, regions)
    assert np.shape(SDIndex_post_onset) == (subjects, regions)

    if normalization:
        empirical_SDIndex_bc = (SDIndex_post_onset - SDIndex_baseline) / np.mean(
            SDIndex_baseline, axis=0
        )
        mean_SDIndex_bc = np.mean(empirical_SDIndex_bc, axis=0)
    else:
        empirical_SDIndex_bc = SDIndex_post_onset - SDIndex_baseline
        mean_SDIndex_bc = np.mean(empirical_SDIndex_bc, axis=0)

    # SD Index
    def average_subs(dict):
        dict["lf"] = np.mean(dict["lf"], axis=0)
        dict["hf"] = np.mean(dict["hf"], axis=0)

        assert np.shape(dict["lf"]) == (regions,)

        return dict

    return empirical_SDIndex_bc, mean_SDIndex_bc


def band_wise_SDI(band, normalization):
    # GFT
    SDI_BC = list()
    # SDI_PO = list()

    for cluster_ in range(number_of_clusters):
        signal = envelope_signal_bandpassed_bc_corrected[f"{band}"][:, cluster_]
        psd = gft(signal)

        # Critical Freq identification for symmetric power dichotomy
        psd_abs_squared = np.power(np.abs(psd), 2)
        assert np.shape(psd_abs_squared) == (subjects, regions, video_duration)

        psd_abs_squared_averaged = np.mean(psd_abs_squared, axis=(0, 2))
        assert np.shape(psd_abs_squared_averaged) == (regions,)

        median_power = np.trapz(psd_abs_squared_averaged) / 2
        sum_of_freqs = 0
        i = 0
        while sum_of_freqs < median_power:
            sum_of_freqs = np.trapz(psd_abs_squared_averaged[:i])
            i += 1
        critical_freq = i - 1
        ### End of critical freq

        # Filters
        low_freqs = np.zeros((regions, regions))
        low_freqs[:, :critical_freq] = eigevecs[:, :critical_freq]

        high_freqs = np.zeros((regions, regions))
        high_freqs[:, critical_freq:] = eigevecs[:, critical_freq:]

        ########################################
        # Signal-filtering empirical data

        lf_signal, hf_signal = signal_filtering(psd, low_freqs, high_freqs)
        empirical_SDIndex_bc, meanSDIndex_bc = signal_to_SDI(
            lf_signal, hf_signal, normalization=normalization
        )

        ########################################
        # #############Surrogate data#############
        surrogate_signal = surrogacy(eigevecs, signal)
        surrogate_psd = [gft(surrogate_signal[n]) for n in range(n_surrogate)]

        assert np.shape(surrogate_psd) == (
            n_surrogate,
            subjects,
            regions,
            video_duration,
        )

        surrogate_lf_signal, surrogate_hf_signal = zip(
            *[
                signal_filtering(surrogate_psd[n], low_freqs, high_freqs)
                for n in range(n_surrogate)
            ]
        )
        assert np.shape(surrogate_lf_signal) == (
            n_surrogate,
            subjects,
            regions,
            video_duration,
        )

        surrogate_SDIndex_bc, _ = zip(
            *[
                signal_to_SDI(
                    surrogate_lf_signal[n],
                    surrogate_hf_signal[n],
                    normalization=normalization,
                )
                for n in range(n_surrogate)
            ]
        )
        assert np.shape(surrogate_SDIndex_bc) == (n_surrogate, subjects, regions)

        ## Comparison between stats and empirical data
        max_surrogate, min_surrogate = np.max(surrogate_SDIndex_bc, axis=0), np.min(
            surrogate_SDIndex_bc, axis=0
        )

        detection_max = np.sum(empirical_SDIndex_bc > max_surrogate, axis=0)
        detection_min = np.sum(empirical_SDIndex_bc < min_surrogate, axis=0)

        x = np.arange(1, 101)
        sf = binom.sf(x, 100, p=0.05)
        thr = np.min(np.where(sf < 0.05 / 360))
        thr = np.floor(subjects / 100 * thr) + 1

        significant_max = (detection_max > thr) * 1
        significant_min = (detection_min > thr) * 1

        idx = np.sort(
            np.unique(
                np.hstack(
                    [np.where(significant_max == 1), np.where(significant_min == 1)]
                )
            )
        )
        significant_SDI_bc = meanSDIndex_bc[idx]

        final_SDI_bc = np.zeros((360,))
        final_SDI_bc[idx] = meanSDIndex_bc[idx]

        # final_SDI_baseline[idx_baseline]  = significant_SDI_baseline
        # final_SDI_post_onset[idx_post_onset]  = significant_SDI_post_onset

        # final_SDI_baseline = np.log2(final_SDI_baseline/np.mean(surrogate_SDI_baseline, axis = (0,1)))
        # final_SDI_post_onset = np.log2(final_SDI_post_onset/np.mean(surrogate_SDI_post_onset, axis = (0,1)))

        SDI_BC.append(final_SDI_bc)
        # SDI_PO.append(mean_SDI_post_onset)

    return SDI_BC  # , SDI_PO


# band_wise_SDI('theta')
# band_wise_SDI('alpha')
# band_wise_SDI('low_beta')
# band_wise_SDI('high_beta')

normalization_condition = True

SDI = defaultdict(dict)
for labels, signal in envelope_signal_bandpassed_bc_corrected.items():
    SDI[f"{labels}"] = band_wise_SDI(f"{labels}", normalization=normalization_condition)


# %%
import matplotlib
from nilearn.regions import signals_to_img_labels

# load nilearn label masker for inverse transform
from nilearn.input_data import NiftiLabelsMasker, NiftiMasker
from nilearn.datasets import fetch_icbm152_2009
from nilearn import image, plotting
from nilearn import datasets
from os.path import join as opj
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm


c = 1
if total_no_of_events == "19_events":
    event_type = ["Audio", "+ve Offset", "-ve offset"]
else:
    event_type = ["C1", "C2", "C3"]
for_j = ["BL", "PO"]

if normalization_condition:
    to_save = f"/homes/v20subra/S4B2/Graph-related_analysis/SDI_results/{total_no_of_events}/BC/"

else:
    to_save = f"/homes/v20subra/S4B2/Graph-related_analysis/SDI_results/{total_no_of_events}/differenced/"

for band, signal_band in SDI.items():
    for i in range(number_of_clusters):

        signal = signal_band[i]

        path_Glasser = "/homes/v20subra/S4B2/GSP/Glasser_masker.nii.gz"

        mnitemp = fetch_icbm152_2009()
        mask_mni = image.load_img(mnitemp["mask"])
        glasser_atlas = image.load_img(path_Glasser)

        signal = np.expand_dims(signal, axis=0)  # add dimension 1 to signal array

        U0_brain = signals_to_img_labels(signal, path_Glasser, mnitemp["mask"])

        plotting.plot_img_on_surf(
            U0_brain,
            colorbar=True,
            title=f"{band} / {event_type[i]} ",
            output_file=f"{to_save}{c}",
        )
        plt.show()
        # U0_brain.to_filename(f'Graph-related_analysis/SDI_results/stat_maps/{band}_{i}_{j}.nii.gz')
        c += 1

# %%

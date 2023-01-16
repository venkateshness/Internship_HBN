#%%
from cProfile import label
import numpy as np
import importlib
import os
import matplotlib.pyplot as plt


os.chdir("/homes/v20subra/S4B2/")
from scipy.stats import binom
from Modular_Scripts import graph_setup

importlib.reload(graph_setup)
from collections import defaultdict
import scipy.linalg as la

laplacian = graph_setup.NNgraph("SC")
[eigvals, eigevecs] = la.eigh(laplacian)

total_no_of_events = "30_events"
number_of_events = 30
envelope_signal_bandpassed_bc_corrected = np.load(
    f"/users2/local/Venkatesh/Generated_Data/25_subjects_new/eloreta_cortical_signal_thresholded/bc_and_thresholded_signal/{total_no_of_events}/envelope_sliced_signal.npz"
)
# envelope_signal_bandpassed_bc_corrected = np.load(
#     "/users2/local/Venkatesh/Generated_Data/25_subjects_new/eloreta_cortical_signal_thresholded/0_percentile.npz",
#     mmap_mode="r",
# )
video_duration = 88
subjects = 25
regions = 360
baseline_in_samples = 25
post_onset_in_samples = 63
n_surrogate = 19
n_clusters = 3

clusters = np.load(
    f"/homes/v20subra/S4B2/AutoAnnotation/dict_of_clustered_events_{total_no_of_events}.npz"
)

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


def signal_to_SDI(lf_signal, hf_signal):  # no change
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
    # SD Index

    def average_subs(dict):
        dict["lf"] = np.mean(dict["lf"], axis=0)
        dict["hf"] = np.mean(dict["hf"], axis=0)

        assert np.shape(dict["lf"]) == (regions,)

        return dict

    mean_normed_baseline = average_subs(normed_baseline_signal)
    mean_normed_post_onset = average_subs(normed_post_onset_signal)

    mean_SDIndex_baseline = SDIndex(mean_normed_baseline)
    mean_SDIndex_post_onset = SDIndex(mean_normed_post_onset)

    return (
        SDIndex_baseline,
        SDIndex_post_onset,
        mean_SDIndex_baseline,
        mean_SDIndex_post_onset,
    )


def band_wise_SDI(band):
    # GFT
    SDI_BL = list()
    SDI_PO = list()
    bc_SDI = list()
    diff_SDI = list()
    raw_SDImap_differenced = list()
    raw_SDImap_bl = list()
    surrogate_baseline_vanilla = list()

    for cluster_ in range(number_of_events):
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
        (
            SDI_baseline,
            SDI_post_onset,
            mean_SDI_baseline,
            mean_SDI_post_onset,
        ) = signal_to_SDI(lf_signal, hf_signal)

        print(np.shape(SDI_baseline))
        ########################################
        #############Surrogate data#############
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

        surrogate_SDI_baseline, surrogate_SDI_post_onset, _, _ = zip(
            *[
                signal_to_SDI(surrogate_lf_signal[n], surrogate_hf_signal[n])
                for n in range(n_surrogate)
            ]
        )
        assert np.shape(surrogate_SDI_baseline) == (n_surrogate, subjects, regions)
        surrogate_baseline_vanilla.append(surrogate_SDI_baseline)

        ## Comparison between stats and empirical data
        max_baseline, min_baseline = np.max(surrogate_SDI_baseline, axis=0), np.min(
            surrogate_SDI_baseline, axis=0
        )
        max_post_onset, min_post_onset = np.max(
            surrogate_SDI_post_onset, axis=0
        ), np.min(surrogate_SDI_post_onset, axis=0)


        raw_SDImap_idx_BL = np.array ((SDI_baseline > max_baseline) *1 + (SDI_baseline < min_baseline) *1, dtype = bool)
        raw_SDImap_idx_PO = np.array ((SDI_post_onset > max_post_onset) *1 + (SDI_post_onset < min_post_onset) *1, dtype = bool)


        final_raw_SDImap_bl = SDI_baseline * raw_SDImap_idx_BL
        final_raw_SDImap_post_onset = SDI_post_onset * raw_SDImap_idx_PO

        # final_raw_SDImap_differenced = raw_SDImap_post_onset - raw_SDImap_bl
        raw_SDImap_bl.append(SDI_baseline)

        detection_max_baseline = np.sum(SDI_baseline > max_baseline, axis=0)
        detection_min_baseline = np.sum(SDI_baseline < min_baseline, axis=0)

        detection_max_post_onset = np.sum(SDI_post_onset > max_post_onset, axis=0)
        detection_min_post_onset = np.sum(SDI_post_onset < min_post_onset, axis=0)

        x = np.arange(1, 101)
        sf = binom.sf(x, 100, p=0.05)
        thr = np.min(np.where(sf < 0.05 / 360))
        thr = np.floor(subjects / 100 * thr) + 1

        significant_max_baseline = (detection_max_baseline > thr) * 1
        significant_min_baseline = (detection_min_baseline > thr) * 1

        significant_max_post_onset = (detection_max_post_onset > thr) * 1
        significant_min_post_onset = (detection_min_post_onset > thr) * 1

        idx_baseline = np.sort(
            np.unique(
                np.hstack(
                    [
                        np.where(significant_max_baseline == 1),
                        np.where(significant_min_baseline == 1),
                    ]
                )
            )
        )
        idx_post_onset = np.sort(
            np.unique(
                np.hstack(
                    [
                        np.where(significant_max_post_onset == 1),
                        np.where(significant_min_post_onset == 1),
                    ]
                )
            )
        )

        significant_SDI_baseline = mean_SDI_baseline[idx_baseline]
        significant_SDI_post_onset = mean_SDI_post_onset[idx_post_onset]
        
        
        final_SDI_baseline = np.ones((360,))
        final_SDI_post_onset = np.ones((360,))

        final_SDI_baseline[idx_baseline] = significant_SDI_baseline
        final_SDI_post_onset[idx_post_onset] = significant_SDI_post_onset

        # print(sum(final_SDI_baseline != 1))

        bc_SDI_values = (
            np.array(final_SDI_post_onset) - np.array(final_SDI_baseline)
        ) / np.mean(final_SDI_baseline)
        diff_SDI_values = np.array(final_SDI_post_onset) - np.array(final_SDI_baseline)

        final_SDI_baseline = np.log2(
            final_SDI_baseline / np.mean(surrogate_SDI_baseline, axis=(0, 1))
        )
        final_SDI_post_onset = np.log2(
            final_SDI_post_onset / np.mean(surrogate_SDI_post_onset, axis=(0, 1))
        )

        SDI_BL.append(final_SDI_baseline)
        SDI_PO.append(final_SDI_post_onset)
        bc_SDI.append(bc_SDI_values)
        diff_SDI.append(diff_SDI_values)

    return raw_SDImap_bl, surrogate_baseline_vanilla #SDI_BL, SDI_PO, bc_SDI, diff_SDI, 


# band_wise_SDI('theta')
# band_wise_SDI('alpha')
# band_wise_SDI('low_beta')
# band_wise_SDI('high_beta')

SDI = defaultdict(dict)
BC = defaultdict(dict)
differenced = defaultdict(dict)
raw_SDImap_baseline = defaultdict(dict)
surrogate_baseline = defaultdict(dict)

for labels, signal in envelope_signal_bandpassed_bc_corrected.items():
    # index_bl, index_po, bc_index, diff_index, raw_SDImap_bl = band_wise_SDI(f"{labels}")
    raw_SDImap_bl, surrogate_bl = band_wise_SDI(f"{labels}")
    # SDI[f"{labels}"] = index_bl, index_po
    # BC[f"{labels}"] = bc_index
    # differenced[f"{labels}"] = diff_index
    # raw_SDImap_baseline[f"{labels}"] = raw_SDImap_bl
    raw_SDImap_baseline[f"{labels}"] = raw_SDImap_bl
    surrogate_baseline[f"{labels}"] = surrogate_bl

counter = 0
dic_order = defaultdict(dict)

for i, j in clusters.items():
    dic_order[f'{i}'] = np.arange(counter, counter+ len(j))
    counter+= len(j)


## Cluster averaging
def cluster_averaging(dict_to_average, is_SDI):
    dic = defaultdict(dict)
    for band_label, band_signal in dict_to_average.items():
        cluster_level = list()
        for event_label, event_index in dic_order.items():
            if is_SDI:
                cluster_level.append( np.mean(  np.array(band_signal)[:, event_index], axis = 1) )

            if is_SDI == False:    
                cluster_level.append( np.mean(  np.array(band_signal)[event_index], axis = 0) )
        
        if is_SDI==False:
            assert np.shape(cluster_level) == (n_clusters, regions)
        
        if is_SDI==True:
            assert np.shape(cluster_level) == (n_clusters, 2, regions)
    
        dic[f'{band_label}'] = cluster_level
    
    return dic

differenced_f =  cluster_averaging(differenced, is_SDI = False)
BC_f =  cluster_averaging(BC, is_SDI = False)
SDI_f =  cluster_averaging(SDI, is_SDI = True)

#%%

np.savez_compressed(f"/users2/local/Venkatesh/Generated_Data/25_subjects_new/SDI/SDI_baseline_vanilla", **raw_SDImap_baseline)
np.savez_compressed(f"/users2/local/Venkatesh/Generated_Data/25_subjects_new/SDI/surrogate_SDI_baseline_vanilla", **surrogate_baseline)

#%%
import matplotlib
from nilearn.regions import signals_to_img_labels
from nilearn.datasets import fetch_icbm152_2009
from nilearn import image, plotting
import matplotlib.pyplot as plt


BL_or_PO = ['BL', 'PO']
if total_no_of_events == "19_events":
    event_type = ["Audio", "+ve Offset", "-ve offset"]
else:
    event_type = ["C1", "C2", "C3"]

counter = 1
for band, signal_band in differenced_f.items():
    for event_group in range(n_clusters):
        # for j in range(2):
            signal = signal_band[event_group]

            path_Glasser = "/homes/v20subra/S4B2/GSP/Glasser_masker.nii.gz"

            mnitemp = fetch_icbm152_2009()
            mask_mni = image.load_img(mnitemp["mask"])
            glasser_atlas = image.load_img(path_Glasser)
            
            signal = np.expand_dims(signal, axis=0)

            U0_brain = signals_to_img_labels(signal, path_Glasser, mnitemp["mask"])

            plot = plotting.plot_img_on_surf(
                U0_brain, title=f"{band} / {event_type[event_group]}", threshold=0.05, output_file = f'/homes/v20subra/S4B2/Graph-related_analysis/SDI_results/{total_no_of_events}/Differenced/{counter}')
            counter+=1
            
            plt.show()
# %%

# video_watching = np.load(
#     "/users2/local/Venkatesh/Generated_Data/25_subjects_new/video_watching_bundle_STC_parcellated.npz"
# )["video_watching_bundle_STC_parcellated"]

if total_no_of_events == "19_events":
    event_type = ["Audio", "+ve Offset", "-ve offset"]
else:
    event_type = ["C1", "C2", "C3"]

envelope_signal_bandpassed_bc_corrected = np.load(
    f"/users2/local/Venkatesh/Generated_Data/25_subjects_new/eloreta_cortical_signal_thresholded/bc_and_thresholded_signal/{total_no_of_events}/0_percentile_with_zscore.npz"
)

counter = 0
for labels, band in envelope_signal_bandpassed_bc_corrected.items():

        percentile = 85
        # video_baseline_averaged = np.mean(band[:, event_type, :, :baseline_in_samples], axis = -1)
        video_po_averaged = np.mean(band[:, :, :, baseline_in_samples :  ], axis = -1)
        
        if total_no_of_events == '19_events':
            video_po_event_averaged = np.mean(video_po_averaged[:, :2], axis = 1)
        
        else:
            video_po_event_averaged = np.mean(video_po_averaged, axis = 1)

        signal = np.mean(video_po_event_averaged , axis=0)
        

        path_Glasser = "/homes/v20subra/S4B2/GSP/Glasser_masker.nii.gz"

        mnitemp = fetch_icbm152_2009()
        mask_mni = image.load_img(mnitemp["mask"])
        glasser_atlas = image.load_img(path_Glasser)

        perc_up = np.percentile(signal, percentile)
        perc_down = np.percentile(signal, 100 - percentile)

        signal_ = np.zeros(360,)
        signal_[signal > perc_up] = signal[signal > perc_up]
        signal_[signal < perc_down] = signal[signal < perc_down]
        
        signal_f = np.expand_dims(signal_, axis=0)

        U0_brain = signals_to_img_labels(signal_f, path_Glasser, mnitemp["mask"])

        plotting.plot_img_on_surf(U0_brain, colorbar=True, title = f'{labels} / CS', threshold = 0.0001)#, output_file = f'/homes/v20subra/S4B2/Graph-related_analysis/SDIvsCorticalSignal/{total_no_of_events}/grand_average/{counter}')
        counter+=1
        plt.show()
# %%


np.shape(surrogate_baseline['theta'])
# %%

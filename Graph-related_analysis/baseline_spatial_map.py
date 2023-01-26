#%%
import numpy as np
from collections import defaultdict
import os
from pandas import cut

from tenacity import retry
os.chdir("/homes/v20subra/S4B2/")

from Modular_Scripts import graph_setup
import importlib
importlib.reload(graph_setup)
from collections import defaultdict
import scipy.linalg as la

envelope_signal_bandpassed = np.load(
    "/users2/local/Venkatesh/Generated_Data/25_subjects_new/eloreta_cortical_signal_thresholded/0_percentile.npz",
    mmap_mode="r",
)

duration = 21250
subjects = 25
regions = 360
event_type = "30_events"
number_of_events = 30
fs = 125
pre_stim_in_samples = 25
post_stim_samples = 63
video_duration_bc = pre_stim_in_samples

clusters = np.load(
    f"/homes/v20subra/S4B2/AutoAnnotation/dict_of_clustered_events_{event_type}.npz"
)

envelope_signal_bandpassed_bc_corrected = np.load(
    f"/users2/local/Venkatesh/Generated_Data/25_subjects_new/eloreta_cortical_signal_thresholded/bc_and_thresholded_signal/{event_type}/0_percentile_with_zscore_events_wise.npz"
)

laplacian = graph_setup.NNgraph("SC")
[eigvals, eigevecs] = la.eigh(laplacian)
time_periods = list()

for group in range(3):
    time_periods.append(clusters[f'{group}'] )

time_samples = sorted(np.hstack(time_periods))


def slicing(band, signal):

    event_wise_slicing = []
    for sample in time_samples:
        
        onset = sample * fs
        prestim = onset - pre_stim_in_samples
        poststim = onset + post_stim_samples
        sliced = signal[:, :, prestim : poststim]
        event_wise_slicing.append(sliced)
    
    event_wise_slicing = np.moveaxis(event_wise_slicing, [0,1,2,3], [1, 0,2, 3])

    return event_wise_slicing



envelope_signal_bandpassed_sliced = defaultdict(dict)
for label, signal in envelope_signal_bandpassed.items():
    envelope_signal_bandpassed_sliced[f'{label}'] = slicing(label, signal)


# np.savez_compressed('/users2/local/Venkatesh/Generated_Data/25_subjects_new/eloreta_cortical_signal_thresholded/bc_and_thresholded_signal/30_events/envelope_sliced_signal.npz', **envelope_signal_bandpassed_sliced)
# %%

empirical_sdi_baseline = np.load("/users2/local/Venkatesh/Generated_Data/25_subjects_new/SDI/SDI_baseline_vanilla.npz")['alpha']
surrogate_sdi_baseline = np.load("/users2/local/Venkatesh/Generated_Data/25_subjects_new/SDI/surrogate_SDI_baseline_vanilla.npz")['alpha']

# %%
from scipy.stats import binom

x = np.arange(1, 101)
sf = binom.sf(x, 100, p=0.05)
thr = np.min(np.where(sf < 0.05 / 360))
thr = np.floor(subjects / 100 * thr) + 1

SDI_f = list()

for i in range(30):

    # surrogate_sdi_baseline = np.moveaxis(surrogate_sdi_baseline, [0, 1, 2, 3, 4], [0, 1, ])
    max_baseline, min_baseline = np.max(surrogate_sdi_baseline[i], axis=0), np.min(surrogate_sdi_baseline[i], axis=0)
        
    detection_max_baseline = np.sum(empirical_sdi_baseline[i] > max_baseline, axis=0)
    detection_min_baseline = np.sum(empirical_sdi_baseline[i] < min_baseline, axis=0)

    significant_max_baseline = (detection_max_baseline > thr) * 1
    significant_min_baseline = (detection_min_baseline > thr) * 1
    


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
    significant_SDI_baseline = empirical_sdi_baseline[i,:,idx_baseline]

    subj_wise_SDI = list()

    for sub in range(subjects):

        final_SDI_baseline = np.ones(360,)
        final_SDI_baseline[idx_baseline] = significant_SDI_baseline[:,sub]


        final_SDI_baseline = np.log2(final_SDI_baseline)
        subj_wise_SDI.append(final_SDI_baseline)
    
    SDI_f.append(subj_wise_SDI)
# %%


import matplotlib
from nilearn.regions import signals_to_img_labels
from nilearn.datasets import fetch_icbm152_2009
from nilearn import image, plotting
import matplotlib.pyplot as plt


counter = 1
for i in range(30):
    signal = np.array(SDI_f)[i,3,:]

    path_Glasser = "/homes/v20subra/S4B2/GSP/Glasser_masker.nii.gz"

    mnitemp = fetch_icbm152_2009()
    mask_mni = image.load_img(mnitemp["mask"])
    glasser_atlas = image.load_img(path_Glasser)

    signal = np.expand_dims(signal, axis=0)

    U0_brain = signals_to_img_labels(signal, path_Glasser, mnitemp["mask"])

    plot = plotting.plot_img_on_surf(
        U0_brain, threshold = 0.1)
    counter+=1

    plt.show()# %%
# %%
coupling = (np.array(SDI_f)<0)*SDI_f
decoupling = (np.array(SDI_f)>0)*SDI_f
# %%
np.shape(SDI_f)
# %%
import matplotlib
from nilearn.regions import signals_to_img_labels
from nilearn.datasets import fetch_icbm152_2009
from nilearn import image, plotting
import matplotlib.pyplot as plt
import nibabel
total_no_of_events = '30_events'

counts = list()
thresholds = [1, 3, 5, 7, 9, 11]
for thr in thresholds:
    signal = np.sum( (np.array(SDI_f)>0) *1, axis = (0))

    signal_f = np.expand_dims(np.sum(1 * (signal > thr), axis=0), axis=0)
    counts.append(signal_f)




BL_or_PO = ['BL', 'PO']
if total_no_of_events == "19_events":
    event_type = ["Audio", "+ve Offset", "-ve offset"]
else:
    event_type = ["C1", "C2", "C3"]

counter = 1

# s = np.sum( (sdi['alpha']>0) *1, axis = (0))


path_Glasser = "/homes/v20subra/S4B2/GSP/Glasser_masker.nii.gz"

mnitemp = fetch_icbm152_2009()
mask_mni = image.load_img(mnitemp["mask"])
glasser_atlas = image.load_img(path_Glasser)

for i in range(len(thresholds)):
    # signal = np.expand_dims(np.sum(s * (s>7), axis=0), axis=0)
    signal = np.expand_dims(counts[i], axis = 2)

    U0_brain = signals_to_img_labels(signal, path_Glasser, mnitemp["mask"])
    # nibabel.nifti1.save(U0_brain,f'/homes/v20subra/S4B2/Graph-related_analysis/OHBM_results/Niftis/{thresholds[i]}')

    plot = plotting.plot_img_on_surf(
        U0_brain, threshold = 1, title = 'baseline')
    counter+=1
    # plt.savefig(f'/homes/v20subra/S4B2/Graph-related_analysis/OHBM_results/Round2/baseline/{i}.png', transparent = True, dpi=800)

    plt.show()


# %%

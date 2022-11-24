#%%
import numpy as np
import importlib
import os
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import binom
from torch import negative


total_no_of_events = "30_events"
number_of_events = 30
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

sdi = np.load(f'/users2/local/Venkatesh/Generated_Data/25_subjects_new/SDI/{total_no_of_events}.npz')



# counter = 0
# dic_order = defaultdict(dict)

# for i, j in clusters.items():
#     dic_order[f'{i}'] = np.arange(counter, counter+ len(j))
#     counter+= len(j)

# #%%
# positive_map = defaultdict(dict)
# negative_map = defaultdict(dict)


# for i in range(n_clusters):
#     signal = sdi['alpha'][dic_order[f'{i}']]

#     signal_pos = np.sum(signal > 0 , axis=0)
#     signal_neg = np.sum( signal < 0, axis=0)
    
#     x = np.arange(1, 101)
#     sf = binom.sf(x, 100, p=0.05)
#     thr = x[np.min(np.where(sf < 0.05 / 360))]
#     thr = np.floor(len(dic_order[f'{i}']) / 100 * thr) + 1

#     for_positive = signal_pos > thr
#     for_negative = signal_neg > thr

#     positive_map[f'{i}'] =  for_positive * 1
#     negative_map[f'{i}'] =  for_negative * 1

# %%
import matplotlib
from nilearn.regions import signals_to_img_labels
from nilearn.datasets import fetch_icbm152_2009
from nilearn import image, plotting
import matplotlib.pyplot as plt
counts = list()
thresholds = [1, 3, 5, 7, 9, 11]
for thr in thresholds:
    signal = np.sum( (sdi['alpha']>0) *1, axis = (0))

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

    plot = plotting.plot_img_on_surf(
        U0_brain, title = f'More decoupled / t = {thresholds[i]} events')
    counter+=1

    plt.show()


# %%


#%%
s = (sdi['alpha']>0)*1

for i in range(number_of_events):
    signal = np.sum(s[i], axis = 0)
    signal = np.expand_dims(signal, axis = 0)
    print(signal)
    U0_brain = signals_to_img_labels(signal, path_Glasser, mnitemp["mask"])

    plot = plotting.plot_img_on_surf(
        U0_brain, threshold = 1, title = f'event {i +1}')
    counter+=1

    plt.show()



# %%
total_no_of_events
# %%

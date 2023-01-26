#%%
import numpy as np
import importlib
import os
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import binom
from torch import negative
import pandas as pd
import seaborn as sns

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
import nibabel

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
        U0_brain, threshold = 1, title ='PO')
    counter+=1
    # plt.savefig(f'/homes/v20subra/S4B2/Graph-related_analysis/OHBM_results/Round2/strongISC/{i}.png', transparent = True, dpi = 800)


    plt.show()



# %%
############event-wise
s = (sdi['alpha']>0)*1

for i in range(number_of_events):
    signal = np.sum(s[i], axis = 0)
    signal = np.expand_dims(signal, axis = 0)

    U0_brain = signals_to_img_labels(signal, path_Glasser, mnitemp["mask"])

    plot = plotting.plot_img_on_surf(
        U0_brain, threshold = 1, title = f'event {i +1}')
    counter+=1

    plt.show()



# %%
total_no_of_events
# %%
from nilearn import datasets, plotting, image, maskers

atlas_yeo_2011 = datasets.fetch_atlas_yeo_2011()
yeo = atlas_yeo_2011.thick_7
glasser = "/homes/v20subra/S4B2/GSP/Glasser_masker.nii.gz"

masker = maskers.NiftiMasker(standardize=False, detrend=False)
masker.fit(glasser)
glasser_vec = masker.transform(glasser)

yeo_vec = masker.transform(yeo)
yeo_vec = np.round(yeo_vec)

matches = []
match = []
best_overlap = []
for i, roi in enumerate(np.unique(glasser_vec)):
    overlap = []
    for roi2 in np.unique(yeo_vec):
        overlap.append(
            np.sum(yeo_vec[glasser_vec == roi] == roi2) / np.sum(glasser_vec == roi)
        )
    best_overlap.append(np.max(overlap))
    match.append(np.argmax(overlap))
    matches.append((i + 1, np.argmax(overlap)))

# %%


n_subject = 9
n_event = 20
aud_roi_rh = 24
aud_roi_lh = 24 + 180

cond = ['coupled', 'decoupled']

dic_of_index_yeo_nw_wise = dict()

for condition in cond:
    if condition == 'coupled':
        s = (sdi['alpha']<0)*1

    if condition == 'decoupled':
        s = (sdi['alpha']>0)*1

    final_array_total = []
    for events in range(number_of_events):
        
        final_array_subjects = []
        for subs in range(subjects):

            sub_event = s[events, subs]

            array_to_plot = []
            for network in range(1,8):
                ids = np.where(np.array(match)==network)
                array_to_plot.append(np.average(sub_event[ids ]))

            final_array_subjects.append(array_to_plot)
            # array_to_plot.append( np.mean ((sub_event[aud_roi_rh], sub_event[aud_roi_lh])))
        final_array_total.append(final_array_subjects)
    
    dic_of_index_yeo_nw_wise[f'{condition}'] = final_array_total


# %%
plt.style.use("fivethirtyeight")

a, b, c = 6, 5, 1
fig = plt.figure(figsize=(35,25))

for cond, signal in dic_of_index_yeo_nw_wise.items():
    signal_f = signal
    c = 1
    for subj in range(subjects):
        plt.subplot(a, b, c)
        err = np.std(np.array(signal_f)[:, subj, :], axis = 0)
        height = np.mean(np.array(signal_f)[:, subj, :], axis = 0)

        if cond == 'decoupled':
            plt.ylim(0, 0.30)
            plt.bar(x = labels, yerr = err, height = height, alpha = 0.4, color = 'r', label = 'decoupled')
        
        else :
            plt.ylim(0, 0.30)
            plt.bar(x = labels, yerr = err, height = height, alpha = 0.2, color = 'b', label = 'coupled')

        plt.legend()
        c+=1

fig.suptitle(f'Condition / Subject-wise / errorbar over events')
# %%

a, b, c = 6, 5, 1
fig = plt.figure(figsize=(35,25))
for cond, signal in dic_of_index_yeo_nw_wise.items():
    signal_f = signal
    c = 1
    
    for events in range(number_of_events):
        
        plt.subplot(a, b, c)
        err = np.std(np.array(signal_f)[events, :, :], axis = 0)
        height = np.mean(np.array(signal_f)[events, :, :], axis = 0)
        if cond == 'decoupled':
            plt.ylim(0, 0.30)
            plt.bar(x = labels, yerr = err,height = height, alpha = 0.4, color = 'r', label = 'decoupled')
        
        else :
            plt.ylim(0, 0.30)
            plt.bar(x = labels, yerr = err, height = height, alpha = 0.2, color = 'b', label = 'coupled')

        plt.legend()
        c+= 1

fig.suptitle(' Condition / Event-wise / errorbar over subjects')
# %%
plt.bar(x = labels, height = np.mean(dic_of_index_yeo_nw_wise['decoupled'], axis = (0,1)), alpha = 0.4, color = 'r', label = 'decoupled')

plt.bar(x = labels, height = np.mean(dic_of_index_yeo_nw_wise['coupled'], axis = (0,1)), alpha = 0.4, color = 'b', label = 'coupled')
plt.legend()

plt.title("Grand average over subjects and events")
# %%
np.shape( dic_of_index_yeo_nw_wise['coupled'] )
# %%
s = (sdi['alpha']>0)*1 
s = (sdi['alpha']<0)*-1 

sns.heatmap(np.array(s)[:,0,:])


# %%
signal = sdi['alpha']

a, b, c = 5, 5, 1
fig = plt.figure(figsize=(45, 35))

for sub in range(subjects):
    plt.subplot(a, b, c)
    sns.heatmap(((signal>0)*1 +(signal<0)*-1)[:, sub, :].T, cmap = 'bwr')

    c+=1

fig.suptitle('SDI subject-wise / decoupling in red')
fig.supxlabel('Events')
fig.supylabel('ROI')
# fig.savefig("/users2/local/Venkatesh/Generated_Data/25_subjects_new/Results/SDI/roi_event_index.jpg", dpi = 500)
# %%


signal = sdi['alpha']

thr = 7

signal_sub_d = np.sum( (signal>0)*1, axis = 1)
signal_sub_c = np.sum( (signal<0)*1, axis = 1)

bin_c = ((signal_sub_d>thr)*1).T 

sns.heatmap()

# %%
# %%

signal = sdi['alpha']
sns.heatmap(   np.sum((signal<0)*1, axis = 0).T)
plt.title('coupled')

plt.show()
sns.heatmap(   np.sum((signal>0)*1, axis = 0).T)
plt.title('decoupled')
plt.show()
# %%
array_for_heatmap = []

thr = [1, 3, 5, 7, 9, 11]
for i in thr:
    array_for_heatmap.append( np.sum((np.sum((signal>0)*1, axis = 0)> i) *1, axis = 0) )

plt.figure(figsize=(25, 15))
sns.heatmap(array_for_heatmap, yticklabels = thr)
# np.savez_compressed('/homes/v20subra/S4B2/Graph-related_analysis/OHBM_results/array_for_heatmap.npz', array_for_heatmap = array_for_heatmap)
# plt.savefig('/homes/v20subra/S4B2/Graph-related_analysis/OHBM_results/decoupling_heatmap.svg', transparent = False)
# %%


#For OHBM

plt.style.use("fivethirtyeight")
signal = sdi['alpha']

cond ='decoupled'
decoupled = signal*(signal>0)

final_sdi_parcellated = []

for event in range(number_of_events):
    subject_wise = []
    for subject in range(subjects):

        signal_final = decoupled[event, subject]


        
        array_to_plot = []
        for network in range(1,8):
                ids = np.where(np.array(match)==network)
                array_to_plot.append(np.average(signal_final[ids ]))            
        
        subject_wise.append(array_to_plot)
    final_sdi_parcellated.append(subject_wise)



#%%

labels = ['Visual', 'SM', 'DA', 'VA', 'Limbic', 'FP', 'DMN']

a, b, c = 6, 5, 1
fig = plt.figure(figsize=(35,25))

signal_f = final_sdi_parcellated
c = 1

for events in range(number_of_events):
    
    plt.subplot(a, b, c)
    err = np.std(np.array(signal_f)[events, :, :], axis = 0)
    height = np.mean(np.array(signal_f)[events, :, :], axis = 0)
    if cond == 'decoupled':
        plt.ylim(0, 0.30)
        plt.bar(x = labels, yerr = err,height = height, alpha = 0.4, color = 'r')
    
    else :
        plt.ylim(0, -0.5)
        plt.bar(x = labels, yerr = err, height = height, alpha = 0.2, color = 'b')

    plt.legend()
    c+= 1

fig.suptitle(f' {cond} / Event-wise / errorbar over subjects')
plt.show()
# %%

a, b, c = 5, 5, 1
fig = plt.figure(figsize=(35,25))

signal_f = final_sdi_parcellated
c = 1
for subj in range(subjects):
    plt.subplot(a, b, c)
    err = np.std(np.array(signal_f)[:, subj, :], axis = 0)
    height = np.mean(np.array(signal_f)[:, subj, :], axis = 0)

    if cond == 'decoupled':
        plt.ylim(0, 0.30)
        plt.bar(x = labels, yerr = err, height = height, alpha = 0.4, color = 'r', label = 'decoupled')
    
    else :
        plt.ylim(0, -0.50)
        plt.bar(x = labels, yerr = err, height = height, alpha = 0.2, color = 'b', label = 'coupled')

    plt.legend()
    c+=1

fig.suptitle(f'{cond} / Subject-wise / errorbar over events')
# %%
np.where(array_for_heatmap[-1] == 4)
# %%
#87, 89, 125
regions = np.load(f"/homes/v20subra/S4B2/GSP/hcp/regions.npy")
regions
# %%
total_no_of_events = '30_events'
import numpy as np

sdi = np.load(f'/users2/local/Venkatesh/Generated_Data/25_subjects_new/SDI/{total_no_of_events}.npz')

# %%
array_of_interest = sdi['theta']
# %%
array_of_interest_binarized = (array_of_interest < 0 )*-1 + (array_of_interest > 0 )*1

# %%
import seaborn as sns
plt.figure(figsize=(25,25))
sns.heatmap(array_of_interest_binarized[:,0].T, cmap = 'seismic')


# %%
plt.figure(figsize=(25,25))
sns.heatmap(((np.sum(array_of_interest_binarized, axis = 1) > 4)*1 + (np.sum(array_of_interest_binarized, axis = 1) < -4)*-1).T, cmap='seismic')

# %%
np.argmax(signal_f)
# %%

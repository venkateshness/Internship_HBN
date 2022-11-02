#%%
from matplotlib.pyplot import axis
import numpy as np
from collections import defaultdict
import mne
from sklearn import cluster
import torch
import importlib
import os
os.chdir('/homes/v20subra/S4B2/')

from Modular_Scripts import graph_setup
importlib.reload(graph_setup)

duration = 21250
subjects = 21
regions = 360
event_type = '30_events'
fs = 125
pre_stim = 25
post_stim = 63
second_in_sample = pre_stim + post_stim
number_of_clusters = 3
graph_type = 'individual'

clusters = np.load(f'/homes/v20subra/S4B2/AutoAnnotation/dict_of_clustered_events_{event_type}.npz')

envelope_signal_bandpassed = np.load(
    '/users2/local/Venkatesh/Generated_Data/25_subjects_new/eloreta_cortical_signal_thresholded/0_percentile.npz', mmap_mode='r')
alpha = envelope_signal_bandpassed['alpha']
theta = envelope_signal_bandpassed['theta']
low_beta = envelope_signal_bandpassed['low_beta']
high_beta = envelope_signal_bandpassed['high_beta']

dict_of_unthresholded_signals_for_all_bands = dict()
dict_of_unthresholded_signals_for_all_bands = {'theta':theta, 'alpha': alpha, 'low_beta':low_beta, 'high_beta':high_beta}

total_subjects = ['NDARAD481FXF','NDARBK669XJQ',
'NDARCD401HGZ','NDARDX770PJK',
'NDAREC182WW2','NDARGY054ENV',
'NDARHP176DPE','NDARLB017MBJ',
'NDARMR242UKQ','NDARNT042GRA',
'NDARRA733VWX','NDARRD720XZK',
'NDARTR840XP1','NDARUJ646APQ',
'NDARVN646NZP','NDARWJ087HKJ',
'NDARXB704HFD','NDARXJ468UGL',
'NDARXJ696AMX','NDARXU679ZE8',
'NDARXY337ZH9','NDARYM257RR6',
'NDARYY218AGA','NDARYZ408VWW','NDARZB377WZJ']

subjects_data_available_for =list()

for i in range(1,25):
     if (os.path.isfile(f'/users2/local/Venkatesh/HBN/CPAC_preprocessed/sub-{total_subjects[i]}_ses-1/functional_to_standard/_scan_rest/_selector_CSF-2mmE-M_aC-CSF+WM-2mm-DPC5_M-SDB_P-2_BP-B0.01-T0.1_C-S-1+2-FD-J0.5/bandpassed_demeaned_filtered_antswarp.nii.gz')):
         subjects_data_available_for.append(total_subjects[i])


envelope_signal_bandpassed_bc_corrected = np.load(f'/users2/local/Venkatesh/Generated_Data/25_subjects_new/eloreta_cortical_signal_thresholded/bc_and_thresholded_signal/{event_type}/0_percentile.npz')

idx_for_the_existing_subjects = np.argwhere(np.isin(total_subjects, subjects_data_available_for)).ravel()


dict_of_unthresholded_signals_for_all_bands_sliced = defaultdict(dict)

for labels, signals in dict_of_unthresholded_signals_for_all_bands.items():
    dict_of_unthresholded_signals_for_all_bands_sliced[f'{labels}'] = dict_of_unthresholded_signals_for_all_bands[f'{labels}'][idx_for_the_existing_subjects]

pli = np.load('/users2/local/Venkatesh/Generated_Data/25_subjects_new/graph_space/pli_graph_21_subjects.npz')

# %%
#helper function
def smoothness_computation(band, laplacian):
    """Smoothness computation subjectwise for each bands.

    Args:
        band (array): signal array per subject

    Returns:
        dict: GSV
    """
    per_subject = list()

    for timepoints in range(second_in_sample): #each timepoints
            signal = band[:,timepoints]

            stage1 = np.matmul(signal.T, laplacian)

            final = np.matmul(stage1, signal)
        
            per_subject.append(final)

    data_to_return = np.array(per_subject).T
    
    assert np.shape( data_to_return   ) == (second_in_sample, )
    return data_to_return


dict_of_sliced_bc_averaged = defaultdict(dict)
from itertools import chain
def slicing_averaging(band):
    """Slicing and Averaging of the cortical signal according to cluster groups. Dim = subjects x regions x time

    Args:
        band (string): frequency bands
    """
    subject_level = list()

    for subject in range(subjects):# Subject-wise

        cluster_level = list()
        for event_label, event_time in clusters.items(): # Cluster-wise
            
            signal = dict_of_unthresholded_signals_for_all_bands[f'{band}'][subject,    :,  :]
            event_level = list()
            
            if graph_type == 'individual':
                laplacian = graph_setup.NNgraph('individual' ,pli[f'{band}'][subject])

            for event in event_time: #event-wise
                signal_sliced_non_bc = signal[:,   event * fs - pre_stim : event * fs + post_stim]
                
                signal_sliced_bc = mne.baseline.rescale(signal_sliced_non_bc, times = np.array(list(range(second_in_sample)))/fs,  baseline = (None, 0.2), mode = 'zscore', verbose = False) # apply baseline correction
                signal_sliced_bc_normalized = signal_sliced_bc/np.diag(laplacian)[:,np.newaxis]
                smoothness = smoothness_computation(signal_sliced_bc_normalized, laplacian = laplacian)

                event_level.append(smoothness)
    

            assert np.shape(event_level) == (len(event_time), second_in_sample)
            cluster_level.append(np.mean(event_level, axis = 0) )
        
        assert np.shape(cluster_level) == (number_of_clusters, second_in_sample)

        subject_level.append(cluster_level)

    dict_of_sliced_bc_averaged[f'{band}'] = subject_level


for labels, signal in  dict_of_unthresholded_signals_for_all_bands.items():
    slicing_averaging(labels)
    assert np.shape(dict_of_sliced_bc_averaged[f'{labels}']) == (subjects, number_of_clusters, second_in_sample)


# %%
np.savez_compressed(f'/users2/local/Venkatesh/Generated_Data/25_subjects_new/graph_space/smoothness_trial_wise_for{event_type}{graph_type}_PLI',**dict_of_sliced_bc_averaged)

#%%

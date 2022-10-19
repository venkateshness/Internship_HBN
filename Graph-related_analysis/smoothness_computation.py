#%%
from matplotlib.pyplot import axis
import numpy as np
from collections import defaultdict
import mne
from sklearn import cluster

import importlib
import os
os.chdir('/homes/v20subra/S4B2/')

from Modular_Scripts import graph_setup
importlib.reload(graph_setup)

duration = 21250
subjects = 25
regions = 360
event_type = '30_events'
fs = 125
pre_stim = 25
post_stim = 63
second_in_sample = pre_stim + post_stim
number_of_clusters = 3
graph = 'SC'

laplacian = graph_setup.NNgraph(graph) # fMRI RS graph
clusters = np.load(f'/homes/v20subra/S4B2/AutoAnnotation/dict_of_clustered_events_{event_type}.npz')

envelope_signal_bandpassed = np.load(
    '/users2/local/Venkatesh/Generated_Data/25_subjects_new/eloreta_cortical_signal_thresholded/0_percentile.npz', mmap_mode='r')
alpha = envelope_signal_bandpassed['alpha']
theta = envelope_signal_bandpassed['theta']
low_beta = envelope_signal_bandpassed['low_beta']
high_beta = envelope_signal_bandpassed['high_beta']

dict_of_unthresholded_signals_for_all_bands = dict()
dict_of_unthresholded_signals_for_all_bands = {'theta':theta, 'alpha': alpha, 'low_beta':low_beta, 'high_beta':high_beta}
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
np.savez_compressed(f'/users2/local/Venkatesh/Generated_Data/25_subjects_new/graph_space/smoothness_trial_wise_for{event_type}{graph}',**dict_of_sliced_bc_averaged)

#%%

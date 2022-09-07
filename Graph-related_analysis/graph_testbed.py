#%%
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
import importlib
from torch import seed
os.chdir('/homes/v20subra/S4B2/')
from tqdm import tqdm

from Modular_Scripts import graph_setup
importlib.reload(graph_setup)
from collections import defaultdict
import networkx

envelope_signal_bandpassed_bc_corrected_thresholded = np.load(f'/users2/local/Venkatesh/Generated_Data/25_subjects_new/eloreta_cortical_signal_thresholded/bc_and_thresholded_signal/98_percentile.npz')

def packaging_bands(signal_array):
    theta = signal_array['theta']
    alpha = signal_array['alpha']
    low_beta = signal_array['low_beta']
    high_beta = signal_array['high_beta']

    dic =  {'theta' : theta, 'alpha' : alpha, 'low_beta' : low_beta, 'high_beta' : high_beta}

    return dic

dic_of_envelope_signals_thresholded = packaging_bands(envelope_signal_bandpassed_bc_corrected_thresholded)

laplacian,_ = graph_setup.NNgraph(graph = 'FC')


subjects = 25
number_of_clusters = 5
fs = 125
pre_stim_in_samples = 25
post_stim_in_samples = 63
seconds_per_event = pre_stim_in_samples + post_stim_in_samples
regions = 360
events = np.load('/homes/v20subra/S4B2/AutoAnnotation/dict_of_clustered_events.npz')


video_duration = seconds_per_event

def smoothness_computation(band, laplacian):
    """The main function that does GFT, function-calls the temporal slicing, frequency summing, pre- post- graph-power accumulating 

    Args:
        band (array): Envelope band to use

    Returns:
        dict: Baseline-corrected ERD for all trials 
    """
    for_all_subjects = list()

    for subject in range(subjects):
        subject_wise = list()
        for timepoints in range(seconds_per_event):
            signal = band[subject,:,timepoints]

            stage1 = np.matmul(signal.T, laplacian)

            final = np.matmul(stage1, signal)
            subject_wise.append(final)
        for_all_subjects.append(subject_wise)
    
    data_to_return = np.array(for_all_subjects).T
    assert np.shape( data_to_return   ) == (video_duration, subjects)
    return data_to_return

def fullstack(graph):
    laplacian_kalofias, _ = graph_setup.NNgraph(graph = graph)
    smoothness_computed_graph = dict()
  
    for labels, signal in dic_of_envelope_signals_thresholded.items():
        placeholder_gsv = list()
  
        for event in range(number_of_clusters):
            signal_for_gsv = np.array(signal)[:, event, :, :]

            signal_normalized = signal_for_gsv/np.diag(laplacian_kalofias)[np.newaxis,:,np.newaxis]
            placeholder_gsv.append( smoothness_computation (  signal_normalized, laplacian_kalofias))

        smoothness_computed_graph[f'{   labels  }'] = placeholder_gsv

    return smoothness_computed_graph

import seaborn as sns
sns.set_theme()
plt.style.use('fivethirtyeight')

def plot(ax, data):
    plt.ylabel('relative variation')
    plt.xticks(ticks = np.arange(0,88,25), labels = ['-200', '0', '200', '400'])
    plt.axvline(25,linestyle = 'dashed')
    mean = np.mean(data, axis = 1)
    sem = scipy.stats.sem(data, axis = 1)

    if ax == 0:
        plt.plot( mean, color='b', label = 'Original FC')
        plt.fill_between(range(seconds_per_event), mean - sem, mean + sem, alpha = 0.2, color = 'b')

    elif ax == 1 :
        plt.plot( mean, color='r', label = 'Original SC')
        plt.fill_between(range(seconds_per_event), mean - sem, mean + sem, alpha = 0.2, color = 'r')
    
    elif ax == 2 :
        plt.plot( mean, color='b', label = 'Kalofias FC', linestyle = 'dashed')
        plt.fill_between(range(seconds_per_event), mean - sem, mean + sem, alpha = 0.2, color = 'b')

    elif ax == 3 :
        plt.plot( mean, color='r', label = 'Kalofias SC', linestyle = 'dashed')
        plt.fill_between(range(seconds_per_event), mean - sem, mean + sem, alpha = 0.2, color = 'r')


# plot(0, data = fullstack('FC')['theta'][0])
plot(1, data = fullstack('SC')['theta'][0])

# plot(2, data = fullstack('kalofiasFC')['theta'][0])
plot(3, data = fullstack('kalofiasSC')['theta'][0])


plt.xlabel("time (ms)")
plt.title('GSV on different unthresholded graphs')
plt.legend()
plt.show()
# %%

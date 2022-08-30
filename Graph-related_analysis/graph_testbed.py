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

envelope_signal_bandpassed_bc_corrected = np.load(f'/users2/local/Venkatesh/Generated_Data/25_subjects_new/eloreta_cortical_signal_thresholded/bc_and_thresholded_signal/0_percentile.npz')

envelope_signal_bandpassed_bc_corrected_thresholded = np.load(f'/users2/local/Venkatesh/Generated_Data/25_subjects_new/eloreta_cortical_signal_thresholded/bc_and_thresholded_signal/98_percentile.npz')

def packaging_bands(signal_array):
    theta = signal_array['theta']
    alpha = signal_array['alpha']
    low_beta = signal_array['low_beta']
    high_beta = signal_array['high_beta']

    dic =  {'theta' : theta, 'alpha' : alpha, 'low_beta' : low_beta, 'high_beta' : high_beta}

    return dic

dic_of_envelope_signals_unthresholded = packaging_bands(envelope_signal_bandpassed_bc_corrected)
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

smoothness_computed = dict()

for labels, signal in dic_of_envelope_signals_thresholded.items():
    placeholder_gsv = list()
    for event in range(number_of_clusters):
        signal_for_gsv = np.array(signal)[:, event, :, :]
        signal_normalized = signal_for_gsv/np.diag(laplacian)[np.newaxis,:,np.newaxis]

        placeholder_gsv.append( smoothness_computation (  signal_normalized, laplacian))

    smoothness_computed[f'{   labels  }'] = placeholder_gsv


########
# G = networkx.erdos_renyi_graph(360, 0.1)
# adjacency = networkx.to_numpy_array(G)

# degree = np.diag(np.sum(adjacency, axis = 0))
# print(np.mean(sum(degree)))
# # adjacency = np.random.rand(360,360)
# # np.fill_diagonal(adjacency, 0)

# # degree = np.diag(np.sum(adjacency, axis = 0))


# laplacian_random_graph = degree - adjacency

laplacian_kalofias, _ = graph_setup.NNgraph(graph = 'SC')

# G = networkx.random_degree_sequence_graph(np.diag(laplacian))
# adjacency = networkx.to_numpy_array(G)
# laplacian_random_graph = np.diag(np.diag(laplacian)) - adjacency

def surrogate_signal(signal, eig_vector_original, is_sc_ignorant):
    all_subject_reconstructed = list()

    random_signs = np.round(np.random.rand(np.shape(eig_vector_original)[1]))
    random_signs[random_signs==0]=-1
    random_signs_diag = np.diag(random_signs)
    
    for subject in range(subjects):
        subject_wise_signal = np.array(signal)[subject]
        
        if is_sc_ignorant:
            eig_vectors_manip = np.matmul(np.fliplr(eig_vector_original), random_signs_diag)
        
        else:
            eig_vectors_manip = np.matmul(eig_vector_original, random_signs_diag)       
        
        spatial = np.array(np.matmul(subject_wise_signal.T, eig_vectors_manip))


        signal_reconstructed = np.matmul(eig_vector_original, spatial.T)
        assert np.shape(signal_reconstructed) == (regions, seconds_per_event)

        all_subject_reconstructed.append(signal_reconstructed)
    return np.array(all_subject_reconstructed)


smoothness_computed_kalofias_graph = dict()

for labels, signal in dic_of_envelope_signals_thresholded.items():
    placeholder_gsv = list()
    for event in range(number_of_clusters):
        signal_for_gsv = np.array(signal)[:, event, :, :]
        
        # [eigvals, eigvecs] = np.linalg.eigh(    laplacian_kalofias  )

        # surrogate = surrogate_signal(signal_for_gsv, eigvecs, is_sc_ignorant=True)

        signal_normalized = signal_for_gsv/np.diag(laplacian_kalofias)[np.newaxis,:,np.newaxis]
        placeholder_gsv.append( smoothness_computation (  signal_normalized, laplacian_kalofias))

    smoothness_computed_kalofias_graph[f'{   labels  }'] = placeholder_gsv

# %%
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
        plt.plot( mean, color='b', label = 'Original')
        plt.fill_between(range(seconds_per_event), mean - sem, mean + sem, alpha = 0.2, color = 'b')

    else :
        plt.plot( mean, color='r', label = 'Kalofias')
        plt.fill_between(range(seconds_per_event), mean - sem, mean + sem, alpha = 0.2, color = 'r')



plot(0, data = smoothness_computed['theta'][0])

plot(1,data = smoothness_computed_kalofias_graph['theta'][0])
plt.xlabel("time (ms)")
plt.title('Signal; Null Setting 1')
plt.legend()
plt.show()
# %%
import pandas as pd
roi_labels = np.hstack([pd.read_excel('/homes/v20subra/S4B2/Graph-related_analysis/Glasser_2016_Table.xlsx',header=None)[2].values[2:]] * 2)

# %%
for time in range(25,seconds_per_event):
    for i in range(360):
        idx = len(np.where(signal_for_gsv[:,:,time][:,i]>0)[0])

        if idx >5:
            print(roi_labels[i])
            print(time)
# %%

roi_labels[np.where(laplacian== np.max(np.diag(laplacian)))[0]]

# %%
#The "curve" is still the same; surrogate signal is the only way to break that
#Synthetic data
#1. if connected: Node with highest degree increases GSV
#2. Degree with highest node has rarely signal for subjects (2 at max)
#3. 

# signal_for_gsv[0][:,0]

#2
idx_max = np.where(np.diag(laplacian)==np.max(np.diag(laplacian)))[0]
for time in range(seconds_per_event):
    print(sum(signal_for_gsv[:,idx_max,time]>0))

# %%
# %%

# %%
signal_for_gsv[0,:,0]
# %%
signal_normalized[0,:,0]

# %%
normed =(signal_for_gsv/np.diag(laplacian)[np.newaxis,:,np.newaxis])[0,:,0]

# %%
normed * np.diag(laplacian)
# %%

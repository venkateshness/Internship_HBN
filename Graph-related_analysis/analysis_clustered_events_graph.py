#%%
import numpy as np
import importlib
import os
import matplotlib.pyplot as plt
from sklearn import cluster
os.chdir('/homes/v20subra/S4B2/')
import scipy
from Modular_Scripts import graph_setup


G = graph_setup.graph_setup_main()
G.compute_fourier_basis()

# %%
envelope_signal_thresholded = np.load('/users2/local/Venkatesh/Generated_Data/25_subjects_copy_FOR_TESTING/eloreta_cortical_signal_thresholded/90_percentile.npz')

alpha = envelope_signal_thresholded['alpha']
theta = envelope_signal_thresholded['theta']
low_beta = envelope_signal_thresholded['low_beta']
high_beta = envelope_signal_thresholded['high_beta']

dic_of_envelope_signals = {'theta' : theta, 'alpha' : alpha, 'low_beta' : low_beta, 'high_beta' : high_beta}
# %%

video_duration = 21250
subjects = 25
regions = 360

def smoothness_computation(band):
    """The main function that does GFT, function-calls the temporal slicing, frequency summing, pre- post- graph-power accumulating 

    Args:
        band (array): Envelope band to use

    Returns:
        dict: Baseline-corrected ERD for all trials 
    """

    laplacian = G.L.toarray()
    
    one = np.array(band).T # dim(one) = entire_video_duration x ROIs x subjects
    two = np.swapaxes(one,axis1=1,axis2=2) # dim (two) = entire_video_duration x subjects x ROIs

    signal = np.expand_dims(two,2) # dim (signal) = entire_video_duration x subjects x 1 x ROIs

    stage1 = np.tensordot(signal,laplacian,axes=(3,0)) # dim (laplacian) = (ROIs x ROIs).... dim (stage1) = same as dim (signal)

    signal_stage2 = np.swapaxes(signal,2,3) # dim(signal_stage2) = (entire_video_duration x subjects x ROIs x 1)
    assert np.shape(signal_stage2) == (video_duration, subjects, regions, 1)

    smoothness_roughness_time_series = np.squeeze( np.matmul(stage1,signal_stage2) ) # dim = entire_video_duration x subjects
    assert np.shape(smoothness_roughness_time_series) == (video_duration, subjects)
    
    return smoothness_roughness_time_series

dic_of_smoothness_signals = dict()

for labels, signal in dic_of_envelope_signals.items():
    dic_of_smoothness_signals[f'{labels}'] = smoothness_computation(signal)
    assert np.shape(dic_of_smoothness_signals[f'{labels}']) == (video_duration, subjects)

#%%

events = np.load('/homes/v20subra/S4B2/AutoAnnotation/dict_of_clustered_events.npz')
number_of_clusters = 5
fs = 125
pre_stim_in_samples = 62
post_stim_in_samples = 63   
seconds_per_event = pre_stim_in_samples + post_stim_in_samples

def smoothness_event_averaging(band, dim_spatial):
    """Averaging given a cluster group of events; input dim = subjects x spatial_dim x entire_video_duration

    Args:
        band (array): 
        ####frequency band (fourier) of the signal (envelope signal computed using Hilbert transform on the eLORETA cortical signal)
        
        dim_spatial(int): The dimension of the spatial space; e.g: regions (ROIs) -- 360

    Returns:
        array: the averaged events; output_dim = subjects x regions x number_of_clusters x seconds_per_event * fs
    """

    all_subject_data_averaged_event = list()

    for subject in range(subjects):
    
        one_subject_data = list()
        for cluster in range(number_of_clusters):
            cluster_groups = events[str(cluster)]
            indices_for_slicing = list()

            for single_cluster_group in range(len(cluster_groups)):
                indices_for_slicing.append( np.arange(  (cluster_groups[single_cluster_group] * fs) - pre_stim_in_samples, (cluster_groups[single_cluster_group] * fs) + post_stim_in_samples))
            
            assert np.shape(band[subject, :, indices_for_slicing]) ==  (len(cluster_groups), seconds_per_event, dim_spatial)
            
            averaged_event_in_a_cluster_group = np.mean( band[subject,:,indices_for_slicing] , axis=0).T


            assert np.shape(averaged_event_in_a_cluster_group) == (dim_spatial,  seconds_per_event)
            one_subject_data.append( averaged_event_in_a_cluster_group)


        one_subject_data_swap_axis =  np.swapaxes (  one_subject_data , 0, 1)
        assert np.shape(one_subject_data_swap_axis) == (dim_spatial, number_of_clusters , seconds_per_event)
        
        all_subject_data_averaged_event.append( one_subject_data_swap_axis  )

    assert np.shape(all_subject_data_averaged_event) == (subjects, dim_spatial, number_of_clusters, seconds_per_event )
    
    return all_subject_data_averaged_event

smoothness_signal_sliced_time_averaged = dict()

for labels, signal in dic_of_smoothness_signals.items():
    smoothness_signal_sliced_time_averaged[f'{labels}']    =   np.squeeze   (   smoothness_event_averaging(   np.expand_dims( signal, 1).T  ,   dim_spatial =   1    )  )
    assert np.shape(    smoothness_signal_sliced_time_averaged[f'{labels}']    ) == (  subjects,  number_of_clusters, seconds_per_event  )
      
# %%


dic_of_baseline_smoothness_signal = dict()

for labels, signal in smoothness_signal_sliced_time_averaged.items():

    erd = list()
    for cluster_group in range(number_of_clusters):
        signal_sliced_per_event = signal[:,cluster_group,:]
        erd_sub = list()

        for subject in range(subjects):
            baseline = signal_sliced_per_event[subject][ : pre_stim_in_samples ]
            baseline_corrected_signal = ( signal_sliced_per_event[subject] - np.mean(baseline)  ) / np.mean(baseline)
            erd_sub.append(baseline_corrected_signal)

        erd.append(erd_sub)
    dic_of_baseline_smoothness_signal[f'{labels}'] = erd
    assert np.shape(erd) == (number_of_clusters, subjects, seconds_per_event)

np.shape(dic_of_baseline_smoothness_signal['theta'])

# %%

# a = 5
# b = 5
# c = 1
# fig = plt.figure(figsize=(25,25))


# for cluster_group in range(number_of_clusters):
#     for labels, signal in dic_of_baseline_smoothness_signal.items():
#         print(labels)
#         baseline_signal = signal[str(cluster_group)]
#         print(np.shape(baseline_signal))

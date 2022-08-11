#%%
from collections import defaultdict
import numpy as np
import importlib
import os
import matplotlib.pyplot as plt
from sklearn import cluster
os.chdir('/homes/v20subra/S4B2/')
import scipy
from Modular_Scripts import graph_setup
importlib.reload(graph_setup)

from collections import defaultdict
from statsmodels.stats.multitest import fdrcorrection, multipletests

G = graph_setup.graph_setup_main()


dic_of_envelope_signals = dict()

percentiles = ['98']

def dicting(percentile):

    envelope_signal_thresholded = np.load(f'/users2/local/Venkatesh/Generated_Data/25_subjects_new/eloreta_cortical_signal_thresholded/bc_and_thresholded_signal/{percentile}_percentile.npz')

    alpha = envelope_signal_thresholded['alpha']
    theta = envelope_signal_thresholded['theta']
    low_beta = envelope_signal_thresholded['low_beta']
    high_beta = envelope_signal_thresholded['high_beta']

    dict = {'theta' : theta, 'alpha' : alpha, 'low_beta' : low_beta, 'high_beta' : high_beta}

    return dict


dic_of_all_signals = {percentiles[0] : dicting(percentiles[0])}
                    #   percentiles[1] : dicting(percentiles[1]),
                    #   percentiles[2] : dicting(percentiles[2]),
                    #   percentiles[3] : dicting(percentiles[3]),
                    #   percentiles[4] : dicting(percentiles[4])}


video_duration = 88
subjects = 25
regions = 360
number_of_clusters = 5

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


dic_of_smoothness_signals = defaultdict(dict)

for labels_outer, signal_outer in dic_of_all_signals.items():
    for labels_inner, signal_inner in signal_outer.items():
        event_level = list()
        for event_group in range(number_of_clusters):

            event_level.append( smoothness_computation(signal_inner[:,event_group,:,:]) )
        dic_of_smoothness_signals[f'{labels_outer}'][f'{labels_inner}'] = event_level
        assert np.shape(dic_of_smoothness_signals[f'{labels_outer}'][f'{labels_inner}']) == (number_of_clusters, video_duration, subjects)


events = np.load('/homes/v20subra/S4B2/AutoAnnotation/dict_of_clustered_events.npz')
number_of_clusters = 5
fs = 125
pre_stim_in_samples = 25
post_stim_in_samples = 63   
seconds_per_event = pre_stim_in_samples + post_stim_in_samples


_5clusters = ['RMS diff(entire event)= >1', '0.5 < RMS < 0.97', '< 0.2', '+ve frame offset', '-ve frame offset']

def plotting(band):
    a = 5
    b = 5
    c = 1
    counter =0
    fig = plt.figure(figsize=(25,15))

    for labels_outer, signal_outer in dic_of_smoothness_signals.items():

            signal_band_wise_percentile_wise = np.swapaxes(signal_outer[f'{band}'], 1, 2)
            assert np.shape(signal_band_wise_percentile_wise) == (number_of_clusters, subjects, seconds_per_event)
            
            for cluster_group in range(number_of_clusters):

                signal_final = signal_band_wise_percentile_wise[cluster_group]

                plt.style.use("fivethirtyeight")
                mean_signal = np.mean(signal_final, axis = 0)
                sem_signal = scipy.stats.sem(signal_final, axis = 0)

                plt.subplot(a, b, c)
                plt.plot(mean_signal)
                plt.fill_between(range(seconds_per_event), mean_signal - sem_signal, mean_signal + sem_signal, alpha = 0.2, label = 'SEM - subjects', color = 'cyan')
                
                plt.axvspan(0, pre_stim_in_samples, label = 'Baseline', color = 'r', alpha = 0.2)
                plt.xticks(np.arange(0,88,25), labels = ['-200', '0', '200', '400'])

                if cluster_group == 3:
                    plt.axvline(pre_stim_in_samples + 30, c = 'g', linestyle = '-.')
                    plt.axvline(pre_stim_in_samples + 40, label = 'Frame change', c = 'g', linestyle = '-.')
                    plt.axvline(pre_stim_in_samples + 50, c = 'g', linestyle = '-.')
                
                if cluster_group == 4:
                    plt.axvline(pre_stim_in_samples - 45, label = 'Frame change', c = 'g', linestyle = '-.')

                plt.axvline(pre_stim_in_samples, label = 'Onset (ISC)', c = 'g', linestyle = 'dashed')
                plt.axvspan(0, pre_stim_in_samples, color = 'r', alpha = 0.2)

                
                pvalues = np.zeros(post_stim_in_samples )
                
                pre_stim_signal = np.array(signal_final)[:,:pre_stim_in_samples]
                assert np.shape(pre_stim_signal) == (subjects, pre_stim_in_samples)                

                # mean_pre_stim_signal = np.mean(pre_stim_signal, axis = 1)

                # for samples in range(pre_stim_in_samples, seconds_per_event ):
                #     signal_signal = np.array(signal_final)[:,samples]
                #     pvalues[samples - pre_stim_in_samples] = scipy.stats.ttest_rel(mean_pre_stim_signal, signal_signal)[1]

                # pvalues_corrected = fdrcorrection(pvalues)[1]
                # # print(sum(pvalues_corrected<=0.05))

                # if c <= 5:
                #     plt.title(f'{_5clusters[cluster_group]}')

                # for pvals_index in range(len(pvalues_corrected)):
                #     if pvals_index in np.where(pvalues_corrected<=0.05)[0]:
                #         # print('yes')
                #         plt.axvline(pvals_index + pre_stim_in_samples , color = 'orange', alpha = 0.2)
              
                if c in [idx for idx in range(1, 41, 5)]:
                    
                    plt.ylabel(f'{percentiles[  counter ]}_percentile',rotation=25, size = 'large', color = 'r')
                    counter += 1 
                c += 1
                plt.legend()
                

    fig.suptitle(f'GSV at varying threshold of the cortical signal (FDR-Corrected pvalues) SC -- {band} band')
    fig.supxlabel('latency (ms) ')
    fig.supylabel('Relative variation')
    # fig.savefig(f'/homes/v20subra/S4B2/Graph-related_analysis/ERD_august/GSV/SC/{band}.jpg')



plotting('theta')
# plotting('alpha')
# plotting('low_beta')
# plotting('high_beta')

# %%
# %%
np.arange(0,88,25)

# %%

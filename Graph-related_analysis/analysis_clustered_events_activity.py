#%%
from cProfile import label
from cgi import test
from turtle import addshape, shape
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import cluster
from sklearn.decomposition import randomized_svd
from torch import rand
sns.set_theme()

import scipy.stats
import importlib
os.chdir('/homes/v20subra/S4B2/')
from Modular_Scripts import graph_setup
importlib.reload(graph_setup)
import numpy as np
from nilearn import datasets, plotting, image, maskers
import networkx as nx
from collections import defaultdict

# Average events (hilbert envelope)


# Unthresholded cortical signal
envelope_signal_bandpassed_bc_corrected = np.load(f'/users2/local/Venkatesh/Generated_Data/25_subjects_new/eloreta_cortical_signal_thresholded/bc_and_thresholded_signal/0_percentile.npz')


########################################################################################################################################
################################## Association of the Glasser ROIs with the 7 networks ################################################# 

atlas_yeo_2011 = datasets.fetch_atlas_yeo_2011()
yeo = atlas_yeo_2011.thick_7
glasser ='/homes/v20subra/S4B2/GSP/Glasser_masker.nii.gz'

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
        overlap.append(np.sum(yeo_vec[glasser_vec == roi] == roi2) / np.sum(glasser_vec == roi))
    best_overlap.append(np.max(overlap))
    match.append(np.argmax(overlap))
    matches.append((i+1, np.argmax(overlap)))

# for ind, roi in enumerate(np.unique(glasser_vec)):
#     print(ind)
#     print(f'roi {int(roi)} in Glasser has maximal overlap with Yeo network {match[ind]} ({best_overlap[ind]})')

##########################################################################################################################################

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


laplacian,_ = graph_setup.NNgraph()


subjects = 25
number_of_clusters = 5
fs = 125
pre_stim_in_samples = 25
post_stim_in_samples = 63
seconds_per_event = pre_stim_in_samples + post_stim_in_samples
regions = 360
events = np.load('/homes/v20subra/S4B2/AutoAnnotation/dict_of_clustered_events.npz')


##########################################################################################################################################################################
################################## Multi-purpose function to temporally slice, average those according to events #########################################################


video_duration = seconds_per_event

def smoothness_computation(band, laplacian):
    """The main function that does GFT, function-calls the temporal slicing, frequency summing, pre- post- graph-power accumulating 
    Args:
        band (array): Envelope band to use
    Returns:
        dict: Baseline-corrected ERD for all trials 
    """

    
    one = np.array(band).T # dim(one) = entire_video_duration x ROIs x subjects
    two = np.swapaxes(one,axis1=1,axis2=2) # dim (two) = entire_video_duration x subjects x ROIs

    signal = np.expand_dims(two,2) # dim (signal) = entire_video_duration x subjects x 1 x ROIs

    stage1 = np.tensordot(signal,laplacian,axes=(3,0)) # dim (laplacian) = (ROIs x ROIs).... dim (stage1) = same as dim (signal)

    signal_stage2 = np.swapaxes(signal,2,3) # dim(signal_stage2) = (entire_video_duration x subjects x ROIs x 1)

    assert np.shape(signal_stage2) == (video_duration, subjects, regions, 1)

    smoothness_roughness_time_series = np.squeeze( np.matmul(stage1,signal_stage2) ) # dim = entire_video_duration x subjects
    assert np.shape(smoothness_roughness_time_series) == (video_duration, subjects)
    
    return smoothness_roughness_time_series


smoothness_computed = dict()

for labels, signal in dic_of_envelope_signals_thresholded.items():
    placeholder_gsv = list()
    for event in range(number_of_clusters):
        signal_for_gsv = np.array(signal)[:, event, :, :]

        placeholder_gsv.append( smoothness_computation (  signal_for_gsv, laplacian))

    smoothness_computed[f'{   labels  }'] = placeholder_gsv

# %%
 

def baseline_correction_network_wise_setup(network):
    dic_of_cortical_signal_baseline_corrected = dict()

    for labels, signal in dic_of_envelope_signals_unthresholded.items():

        cortical_signal_baseline_corrected = list()

        for subject in range(subjects):
            event_group = list()

            for cluster_group in range(number_of_clusters):
                per_subject_per_cluster_group_cortical_signal   =    np.array(signal)  [subject,  cluster_group, :,  :]
                assert np.shape(    per_subject_per_cluster_group_cortical_signal    ) == ( regions,    seconds_per_event )

                indices_of_roi_belonging_network_group  = np.where(   np.array(match)== network  )[0]
                rois_belonging_to_a_network_averaged = np.mean(  per_subject_per_cluster_group_cortical_signal   [indices_of_roi_belonging_network_group],   axis = 0 )
                assert np.shape(    rois_belonging_to_a_network_averaged    ) == (seconds_per_event, )

                event_group.append(rois_belonging_to_a_network_averaged)
            
            cortical_signal_baseline_corrected.append(  event_group)
        assert np.shape(cortical_signal_baseline_corrected) == (subjects,   number_of_clusters, seconds_per_event)

        dic_of_cortical_signal_baseline_corrected[f'{labels}']  =   cortical_signal_baseline_corrected


    return dic_of_cortical_signal_baseline_corrected



dic_of_cortical_signal_baseline_corrected_nw  = dict()
for i in range(1, 8):
    dic_of_cortical_signal_baseline_corrected_nw[f'{i}']    =   baseline_correction_network_wise_setup(network = i)


bands = ['theta', 'alpha', 'low_beta', 'high_beta']
aud_ROI_label = 24
for i in bands:
    dic_of_cortical_signal_baseline_corrected_nw['8'] = np.squeeze( np.mean (dic_of_envelope_signals_unthresholded[f'{i}'][:,:,[aud_ROI_label, aud_ROI_label + 179],:], axis = 2) )

from statsmodels.stats.multitest import fdrcorrection, multipletests

_7_networks = ['GSV','Visual', 'Somatomotor', 'Dorsal Attention', 'Ventral Attention', 'Auditory C']
_5clusters = ['RMS diff(entire event)= >1', '0.5 < RMS < 0.97', '< 0.2', '+ve frame offset', '-ve frame offset']

def plotting(band):
    a = 6
    b = 5
    c = 1
    fig = plt.figure(figsize=(25, 25))
    for i in range(6):

        for cluster_group in range(number_of_clusters):
            plt.style.use("fivethirtyeight")

            plt.subplot(a,  b,  c)

            if i==0:
                label_band = list(smoothness_computed.keys())[i]
                signal = (np.array(  smoothness_computed[f'{band}'] ) [cluster_group,:,:]).T
            
            else:
                label_band = list(dic_of_cortical_signal_baseline_corrected_nw.keys())[i-1]
                signal = np.array(  dic_of_cortical_signal_baseline_corrected_nw[f'{label_band}'][f'{band}'] ) [:,cluster_group,:]
            
            assert np.shape(signal) == (subjects, seconds_per_event)

            mean_signal = np.mean( signal, axis = 0 )
            assert np.shape(mean_signal) == (seconds_per_event,)
            
            sem_signal = scipy.stats.sem(  signal, axis = 0 )
            assert np.shape(sem_signal) == (seconds_per_event,)
            
            plt.xticks(np.arange(0,88,25), labels = ['-200', '0', '200', '400'])
            
            plt.plot(   mean_signal    )
            plt.fill_between    (   range( seconds_per_event ), mean_signal - sem_signal, mean_signal + sem_signal, alpha = 0.2, label = 'SEM - subjects')

            if cluster_group == 3:
                plt.axvline(pre_stim_in_samples + 30, label = 'Frame change', c = 'g', linestyle = '-.')
                plt.axvline(pre_stim_in_samples + 40, c = 'g', linestyle = '-.')
                plt.axvline(pre_stim_in_samples + 50, c = 'g', linestyle = '-.')
            
            plt.axvline(pre_stim_in_samples + 12, c = 'c', linestyle = '-')
            plt.axvspan(0, pre_stim_in_samples , alpha = 0.2, color = 'r', label = 'Baseline')
            plt.axvline(pre_stim_in_samples, label = 'Onset (ISC)', c = 'g', linestyle = 'dashed')
            
            if c in [idx for idx in range(1, 41, 5)]:
                plt.ylabel(f'{_7_networks[i]}',rotation=25, size = 'large', color = 'r')
            
            if c <= 5:
                plt.title(f'{_5clusters[cluster_group]}')

        
            signal_baseline = signal[:pre_stim_in_samples]
            assert np.shape(signal_baseline) == (subjects, seconds_per_event)
            
            signal_baseline_averaged = np.mean(signal_baseline, axis = 1)
            assert np.shape(signal_baseline_averaged) == (subjects,)

            pvalues = np.zeros(post_stim_in_samples )

            for samples in range(pre_stim_in_samples, seconds_per_event ):
                signal_post_onset = signal[:,samples]
                if i == 0:
                    
                    pvalues[samples - pre_stim_in_samples] = scipy.stats.ttest_rel (signal_post_onset, signal_baseline_averaged,  )[1]
                else:   
                    pvalues[samples - pre_stim_in_samples] = scipy.stats.ttest_1samp(signal_post_onset, popmean=0)[1]
            
            pvalues_corrected = multipletests(pvalues, method = "bonferroni")[1]
            # print(sum(pvalues_corrected<=0.05))
            # pvalues_corrected = fdrcorrection(pvalues)[1]
            # print(sum(pvalues_corrected<=0.05))

            for pvals_index in range(len(pvalues_corrected)):
                if pvals_index in np.where(pvalues_corrected<=0.05)[0]:
                    plt.axvline(pvals_index + pre_stim_in_samples , color = 'orange', alpha = 0.2)

            plt.legend()
            c += 1
    fig.supylabel('relative variation')
    fig.suptitle(f'GSV & activity in Yeo NW / FC graph. {band} band - BonF-corrected')
    fig.supxlabel('latency (ms)')
    # fig.savefig(f'/homes/v20subra/S4B2/Graph-related_analysis/ERD_august/GSV/FC/{band}.jpg')

plotting('theta')
plotting('alpha')
plotting('low_beta')
plotting('high_beta')


# %%
test_signal = np.array(dic_of_envelope_signals_thresholded['low_beta'])[:,3,:,:]

for_all_subjects = list()

for subject in range(subjects):
    subject_wise = list()
    for timepoints in range(seconds_per_event):
        sig = test_signal[subject,:,timepoints]

        stage1 = np.matmul(sig.T, laplacian.numpy())

        final = np.matmul(stage1, sig)
        subject_wise.append(final)
    for_all_subjects.append(subject_wise)

plt.plot(np.mean(np.vstack(for_all_subjects).T,axis=1))
# %%
# %%
plt.plot(np.mean(smoothness_computed['low_beta'][3],axis=1))

# %%

#%%

import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import cluster
sns.set_theme()

import mne
import warnings
warnings.filterwarnings("ignore")

import scipy.stats
import importlib
os.chdir('/homes/v20subra/S4B2/')

from Modular_Scripts import graph_setup
importlib.reload(graph_setup)
import numpy as np
from nilearn import datasets, plotting, image, maskers
from collections import defaultdict
from statsmodels.stats.multitest import fdrcorrection, multipletests

event_type = '30_events'
# Unthresholded cortical signal
envelope_signal_bandpassed_bc_corrected = np.load(f'/users2/local/Venkatesh/Generated_Data/25_subjects_new/eloreta_cortical_signal_thresholded/bc_and_thresholded_signal/{event_type}/0_percentile.npz')


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



##########################################################################################################################################

envelope_signal_bandpassed_bc_corrected_thresholded = np.load(f'/users2/local/Venkatesh/Generated_Data/25_subjects_new/eloreta_cortical_signal_thresholded/bc_and_thresholded_signal/{event_type}/98_percentile.npz')

def packaging_bands(signal_array):
    """Package into the frequency band in a dic.

    Args:
        signal_array (array): the signal array of diff bands

    Returns:
        dict: signal assembled into frequency bands
    """
    theta = signal_array['theta']
    alpha = signal_array['alpha']
    low_beta = signal_array['low_beta']
    high_beta = signal_array['high_beta']

    dic =  {'theta' : theta, 'alpha' : alpha, 'low_beta' : low_beta, 'high_beta' : high_beta}

    return dic

dic_of_envelope_signals_unthresholded = packaging_bands(envelope_signal_bandpassed_bc_corrected)
dic_of_envelope_signals_thresholded = packaging_bands(envelope_signal_bandpassed_bc_corrected_thresholded)


laplacian = graph_setup.NNgraph() # fMRI RS graph

subjects = 25
number_of_clusters = 3
fs = 125
pre_stim_in_samples = 25
post_stim_in_samples = 63
seconds_per_event = pre_stim_in_samples + post_stim_in_samples
regions = 360


###################################################################################################################
################################## Smoothness computation #########################################################


video_duration = seconds_per_event

#helper function
def smoothness_computation(band, laplacian):
    """Smoothness computation subjectwise for each bands.

    Args:
        band (array): signal array per subject

    Returns:
        dict: GSV
    """
    per_subject = list()
    for event in range(number_of_clusters):# each cluster group
        per_event = list()
        
        for timepoints in range(seconds_per_event): #each timepoints
            signal = band[event, :,timepoints]

            stage1 = np.matmul(signal.T, laplacian)

            final = np.matmul(stage1, signal)
            per_event.append(final)
        
        per_subject.append(per_event)

    data_to_return = np.array(per_subject).T
    assert np.shape( data_to_return   ) == (video_duration, number_of_clusters)
    return data_to_return


#Compute GSV for each subjects
smoothness_computed = defaultdict(dict)

for labels, signal in dic_of_envelope_signals_thresholded.items():
    smoothness_subject = list()

    for subject in range(subjects):
        
        signal_normalized = dic_of_envelope_signals_thresholded[f'{labels}'][subject]/np.diag(laplacian)[np.newaxis, :, np.newaxis]
        
        smoothness_subject.append(smoothness_computation(signal_normalized, laplacian))

    smoothness_computed[f'{labels}'] = np.array(smoothness_subject)


##############################################################################################################################

def yeo_network_wise_setup(network): # Network wise
    """Group the ROIs according to Yeo networks 

    Args:
        network (string): Yeo Networks e.g : Visual, DMN, etc

    Returns:
        _type_: _description_
    """
    dic_of_cortical_signal_baseline_corrected = dict()

    for labels, signal in dic_of_envelope_signals_unthresholded.items(): # take the cortical unthresholded signal

        cortical_signal_baseline_corrected = list()

        for subject in range(subjects):# each subject
            event_group = list()

            for cluster_group in range(number_of_clusters): # each cluster

                per_subject_per_cluster_group_cortical_signal   =    np.array(signal)  [subject,  cluster_group, :,  :] # take the signal of interest
                assert np.shape(    per_subject_per_cluster_group_cortical_signal    ) == ( regions,    seconds_per_event )

                indices_of_roi_belonging_network_group  = np.where(   np.array(match)== network  )[0] #indices of ROI given a network
                rois_belonging_to_a_network_averaged = np.mean(  per_subject_per_cluster_group_cortical_signal   [indices_of_roi_belonging_network_group],   axis = 0 )
                assert np.shape(    rois_belonging_to_a_network_averaged    ) == (seconds_per_event, )

                event_group.append(rois_belonging_to_a_network_averaged)
            
            cortical_signal_baseline_corrected.append(  event_group)
        assert np.shape(cortical_signal_baseline_corrected) == (subjects,   number_of_clusters, seconds_per_event)

        dic_of_cortical_signal_baseline_corrected[f'{labels}']  =   cortical_signal_baseline_corrected


    return dic_of_cortical_signal_baseline_corrected


# trigger the network setup with network label
dic_of_cortical_signal_baseline_corrected_nw  = defaultdict(dict)
for i in range(1, 8):
    dic_of_cortical_signal_baseline_corrected_nw[f'{i}']    =   yeo_network_wise_setup(network = i)

# In addition of 7 networks, STG
bands = ['theta', 'alpha', 'low_beta', 'high_beta']
aud_ROI_label = 24
for band in bands: # cortical signal bands
    dic_of_cortical_signal_baseline_corrected_nw['8'][f'{band}'] = np.squeeze( np.mean (dic_of_envelope_signals_unthresholded[f'{band}'][:,:,[aud_ROI_label, aud_ROI_label + 179],:], axis = 2) )


def std(sig):
    low_bound = np.mean(sig, axis = 0) - 2 * np.std(sig, axis = 0)
    upper_bound = np.mean(sig, axis = 0) + 2 * np.std(sig, axis = 0)
    return low_bound, upper_bound

    
########################################################################################################################################################################



_7_networks = ['GSV','Visual', 'Somatomotor', 'Dorsal Attention', 'Ventral Attention', 'Limbic', 'FrontoPar','DMN', 'Auditory C']

if event_type == '30_events':
    _5clusters = ['Cluster 1', 'Cluster 2', 'Cluster 3']

if event_type == '19_events':
    _5clusters = ['Audio', '+ve frame offset', '-ve frame offset']


def plotting(band):
    """Plotting Network x cluster plots, alongside GSV. 

    Args:
        band (string): band label
    """
    a = 9
    b = 3
    counter = 1

    fig = plt.figure(figsize=(30, 30))
    for i in [ i for i in range(9) if i != 5 ]: # neglecting Limbic network 

        for cluster_group in range(number_of_clusters):

            plt.subplot(a,  b,  counter)
            plt.style.use('fivethirtyeight')
            if i == 0:# Plot GSV on top
                signal = smoothness_computed[f'{band}'][:,:,cluster_group]

            if i>0:# rest for cortical activity

                label_band = list(dic_of_cortical_signal_baseline_corrected_nw.keys())[i-1]
                signal = np.array(  dic_of_cortical_signal_baseline_corrected_nw[f'{label_band}'][f'{band}'] ) [:,cluster_group,:]

            assert np.shape(signal) == (subjects, seconds_per_event)

            mean_signal = np.mean( signal, axis = 0 )
            assert np.shape(mean_signal) == (seconds_per_event,)
            
            sem_signal = scipy.stats.sem(  signal, axis = 0 )
            assert np.shape(sem_signal) == (seconds_per_event,)
            
            plt.xticks(np.arange(0, video_duration, pre_stim_in_samples), labels = ['-200', '0', '200', '400'])
            
            plt.plot(   mean_signal    )
            plt.fill_between    (   range( seconds_per_event ), mean_signal - sem_signal, mean_signal + sem_signal, alpha = 0.2, label = 'SEM - subjects')

            if event_type == '19_events':
                if cluster_group == 1:
                    # plt.axvline(pre_stim_in_samples + 30, label = 'Frame change', c = 'g', linestyle = '-.')
                    plt.axvline(pre_stim_in_samples + 40, label = 'Frame change',c = 'g', linestyle = '-.')
                    # plt.axvline(pre_stim_in_samples + 50, c = 'g', linestyle = '-.')
            
            plt.axvspan(0, pre_stim_in_samples , alpha = 0.2, color = 'r', label = 'Baseline')
            plt.axvline(pre_stim_in_samples, label = 'Onset (ISC)', c = 'g', linestyle = 'dashed')
            
            if counter in [idx for idx in range(1, number_of_clusters * 8, number_of_clusters)]:
                plt.ylabel(f'{_7_networks[i]}',rotation=25, size = 'large', color = 'r')
            
            if counter <= 3:
                plt.title(f'{_5clusters[cluster_group]}')

        
            signal_baseline = signal[:pre_stim_in_samples]
            assert np.shape(signal_baseline) == (subjects, seconds_per_event)
            
            signal_baseline_averaged = np.mean(signal_baseline, axis = 1)
            assert np.shape(signal_baseline_averaged) == (subjects,)

            pvalues = np.zeros(post_stim_in_samples )

            for samples in range(pre_stim_in_samples, seconds_per_event ):
                signal_post_onset = signal[:,samples]
                
                pvalues[samples - pre_stim_in_samples] = scipy.stats.ttest_1samp(signal_post_onset, popmean=0)[1]
            
            pvalues_corrected = multipletests(pvalues, method = "bonferroni")[1]
            # t, c, c_pv, h0 = mne.stats.permutation_cluster_1samp_test(signal, adjacency = None, n_permutations=1)

            # print(sum(pvalues_corrected<=0.05))
            # pvalues_corrected = fdrcorrection(pvalues)[1]
            
            
            # if len(c)>0:
            #     idx =np.argwhere(c_pv<=0.05)
            #     if len(idx)>0:
            #         idx = np.hstack(np.array(idx))

            #         for id in idx:
            #             # print(np.squeeze(c)[id].min())
            #             # print(np.squeeze(c)[id].max())
            #             plt.axvspan(np.squeeze(c)[id].min(), np.squeeze(c)[id].max(), color = 'green', alpha =0.1)

            for pvals_index in range(len(pvalues_corrected)):
                if pvals_index in np.where(pvalues_corrected<=0.05)[0]:
                    plt.axvline(pvals_index + pre_stim_in_samples , color = 'orange', alpha = 0.2)

            plt.legend()
            counter += 1
    fig.supylabel('relative variation')
    fig.suptitle(f'{event_type}/ GSV + Cortical activity in Yeo NW/ {band} band / corrected, n_perm = 10000')
    fig.supxlabel('latency (ms)')
    # fig.savefig(f'/homes/v20subra/S4B2/Graph-related_analysis/ERD_oct/intra-cluster/{event_type}/{band}.jpg')

plotting('theta')
plotting('alpha')
plotting('low_beta')
plotting('high_beta')



# %%

### ANOVA
labels_ = ['GSV','Visual', 'Somatomotor', 'Dorsal Attention', 'Ventral Attention', 'Limbic', 'FrontoPar','DMN', 'Auditory C']

if event_type == '30_events':
    _3clusters = ['Cluster 1', 'Cluster 2', 'Cluster 3']

if event_type == '19_events':
    _3clusters = ['Audio', '+ve frame offset', '-ve frame offset']



def band_wise_stats(band):
    a, b, counter = 2, 4, 1
    fig = plt.figure(figsize = (25,25))
    for nw in [ nw for nw in range(9) if nw != 5 ]:
        if nw == 0:
            signal_for_anova = smoothness_computed[f'{band}']
            signal_for_anova = np.swapaxes(signal_for_anova, 1, 2)
            
            signal_for_anova_ = [signal_for_anova[:,0,:], signal_for_anova[:,1,:], signal_for_anova[:,2,:]]


        if nw >0:
            signal_for_anova = np.array(dic_of_cortical_signal_baseline_corrected_nw[f'{nw}'][f'{band}'])
            signal_for_anova_ = [signal_for_anova[:,0,:], signal_for_anova[:,1,:], signal_for_anova[:,2,:]]

        plt.subplot(a, b, counter)
        
        F_obs, c, c_pv, H0= mne.stats.permutation_cluster_test(signal_for_anova_, tail = 0, adjacency=None, out_type='mask', n_permutations=10000 )


     
        print("ANOVA signal shape",np.shape(signal_for_anova_))
        signal_c1 = signal_for_anova_[0]
        plt.plot(np.mean (signal_c1, axis = 0), label = _3clusters[0])
        c1 = std(signal_c1)
        plt.fill_between(range(seconds_per_event), c1[0], c1[1], alpha = 0.2)

        signal_c2 = signal_for_anova_[1]
        plt.plot(np.mean (signal_c2, axis = 0), label = _3clusters[1])
        c2 = std(signal_c2)
        plt.fill_between(range(seconds_per_event), c2[0], c2[1], alpha = 0.2)

        signal_c3 = signal_for_anova_[2]
        plt.plot(np.mean (signal_c3, axis = 0), label = _3clusters[2])
        c3 = std(signal_c3)
        plt.fill_between(range(seconds_per_event), c3[0], c3[1], alpha = 0.2)

        if len(c)>0:
            idx =np.argwhere(c_pv<=0.05)
            if len(idx)>0:
                for i in idx:
                    print(i[0])
                    plt.axvspan(c[i[0]][0].start, c[i[0]][0].stop, color = 'green', alpha =0.1)

        plt.legend()
        plt.title(f"{labels_[nw]}")
        plt.axvline(pre_stim_in_samples, linestyle = '-.')
        counter+=1

    fig.supxlabel("time (ms)")
    fig.suptitle(f"ANOVA; corrected with n_perms = 10000 / 95%CI / band = {band} / # event = {event_type}")
    # fig.savefig(f'/homes/v20subra/S4B2/Graph-related_analysis/ERD_oct/Anova_group_wise/{band}')

band_wise_stats("theta")
band_wise_stats("alpha")
band_wise_stats("low_beta")
band_wise_stats("high_beta")


# %%
###Post-hoc
##Paired t-test

# 1 vs 2; 1 vs 3; 2 vs 3
cluster_pairs = [(1, 2), (1, 3), (2, 3)]
_3clusters = ['1 vs 2', '1 vs 3', '2 vs 3']
def plotting_pairs(band):
    """Plotting Network x cluster plots, alongside GSV. 

    Args:
        band (string): band label
    """
    a = 9
    b = 3
    counter = 1

    fig = plt.figure(figsize=(30, 30))
    for i in [ i for i in range(9) if i != 5 ]: # neglecting Limbic network 

        for cluster_group in range(len(cluster_pairs)):
            print(cluster_pairs[cluster_group][0]-1)
            print(cluster_pairs[cluster_group][1]-1)
            plt.subplot(a,  b,  counter)
            plt.style.use('fivethirtyeight')
            if i == 0:# Plot GSV on top
                signal_cluster_1 = smoothness_computed[f'{band}'][:,:,cluster_pairs[cluster_group][0]-1]
                signal_cluster_2 = smoothness_computed[f'{band}'][:,:,cluster_pairs[cluster_group][1]-1]
                signal = signal_cluster_1 - signal_cluster_2

            if i>0:# rest for cortical activity

                label_band = list(dic_of_cortical_signal_baseline_corrected_nw.keys())[i-1]
                signal_cluster_1 = np.array(  dic_of_cortical_signal_baseline_corrected_nw[f'{label_band}'][f'{band}'] ) [:,cluster_pairs[cluster_group][0]-1,:]
                signal_cluster_2 = np.array(  dic_of_cortical_signal_baseline_corrected_nw[f'{label_band}'][f'{band}'] ) [:,cluster_pairs[cluster_group][1]-1,:]
                signal = signal_cluster_1 - signal_cluster_2

            assert np.shape(signal) == (subjects, seconds_per_event)


            def sem_and_plotting(sig, c = 'b', label = None):

                mean_signal = np.mean( sig, axis = 0 )
                assert np.shape(mean_signal) == (seconds_per_event,)
                
                sem_signal = scipy.stats.sem(  sig, axis = 0 )
                assert np.shape(sem_signal) == (seconds_per_event,)
            
                plt.xticks(np.arange(0, video_duration, pre_stim_in_samples), labels = ['-200', '0', '200', '400'])
                
                plt.plot(   mean_signal    , c = c, label = label)
                plt.fill_between    (   range( seconds_per_event ), mean_signal - sem_signal, mean_signal + sem_signal, alpha = 0.2, label = 'SEM - subjects', color = c)

            sem_and_plotting(signal_cluster_1, c = 'orange', label = f'C {cluster_pairs[cluster_group][0]}')
            sem_and_plotting(signal_cluster_2, label = f'C {cluster_pairs[cluster_group][1]}')

            if event_type == '19_events':
                if cluster_group == 1:
                    plt.axvline(pre_stim_in_samples + 40, label = 'Frame change',c = 'g', linestyle = '-.')
            
            plt.axvspan(0, pre_stim_in_samples , alpha = 0.2, color = 'r', label = 'Baseline')
            plt.axvline(pre_stim_in_samples, label = 'Onset (ISC)', c = 'g', linestyle = 'dashed')
            
            if counter in [idx for idx in range(1, number_of_clusters * 8, number_of_clusters)]:
                plt.ylabel(f'{_7_networks[i]}',rotation=25, size = 'large', color = 'r')
            
            if counter <= 3:
                plt.title(f'{_3clusters[cluster_group]}')

        
            signal_baseline = signal[:pre_stim_in_samples]
            assert np.shape(signal_baseline) == (subjects, seconds_per_event)
            
            signal_baseline_averaged = np.mean(signal_baseline, axis = 1)
            assert np.shape(signal_baseline_averaged) == (subjects,)

            pvalues = np.zeros(post_stim_in_samples )


            t, c, c_pv, h0 = mne.stats.permutation_cluster_1samp_test(signal, adjacency = None, n_permutations=10000)
            if len(c)>0:
                idx =np.argwhere(c_pv<=0.05)
                if len(idx)>0:
                    idx = np.hstack(np.array(idx))

                    for id in idx:
                        plt.axvspan(np.squeeze(c)[id].min(), np.squeeze(c)[id].max(), color = 'green', alpha =0.1)

            plt.legend()
            counter += 1
    fig.supylabel('relative variation')
    fig.suptitle(f'{event_type}/ Paired t-test {band} band / corrected, n_perm = 10000')
    fig.supxlabel('latency (ms)')
    # fig.savefig(f'/homes/v20subra/S4B2/Graph-related_analysis/ERD_oct/paired/{event_type}/{band}.jpg')
    
plotting_pairs('theta')
plotting_pairs('alpha')
plotting_pairs('low_beta')
plotting_pairs('high_beta')

# %%
signal_cluster_1 = smoothness_computed['theta'][:,:,0]
signal_cluster_2 = smoothness_computed['theta'][:,:,1]
signal_cluster_3 = smoothness_computed['theta'][:,:,2]
# %%
for i in range(25):
    plt.hist(signal_cluster_1[:,i], bins=20, label ='C1')
    plt.hist(signal_cluster_2[:,i], bins=20, label = 'C2')
    plt.hist(signal_cluster_3[:,i], bins=20, label = 'C3')
    plt.hist(np.hstack( np.array([signal_cluster_1, signal_cluster_2, signal_cluster_3])[:,:,i] ), alpha = 0.1, bins = 20, label = 'overall distri')
    plt.legend()
    plt.title(f'sample {i}')
    plt.show()
# %%
plt.hist(np.hstack( np.array([signal_cluster_1, signal_cluster_2, signal_cluster_3])[:,:,0] ))
# %%

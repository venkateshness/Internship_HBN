#%%
from cProfile import label
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import cluster
sns.set_theme()

import scipy.stats
import importlib
os.chdir('/homes/v20subra/S4B2/')
from Modular_Scripts import graph_setup

import numpy as np
from nilearn import datasets, plotting, image, maskers
# Average events (hilbert envelope)

# Unthresholded cortical signal
envelope_signal_bandpassed = np.load(
    '/users2/local/Venkatesh/Generated_Data/25_subjects_copy_FOR_TESTING/envelope_signal_bandpassed_with_beta_dichotomy.npz', mmap_mode='r')

alpha = envelope_signal_bandpassed['alpha']
low_beta = envelope_signal_bandpassed['lower_beta']
high_beta = envelope_signal_bandpassed['higher_beta']
theta = envelope_signal_bandpassed['theta']



dic_of_envelope_signals = {'theta' : theta, 'alpha' : alpha, 'low_beta' : low_beta, 'high_beta' : high_beta}

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



G = graph_setup.graph_setup_main()
G.compute_fourier_basis()

# %%
subjects = 25
number_of_clusters = 5
fs = 125
pre_stim_in_samples = 62 
post_stim_in_samples = 63
seconds_per_event = pre_stim_in_samples + post_stim_in_samples
regions = 360
events = np.load('/homes/v20subra/S4B2/AutoAnnotation/dict_of_clustered_events.npz')


##########################################################################################################################################################################
################################## Multi-purpose function to temporally slice, average those according to events #########################################################

def averaging_events_cluster_groups(band, dim_spatial):
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

cortico_signal_sliced_time_averaged = dict()

for labels, signal in dic_of_envelope_signals.items():
    cortico_signal_sliced_time_averaged[f'{labels}']    =   averaging_events_cluster_groups(    signal  ,   dim_spatial =   regions    )
    assert np.shape(    cortico_signal_sliced_time_averaged[f'{labels}']    ) == (  subjects,   regions,    number_of_clusters, seconds_per_event  )

##########################################################################################################################################################################


video_duration = fs * 170
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

smoothness_computed = dict()
smoothness_per_cluster = dict()

for labels, signal in dic_of_envelope_signals.items():
    smoothness_computed[f'{   labels  }'] = smoothness_computation (  signal )
    smoothness_per_cluster[f'{  labels  }'] = np.squeeze( averaging_events_cluster_groups( np.expand_dims(smoothness_computed[f'{labels}'].T ,axis=1), 1) )

# %%
# _100ms_in_samples = int( fs / 10 )
# _900ms_in_samples = fs - _100ms_in_samples

dic_of_erd_smoothness_signal = dict()

for labels, signal in smoothness_per_cluster.items():

    erd = list()
    for cluster_group in range(number_of_clusters):
        signal_sliced_per_event = signal[:,cluster_group,:]
        erd_sub = list()

        for subject in range(subjects):
            baseline = signal_sliced_per_event[subject][ : pre_stim_in_samples ]
            baseline_corrected_signal = ( signal_sliced_per_event[subject] - np.mean(baseline)  ) / np.mean(baseline)
            erd_sub.append(baseline_corrected_signal)

        erd.append(erd_sub)
    dic_of_erd_smoothness_signal[f'{labels}'] = erd
    assert np.shape(erd) == (number_of_clusters, subjects, seconds_per_event)
# %%

a = 5
b = 5
c = 1
fig = plt.figure(figsize=(25, 25))

for labels, signal in dic_of_erd_smoothness_signal.items():

    for cluster_group in range(number_of_clusters):
        
        baseline_corrected_signal = signal[cluster_group]
        
        plt.subplot(a, b, c)
        plt.style.use('fivethirtyeight')
        mean_bline_corrected = np.mean(baseline_corrected_signal,axis = 0)
        sem_bline_corrected = scipy.stats.sem(baseline_corrected_signal)
        plt.plot(mean_bline_corrected, color = 'b')
        
        if cluster_group == 3:
            plt.axvline(pre_stim_in_samples + 30, label = 'Frame change', c = 'g', linestyle = '-.')
            plt.axvline(pre_stim_in_samples + 40, label = 'Frame change', c = 'g', linestyle = '-.')
            plt.axvline(pre_stim_in_samples + 50, label = 'Frame change', c = 'g', linestyle = '-.')
        if cluster_group == 4:
            plt.axvline(pre_stim_in_samples - 45, label = 'Frame change', c = 'g', linestyle = '-.')
        
        plt.fill_between(range(seconds_per_event), mean_bline_corrected + sem_bline_corrected, mean_bline_corrected - sem_bline_corrected, alpha = 0.2, label = 'SEM - subjects')
        plt.axvspan(0, pre_stim_in_samples, alpha = 0.2, color = 'r', label = 'Baseline')
        plt.axvline(pre_stim_in_samples, label = 'Onset (ISC)', c = 'g', linestyle = 'dashed')
    
        # plt.xticks( list(range(0, seconds_per_event * fs + fs , fs)), labels = [-1000, 0, 1000, 2000] )
        plt.title('')
    
        plt.legend()
        c += 1



# fig.supylabel('Relative GSV')
# fig.supxlabel('Latency (ms)')
# plt.show()
# %%


def baseline_correction_network_wise_setup(network):
    dic_of_cortical_signal_baseline_corrected = dict()

    for labels, signal in cortico_signal_sliced_time_averaged.items():

        cortical_signal_baseline_corrected = list()

        for subject in range(subjects):
            event_group = list()

            for cluster_group in range(number_of_clusters):
                per_subject_per_cluster_group_cortical_signal   =    np.array(signal)  [subject,   :,  cluster_group,  :]
                assert np.shape(    per_subject_per_cluster_group_cortical_signal    ) == ( regions,    seconds_per_event )

                indices_of_roi_belonging_network_group  = np.where(   np.array(match)== network  )[0]
                rois_belonging_to_a_network_averaged = np.mean(  per_subject_per_cluster_group_cortical_signal   [indices_of_roi_belonging_network_group],   axis = 0 )
                assert np.shape(    rois_belonging_to_a_network_averaged    ) == (seconds_per_event, )

                ###############################################################
                ################# Baseline Correction #########################

                baseline_signal = rois_belonging_to_a_network_averaged[ :  pre_stim_in_samples]
                baseline_correction = (rois_belonging_to_a_network_averaged -   np.mean(baseline_signal))   /   np.mean(baseline_signal)
                ###############################################################
                ###############################################################


                event_group.append(baseline_correction)
            
            cortical_signal_baseline_corrected.append(  event_group)
        assert np.shape(cortical_signal_baseline_corrected) == (subjects,   number_of_clusters, seconds_per_event)

        dic_of_cortical_signal_baseline_corrected[f'{labels}']  =   cortical_signal_baseline_corrected


    return dic_of_cortical_signal_baseline_corrected



dic_of_cortical_signal_baseline_corrected_nw  = dict()
for i in range(1, 8):
    dic_of_cortical_signal_baseline_corrected_nw[f'{i}']    =   baseline_correction_network_wise_setup(network = i)
# %%
from statsmodels.stats.multitest import fdrcorrection, multipletests

_7_networks = ['GSV','Visual', 'Somatomotor', 'Dorsal Attention', 'Ventral Attention', 'Limbic', 'Frontoparietal','DMN']
_5clusters = ['Positive RMS', 'Fully negative RMS', 'Mixed', '+ve frame offset', '-ve frame offset']

def plotting(band):
    a = 5
    b = 5
    c = 1
    fig = plt.figure(figsize=(25, 25))
    for i in range(5):

        for cluster_group in range(number_of_clusters):
            plt.subplot(a,  b,  c)

            if i==0:
                label_band = list(dic_of_erd_smoothness_signal.keys())[i]
                signal = np.array(  dic_of_erd_smoothness_signal[f'{band}'] ) [cluster_group,:,:]
            
            else:
                label_band = list(dic_of_cortical_signal_baseline_corrected_nw.keys())[i-1]
                signal = np.array(  dic_of_cortical_signal_baseline_corrected_nw[f'{label_band}'][f'{band}'] ) [:,cluster_group,:]
            
            assert np.shape(signal) == (subjects, seconds_per_event)

            mean_signal = np.mean( signal, axis = 0 )
            assert np.shape(mean_signal) == (seconds_per_event,)
            
            sem_signal = scipy.stats.sem(  signal, axis = 0 )
            assert np.shape(sem_signal) == (seconds_per_event,)
            
            # plt.xticks( list(range(0, seconds_per_event * fs + fs , fs)), labels = [-1000, 0, 1000, 2000] )
            
        
            plt.plot(   mean_signal    )
            plt.fill_between    (   range( seconds_per_event ), mean_signal - sem_signal, mean_signal + sem_signal, alpha = 0.2, label = 'SEM - subjects')

            if cluster_group == 3:
                plt.axvline(pre_stim_in_samples + 30, label = 'Frame change', c = 'g', linestyle = '-.')
                plt.axvline(pre_stim_in_samples + 40, label = 'Frame change', c = 'g', linestyle = '-.')
                plt.axvline(pre_stim_in_samples + 50, label = 'Frame change', c = 'g', linestyle = '-.')
            
            if cluster_group == 4:
                plt.axvline(pre_stim_in_samples - 45, label = 'Frame change', c = 'g', linestyle = '-.')

            plt.axvspan(0, pre_stim_in_samples , alpha = 0.2, color = 'r', label = 'Baseline')
            plt.axvline(pre_stim_in_samples, label = 'Onset (ISC)', c = 'g', linestyle = 'dashed')
            
            if c in [idx for idx in range(1, 41, 5)]:
                plt.ylabel(f'{_7_networks[i]}',rotation=25, size = 'large', color = 'r')
            
            if c <= 5:
                plt.title(f'{_5clusters[cluster_group]}')

        
            signal_baseline = signal[:pre_stim_in_samples]
            signal_baseline_averaged = np.mean(signal_baseline, axis = 1)
            
            pvalues = np.zeros(post_stim_in_samples )

            for samples in range(pre_stim_in_samples, seconds_per_event ):
                signal_signal = signal[:,samples]
                pvalues[samples - pre_stim_in_samples] = scipy.stats.ttest_rel(signal_baseline_averaged, signal_signal)[1]
            
            # pvalues_corrected = multipletests(pvalues, method = "bonferroni")[1]
            # print(sum(pvalues_corrected<=0.05))
            pvalues_corrected = fdrcorrection(pvalues)[1]
            print(sum(pvalues_corrected<=0.05))

            for pvals_index in range(len(pvalues_corrected)):
                if pvals_index in np.where(pvalues_corrected<=0.05)[0]:
                    print('yes')
                    plt.axvline(pvals_index + pre_stim_in_samples , color = 'orange', alpha = 0.2)

            plt.legend()
            c += 1
    fig.supylabel('relative variation')
    fig.suptitle(f'{band} band')
    fig.supxlabel('latency (ms)')
    # fig.savefig(f'/homes/v20subra/S4B2/Graph-related_analysis/ERD_7_networks/{band}.jpg')

plotting('theta')
plotting('alpha')
plotting('low_beta')
plotting('high_beta')

# %%

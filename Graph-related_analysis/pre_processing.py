#%%
from cProfile import label
from collections import defaultdict
from email.policy import default
import nilearn
from nilearn.input_data import NiftiLabelsMasker
import scipy # NiftiMasker
import seaborn as sns
from nilearn.connectome import ConnectivityMeasure
import numpy as np
import pickle
import torch
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import librosa

sns.set_theme()

subjects = 21
number_of_clusters = 3
fs = 125
pre_stim_in_samples = 25
post_stim_in_samples = 63
seconds_per_event = pre_stim_in_samples + post_stim_in_samples
video_duration = seconds_per_event
regions = 360
event_type = '30_events'
path_Glasser = '/homes/v20subra/S4B2/GSP/Glasser_masker.nii.gz'

def parcellation(mask,img):
    # Glasser is a reference map for regions in the brain. It splits the brain into 360 regions.
    glassermasker = NiftiLabelsMasker(labels_img=path_Glasser,mask_img=mask,standardize=True)
    parcellated = glassermasker.fit_transform(img)
    return parcellated

def connectivitymeasure(which_subject_after_parcellation):
    correlation_measure = ConnectivityMeasure(kind='correlation')
    correlation_matrix = correlation_measure.fit_transform([which_subject_after_parcellation])[0]# 25 indiv 

    np.fill_diagonal(correlation_matrix, 0)

    return correlation_matrix


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


envelope_signal_bandpassed_bc_corrected_sliced = defaultdict(dict)

for labels, signals in envelope_signal_bandpassed_bc_corrected.items():
    envelope_signal_bandpassed_bc_corrected_sliced[f'{labels}'] = envelope_signal_bandpassed_bc_corrected[f'{labels}'][idx_for_the_existing_subjects]


def NNgraph(corr_mat):
    """Nearest Neighbour graph Setup.

    Returns:
        Matrix of floats: A weight matrix for the thresholded graph
    """
    
    connectivity = corr_mat

    graph = torch.from_numpy(connectivity)
    graph.fill_diagonal_(0)
    knn_graph = graph
    

    degree = torch.tensor(np.diag(sum(connectivity!=0)))#torch.diag(knn_graph.sum(dim = 0))
    adjacency = knn_graph
    laplacian   = degree - adjacency
    # values, eigs = torch.linalg.eigh(laplacian)
    return laplacian, adjacency

def smoothness_computation(band, laplacian):
    """The main function that does GFT, function-calls the temporal slicing, frequency summing, pre- post- graph-power accumulating 

    Args:
        band (array): Envelope band to use

    Returns:
        dict: Baseline-corrected ERD for all trials 
    """
    per_subject = list()
    for event in range(number_of_clusters):
        per_event = list()
        for timepoints in range(seconds_per_event):
            signal = band[event, :,timepoints]

            stage1 = np.matmul(signal.T, laplacian)

            final = np.matmul(stage1, signal)
            per_event.append(final)
        
        per_subject.append(per_event)

    data_to_return = np.array(per_subject).T
    assert np.shape( data_to_return   ) == (video_duration, number_of_clusters)
    return data_to_return


i = 0

# parcellated_data = list()
# for subject_label in tqdm(subjects_data_available_for):
#     img = f'/users2/local/Venkatesh/HBN/CPAC_preprocessed/sub-{subject_label}_ses-1/functional_to_standard/_scan_rest/_selector_CSF-2mmE-M_aC-CSF+WM-2mm-DPC5_M-SDB_P-2_BP-B0.01-T0.1_C-S-1+2-FD-J0.5/bandpassed_demeaned_filtered_antswarp.nii.gz'
#     mask = f'/users2/local/Venkatesh/HBN/CPAC_preprocessed/sub-{subject_label}_ses-1/functional_brain_mask_to_standard/_scan_rest/sub-{subject_label}_task-rest_bold_calc_resample_volreg_mask_antswarp.nii.gz'
    
#     parcellated = parcellation(mask, img)

#     parcellated_data.append(parcellated)

parcellated_data = np.load("/users2/local/Venkatesh/Generated_Data/25_subjects_new/graph_space/21_subject_fmri_parcellated.npz")['parcellated_data']

gsv_original_all_bands = defaultdict(dict)

for labels, signal in envelope_signal_bandpassed_bc_corrected_sliced.items():
    gsv_original_graph = list()

    for subject in range(subjects):
        
        # print('running correlation_matrix')
        correlation_matrix = connectivitymeasure(parcellated_data[subject])

        # print('running weight_matrix_after_NN')
        laplacian, adjacency = NNgraph(correlation_matrix)
        
        
        signal_normalized = envelope_signal_bandpassed_bc_corrected_sliced[f'{labels}'][subject]/np.diag(laplacian)[np.newaxis, :, np.newaxis]
        gsv_original_graph.append(smoothness_computation(signal_normalized, laplacian))

    gsv_original_all_bands[f'{labels}'] = gsv_original_graph


from scipy import io as sio

gsv_resting_state_all_bands = defaultdict(dict)

for labels, signal in envelope_signal_bandpassed_bc_corrected_sliced.items():
    gsv_resting_state = list()

    for i in range(subjects):
        learnt_weights = np.load('/users2/local/Venkatesh/Generated_Data/25_subjects_new/graph_space/pli_graph_21_subjects.npz')[f'{labels}'][i]

        degree = np.diag(sum(learnt_weights!=0))
        laplacian_pli = degree - learnt_weights
        
        signal_normalized = envelope_signal_bandpassed_bc_corrected_sliced[f'{labels}'][i]/np.diag(degree)[np.newaxis,:, np.newaxis]
        
        gsv_resting_state.append(smoothness_computation(signal_normalized, laplacian_pli) )
    
    gsv_resting_state_all_bands[f'{labels}'] = gsv_resting_state
    

from statsmodels.stats.multitest import fdrcorrection, multipletests

# _5clusters = ['Audio', '+ve frame offset', '-ve frame offset']
_5clusters = ['Cluster 1', 'Cluster 2', 'Cluster 3']
#########################################################################################################
###################################RMS post-hoc difference###################################
samples, sample_rate = librosa.load('/homes/v20subra/S4B2/Despicable Me-HQ.wav',sr=None)
samples_normed = (samples - np.average(samples))/np.std(samples)
rms = librosa.feature.rms(y=samples_normed,hop_length=386,frame_length=1000)

duration = 170
fs = 48000

events = np.load('/homes/v20subra/S4B2/AutoAnnotation/dict_of_clustered_events.npz')
rms_all_groups = list()
for i,j in events.items():
    group = list()
    for second in j:
        group.append(np.sqrt(np.square(np.average(samples_normed[fs * second : fs * second + int(fs/2)]) - np.average(samples_normed[fs * second - int(fs/5) : fs * second]))))
    rms_all_groups.append(np.average(group))

#########################################################################################################


def mean_std(signal):
    mean = np.mean(signal, axis= 0)
    sem = scipy.stats.sem(signal, axis= 0)
    # std = np.std(signal, axis=0)

    return mean, sem


a,b,c =  5, 3, 1
fig,ax = plt.subplots(figsize = (20,20))

for labels in gsv_original_all_bands.keys():

    for event_no in range(number_of_clusters):

        mean_orig, sem_orig = mean_std(np.array(gsv_original_all_bands[f'{labels}'])[:,:,event_no])
        # mean_kalo, sem_kalo = mean_std(np.array(gsv_resting_state_all_bands[f'{labels}'])[:,:,event_no])

        plt.style.use('fivethirtyeight')
        plt.subplot(a, b, c)
        plt.plot(mean_orig)
        plt.fill_between(range(video_duration), mean_orig-sem_orig, mean_orig+sem_orig, alpha = 0.2)

        # plt.plot(mean_kalo, label = 'on PLI RSFC', alpha = 0.5)
        # plt.fill_between(range(video_duration), mean_kalo-sem_kalo, mean_kalo+sem_kalo, alpha = 0.2)
        # plt.xticks(ticks = np.arange(0,88,25), labels = ['-200', '0', '200', '400'])
        plt.axvline(25,linestyle = 'dashed')
        
        if labels == 'theta':
            plt.title(f'{_5clusters[event_no]}')
        
        if c in [idx for idx in range(1, number_of_clusters * 4, number_of_clusters )]: 
                plt.ylabel(f'{labels}',rotation=25, size = 'large', color = 'r')
        
        plt.legend()
        c+=1
        
        _, pvalues = scipy.stats.ttest_rel(np.array(gsv_resting_state_all_bands[f'{labels}'])[:,pre_stim_in_samples:,event_no], np.array(gsv_original_all_bands[f'{labels}'])[:,pre_stim_in_samples:,event_no])
        pvalues_corrected = multipletests(pvalues, method = "bonferroni")[1]
        idx = np.argwhere(pvalues_corrected<=0.05)

        if len(idx)>0:
            [plt.axvline(_x, linewidth=1, color='g') for _x in (idx)+25]

fig.suptitle(f'{event_type} / GSV on fMRI RSFC / cortical signal = unthresholded, with Bonferroni correction')
fig.supylabel('relative variation')
fig.supxlabel('time lag (ms)')
fig.savefig(f'/homes/v20subra/S4B2/Graph-related_analysis/ERD_august/GSV/FC/{event_type}/GSV.jpg')
# %%

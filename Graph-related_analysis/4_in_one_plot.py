#%%
from cProfile import label
from collections import defaultdict
from cv2 import mean, norm
from matplotlib import axis, colors
import numpy as np
import scipy
import matplotlib.pyplot as plt
import scipy.stats
import matplotlib as mpl
import seaborn as sns

from nilearn.regions import signals_to_img_labels  
# load nilearn label masker for inverse transform
from nilearn.input_data import NiftiLabelsMasker, NiftiMasker
from nilearn.datasets import fetch_icbm152_2009
from nilearn import image, plotting
from nilearn import datasets
import nilearn
import os
import matplotlib.colors as colors

# We are gonna make four different figures,
# A) raw Envelope signal (theta band)
# B) Graph power (low frequency and theta band) (graph_power.py)
# C) Graph smoothness on envelope theta band (smoothness.py)
# D) Graph smoothness on eLORETA cortical signal wide-band (smoothness.py)
#%%
def slicing(what_to_slice, where_to_slice, axis):
    """Temporal Slicing. Function to slice at the temporal axis.
    Args:
        what_to_slice (array): The array to do the temporal slicing for; the input dim = subject x entire video duration
        where_to_slice (array of "range"): The period(s) when to do the temporal slicing in; dim = 1D array
        axis (int): The axis of the array to do the temporal slicing on

    Returns:
        array: An array temporally sliced; dim = subject x `total_duration_in_samples` (375 = 3s)
    """
    array_to_append = list()
    if axis > 2:
        array_to_append.append(what_to_slice[:, :, where_to_slice])
    else:
        print( "size for the what_to_slice:", np.shape(what_to_slice))
        array_to_append.append(what_to_slice[:, where_to_slice])
    return array_to_append

def stats_SEM(freqs):
    """SEM estimation -- Standard error of the Mean

    Args:
        freqs (dict): The grand-averaged graph smoothness to apply the SEM on

    Returns:
        array: SEMed graph smoothness 
    """
    return scipy.stats.sem(freqs, axis = 1)

trials = [8, 56, 68, 74, 86, 132, 162]
fs = 125
subjects = 25
baseline_duration_of_900ms_in_samples = 113# baseline = -1000ms to -100ms, so 900ms; since fs = 125, 900 ms = 113 samples
total_duration_in_samples = 375
total_roi = 360

dic_for_envelope_signal_plot = defaultdict()   

envelope_signal_bandpassed = np.load(
    '/users2/local/Venkatesh/Generated_Data/25_subjects_copy_FOR_TESTING/envelope_signal_bandpassed_with_beta_dichotomy.npz', mmap_mode='r')


"""
Code below for generating plot A
"""
def env_band(band):
    for i in range(len(trials)): # looping over each trials
            indices = np.hstack([
                np.arange(trials[i] * fs - fs, trials[i] * fs + 2 * fs)])

            env = np.squeeze(slicing(envelope_signal_bandpassed[f'{band}'] , indices, axis = 3))
            env_reordered = np.squeeze(np.moveaxis(env, [0, 1, 2], [0, 2, 1])) #swapping axis for easier handling
            assert np.shape(env_reordered) == (subjects, total_duration_in_samples, total_roi)

            env_reordered_baseline = env_reordered[:,:baseline_duration_of_900ms_in_samples,:]
            env_reordered_baseline_averaged = np.expand_dims(np.mean(env_reordered_baseline,axis=1),1) # adding dim to match the dim for both subtracting elements
            baseline_done =  (env_reordered - env_reordered_baseline_averaged)/env_reordered_baseline_averaged
            
            assert np.shape(baseline_done) == (subjects, total_duration_in_samples, total_roi)
            roi_averaged = np.mean(baseline_done, axis = 2)
            assert np.shape(roi_averaged) == (subjects, total_duration_in_samples)

            dic_for_envelope_signal_plot[i] = roi_averaged

    trial_averaged = np.mean(list(dic_for_envelope_signal_plot.values()),axis = 0)
    assert np.shape(trial_averaged) == (subjects, total_duration_in_samples)

    a, b, c = 2, 2, 1
    fig = plt.figure(figsize= (25, 25))

    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize='x-small')
    plt.rc('ytick', labelsize='x-small')

    plt.tick_params(left=False, labelleft=False) #remove ticks
    plt.box(False) #remove box
    plt.style.use('fivethirtyeight') # For better style

    A_mean = np.mean(trial_averaged, axis=0)
    assert np.shape(A_mean) == (total_duration_in_samples,)
    plt.subplot(a, b, c)

    plt.plot(A_mean,color = 'r')

    A_sem = stats_SEM(trial_averaged.T)

    plt.fill_between(range(total_duration_in_samples), A_mean - A_sem, A_mean + A_sem, alpha = 0.2, label = 'SEM')
    plt.xticks(np.arange(0, total_duration_in_samples+1, 62.5), np.arange(-1000, 2500, 500))
    plt.xlabel('time (ms)')
    plt.axvspan(xmin = 0, xmax = baseline_duration_of_900ms_in_samples,
                color='r', alpha = 0.2, label = 'Baseline period')
    plt.title(f'Envelope {band} band',fontsize = 25)
    plt.legend()
    return A_mean, A_sem

# """
# The code block for generating B - D
# """
# def plotting(file, c):
#     plt.subplot(a, b, c)
#     loaded_file = np.load(f'/users2/local/Venkatesh/Generated_Data/25_subjects_copy_FOR_TESTING/power_smoothness_4_in_one_plot/{file}.npz',allow_pickle=True)[f'{file}']
#     sem = np.ravel(loaded_file)[0]['sem']
#     average = np.ravel(loaded_file)[0]['average']
    
#     plt.plot(average, 'r')
#     plt.fill_between(range(total_duration_in_samples), average - sem, average + sem, alpha = 0.2, label = 'SEM')
#     plt.xticks(np.arange(0, total_duration_in_samples+1, 62.5), np.arange(-1000, 2500, 500))
#     plt.xlabel('time (ms)')
#     plt.axvspan(xmin = 0, xmax = baseline_duration_of_900ms_in_samples,
#                 color='r', alpha = 0.2, label = 'Baseline period')
#     plt.title(f'{file}',fontsize = 25)
#     plt.legend()

# plotting('gPower', 2)
# # plotting('smoothness_theta', 3)
# # plotting('smoothness_wideband', 4)
# plt.tight_layout()

# fig.savefig('/homes/v20subra/S4B2/Graph-related_analysis/4_in_one',bbox_inches=None)
# %%
os.chdir('/homes/v20subra/S4B2/')
path_Glasser = 'GSP/Glasser_masker.nii.gz'



def plot(band, A_mean, A_sem):
    dic_for_envelope_signal_plot = defaultdict(dict)   

    for i in range(len(trials)): # looping over each trials
            indices = np.hstack([
                np.arange(trials[i] * fs - fs, trials[i] * fs + 2 * fs)])

            env = np.squeeze(slicing(envelope_signal_bandpassed[f'{band}'] , indices, axis = 3))
            env_reordered = np.squeeze(np.moveaxis(env, [0, 1, 2], [0, 2, 1])) #swapping axis for easier handling
            assert np.shape(env_reordered) == (subjects, total_duration_in_samples, total_roi)

            env_reordered_baseline = env_reordered[:,:baseline_duration_of_900ms_in_samples,:]
            env_reordered_baseline_averaged = np.expand_dims(np.mean(env_reordered_baseline,axis=1),1) # adding dim to match the dim for both subtracting elements
            baseline_done =  (env_reordered - env_reordered_baseline_averaged)/env_reordered_baseline_averaged
            
            assert np.shape(baseline_done) == (subjects, total_duration_in_samples, total_roi)
            sub_averaged = np.mean(baseline_done, axis = 0)
            assert np.shape(sub_averaged) == (total_duration_in_samples, total_roi)

            dic_for_envelope_signal_plot['average'][i] = sub_averaged
            sub_std = np.std(baseline_done, axis = 0)

            dic_for_envelope_signal_plot['std'][i] = sub_std


    trial_averaged_sub_avg = np.mean(list(dic_for_envelope_signal_plot['average'].values()),axis = 0)
    trial_averaged_sub_std = np.mean(list(dic_for_envelope_signal_plot['std'].values()),axis = 0)

    assert np.shape(trial_averaged_sub_avg) == (total_duration_in_samples, total_roi)

    
    fig = plt.figure(figsize=(15,10))
    ax = plt.subplot(211)
    ax2 = ax.twinx()
    norm = colors.TwoSlopeNorm(vcenter=0)
    sns.heatmap(trial_averaged_sub_avg.T,ax=ax, cmap='bwr',norm=norm)


    ax2.plot(A_mean,color = 'r')
    
    print(np.shape(list(dic_for_envelope_signal_plot['average'].values())))
   
    ax2.fill_between(range(total_duration_in_samples), A_mean - A_sem, A_mean + A_sem ,color = 'b',alpha=0.2)
    ax.set_title('Average thru subject')
    ax.axvline(125, c = 'green')
    ax.set_xticks(np.arange(0, total_duration_in_samples+1, 62.5))
    ax.set_xticklabels([])

    axx = plt.subplot(212)
    axx2 = axx.twinx()
    sns.heatmap(trial_averaged_sub_std.T, ax=axx)
    axx2.plot(A_mean,color = 'r')
    axx2.fill_between(range(total_duration_in_samples), A_mean - A_sem, A_mean + A_sem ,color = 'b',alpha=0.2)
    axx2.set_xticks(np.arange(0, total_duration_in_samples+1, 62.5))
    axx2.set_xticklabels(np.arange(-1000, 2500, 500))
    axx2.axvline(125, c = 'green')
    axx2.set_title("STD thru subjects")

    # plt.title('STD thru subjects')
    # plt.suptitle(f'{band}')
    # plt.xticks(np.arange(0, total_duration_in_samples+1, 62.5), np.arange(-1000, 2500, 500))
    # plt.xlabel('time (ms)')
    # plt.axvline(125, c = 'green')
    fig.supylabel('ROI')
    fig.suptitle(f'{band} Envelope obtained from the cortical activation signal')
    fig.supxlabel('time (ms)')
    plt.show()
    fig.savefig(f'/homes/v20subra/S4B2/Graph-related_analysis/Functional_graph_setup/eloreta/{band}')
bands = ['theta', 'alpha', 'lower_beta', 'higher_beta']

for i in range(len(bands)):
    avg,sem =env_band(bands[i])
    plot(bands[i], avg, sem)

# %%

sourceCCA = np.load('/users2/local/Venkatesh/Generated_Data/25_subjects_copy_FOR_TESTING/sourceCCA_ISC_8s_window.npz')['sourceISC']
noise_floor = np.load('/users2/local/Venkatesh/Generated_Data/25_subjects_copy_FOR_TESTING/noise_floor_8s_window.npz')['isc_noise_floored']
significance = np.where(np.max(np.array(noise_floor)[:,0,:],axis=0)<sourceCCA[0])[0]


# %%

def axvspan():
    for i in range(len(trials)-1):
        plt.axvspan( xmin= trials[i]-1, xmax = trials[i], color='b', alpha = 0.2)
        plt.axvspan( xmin= trials[i], xmax = trials[i]+2, color='r', alpha = 0.2)
    plt.axvspan( xmin= trials[-1]-1, xmax = trials[-1], color='b', alpha = 0.2, label= 'Pre-stimulus')
    plt.axvspan( xmin= trials[-1], xmax = trials[-1]+2, color='r', alpha = 0.2, label = 'Post-stimulus')


fig = plt.figure(figsize=(15,10))
plt.plot(noise_floor[:,0,:].T,c='grey',alpha=0.2)
plt.plot(sourceCCA[0])
plt.plot(significance,sourceCCA[0][significance],c='r',marker='o',ls="", label ="Significant ISC; p < 0")
plt.xlabel('time (s)')
plt.ylabel('ISC coefficients')
plt.title("ISC First component")
axvspan()
plt.legend()
plt.show()
fig.savefig('/homes/v20subra/S4B2/3Source_Inversion_full_stack/first_comp')
# %%
plt.axvspan()

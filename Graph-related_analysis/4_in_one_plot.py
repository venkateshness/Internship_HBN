#%%
from cProfile import label
from collections import defaultdict
import numpy as np
import scipy
import matplotlib.pyplot as plt
import scipy.stats
import matplotlib as mpl
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

for i in range(len(trials)): # looping over each trials
        indices = np.hstack([
            np.arange(trials[i] * fs - fs, trials[i] * fs + 2 * fs)])

        env = np.squeeze(slicing(envelope_signal_bandpassed['theta'] , indices, axis = 3))
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
plt.figure(figsize= (25, 25))

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
plt.title('Envelope Theta band',fontsize = 25)
plt.legend()

"""
The code block for generating B - D
"""
def plotting(file, c):
    plt.subplot(a, b, c)
    loaded_file = np.load(f'/users2/local/Venkatesh/Generated_Data/25_subjects_copy_FOR_TESTING/power_smoothness_4_in_one_plot/{file}.npz',allow_pickle=True)[f'{file}']
    sem = np.ravel(loaded_file)[0]['sem']
    average = np.ravel(loaded_file)[0]['average']
    
    plt.plot(average, 'r')
    plt.fill_between(range(total_duration_in_samples), average - sem, average + sem, alpha = 0.2, label = 'SEM')
    plt.xticks(np.arange(0, total_duration_in_samples+1, 62.5), np.arange(-1000, 2500, 500))
    plt.xlabel('time (ms)')
    plt.axvspan(xmin = 0, xmax = baseline_duration_of_900ms_in_samples,
                color='r', alpha = 0.2, label = 'Baseline period')
    plt.title(f'{file}',fontsize = 25)
    plt.legend()

plotting('gPower', 2)
plotting('smoothness_theta', 3)
plotting('smoothness_wideband', 4)
plt.tight_layout()


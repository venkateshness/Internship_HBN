# %%
import scipy.stats as st
from cProfile import label
from collections import defaultdict
from email.mime import base
from operator import index
from re import I
from tkinter import Y
from turtle import color, pos, shape
from click import style
from cv2 import accumulate, norm
import numpy as np
import matplotlib.pyplot as plt
from pygsp import graphs, filters
from pygsp import plotting as gsp_plt
from nilearn import image, plotting, datasets


from pathlib import Path
from scipy import io as sio
from pygsp import graphs

import scipy
import torch
import pickle
import seaborn as sns
import pandas as pd
from statannot import add_stat_annotation
import matplotlib as mpl
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict
import ptitprince as pt

sns.set_theme()
############################################################
##########Getting the Graph ready###########################
############################################################


def graph_setup(unthresholding, weights):
    """Function to finalize the graph setup -- with options to threshold the graph by overriding the graph weights

    Args:
        unthresholding (bool): do you want a graph unthresholded ?
        weights (Float): Weight matrix

    Returns:
        Graph: Returns the Graph, be it un- or thresholded, which the latter is done using the 8Nearest-Neighbour
    """
    coordinates = sio.loadmat(
        '/homes/v20subra/S4B2/GSP/Glasser360_2mm_codebook.mat')['codeBook']

    G = graphs.Graph(weights, gtype = 'HCP subject',
                     lap_type = 'combinatorial', coords = coordinates)
    G.set_coordinates('spring')
    print('{} nodes, {} edges'.format(G.N, G.Ne))

    if unthresholding:
        pickle_file = '/homes/v20subra/S4B2/GSP/MMP_RSFC_brain_graph_fullgraph.pkl'

        with open(pickle_file, 'rb') as f:
            [connectivity] = pickle.load(f)
        np.fill_diagonal(connectivity, 0)

        G = graphs.Graph(connectivity)
        print(G.is_connected())
        print('{} nodes, {} edges'.format(G.N, G.Ne))

    return G


def NNgraph():
    """Nearest Neighbour graph Setup.

    Returns:
        Matrix of floats: A weight matrix for the thresholded graph
    """

    pickle_file = '/homes/v20subra/S4B2/GSP/MMP_RSFC_brain_graph_fullgraph.pkl'

    with open(pickle_file, 'rb') as f:
        [connectivity] = pickle.load(f)
    np.fill_diagonal(connectivity, 0)

    graph = torch.from_numpy(connectivity)
    knn_graph = torch.zeros(graph.shape)
    for i in range(knn_graph.shape[0]):
        graph[i, i] = 0
        best_k = torch.sort(graph[i, :])[1][-8:]
        knn_graph[i, best_k] = 1
        knn_graph[best_k, i] = 1

    degree = torch.diag(torch.pow(knn_graph.sum(dim = 0), -0.5))

    weight_matrix_after_NN = torch.matmul(
        degree, torch.matmul(knn_graph, degree))
    return weight_matrix_after_NN


G = graph_setup(False, NNgraph())
G.compute_fourier_basis()
# %%
index = [8, 56, 68, 74, 86, 132, 162]
fs = 125
subjects = 25# baseline = -1000ms to -100ms, so 900ms; since fs = 125, 900 ms = 113 samples
baseline_duration_of_900ms_in_samples = 113
total_duration_in_samples = 375
low_freq_range = np.arange(1, 51)
med_freq_range = np.arange(51, 200)
high_freq_range = np.arange(200, 360)
pvalues = list()
env_bands = ['Low', 'Med', 'High']

envelope_signal_bandpassed = np.load(
    '/users2/local/Venkatesh/Generated_Data/25_subjects_copy_FOR_TESTING/envelope_signal_bandpassed_with_beta_dichotomy.npz', mmap_mode='r')

alpha = envelope_signal_bandpassed['alpha']
low_beta = envelope_signal_bandpassed['lower_beta']
high_beta = envelope_signal_bandpassed['higher_beta']
theta = envelope_signal_bandpassed['theta']


def slicing(what_to_slice, where_to_slice, axis):
    """Temporal Slicing. Function to slice at the temporal axis.
    Args:
        what_to_slice (array): The array to do the temporal slicing for; **dim = subject x frequency x whole_time_duration**
        where_to_slice (array of "range"): The period(s) when to do the temporal slicing in; dim = 1D array
        axis (int): The axis of the array to do the temporal slicing on

    Returns:
        array: An array temporally sliced
    """
    array_to_append = list()
    if axis > 2:
        array_to_append.append(what_to_slice[:, :, where_to_slice])
    else:
        array_to_append.append(what_to_slice[:, where_to_slice])
    return array_to_append


def slicing_time(freqs, indices_whole):
    """A precursor function to do temporal slicing. 
    Dimension inflow = subject x frequency x whole_time_duration

    Args:
        freqs (array): The GFT-ed frequency to slice
        indices_whole (array of range):  Indices for the time period containing baseline, stimulus data

    Returns:
        items_whole: A sliced array for the pre-strong period. 
        Dim outflow = subject x frequency x len(item_pre_strong)

    """
    items_whole = np.squeeze(slicing(freqs, indices_whole, axis = 3))
    return items_whole


def sum_freqs(freqs, axis):
    """Integrating frequency. Applying the L2 norm and summing the frequency power. Dim = subj x frequency x sliced_time

    Args:
        freqs (array): Frequency to sum
        axis (int): Axis of the Frequency

    Returns:
        array: Summed frequency power array (dim = subjects x 1 x sliced_time)
    """
    return np.linalg.norm(freqs, axis = axis)  # L2 norm


def baseline(whole):
    """ ERD baseline setup. Dim = sub x sliced_time

        Args:
            whole (array): Pre- and Post-stimulus/strong; duration = -1 to 2, so 3 seconds

        Returns:
            array : Subject-wise ERD setup. Dim = sub x sliced_time
    """
    pre = whole[:, :baseline_duration_of_900ms_in_samples]
    print(np.shape(np.mean(pre, axis = 1)))
    return np.array((whole.T - np.mean(pre, axis = 1))/np.mean(pre, axis = 1))


def averaging_time(freqs, axis=-1):
    """Temporal Average. Dim = sub x sliced_time

    Args:
        freqs (array): Array of which graph frequency to average temporally
        axis (int, optional): Temporal axis. Defaults to -1.

    Returns:
        array: Temporally averaged (dim =  sub x 1)
    """
    return np.average(freqs.T, axis = axis)


def stats_SEM(freqs):
    """SEM estimation -- Standard error of the Mean

    Args:
        freqs (dict): The grand-averaged graph power to apply the SEM on

    Returns:
        array: SEMed graph power 
    """
    return scipy.stats.sem(freqs, axis = 1)


def accumulate_freqs(power_pre_low, power_pre_med, power_pre_high):
    """Concatenate after baseline setup the pre and post; across graph frequencies

    Args:
        freq_pre_low (array): Pre-stimulus; low frequency
        freq_post_low (array): Post-stimulus; low frequency
        freq_pre_med (array): Pre-stimulus; medium frequency
        freq_post_med (array): Post-stimulus; medium frequency
        freq_pre_high (array): Pre-stimulus; high frequency
        freq_post_high (array): Post-stimulus; high frequency


    Returns:
        dict: concatenated graph power of baseline, post-stimulus for all three graph frequencies
    """
    dic_append_everything = defaultdict(dict)
    for i in range(3):
        if i == 0:
            power_after_baseline = baseline(power_pre_low)
        elif i == 1:
            power_after_baseline = baseline(power_pre_med)
        else:
            power_after_baseline = baseline(power_pre_high)

            print(np.shape(power_after_baseline))
        
        dic_append_everything[i] = power_after_baseline

    return dic_append_everything


def master(band):
    """The main function that does GFT, function-calls the temporal slicing, frequency summing, pre- post- graph-power accumulating 

    Args:
        band (array): Envelope band to use

    Returns:
        dict: Baseline-corrected ERD for all trials 
    """
    one_freq = list()
    GFTed_cortical_signal = [G.gft(np.array(band[i])) for i in range(subjects)] # Applying GFT on the cortical signal, for all the subjects

    GFT_one_frequency = np.array(GFTed_cortical_signal)[:, np.arange(0,150,50), :]

    GFTed_cortical_signal_low_freq = np.array(GFTed_cortical_signal)[
        :, low_freq_range, :]
    GFTed_cortical_signal_medium_freq = np.array(GFTed_cortical_signal)[
        :, med_freq_range, :]
    GFTed_cortical_signal_high_freq = np.array(GFTed_cortical_signal)[
        :, high_freq_range, :]

    dic_accumulated = defaultdict(dict)

    for i in range(len(index)): # looping over each trials
        indices_whole = np.hstack([
            np.arange(index[i] * fs - fs, index[i] * fs + 2 * fs)]) # taking data from -1 to +2 seconds, thus 3s in total
        print("the whole-indices length", np.shape(indices_whole)) # the indices for the said 3 seconds
        
        low_freq_whole = slicing_time(
            GFTed_cortical_signal_low_freq, indices_whole) 
        med_freq_whole = slicing_time(
            GFTed_cortical_signal_medium_freq, indices_whole)
        high_freq_whole = slicing_time(
            GFTed_cortical_signal_high_freq, indices_whole)
        
        one_freq_whole = slicing_time(GFT_one_frequency, indices_whole)
        print(np.shape(one_freq_whole))
        assert np.shape(one_freq_whole) == (subjects, 3,  len(indices_whole))
        assert np.shape(low_freq_whole) == (subjects,len(low_freq_range),len(indices_whole)) # check for the integrity of the slicing


        low_freq_f_summed_whole = sum_freqs(low_freq_whole, axis = 1)
        med_freq_f_summed_whole = sum_freqs(med_freq_whole, axis = 1)
        high_freq_f_summed_whole = sum_freqs(high_freq_whole, axis = 1)

        assert np.shape(low_freq_f_summed_whole) == (subjects,len(indices_whole)) # Check for the integrity of summing the frequency

        dic_accumulated[f'{index[i]}'] = accumulate_freqs(
            low_freq_f_summed_whole, med_freq_f_summed_whole, high_freq_f_summed_whole)

        one_freq.append(one_freq_whole)

        print(np.shape(GFTed_cortical_signal_low_freq))


    return dic_accumulated, one_freq

dic, one_freq = master(theta)
band = 'Theta'

"""
The `dic` contains the graph power(or smoothness) for all the trials, and likewise for all the graph frequencies during the stimulus periods of interest.
So the following steps are specific trial-wise, graph-frequency-wise analyses
"""

df_for_plotting_purpose = pd.DataFrame(columns=['gPower', 'gFreqs'])


def ttest(pre_stim, post_stim):
    """Stats

    Args:
        pre_stim (array): ERD for the pre stimulus
        post_stim (array): ERD for the post stimulus

    Returns:
        list/array: t and p values
    """
    
    return scipy.stats.ttest_rel(pre_stim, post_stim)


def averaging_ERD(which_freq, env_band):
    """Averaging ERD for all the trials, creating DF, sometimes plotting

    Args:
        which_freq (array): The graph frequency to average the trials with 
        env_band (string): String of the envelope band name for creating a dataframe

    Returns:
        averaged: the grand-average of ERD
        total: averaged ERP across trials
        dic2: dictionary containing concatenated ERD for pre- and post- stimulus, etc 
    """
    fig = plt.figure(figsize=(45, 25))

    total = (dic['8'][which_freq] + dic['56'][which_freq] + dic['68'][which_freq] + dic['74']
             [which_freq] + dic['86'][which_freq] + dic['132'][which_freq] + dic['162'][which_freq]) / len(index)

    pre_stimulus = np.mean(total[:fs, :], axis = 0) # prestimulus = -100ms to 0s; thus 1s, which is 125samples

    print(np.mean(total[:baseline_duration_of_900ms_in_samples, :], axis = 0)) # sanity check whether the average during the baseline period is 0
    
    post_stimulus = np.mean(total[fs:, :], axis = 0 )# post-stimulus = 0 to 2s
    
    pvalues.append(ttest(pre_stimulus, post_stimulus)[1])
    # a, b, c = 5, 5, 1
    # for i in range(25):
    #     plt.subplot(a,b,c)
    #     plt.plot(total[:,i])
    #     plt.axvline(125)
    #     plt.xticks(np.arange(0,total_duration_in_samples+1,62.5),np.arange(-1000,2500,500))
    #     plt.xlabel('time (ms)')
    #     plt.axvspan(xmin = 0,xmax = 113,color='r',alpha = 0.2)
    #     c+ = 1
    # plt.suptitle('Theta band/Low Frequency/Averaged trials subject-wise')
    # fig.supxlabel("Relative power difference")

    # the following code is for the Violin plot, in a dataframe
    print(np.shape(total))
    averaged = np.mean(total, axis = 1)

    dic2 = defaultdict(dict)
    dic2['gPower'] = np.squeeze(np.concatenate([pre_stimulus, post_stimulus]).T) # vertically appending the pre and post stimulus data for the subjects, so subjects times each
    dic2['stim_group'] = np.squeeze(np.concatenate(
        [['pre-']*subjects, ['post-']*subjects]).T)
    dic2['gFreqs'] = np.squeeze(np.concatenate([[f'{env_band}']*2*subjects])).T # len(pre_stim) + len(post_stim) =  2 * len(subjects)
    return dic2, averaged, stats_SEM(total) 




a = 2
b = 2
c = 1
fig = plt.figure(figsize=(25, 25))
"""
Averaging ERD trials gFreq-wise
"""
for i in range(3): # iterating over graph frequencies; 0 = low, 1 = Med; 2 = High gFreqs
    the_returned, averaged, sem = averaging_ERD(
        which_freq = i, env_band = env_bands[i])

    df_for_plotting_purpose = pd.concat([pd.DataFrame(the_returned), df_for_plotting_purpose], ignore_index = True)
    ax = fig.add_subplot(a, b, i+1)
    ax.plot(averaged, color='r')
    ax.fill_between(range(total_duration_in_samples),
                    averaged-sem, averaged+sem, alpha = 0.2)
    ax.axvline(125)
    ax.set_xticks(np.arange(0, total_duration_in_samples+1, 62.5))
    ax.set_xticklabels(np.arange(-1000, 2500, 500))
    ax.set_xlabel('time (ms)')
    ax.axvspan(xmin = 0, xmax = baseline_duration_of_900ms_in_samples,
               color='r', alpha = 0.2)
    ax.set_title(f'g{env_bands[i]} freq')
    if i == 0:
            dic_to_write = defaultdict(dict)
            dic_to_write['gPower']['average'] = averaged
            dic_to_write['gPower']['sem'] = sem

            np.savez_compressed(f'/users2/local/Venkatesh/Generated_Data/25_subjects_copy_FOR_TESTING/power_smoothness_4_in_one_plot/gPower', **dic_to_write )


width = 0.35
ort = "v"
pal = "blue_red_r"
ax = fig.add_subplot(a, b, 4)

pt.RainCloud(hue="stim_group", y="gPower", x="gFreqs", palette=['C0', 'C1'], data = df_for_plotting_purpose, width_viol=.7,
             ax = ax, orient = ort, alpha=.45, dodge = True)
add_stat_annotation(ax, data = df_for_plotting_purpose, y="gPower", x="gFreqs", hue="stim_group",
                    box_pairs=[(("Low", "pre-"), ("Low", "pre-")),
                               (("Med", "pre-"), ("Med", "pre-")),
                               (("High", "pre-"), ("High", "pre-"))
                               ],
                    perform_stat_test = False, pvalues = pvalues, text_format='star', loc='outside', verbose = 2)
fig.suptitle(
    f'ERD of graph power for the averaged trials across subjects -- {band}')
fig.supylabel('The relative power difference')
# fig.savefig(f'/homes/v20subra/S4B2/Graph-related_analysis/ERD/{band}')

index_in_str = [str(i) for i in index]#  converting the indices into string for parsing from the dictionary
dic_parse_freq_wise = [dic.get(key)[0] for key in index_in_str] # 0 = Low gfreq; 1 and 2 = Med and High gFreq respectively



a, b, c = 4, 2, 1
"""
Iterating over trials, for plotting 
"""
fig = plt.figure(figsize=(25, 25))
for i in range(7):
    plt.subplot(a, b, c)
    mean, sigma = np.mean(np.array(dic_parse_freq_wise)[i, :, :], axis = 1), np.std(
        np.array(dic_parse_freq_wise)[i, :, :], axis = 1)
    print(np.shape(np.array(dic_parse_freq_wise)))
    conf_int_a = scipy.stats.norm.interval(0.95, loc = mean, scale = sigma)

    plt.plot(np.array(dic_parse_freq_wise)[i, :, :])
    plt.plot(mean, color='r', linewidth = 5, label='Mean (subjects)')
    plt.fill_between(range(total_duration_in_samples), np.array(
        conf_int_a).T[:, 0], np.array(conf_int_a).T[:, 1], alpha = 0.2, label='95% CI')
    plt.xticks(np.arange(0, total_duration_in_samples+1, 62.5),
               np.arange(-1000, 2500, 500))
    plt.axvline(125, c='g', linewidth = 3)
    plt.axvspan(xmin = 0, xmax = baseline_duration_of_900ms_in_samples,
                color='r', alpha = 0.2)
    c += 1
    plt.legend()

fig.supylabel('Relative power difference')
fig.supxlabel('time (ms)')
fig.suptitle(f'ERD trial-wise -- {band}')
# fig.savefig(
#     f'/homes/v20subra/S4B2/Graph-related_analysis/Functional_graph_setup/Results_ERD_trial_wise/{band}')



# %%
fig = plt.figure(figsize=(15,10))
plt.style.use('fivethirtyeight') # For better style

a,b,c = 3, 1, 1
for i in range(3):
    plt.subplot(a,b,c)
    trial_avg = np.mean(np.abs(one_freq), axis=0)
    mean = np.mean(trial_avg[:,i,:],axis=0)
    plt.plot(mean)
    sem = stats_SEM(trial_avg[:,i,:].T)
    plt.fill_between(range(total_duration_in_samples), mean - sem, mean + sem, alpha = 0.2, label = 'SEM of subjects')
    c+=1
    plt.title(f'gFrequency {i*50}') 
    plt.xticks(np.arange(0, total_duration_in_samples+1, 62.5),
               np.arange(-1000, 2500, 500))
    plt.axvline(125, c='g', linewidth = 3)
    plt.axvspan(xmin = 0, xmax = baseline_duration_of_900ms_in_samples,
                color='r', alpha = 0.2)
    plt.legend()
fig.supxlabel('time (ms)')
fig.supylabel('graph power')
fig.suptitle('Envelope Theta band')
plt.tight_layout()

fig.savefig(f'/homes/v20subra/S4B2/Graph-related_analysis/{band}')
# %%
np.shape(one_freq)
# %%

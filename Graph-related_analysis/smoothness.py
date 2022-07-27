# %%




from turtle import color, shape
from matplotlib import colorbar
import scipy.stats as st
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from pygsp import graphs, filters
from pygsp import plotting as gsp_plt

from scipy import io as sio
from pygsp import graphs

import scipy
import torch
import pickle
import seaborn as sns
import pandas as pd
from statannot import add_stat_annotation
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


trials = [8, 56, 68, 74, 86, 132, 162]
fs = 125
subjects = 25# baseline = -1000ms to -100ms, so 900ms; since fs = 125, 900 ms = 113 samples
baseline_duration_of_900ms_in_samples = 113
total_duration_in_samples = 375
low_freq_range = np.arange(1, 51)
med_freq_range = np.arange(51, 200)
high_freq_range = np.arange(200, 360)
pvalues = list()
gFreqs = ['Low', 'Med', 'High']

envelope_signal_bandpassed = np.load(
    '/users2/local/Venkatesh/Generated_Data/25_subjects_copy_FOR_TESTING/envelope_signal_bandpassed_with_beta_dichotomy.npz', mmap_mode='r')

eloreta_signal = np.load('/users2/local/Venkatesh/Generated_Data/25_subjects/video_watching_bundle_STC_parcellated.npz')['video_watching_bundle_STC_parcellated']

alpha = envelope_signal_bandpassed['alpha']
low_beta = envelope_signal_bandpassed['lower_beta']
high_beta = envelope_signal_bandpassed['higher_beta']
theta = envelope_signal_bandpassed['theta']


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


def slicing_time(coefficients, indices):
    """A precursor function to do temporal slicing. 
    Dimension inflow = subject x entire video duration

    Args:
        coefficients (array): The smoothness-estimated array
        indices(array of range):  Indices for the time period containing baseline, stimulus data

    Returns:
        items_whole: A sliced array for the pre-strong period. 
        Dim outflow = subject x `total_duration_in_samples` 

    """
    items_whole = np.squeeze(slicing(coefficients, indices, axis = 2))
    return items_whole


def baseline(whole):
    """ ERD baseline setup. Dim = sub x sliced_time

        Args:
            whole (array): Pre- and Post-stimulus/strong; duration = -1 to 2, so 3 seconds

        Returns:
            array : Subject-wise ERD setup. Dim = sub x sliced_time
    """
    pre = whole[:, :baseline_duration_of_900ms_in_samples]
    return np.array((whole.T - np.mean(pre, axis = 1))/np.mean(pre, axis = 1))


def stats_SEM(freqs):
    """SEM estimation -- Standard error of the Mean

    Args:
        freqs (dict): The grand-averaged graph smoothness to apply the SEM on

    Returns:
        array: SEMed graph smoothness 
    """
    return scipy.stats.sem(freqs, axis = 1)


def smoothness_baseline_setup(smoothness_input_for_baseline_setup):
    """Baseline setup the pre and post

    Args:
        smoothness_input_for_baseline_setup (array): smoothness-estimated and time-sliced array


    Returns:
        dict: baseline setup, subject-wise; dim =  sliced_time x subjects
    """
    smoothness_after_baseline = defaultdict(dict)
    smoothness_after_baseline = baseline(smoothness_input_for_baseline_setup)

    return smoothness_after_baseline


def master(band):
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
    print(np.shape(signal_stage2))

#   ( entire_video_duration x subjects x 1 x ROIs) x (entire_video_duration x subjects x ROIs x 1) 

#######################################
###TESTING############################
# a = np.random.random((21250,25,1,360))
# b = np.random.random((21250,25,360,1))
#######################################


    smoothness_roughness_time_series = np.squeeze( np.matmul(stage1,signal_stage2) ) # dim = entire_video_duration x subjects
    print("np.shape(smoothness_roughness_time_series):",np.shape(smoothness_roughness_time_series))

    dic_accumulated = defaultdict(dict)

    for i in range(len(trials)): # looping over each trials
        indices = np.hstack([
            np.arange(trials[i] * fs - fs, trials[i] * fs + 2 * fs)]) # taking data from -1 to +2 seconds, thus 3s in total
        print("the whole-indices length", np.shape(indices)) # the indices for the said 3 seconds
        smoothness_sliced = slicing_time( smoothness_roughness_time_series.T, indices= indices)
 
        assert np.shape(smoothness_sliced) == (subjects,len(indices)) # check for the integrity of the slicing
        
        dic_accumulated[f'{trials[i]}'] = smoothness_baseline_setup(smoothness_sliced)

    return dic_accumulated




def ttest(pre_stim, post_stim):
    """Stats

    Args:
        pre_stim (array): ERD for the pre stimulus
        post_stim (array): ERD for the post stimulus

    Returns:
        list/array: t and p values
    """
    
    return scipy.stats.ttest_rel(pre_stim, post_stim)


def averaging_ERD():
    """Averaging ERD for all the trials, creating DF, sometimes plotting

    Returns:
        averaged: the grand-average of ERD
        total: averaged ERP across trials
        dic2: dictionary containing concatenated ERD for pre- and post- stimulus, etc 
    """

    fig = plt.figure(figsize=(45, 25))
    total = (dic['8'] + dic['56'] + dic['68'] + dic['74']
              + dic['86'] + dic['132'] + dic['162']) / len(trials)
    pre_stimulus = np.mean(total[:fs, :], axis = 0) # prestimulus = -100ms to 0s; thus 1s, which is 125samples

    print(np.mean(total[:baseline_duration_of_900ms_in_samples, :], axis = 0)) # sanity check whether the average during the baseline period is 0
    
    post_stimulus = np.mean(total[fs:, :], axis = 0 )# post-stimulus = 0 to 2s
    
    pvalues.append(ttest(pre_stimulus, post_stimulus)[1])

    # a, b, c = 5, 5, 1
    # fig = plt.figure(figsize=(25, 25))
    # for i in range(subjects):# iterating over subjects
    #     mean, sigma = np.mean(np.array(list(dic.values()))[:, :, i], axis = 0), np.std(
    #         np.array(list(dic.values()))[:, :, i], axis = 0)
    #     assert np.shape(mean) == (375,)
    #     conf_int_a = scipy.stats.norm.interval(0.95, loc = mean, scale = sigma)

    #     plt.subplot(a,b,c)
    #     plt.plot(mean,  color='r', linewidth = 5, label='Mean')
    #     plt.plot((np.array(list(dic.values()))[:,:,i]).T)
    #     plt.fill_between(range(total_duration_in_samples), np.array(
    #         conf_int_a).T[:, 0], np.array(conf_int_a).T[:, 1], alpha = 0.2, label='95% CI')
    #     plt.axvline(125)
    #     plt.xticks(np.arange(0, total_duration_in_samples+1, 62.5), np.arange(-1000, 2500, 500))
    #     plt.axvspan(xmin = 0, xmax = 113, color='r', alpha = 0.2)
    #     c+= 1
    #     plt.legend()
    
    
    # fig.suptitle(f'ERD across trials subject-wise--{band}')
    # fig.supylabel("Relative smoothness")
    # fig.supxlabel('time (ms)')
    # fig.savefig(f'/homes/v20subra/S4B2/Graph-related_analysis/Functional_graph_setup/Results_ERD_across_trial/{band}_smoothness')

    # the following code is for the Raincloud plot, in a dataframe
    averaged_subject = np.mean(total, axis = 1)

    dic2 = defaultdict(dict)
    dic2['gSmoothness'] = np.squeeze(np.concatenate([pre_stimulus, post_stimulus]).T) # vertically appending the pre and post stimulus data for the subjects, so subjects times each
    dic2['stim_group'] = np.squeeze(np.concatenate(
        [['pre-'] * subjects, ['post-'] * subjects]).T)
    return dic2, averaged_subject, stats_SEM(total), total




dic_of_env_bands = { 'smoothness_theta': theta, 'smoothness_wideband': eloreta_signal}#, 'alpha':alpha, 'lower_beta' : low_beta, 'upper_beta' : high_beta} # ALL the hilbert-transformed envelopes


"""
The following block is where all the code gets called;

"""


for i in range( len (dic_of_env_bands.keys() ) ): # looping over all the envelope bands
    band = list(dic_of_env_bands.keys())[i]
    dic = master( dic_of_env_bands[  band  ]  ) # the main... master function is called given a single envelope
    
    assert np.shape(dic['8']) == (total_duration_in_samples, subjects)  # sanity check, done for one trial.. will be the same for the remaining
    
    """
    The `dic` contains the smoothness for all the trials during the stimulus periods of interest. The following steps are specific trial-wise analyses
    """
    pvalues = list() # appending the pvalues from the stat tests

    a = 2
    b = 2
    c = 1
    """
    Averaging ERD trials
    """
    dic_for_plotting_purpose, grand_average, grand_sem, total = averaging_ERD() # Averaging the ERD trials

    assert np.shape(grand_average) == (total_duration_in_samples,)
    assert np.shape(grand_sem) == (total_duration_in_samples,)

    dic_to_write = defaultdict(dict)
    dic_to_write[f'{list(dic_of_env_bands.keys())[i]}']['average'] = grand_average
    dic_to_write[f'{list(dic_of_env_bands.keys())[i]}']['sem'] = grand_sem

    np.savez_compressed(f'/users2/local/Venkatesh/Generated_Data/25_subjects_copy_FOR_TESTING/power_smoothness_4_in_one_plot/{list(dic_of_env_bands.keys())[i]}', **dic_to_write )

    fig = plt.figure(figsize=(25, 15))
    ax = fig.add_subplot(a, b, 1)

    ax.plot(grand_average, color='r')
    ax.fill_between(range(total_duration_in_samples),
                    grand_average-grand_sem, grand_average+grand_sem, alpha = 0.2)
    ax.axvline(125) # The stimulus onset, in samples
    ax.set_xticks(np.arange(0, total_duration_in_samples+1, 62.5))
    ax.set_xticklabels(np.arange(-1000, 2500, 500))
    ax.set_xlabel('time (ms)')
    ax.axvspan(xmin = 0, xmax = baseline_duration_of_900ms_in_samples,
                color='r', alpha = 0.2)
    
    # width = 0.35
    # ort = "v" # orientation of the raincloud plot
    # pal = "blue_red_r"
    # ax = fig.add_subplot(a, b, 2)
    # pt.RainCloud(x="stim_group", y="gSmoothness", palette=['C0', 'C1'], data = dic_for_plotting_purpose, ax = ax,
    #             orient = ort, alpha=.45, dodge = True)
    # add_stat_annotation(ax, data = dic_for_plotting_purpose, y="gSmoothness", x="stim_group",
    #                     box_pairs=[(( "pre-"), ("post-"))],
    #                     perform_stat_test = False, pvalues = pvalues, text_format='star', loc='outside', verbose = 2)
    # fig.suptitle(
    #     f'ERD of graph smoothness for the averaged trials across subjects -- {band}')
    # fig.supylabel('The relative Smoothness difference')
    # # fig.savefig(f'/homes/v20subra/S4B2/Graph-related_analysis/ERD/{band}_smoothness')

    # index_in_str = [str(i) for i in trials]#  converting the indices into string for parsing from the dictionary
    # dic_parse_all_trials = [dic.get(key) for key in index_in_str] # parsing and loading all the trials into this dict
    # ax = fig.add_subplot(a, b, 3)

    # sns.heatmap(total.T*100,  cmap='seismic',vmin=-600, vmax = 600)
    # ax.axvline(125) # The stimulus onset, in samples
    # ax.set_xticks(np.arange(0, total_duration_in_samples+1, 62.5))
    # ax.set_xticklabels(np.arange(-1000, 2500, 500))



    # a, b, c = 4, 2, 1
    # """
    # Iterating over trials, for plotting 
    # """
    # fig = plt.figure(figsize=(25, 25))
    # for i in range(7):
    #     plt.subplot(a, b, c)
    #     mean, sigma = np.mean(np.array(dic_parse_all_trials)[i, :, :], axis = 1), np.std(
    #         np.array(dic_parse_all_trials)[i, :, :], axis = 1)
    #     print(np.shape(np.array(dic_parse_all_trials)))
    #     conf_int_a = scipy.stats.norm.interval(0.95, loc = mean, scale = sigma) # provides CI for upper and lower bound

    #     plt.plot(np.array(dic_parse_all_trials)[i, :, :])
    #     plt.plot(mean, color='r', linewidth = 5, label='Mean (subjects)')
    #     plt.fill_between(range(total_duration_in_samples), np.array(
    #         conf_int_a).T[:, 0], np.array(conf_int_a).T[:, 1], alpha = 0.2, label='95% CI') # CI [0] and CI[1] = upper and lower bound for CI
    #     plt.xticks(np.arange(0, total_duration_in_samples+1, 62.5),
    #             np.arange(-1000, 2500, 500))
    #     plt.axvline(125, c='g', linewidth = 3)
    #     plt.axvspan(xmin = 0, xmax = baseline_duration_of_900ms_in_samples,
    #                 color='r', alpha = 0.2)
    #     c += 1
    #     plt.legend()
    # fig.suptitle(f'ERD of smoothness trial-wise -- {band}')
    # fig.supylabel('Smoothness')
    # fig.supxlabel('time (ms)')
    # # fig.savefig(f'/homes/v20subra/S4B2/Graph-related_analysis/Functional_graph_setup/Results_ERD_trial_wise/{band}_smoothness')


#%%


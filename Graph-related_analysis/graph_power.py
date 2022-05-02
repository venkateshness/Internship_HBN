#%%
from cProfile import label
from collections import defaultdict
from operator import index
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
def graph_setup(unthresholding,weights):
    """Function to finalize the graph setup -- with options to threshold the graph by overriding the graph weights

    Args:
        unthresholding (bool): do you want a graph unthresholded ?
        weights (Float): Weight matrix

    Returns:
        Graph: Returns the Graph, be it un- or thresholded, which the latter is done using the 8Nearest-Neighbour
    """
    coordinates = sio.loadmat('/homes/v20subra/S4B2/GSP/Glasser360_2mm_codebook.mat')['codeBook'] 

    G=graphs.Graph(weights,gtype='HCP subject',lap_type='combinatorial',coords=coordinates) 
    G.set_coordinates('spring')
    print('{} nodes, {} edges'.format(G.N, G.Ne))

    if unthresholding:
        pickle_file = '/homes/v20subra/S4B2/GSP/MMP_RSFC_brain_graph_fullgraph.pkl'

        with open(pickle_file, 'rb') as f:
                    [connectivity]= pickle.load(f)
        np.fill_diagonal(connectivity,0)

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
                [connectivity]= pickle.load(f)
    np.fill_diagonal(connectivity,0)
    
    graph = torch.from_numpy(connectivity)
    knn_graph = torch.zeros(graph.shape)
    for i in range(knn_graph.shape[0]):
        graph[i,i] = 0
        best_k = torch.sort(graph[i,:])[1][-8:]
        knn_graph[i, best_k] = 1
        knn_graph[best_k, i] = 1
        
    degree = torch.diag(torch.pow(knn_graph.sum(dim = 0), -0.5))

    weight_matrix_after_NN = torch.matmul(degree, torch.matmul(knn_graph, degree))
    return weight_matrix_after_NN


G = graph_setup(False,NNgraph())
G.compute_fourier_basis()
#%%
envelope_signal_bandpassed = np.load('/users2/local/Venkatesh/Generated_Data/25_subjects_copy_FOR_TESTING/envelope_signal_bandpassed_low_high_beta.npz', mmap_mode='r')

alpha = envelope_signal_bandpassed['alpha']
low_beta = envelope_signal_bandpassed['lower_beta']
high_beta = envelope_signal_bandpassed['higher_beta']
theta = envelope_signal_bandpassed['theta']


def slicing(what_to_slice,where_to_slice,axis):
    """Temporal Slicing. Function to slice at the temporal axis.

    Args:
        what_to_slice (array): The array to do the temporal slicing for
        where_to_slice (array of "range"): The period(s) when to do the temporal slicing in
        axis (int): The axis of the array to do the temporal slicing on 

    Returns:
        array: An array temporally sliced
    """ 
    array_to_append = list()
    if axis >2:
        array_to_append.append ( what_to_slice[:,:,where_to_slice] )
    else:
        array_to_append.append ( what_to_slice[:,where_to_slice] )
    return array_to_append


def slicing_time(freqs,indices_pre_strong,indices_post_strong):
    """A precursor function to do temporal slicing. Dim = subj x frequency x whole_time_duration

    Args:
        freqs (array): The GFT-ed frequency to slice
        indices_pre_strong (array of range):  Indices for the pre-strong time period
        indices_post_strong (array of range): Indices for the post-strong time period

    Returns:
        items_pre_strong: A sliced array for the pre-strong period. Dim = subj x frequency x len(item_pre_strong)
        items_post_strong: A sliced array for the post-strong period. Dim = subj x frequency x len(item_post_strong)

    """
    items_pre_strong = np.squeeze(slicing(freqs,indices_pre_strong,axis=3))
    items_post_strong = np.squeeze(slicing(freqs,indices_post_strong,axis=3))
    return items_pre_strong,items_post_strong


def sum_freqs(freqs,axis):
    """Integrating frequency. Applying the L2 norm and summing the frequency power. Dim = subj x frequency x sliced_time

    Args:
        freqs (array): Frequency to sum
        axis (int): Axis of the Frequency

    Returns:
        array: Summed frequency power array (dim = subjects x 1 x sliced_time)
    """
    return np.sum(np.sqrt(np.power(freqs,2)),axis=axis)#L2 norm



def baseline(pre,post):
    """ERD baseline setup. Dim = sub x sliced_time

    Args:
        pre (array): Pre-stimulus/strong
        post (array): Post-stimulus/strong

    Returns:
        array : Subject-wise ERD setup. Dim = sub x sliced_time
    """
    return np.array((post.T - np.mean(pre.T))/np.mean(pre.T))


def averaging_time(freqs,axis=-1):
    """Temporal Average. Dim = sub x sliced_time

    Args:
        freqs (array): Array of which graph frequency to average temporally
        axis (int, optional): Temporal axis. Defaults to -1.

    Returns:
        array: Temporally averaged (dim =  sub x 1)
    """
    return np.average(freqs.T,axis=axis)


def stats_SEM(freqs):
    """SEM estimation -- Standard error of the Mean

    Args:
        freqs (dict): The grand-averaged graph power to apply the SEM on

    Returns:
        array: SEMed graph power 
    """
    return scipy.stats.sem(freqs,axis=1)#/np.sqrt(25)

def accumulate_freqs(freq_pre_low,freq_post_low,freq_pre_med,freq_post_med,freq_pre_high,freq_post_high):
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
        if i ==0:
            freq = np.concatenate([baseline(freq_pre_low,freq_pre_low),baseline(freq_pre_low,freq_post_low)])
        elif i ==1:
            freq = np.concatenate([baseline(freq_pre_med,freq_pre_med),baseline(freq_pre_med,freq_post_med)])
        else:
            freq = np.concatenate([baseline(freq_pre_high,freq_pre_high),baseline(freq_pre_high,freq_post_high)])
        dic_append_everything[i]= freq
        
    return dic_append_everything

index = [8,56,68,74,86,132,162]

def master(band):
    """The main function that does GFT, function-calls the temporal slicing, frequency summing, pre- post- graph-power accumulating 

    Args:
        band (array): Envelope band to use

    Returns:
        dict: Baseline-corrected ERD for all trials 
    """
    GFTed_cortical_signal = [G.gft(np.array(band[i])) for i in range(25)]

    GFTed_cortical_signal_low_freq = np.array(GFTed_cortical_signal)[:,1:51,:]
    GFTed_cortical_signal_medium_freq = np.array(GFTed_cortical_signal)[:,51:200,:]
    GFTed_cortical_signal_high_freq = np.array(GFTed_cortical_signal)[:,200:,:]

    
    dic_accumulated = defaultdict(dict)


    for i in range(len(index)):
        indices_pre_strong = np.hstack([
        np.arange(index[i]-1*125,index[i]-1*125+113)])
        print("the pre-indices length",np.shape(indices_pre_strong))
        indices_post_strong =  np.hstack([
        np.arange(index[i]*125-13,(index[i]+2)*125)])
        print("the post-indices length",np.shape(indices_post_strong))

        low_freq_pre, low_freq_post = slicing_time(GFTed_cortical_signal_low_freq,indices_pre_strong,indices_post_strong)
        med_freq_pre, med_freq_post = slicing_time(GFTed_cortical_signal_medium_freq,indices_pre_strong,indices_post_strong)
        high_freq_pre, high_freq_post = slicing_time(GFTed_cortical_signal_high_freq,indices_pre_strong,indices_post_strong)


        low_freq_pre_f_summed, low_freq_post_f_summed = sum_freqs(low_freq_pre,axis=1),sum_freqs(low_freq_post,axis=1)
        med_freq_pre_f_summed, med_freq_post_f_summed = sum_freqs(med_freq_pre,axis=1),sum_freqs(med_freq_post,axis=1)
        high_freq_pre_f_summed, high_freq_post_f_summed = sum_freqs(high_freq_pre,axis=1),sum_freqs(high_freq_post,axis=1)

        dic_accumulated[f'{index[i]}'] =accumulate_freqs(low_freq_pre_f_summed,low_freq_post_f_summed,med_freq_pre_f_summed,med_freq_post_f_summed,high_freq_pre_f_summed,high_freq_post_f_summed)

    return dic_accumulated


dic = master(theta,'Theta')

df = pd.DataFrame(columns=['gPower','gFreqs'])
to_df = defaultdict(dict)
def ttest(pre_stim, post_stim):
    """Stats

    Args:
        pre_stim (array): ERD for the pre stimulus
        post_stim (array): ERD for the post stimulus

    Returns:
        list/array: t and p values
    """
    return scipy.stats.ttest_rel(pre_stim,post_stim)

pvalues_slicing = list()
def freq_plot(which_freq,env_band):
    """Averaging ERD for all the trials, creating DF, sometimes plotting

    Args:
        which_freq (array): The graph frequency to average the trials with 
        env_band (string): String of the envelope band name for creating a dataframe

    Returns:
        averaged: the grand-average of ERD
        total: averaged ERP across trials
        dic2: dictionary containing concatenated ERD for pre- and post- stimulus, etc 
    """
    fig =plt.figure(figsize=(45,25))

    total = (dic['8'][which_freq] +dic['56'][which_freq] +dic['68'][which_freq] + dic['74'][which_freq] + dic['86'][which_freq] + dic['132'][which_freq] + dic['162'][which_freq])/ len(index)
    pre_total = np.mean(total[:125,:],axis=0)
    post_total = np.mean(total[125:,:],axis=0)
    pvalues_slicing.append(ttest(pre_total,post_total)[1])
    # a,b,c = 5,5,1
    # for i in range(25):
    #     plt.subplot(a,b,c)
    #     plt.plot(total[:,i])
    #     plt.axvline(125)
    #     plt.xticks(np.arange(0,376,62.5),np.arange(-1000,2500,500))
    #     plt.xlabel('time (ms)')
    #     plt.axvspan(xmin=0,xmax=113,color='r',alpha=0.2)
    #     c+=1
    # plt.suptitle('Theta band/Low Frequency/Averaged trials subject-wise')
    # fig.supxlabel("Relative power difference")
    # if env_band=="Low":
    #     fig.savefig('/homes/v20subra/S4B2/Graph-related_analysis/ERD/Theta_low_subjwise')
    averaged = np.mean(total,axis=1)

    dic2 = defaultdict(dict)
    dic2['gPower'] = np.squeeze(np.concatenate([pre_total,post_total]).T)
    dic2['stim_group'] =np.squeeze(np.concatenate([['pre-']*25,['post-']*25]).T)
    dic2['gFreqs'] = np.squeeze(np.concatenate([[f'{env_band}']*50])).T
    return dic2, averaged,stats_SEM(total)

sns.set_theme()

env_bands = ['Low','Med','High']

a = 2
b = 2
c = 1
fig = plt.figure(figsize=(25,25))

for i in range(3):
        the_returned, averaged, sem =freq_plot(which_freq=i,env_band=env_bands[i])
        
        df = pd.concat([pd.DataFrame(the_returned),df],ignore_index=True)
        ax = fig.add_subplot(a,b,i+1)
        ax.plot(averaged,color='r')
        ax.fill_between(range(251+125),averaged-sem,averaged+sem,alpha=0.2)
        ax.axvline(125)
        ax.set_xticks(np.arange(0,376,62.5))
        ax.set_xticklabels(np.arange(-1000,2500,500))
        ax.set_xlabel('time (ms)')
        ax.axvspan(xmin=0,xmax=113,color='r',alpha=0.2)
        ax.set_title(f'g{env_bands[i]} freq')

width = 0.35
ort = "v"; pal = "blue_red_r"
ax = fig.add_subplot(a,b,4)

pt.RainCloud(hue="stim_group",y="gPower",x="gFreqs",palette = ['C0','C1'],data=df, width_viol = .7,
          ax=ax,orient = ort , alpha = .45, dodge = True)
add_stat_annotation(ax, data=df, y="gPower",x="gFreqs", hue="stim_group",
                    box_pairs=[(("Low", "pre-"), ("Low", "pre-")),
                                 (("Med", "pre-"), ("Med", "pre-")),
                                 (("High", "pre-"), ("High", "pre-"))
                                ],
                    perform_stat_test=False, pvalues=pvalues_slicing, text_format='star', loc='outside', verbose=2)
fig.suptitle('ERD of graph power for the averaged trials across subjects -- Theta')
fig.supylabel('The relative power difference')
fig.savefig('/homes/v20subra/S4B2/Graph-related_analysis/ERD/Theta')

# %%

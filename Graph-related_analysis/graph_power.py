#%%
from turtle import pos
from cv2 import norm
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

sns.set_theme()
############################################################
##########Getting the Graph ready###########################
############################################################ 
def graph_setup(unthresholding, percentage,weights):
    path_Glasser='/homes/v20subra/S4B2/GSP/Glasser_masker.nii.gz'
    res_path=''


    # Load structural connectivity matrix
    # connectivity = sio.loadmat('/homes/v20subra/S4B2/GSP/SC_avg56.mat')['SC_avg56']
 
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
    # laplacian = torch.eye(graph.shape[0]) - torch.matmul(degree, torch.matmul(knn_graph, degree))
    # values, eigs = torch.linalg.eigh(laplacian)

    weight_matrix_after_NN = torch.matmul(degree, torch.matmul(knn_graph, degree))
    return weight_matrix_after_NN


G = graph_setup(False,66,NNgraph())
G.compute_fourier_basis()

#%%
envelope_signal_bandpassed = np.load('/users2/local/Venkatesh/Generated_Data/25_subjects_copy_FOR_TESTING/envelope_signal_bandpassed_low_high_beta.npz', mmap_mode='r')

alpha = envelope_signal_bandpassed['alpha']
low_beta = envelope_signal_bandpassed['lower_beta']
high_beta = envelope_signal_bandpassed['higher_beta']
theta = envelope_signal_bandpassed['theta']
# %%
GFTed_cortical_signal = [G.gft(np.array(high_beta[i])) for i in range(25)]
# %%
GFTed_cortical_signal_low_freq = np.array(GFTed_cortical_signal)[:,1:51,:]
GFTed_cortical_signal_medium_freq = np.array(GFTed_cortical_signal)[:,51:200,:]
GFTed_cortical_signal_high_freq = np.array(GFTed_cortical_signal)[:,200:,:]

# %%
indices_pre_strong = np.hstack([np.arange(48*125,55*125),np.arange(76*125,78*125),np.arange(97*125,103*125)])
indices_post_strong =  np.hstack([np.arange(55*125,62*125),np.arange(78*125,80*125),np.arange(103*125,109*125)])
#%%
def slicing(what_to_slice,where_to_slice,axis):
    array_to_append = list()
    if axis >2:
        array_to_append.append ( what_to_slice[:,:,where_to_slice] )
    else:
        array_to_append.append ( what_to_slice[:,where_to_slice] )
    return array_to_append


def slicing_freqs(freqs):

    print("indices length for Pre-Strong ISC is: ", len(indices_pre_strong)/125)

    items_pre_strong = np.squeeze(slicing(freqs,indices_pre_strong,axis=3))

    print("indices length for Post-Strong ISC is: ", len(indices_post_strong)/125)

    items_post_strong = np.squeeze(slicing(freqs,indices_post_strong,axis=3))

    return items_pre_strong,items_post_strong

low_freq_pre, low_freq_post = slicing_freqs(GFTed_cortical_signal_low_freq)
med_freq_pre, med_freq_post = slicing_freqs(GFTed_cortical_signal_medium_freq)
high_freq_pre, high_freq_post = slicing_freqs(GFTed_cortical_signal_high_freq)

# %%

def averaging_time(freqs):
    return np.average(freqs,axis=-1)

low_freq_pre_t_averaged, low_freq_post_t_averaged = averaging_time(low_freq_pre),averaging_time(low_freq_post)
med_freq_pre_t_averaged, med_freq_post_t_averaged = averaging_time(med_freq_pre),averaging_time(med_freq_post)
high_freq_pre_t_averaged, high_freq_post_t_averaged = averaging_time(high_freq_pre),averaging_time(high_freq_post)

# %%

def sum_freqs(freqs,axis):
    return np.sum(np.sqrt(np.power(freqs,2)),axis=axis)#L2 norm

low_freq_pre_t_averaged_summed, low_freq_post_t_averaged_summed = sum_freqs(low_freq_pre_t_averaged,axis=-1),sum_freqs(low_freq_post_t_averaged,axis=-1)
med_freq_pre_t_averaged_summed, med_freq_post_t_averaged_summed = sum_freqs(med_freq_pre_t_averaged,axis=-1),sum_freqs(med_freq_post_t_averaged,axis=-1)
high_freq_pre_t_averaged_summed, high_freq_post_t_averaged_summed = sum_freqs(high_freq_pre_t_averaged,axis=-1),sum_freqs(high_freq_post_t_averaged,axis=-1)

# %%
def ttest(band1, band2):
    return scipy.stats.ttest_rel(band1,band2)
print("high" ,ttest(high_freq_pre_t_averaged_summed, high_freq_post_t_averaged_summed) )
print ("medium", ttest(med_freq_pre_t_averaged_summed, med_freq_post_t_averaged_summed) ) 
print ("low:",ttest(low_freq_pre_t_averaged_summed, low_freq_post_t_averaged_summed))

pvalues = [ttest(low_freq_pre_t_averaged_summed, low_freq_post_t_averaged_summed),
ttest(med_freq_pre_t_averaged_summed, med_freq_post_t_averaged_summed),
ttest(high_freq_pre_t_averaged_summed, high_freq_post_t_averaged_summed) ]

# %%

def stats_SEM(freqs):
        print(np.shape(freqs))
        return scipy.stats.sem(freqs,axis=1)#/np.sqrt(25)


mpl.rcParams['font.family'] = 'Arial'

plt.rc('font', family='serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')

fig,ax = plt.subplots(figsize=(10,10))
labels = ['Low', 'Med', 'High']

x = np.arange(len(labels))  # the label locations
width=0.35

# low = stats_SEM(np.squeeze(smoothness_roughness_time_series_dict['theta']))
# med = stats_SEM(np.squeeze(smoothness_roughness_time_series_dict['low_beta']))
# hig = stats_SEM(np.squeeze(smoothness_roughness_time_series_dict['high_beta']))

# error_alpha = stats_SEM(np.squeeze(smoothness_roughness_time_series_dict['alpha']))

pre_strong = [low_freq_pre_t_averaged_summed, med_freq_pre_t_averaged_summed, high_freq_pre_t_averaged_summed]
post_strong = [low_freq_post_t_averaged_summed, med_freq_post_t_averaged_summed, high_freq_post_t_averaged_summed]

# post_strong = [np.average(np.squeeze(smoothness_roughness_time_series_dict['theta']),axis=1)[1],
#             np.average(np.squeeze(smoothness_roughness_time_series_dict['alpha']),axis=1)[1],
#             np.average(np.squeeze(smoothness_roughness_time_series_dict['low_beta']),axis=1)[1],
#             np.average(np.squeeze(smoothness_roughness_time_series_dict['high_beta']),axis=1)[1]]


ax.violinplot(positions= x - width/2, dataset = pre_strong,widths=width,showextrema=True,showmeans=True )
ax.violinplot(positions=x + width/2, dataset =post_strong,widths=width,showextrema=True,showmeans=True)
ax.legend(['Pre','post'])
data = pd.DataFrame({'labels':labels,'graphPower':np.sum(pre_strong,axis=1)})

data2 = pd.DataFrame({'labels':labels,'graphPower':np.sum(post_strong,axis=1)})
data_fin = data.append(data2,ignore_index=True)
data_fin['cond'] = ['Pre','Pre','Pre','Post','Post','Post']
pvalues_slicing =[pvalues[i][1] for i in range(3)]
add_stat_annotation(ax,data=data_fin, y='graphPower', x ='labels', hue='cond',
                    box_pairs=[(("Low", "Post"), ("Low", "Pre")),
                    (("Med", "Post"), ("Med", "Pre")),
                    (("High", "Post"), ("High", "Pre"))],
                                 perform_stat_test=False, pvalues=pvalues_slicing,
line_offset_to_box=0.25, line_offset=0.1, line_height=0.05, text_format='star', loc='outside', verbose=2)

plt.tight_layout()
plt.legend()
plt.xticks(x,labels)
plt.ylabel('gPower')
plt.xlabel('gFrequency bands')
plt.title('Upper Beta')
plt.savefig('/homes/v20subra/S4B2/Graph-related_analysis/Functional_graph_setup/Results_graph_power/upper')
# %%
# sns.heatmap( np.average(np.concatenate([pre_strong_cortical,post_strong_cortical],axis=2),axis=0)[1:100])
# plt.axvline(x=900,linestyle='--')
# %%

low_freq_pre_summed, low_freq_post_summed = sum_freqs(low_freq_pre,axis=1),sum_freqs(low_freq_pre,axis=1)

plt.plot(np.average(np.concatenate([low_freq_pre_summed,low_freq_post_summed],axis=-1).T,axis=1)[:])
# %%
def load_smoothness(band):
    smoothness_roughness_time_series = np.load(f'/users2/local/Venkatesh/Generated_Data/25_subjects_copy_FOR_TESTING/smoothness_time_series{band}.npz')['smoothness_time_series']
    return np.squeeze(smoothness_roughness_time_series)
smoothness = load_smoothness('theta')

# %%
smoothness_concatenated = np.concatenate ([np.squeeze(slicing(smoothness.T,indices_pre_strong,axis=2)),
 np.squeeze(slicing(smoothness.T,indices_post_strong,axis=2))],axis=-1)[:,800:1100]

# %%

def normalisation(data):
    
    normalised =(data - np.min(data)/(np.max(data) - np.min(data)))
    return normalised
from sklearn.preprocessing import StandardScaler
b = np.concatenate([low_freq_pre_summed,low_freq_post_summed],axis=-1)[:,800:1100]
a = normalisation(b)#StandardScaler().fit(b).transform(b)
print(a)
# %%
plt.plot(np.mean(a,axis=0))
sem = scipy.stats.sem(a,axis=0)
print(sem)
plt.fill_between(range(300),np.mean(a,axis=0)-sem,np.mean(a,axis=0)+sem,color='r',alpha=0.2)
plt.axvspan(75,200,alpha=0.2)
# plt.axvline(6*125)
# %%
sc_scaled = normalisation(smoothness_concatenated)#StandardScaler().fit(smoothness_concatenated.T).transform(smoothness_concatenated.T)

# %%
plt.plot(np.mean(sc_scaled,axis=0))
sem_sc_scaled = scipy.stats.sem(sc_scaled)
plt.fill_between(range(300),np.mean(sc_scaled,axis=0)-sem_sc_scaled,np.mean(sc_scaled,axis=0)+sem_sc_scaled,color='r',alpha=0.2)# %%
plt.axvspan(75,200,alpha=0.2)
# plt.axvline(6*125)

# %%


isc_results_source = np.load('/users2/local/Venkatesh/Generated_Data/25_subjects_copy_FOR_TESTING/sourceCCA_ISC_8s_window.npz')['sourceISC']
noise_SI = np.load('/users2/local/Venkatesh/Generated_Data/25_subjects_copy_FOR_TESTING/noise_floor_8s_window.npz')['isc_noise_floored']
comp=1
significance = np.array(np.where(np.max(np.array(noise_SI)[:,1,:],axis=0)<isc_results_source[1]))

#%%
plt.plot(range(48,62),isc_results_source[comp][48:62], label='8-s window')
# plt.fill_between(range(48,72),np.max(np.array(noise_SI)[:,comp,:],axis=0).T,np.min(np.array(noise_SI)[:,comp,:],axis=0).T,color ='grey',alpha=0.8)
# plt.plot(significance,isc_results_source[comp][significance],
#             marker='o', ls="",color='red',markersize=4)
# plt.legend(loc="upper left")


#%%
from turtle import pos
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
vw_bundle = np.load('/users2/local/Venkatesh/Generated_Data/25_subjects_copy_FOR_TESTING/video_watching_bundle_STC_parcellated.npz')['video_watching_bundle_STC_parcellated']

GFTed_cortical_signal = [G.gft(np.array(vw_bundle[i])) for i in range(25)]
# %%
GFTed_cortical_signal_low_freq = np.array(GFTed_cortical_signal)[:,1:51,:]
GFTed_cortical_signal_medium_freq = np.array(GFTed_cortical_signal)[:,51:200,:]
GFTed_cortical_signal_high_freq = np.array(GFTed_cortical_signal)[:,200:,:]

# %%
def slicing(what_to_slice,where_to_slice):
    array_to_append = list()

    array_to_append.append ( what_to_slice[:,:,where_to_slice] )
    return array_to_append


def slicing_freqs(freqs):
    indices_pre_strong = np.hstack([np.arange(48*125,55*125),np.arange(76*125,78*125),np.arange(97*125,103*125)])

    print("indices length for Pre-Strong ISC is: ", len(indices_pre_strong)/125)

    items_pre_strong = np.squeeze(slicing(freqs,indices_pre_strong))

    indices_post_strong = np.hstack([np.arange(55*125,62*125),np.arange(78*125,80*125),np.arange(103*125,109*125)])
    print("indices length for Post-Strong ISC is: ", len(indices_post_strong)/125)

    items_post_strong = np.squeeze(slicing(freqs,indices_post_strong))

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

def sum_freqs(freqs):
    return np.sum(np.sqrt(np.power(freqs,2)),axis=-1)

low_freq_pre_t_averaged_summed, low_freq_post_t_averaged_summed = sum_freqs(low_freq_pre_t_averaged),sum_freqs(low_freq_post_t_averaged)
med_freq_pre_t_averaged_summed, med_freq_post_t_averaged_summed = sum_freqs(med_freq_pre_t_averaged),sum_freqs(med_freq_post_t_averaged)
high_freq_pre_t_averaged_summed, high_freq_post_t_averaged_summed = sum_freqs(high_freq_pre_t_averaged),sum_freqs(high_freq_post_t_averaged)

# %%
def ttest(band1, band2):
    return scipy.stats.ttest_rel(band1,band2)
print( ttest(high_freq_pre_t_averaged_summed, high_freq_post_t_averaged_summed) )
print ( ttest(med_freq_pre_t_averaged_summed, med_freq_post_t_averaged_summed) ) 
ttest(low_freq_pre_t_averaged_summed, low_freq_post_t_averaged_summed)

# %%

def stats_SEM(freqs):
        print(np.shape(freqs))
        return scipy.stats.sem(freqs,axis=1)#/np.sqrt(25)


# mpl.rcParams['font.family'] = 'Arial'

plt.rc('font', family='serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')

fig,ax = plt.subplots(figsize=(6,5))
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


ax.bar(x - width/2, np.sum(pre_strong,axis=-1), width, yerr=stats_SEM(pre_strong) ,label='Pre-Strong', align='center',color='C1')
ax.bar(x + width/2, np.sum(post_strong,axis=-1), width,yerr =stats_SEM(post_strong), label='Post-Strong', align='center', color='C0')


# data = pd.DataFrame({'labels':labels,'Smoothness':weak_isc})

# data2 = pd.DataFrame({'labels':labels,'Smoothness':strong_isc})
# data_fin = data.append(data2,ignore_index=True)
# data_fin['cond'] = ['Weak','Weak','Weak','Weak','Strong','Strong','Strong','Strong']
# pvalues_slicing =[pvalues[i][1] for i in range(4)]
# add_stat_annotation(ax,data=data_fin, y='Smoothness', x ='labels', hue='cond',
#                     box_pairs=[(("Theta", "Strong"), ("Theta", "Weak")),
#                     (("Alpha", "Strong"), ("Alpha", "Weak")),
#                     (("Lower Beta", "Strong"), ("Lower Beta", "Weak")),
#                     (("Upper Beta", "Strong"), ("Upper Beta", "Weak"))],
#                                  perform_stat_test=False, pvalues=pvalues_slicing,
# line_offset_to_box=0.25, line_offset=0.1, line_height=0.05, text_format='simple', loc='inside', verbose=2)

plt.tight_layout()
plt.legend()
plt.xticks(x,labels)
plt.ylabel('gPower')
plt.xlabel('Frequency bands')
# plt.title('HCP FC thresholded / 8s-window / 1st ISC component / weak = 170s - strong')

# %%

print(stats_SEM(pre_strong))
stats_SEM(post_strong)

# %%

#%%
from cProfile import label
from fileinput import filename
from logging import error
from signal import signal
from turtle import color
from cv2 import Laplacian
from nbformat import from_dict
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import size
from pygsp import graphs, filters
from pygsp import plotting as gsp_plt

from nilearn import image, plotting, datasets
import numpy as np
import matplotlib as mpl

from pathlib import Path
from scipy import io as sio
from pygsp import graphs
from seaborn.utils import axis_ticklabels_overlap
from scipy.sparse import csr_matrix
import scipy
import matplotlib
import seaborn as sns
from scipy import stats

import pickle

import pandas as pd
from scipy.stats import boxcox
import scipy
import torch

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

###############################
####Decomposing into eigenmodes
###############################
G.compute_fourier_basis()

# %%

#########################################
######Utility function###################
#########################################
def mean_std(freq):
    
    d = freq
    print("d",np.shape(d))
    mean_t = np.mean(d,axis=1)
    std_t = 2 * np.std(d,axis=1)
    top = mean_t + std_t
    bottom = mean_t - std_t
    
    return mean_t,std_t,top,bottom


def slicing(what_to_slice,where_to_slice):
    array_to_append = list()

    array_to_append.append ( what_to_slice[:,where_to_slice] )
    return array_to_append



# %%
noise_floor_source = np.load('/users2/local/Venkatesh/Generated_Data/25_subjects/noise_floor_source.npz')
for i in noise_floor_source:
    print(i)

#%%

# isc_result = np.load('/users2/local/Venkatesh/Generated_Data/25_subjects/sourceCCA_ISC.npz')['sourceISC']
# noise_floor_source = np.load('/users2/local/Venkatesh/Generated_Data/25_subjects/noise_floor_source.npz')['sourceCCA']
#isc_result = np.load('/users2/local/Venkatesh/Generated_Data/25_subjects_copy_FOR_TESTING/sourceCCA_ISC_8s_window.npz')['sourceISC']
#noise_floor_source = np.load('/users2/local/Venkatesh/Generated_Data/25_subjects_copy_FOR_TESTING/noise_floor_8s_window.npz')['isc_noise_floored']
isc_result = np.load('/users2/local/Venkatesh/Generated_Data/25_subjects_copy_FOR_TESTING/sourceCCA_ISC_8s_window.npz')['sourceISC']
noise_floor_source = np.load('/users2/local/Venkatesh/Generated_Data/25_subjects_copy_FOR_TESTING/noise_floor_8s_window.npz')['isc_noise_floored']

# isc_sliced = isc_result[0,:][np.hstack([np.arange(0,5),np.arange(7,9),np.arange(13,17), np.arange(26,30),np.arange(33,38),np.arange(40,44),np.arange(58,59),np.arange(62,66),
#                         np.arange(70,74),np.arange(85,86),np.arange(88,95),np.arange(99,100),np.arange(104,110),np.arange(118,124),np.arange(127,128),np.arange(129,133),np.arange(144,145),np.arange(148,153),np.arange(155,158)])]




envelope_signal_bandpassed = np.load('/users2/local/Venkatesh/Generated_Data/25_subjects_copy_FOR_TESTING/envelope_signal_bandpassed_low_high_beta.npz', mmap_mode='r')

alpha = envelope_signal_bandpassed['alpha']
low_beta = envelope_signal_bandpassed['lower_beta']
high_beta = envelope_signal_bandpassed['higher_beta']
theta = envelope_signal_bandpassed['theta']


#%%
from statannot import add_stat_annotation
import pandas as pd

# def barplot(data_for_strong,data_for_weak,title,pvalues,band):
#     labels = ['Theta','Beta','Alpha']
#     x = np.arange(len(labels))  # the label locations
#     width = 1.5# the width of the bars

#     sns.set_theme()
#     def stats_SEM(freqs):
#         print(np.shape(freqs))
#         return scipy.stats.sem(freqs,axis=1)#/np.sqrt(25)

#     std_err_weak = np.hstack([stats_SEM(data_for_weak)])
#     std_err_strong = np.hstack([stats_SEM(data_for_strong)])


  
#     data = [data_for_strong,data_for_weak]
#     error = [std_err_strong,std_err_weak]
#     print(np.shape(np.array(error)))
#     fig, ax = plt.subplots()
#     ax.bar(x,np.average(np.squeeze(data).T,axis=0),yerr=np.squeeze(np.array(error)),color=('C0','C1'))
    
#     y = "Smoothness coefficients"
#     order = ["Strong ISC", "Weak ISC"]
#     print(np.shape(data_for_strong))
#     df = pd.DataFrame([np.average(data_for_strong),np.average(data_for_weak)],columns=['Smoothness coefficients'])
#     df['conditions'] = ["Strong ISC", "Weak ISC"]

#     plt.ylabel(y)
#     plt.title(title)
#     plt.xticks(x,labels)
#     plt.show()
#     #fig.savefig(f'/homes/v20subra/S4B2/graph_analysis_during_PhD/Gretsi/{band}.png')



smoothness_roughness_time_series = list()
pvalues = list()
def master(signal_to_calculate_smoothness,band):
    G.compute_laplacian('combinatorial')
    laplacian = G.L.toarray()



    one = np.array(signal_to_calculate_smoothness).T
    two = np.swapaxes(one,axis1=1,axis2=2)
    signal = np.expand_dims(two,2)

    stage1 = np.tensordot(signal,laplacian,axes=(3,0))
    print(np.shape(stage1))

    signal_stage2 = np.swapaxes(signal,2,3)
    print(np.shape(signal_stage2))


    smoothness_roughness_time_series = np.matmul(stage1,signal_stage2)
    # np.savez_compressed(f'/users2/local/Venkatesh/Generated_Data/25_subjects_copy_FOR_TESTING/smoothness_time_series{band}',smoothness_time_series=smoothness_roughness_time_series)
    print("np.shape(smoothness_roughness_time_series):",np.shape(smoothness_roughness_time_series))

# [[  1   7   8   9  10  12  13  32  39  40  41  42  43  46  56  60  87  88
#    89  90 127 134 135 136 137 138 140 141 158 165 166]]
# [[  8  19  52  53  54  55  56  57  58  59  60  61  78  80 103 104 105 106
#   107 147 149 150 155 156 157]]
#   axes.axvspan(3,6,alpha=0.1, color='C1')
#         axes.axvspan(16,19,alpha=0.1, color='C1')
#         axes.axvspan(25,28,alpha=0.1, color='C1')
#         axes.axvspan(50,53,alpha=0.1, color='C1')
#         axes.axvspan(61,70,alpha=0.1, color='C1')

#         axes.axvspan(79,83,alpha=0.1, color='C1')
#         axes.axvspan(117,120,alpha=0.1, color='C1')
#         axes.axvspan(129,132,alpha=0.1, color='C1')

#array([131, 132,  51, 162,  52,  64,   6,  27, 148, 149,  70,  80,  81,
    #    130,  82,  53,  18, 116, 163, 101,  63, 122, 110, 150, 160, 119,
    #      5,  26,  83, 103])


#[3, 4, 5, 6, 17, 18, 25, 26, 27, 51, 52, 53, 61, 62, 63, 64, 65, 66, 70, 71, 72, 74, 75, 79, 80, 81, 82, 83, 100, 101, 102, 103, 104, 109, 110, 116, 117, 118, 119, 120, 122, 
# 130, 131, 132, 147, 148, 149, 150, 159, 160, 161, 162, 163]

#First two components
    # items_weak = np.hstack([np.arange(3*125,7*125),np.arange(17*125,19*125),np.arange(25*125,28*125),np.arange(51*125,54*125),np.arange(61*125,67*125),
    #                     np.arange(70*125,73*125),np.arange(74*125,76*125),np.arange(79*125-62,79*125+63),np.arange(80*125,84*125),np.arange(100*125,105*125),np.arange(104*125-62,104*125+63),np.arange(110*125-62,110*125+63),
    #                     np.arange(116*125,121*125),np.arange(122*125-62,122*125+63),np.arange(130*125,133*125),np.arange(147*125,151*125),
    #                     np.arange(159*125,164*125)])
#second component
#[6, 18, 27, 51, 52, 53, 63, 64, 70, 80, 81, 82, 101, 110, 116, 122, 130, 131, 132, 148, 149, 150, 160, 162, 163]
    # items_weak = np.hstack([np.arange(6*125-62,6*125+63),np.arange(18*125-62,18*125+63),np.arange(27*125-62,27*125+63),np.arange(51*125,54*125),np.arange(63*125,65*125),
    # np.arange(70*125-62,70*125+63),np.arange(80*125,83*125),np.arange(101*125-62,101*125+63),np.arange(110*125-62,110*125+63),np.arange(116*125-62,116*125+63),np.arange(122*125-62,122*125+63),
    # np.arange(130*125,133*125),np.arange(148*125,151*125),np.arange(162*125,164*125),np.arange(160*125-62,160*125+63)])
#third component
#[5, 6, 18, 25, 26, 27, 51, 52, 53, 62, 63, 64, 70, 74, 80, 81, 82, 83, 101, 103, 109, 110, 116, 119, 122, 130, 131, 132, 148, 149, 150, 160, 162, 163]
    # items_weak = np.hstack([np.arange(5*125,7*125),np.arange(18*125-62,18*125+63),np.arange(25*125,28*125),np.arange(51*125,54*125),np.arange(62*125,65*125),np.arange(70*125-62,70*125+63),
    # np.arange(74*125-62,74*125+63),np.arange(80*125,84*125),np.arange(101*125-62,101*125+63),np.arange(103*125-62,103*125+63),np.arange(109*125,111*125),np.arange(116*125-62,116*125+63),np.arange(119*125-62,119*125+63),
    # np.arange(122*125-62,122*125+63),np.arange(131*125,133*125),np.arange(148*125,151*125),np.arange(160*125,164*125)])

#first comp

    # items_weak = np.hstack([np.arange(5*125,7*125),np.arange(18*125-62,18*125+63),np.arange(26*125,28*125),np.arange(51*125,54*125),np.arange(63*125,65*125),
    #                     np.arange(70*125-62,70*125+63),np.arange(74*125-62,74*125+63),np.arange(80*125,84*125),np.arange(101*125-62,101*125+63),np.arange(103*125-62,103*125+63),np.arange(110*125-62,110*125+63),
    #                     np.arange(116*125-62,116*125+63),np.arange(119*125-62,119*125+63),np.arange(122*125-62,122*125+63),np.arange(130*125,133*125),np.arange(148*125,151*125),np.arange(160*125-62,160*125+63),
    #                     np.arange(162*125,164*125)])
# 8s window
#first
#     items_weak = np.hstack([np.arange(3*125-62,3*125+63),np.arange(17*125,19*125),np.arange(22*125-62,22*125+63),np.arange(24*125-62,24*125+63),np.arange(48*125-62,48*125+63),np.arange(52*125-62,52*125+63),np.arange(60*125,62*125),
# np.arange(64*125,68*125),np.arange(71*125,74*125),np.arange(77*125,82*125),np.arange(98*125,102*125),np.arange(103*125,107*125),np.arange(109*125,111*125),np.arange(112*125,120*125),
# np.arange(129*125-62,129*125+63),np.arange(131*125-62,131*125+63),np.arange(143*125-62,143*125+63),np.arange(145*125-62,145*125+63),np.arange(147*125,152*125),np.arange(159*125,161*125)])

#     items_weak = np.hstack([np.arange(2*125,8*125),np.arange(11*125,30*125),np.arange(32*125,35*125),np.arange(48*125,56*125),np.arange(59*125,68*125),np.arange(70*125,74*125),
# np.arange(79*125,86*125),np.arange(94*125,123*125),np.arange(125*125,132*125),np.arange(142*125,155*125),np.arange(157*125,162*125),np.arange(167*125,169*125)])

    items_weak = np.hstack([np.arange(48*125,55*125),np.arange(76*125,78*125),np.arange(97*125,103*125)])


#[1, 7, 8, 9, 10, 12, 13, 19, 32, 39, 40, 41, 42, 43, 46, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 78, 80, 87, 88, 89, 90, 103, 104, 105, 106, 107,
#  127, 134, 135, 136, 137, 138, 140, 141, 147, 149, 150, 155, 156, 157, 158, 165, 166]

#First two components
    # items_strong = np.hstack([np.arange(1*125-62,1*125+63),np.arange(7*125,11*125), np.arange(12*125,14*125),  np.arange(19*125-62,19*125+63), np.arange(32*125-62,32*125+63),np.arange(39*125,44*125),
    #  np.arange(46*125-62,46*125+63), np.arange(52*125,62*125),np.arange(78*125-62,78*125+63),np.arange(80*125-62,80*125+63),np.arange(87*125,91*125),np.arange(103*125,108*125),
    #  np.arange(127*125-62,127*125+63),np.arange(134*125,139*125),np.arange(140*125,142*125),np.arange(147*125-62,147*125+63),np.arange(149*125,151*125),np.arange(155*125,159*125),np.arange(165*125,167*125)])

#Second component
#rray([  8,  19,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  78,
        # 80, 103, 104, 105, 106, 107, 147, 149, 150, 155, 156, 157])
    # items_strong = np.hstack([np.arange(8*125-62,8*125+63),np.arange(19*125-62,19*125+63),np.arange(52*125,62*125),np.arange(78*125-62,78*125+63),np.arange(80*125-62,80*125+63),np.arange(103*125,108*125),
    # np.arange(147*125-62,147*125+63),np.arange(149*125,151*125),np.arange(155*125,158*125)])
#third component
#array([ 13,  33,  34,  35,  36,  37,  57,  58,  65,  66,  67,  68,  69,
    #     73,  74,  81,  83,  94, 111, 113, 119, 130, 133, 134, 135, 136,
    #    137, 138, 141, 143, 144, 145, 146, 160])
    # items_strong = np.hstack([np.arange(13*125-62,13*125+63),np.arange(33*125,38*125),np.arange(57*125,59*125),np.arange(65*125,70*125),np.arange(73*125,75*125),np.arange(81*125-62,81*125+63),
    # np.arange(83*125-62,83*125+63),np.arange(94*125-62,94*125+63),np.arange(111*125-62,111*125+63),np.arange(113*125-62,113*125+63),np.arange(119*125-62,119*125+63),np.arange(130*125-62,130*125+63),np.arange(133*125,139*125),
    # np.arange(141*125-62,141*125+63),np.arange(143*125,147*125),np.arange(160*125-62,160*125+63)])

# #first comp
#     items_strong = np.hstack([np.arange(1*125-62,1*125+63),np.arange(7*125,11*125), np.arange(12*125,14*125), np.arange(39*125,44*125),np.arange(87*125,91*125),np.arange(134*125,139*125),np.arange(140*125,142*125),np.arange(165*125,167*125),
#     np.arange(32*125-62,32*125+63),np.arange(46*125-62,46*125+63),np.arange(56*125-62,56*125+63),np.arange(60*125-62,60*125+63),np.arange(127*125-62,127*125+63),np.arange(158*125-62,158*125+63)])

# 8-s time window
    # items_strong = np.hstack([np.arange(1*125-62,1*125+63),np.arange(8*125,11*125),np.arange(30*125-62,30*125+63),np.arange(35*125,45*125),np.arange(46*125,48*125),np.arange(56*125,59*125),
    # np.arange(68*125,70*125),np.arange(74*125,76*125),np.arange(86*125,92*125),np.arange(93*125-62,93*125+63),np.arange(123*125,125*125),np.arange(132*125,142*125),np.arange(155*125,157*125),
    # np.arange(162*125,167*125)])

    items_strong = np.hstack([np.arange(55*125,62*125),np.arange(78*125,80*125),np.arange(103*125,109*125)])

    print("strong:",len(items_strong)/125)
    print(len(items_weak)/125)
    print(band)
    smoothness_roughness_time_series_weak = slicing(np.squeeze(smoothness_roughness_time_series).T,items_weak)
    smoothness_roughness_time_series_strong = slicing(np.squeeze(smoothness_roughness_time_series).T,items_strong)


    to_append = list()
    for i in range(0,len(smoothness_roughness_time_series),125):
            to_append.append(np.average(np.squeeze(smoothness_roughness_time_series)[i:i+125,:],axis=0))
    

    squeezed_weak = np.squeeze(smoothness_roughness_time_series_weak)
    squeezed_strong = np.squeeze(smoothness_roughness_time_series_strong)

    def average(which_condition):
        to_append_average = list()
        for i in range(0,np.shape(smoothness_roughness_time_series_strong)[1],125):
            to_append_average.append(np.average(which_condition[:,i:i+125],axis=1))
        return to_append_average

    squeezed_strong_averaged = average(squeezed_strong)
    squeezed_weak_averaged = average(squeezed_weak)

    print("squeezed_weak_averaged:",np.shape(squeezed_weak_averaged))
    ttest1 = scipy.stats.ttest_rel(boxcox(squeezed_strong_averaged[0],lmbda=0),boxcox(squeezed_weak_averaged[0],lmbda=0))
    print((ttest1))
    pvalues.append(ttest1)
    # for i in range(25):
    #     r, p = scipy.stats.pearsonr(boxcox(np.array(to_append)[:,i],lmbda=0),isc_result[0])
    #     print(f'r ={r}, p ={p}')

    #barplot(squeezed_strong_averaged,squeezed_weak_averaged,f'Smoothness for the envelope {band}-band',pvalues=ttest1[1],band=band)
    return  squeezed_strong_averaged, squeezed_weak_averaged

smoothness_roughness_time_series_dict = dict()
# smoothness_roughness_time_series =master(signal_to_calculate_smoothness=beta,band='beta')
smoothness_roughness_time_series_dict['theta'] = master(signal_to_calculate_smoothness=theta,band='theta')
smoothness_roughness_time_series_dict['alpha'] = master(signal_to_calculate_smoothness=alpha,band='alpha')
smoothness_roughness_time_series_dict['low_beta'] = master(signal_to_calculate_smoothness=low_beta,band='lower_beta')
smoothness_roughness_time_series_dict['high_beta'] = master(signal_to_calculate_smoothness=high_beta,band='higher_beta')

# smoothness_roughness_time_series_dict_conditions =dict()
# smoothness_roughness_time_series_dict_conditions['theta'] = master(signal_to_calculate_smoothness=theta,band='theta')
# smoothness_roughness_time_series_dict_conditions['alpha'] = master(signal_to_calculate_smoothness=alpha,band='alpha')
# smoothness_roughness_time_series_dict_conditions['beta'] = master(signal_to_calculate_smoothness=beta,band='beta')
#%%
np.shape(smoothness_roughness_time_series)

# %%

smoothness_roughness_time_series = np.load('/users2/local/Venkatesh/Generated_Data/25_subjects_copy_FOR_TESTING/smoothness_time_series.npz')['smoothness_time_series']
smoothness_roughness_time_series_averaged_temporally = list()
for i in range(0,21250,125):
     smoothness_roughness_time_series_averaged_temporally.append(np.average(np.squeeze(smoothness_roughness_time_series)[i:i+125,:],axis=0))


#%%
print (np.array(np.where(np.array(noise_floor_source)[:,0,:]<=0.05)))

import seaborn as sns
sns.set_theme()
def plot(comp,band):
    significance = np.array(np.where(np.max(np.array(noise_floor_source)[:,comp,:],axis=0)<isc_result[comp]))
    print(len(significance[0]))
    def drawingline(axes):
        
# [[  1   7   8   9  10  12  13  32  39  40  41  42  43  46  56  60  87  88
#    89  90 127 134 135 136 137 138 140 141 158 165 166]]
# [[  8  19  52  53  54  55  56  57  58  59  60  61  78  80 103 104 105 106
#   107 147 149 150 155 156 157]]


#([  3,   4,   5,   6,  17,  18,  19,  25,  26,  27,  51,  52,  53,
    #     61,  62,  63,  64,  65,  66,  70,  71,  72,  74,  75,  76,  79,
    #     80,  81,  82,  83,  84, 100, 101, 102, 103, 104, 109, 110, 113,
    #    116, 117, 118, 119, 120, 121, 122, 129, 130, 131, 132, 147, 148,
    #    149, 150, 151, 152, 159, 160, 161, 162, 163])


#array([131, 132,  51, 162,  52,  64,   6,  27, 148, 149,  70,  80,  81,
    #    130,  82,  53,  18, 116, 163, 101,  63, 122, 110, 150, 160, 119,
    #      5,  26,  83, 103])



        axes.axvspan(5,6,alpha=0.1, color='C1')
        axes.axvspan(18,18,alpha=0.1, color='C1')
        axes.axvspan(26,28,alpha=0.1, color='C1')
        axes.axvspan(51,53,alpha=0.1, color='C1')
        axes.axvspan(63,64,alpha=0.1, color='C1')
        axes.axvspan(70,70,alpha=0.1, color='C1')
        axes.axvspan(74,74,alpha=0.1, color='C1')

        axes.axvspan(80,83,alpha=0.1, color='C1')

        axes.axvspan(101,101,alpha=0.1, color='C1')
        axes.axvspan(101,101,alpha=0.1, color='C1')
        axes.axvspan(103,103,alpha=0.1, color='C1')
        axes.axvspan(110,110,alpha=0.1, color='C1')
        axes.axvspan(116,116,alpha=0.1, color='C1')
        axes.axvspan(119,119,alpha=0.1, color='C1')
        axes.axvspan(122,122,alpha=0.1, color='C1')
        axes.axvspan(130,132,alpha=0.1, color='C1')
        axes.axvspan(148,150,alpha=0.1, color='C1')
        axes.axvspan(160,160,alpha=0.1, color='C1')
        axes.axvspan(162,163,alpha=0.1, color='C1')



        axes.axvspan(7,14,alpha=0.1, color='C0')
        axes.axvspan(40,43,alpha=0.1, color='C0')
        axes.axvspan(87,90,alpha=0.1, color='C0')
        axes.axvspan(134,138,alpha=0.1, color='C0')
        axes.axvspan(140,141,alpha=0.1, color='C0')
        axes.axvspan(158,159,alpha=0.1, color='C0')
    
  

    fig = plt.figure(figsize = (10,10))
    ax1 = plt.subplot(211, frameon=False)
    ax2 = plt.subplot(212, frameon=False)

    ax1.plot(range(170),isc_result[comp])
    ax1.fill_between(range(170),np.max(np.array(noise_floor_source)[:,comp,:],axis=0).T,np.min(np.array(noise_floor_source)[:,comp,:],axis=0).T,color ='grey',alpha=0.8)
    ax1.plot(significance,isc_result[comp][significance],
                marker='o', ls="",color='red',markersize=4)

    drawingline(ax1)

    # print(np.shape(smoothness_roughness_time_series_averaged_temporally))

    # mean_t = np.average(np.squeeze(smoothness_roughness_time_series_averaged_temporally),axis=1)
    # std_t = scipy.stats.sem(np.squeeze(smoothness_roughness_time_series_averaged_temporally),axis=1)
    # ax2.plot(mean_t)
    # ax2.fill_between(range(170),mean_t-std_t,mean_t+std_t, color='b', alpha=.3)

    drawingline(ax2)
    ax1.set_xticklabels([])
    
    fig.text(0.5, 0.04, 'time (s)', ha='center')
    fig.text(0.04, 0.75, 'ISC coefficients', va='top', rotation='vertical')
    fig.text(0.04, 0.25, 'Smoothness coefficients', va='bottom', rotation='vertical')
    plt.show()
    # fig.savefig(f'/homes/v20subra/S4B2/graph_analysis_during_PhD/Gretsi/{band}_smoothness.png')
plot(0,'beta')

# %%
def stats_SEM(freqs):
        print(np.shape(freqs))
        return scipy.stats.sem(freqs,axis=1)#/np.sqrt(25)


mpl.rcParams['font.family'] = 'Arial'

plt.rc('font', family='serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')

fig,ax = plt.subplots(figsize=(6,5))
labels = ['Theta', 'Alpha', 'Lower Beta','Upper Beta']

x = np.arange(len(labels))  # the label locations
width=0.35
error_theta = stats_SEM(np.squeeze(smoothness_roughness_time_series_dict['theta']))
error_low_beta = stats_SEM(np.squeeze(smoothness_roughness_time_series_dict['low_beta']))
error_high_beta = stats_SEM(np.squeeze(smoothness_roughness_time_series_dict['high_beta']))

error_alpha = stats_SEM(np.squeeze(smoothness_roughness_time_series_dict['alpha']))

strong_isc = [np.squeeze(smoothness_roughness_time_series_dict['theta'])[0],
            np.squeeze(smoothness_roughness_time_series_dict['alpha'])[0],
            np.squeeze(smoothness_roughness_time_series_dict['low_beta'])[0],
            np.squeeze(smoothness_roughness_time_series_dict['high_beta'])[0]]

weak_isc = [np.squeeze(smoothness_roughness_time_series_dict['theta'])[1],
            np.squeeze(smoothness_roughness_time_series_dict['alpha'])[1],
            np.squeeze(smoothness_roughness_time_series_dict['low_beta'])[1],
            np.squeeze(smoothness_roughness_time_series_dict['high_beta'])[1]]

def iqr(data):

    quartile1, medians, quartile3 = np.percentile(data, [25, 50, 75], axis=1)
    return quartile1,medians, quartile3


# ax.bar(x - width/2, strong_isc, width, label='Strong ISC',yerr=[error_theta[0],error_alpha[0],error_low_beta[0],error_high_beta[0]], align='center',color='C0')
# ax.bar(x + width/2, weak_isc, width, label='Weak ISC',yerr=[error_theta[1],error_alpha[1],error_low_beta[1],error_high_beta[1]], align='center', color='C1')
ax.violinplot(positions= x - width/2, dataset = strong_isc,widths=width,showextrema=True,showmeans=True )

ax.violinplot(positions=x + width/2, dataset =weak_isc,widths=width,showextrema=True,showmeans=True)

# ax2.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)

data = pd.DataFrame({'labels':labels,'Smoothness':np.sum(weak_isc,axis=1)})

data2 = pd.DataFrame({'labels':labels,'Smoothness':np.sum(strong_isc,axis=1)})
data_fin = data.append(data2,ignore_index=True)
data_fin['cond'] = ['Weak','Weak','Weak','Weak','Strong','Strong','Strong','Strong']
pvalues_slicing =[pvalues[i][1] for i in range(4)]
add_stat_annotation(ax,data=data_fin, y='Smoothness', x ='labels', hue='cond',
                    box_pairs=[(("Theta", "Strong"), ("Theta", "Weak")),
                    (("Alpha", "Strong"), ("Alpha", "Weak")),
                    (("Lower Beta", "Strong"), ("Lower Beta", "Weak")),
                    (("Upper Beta", "Strong"), ("Upper Beta", "Weak"))],
                                 perform_stat_test=False, pvalues=pvalues_slicing,
line_offset_to_box=0.25, line_offset=0.1, line_height=0.05, text_format='star', loc='outside', verbose=2)
plt.legend(['Strong ISC','Weak ISC'])

plt.tight_layout()
plt.legend()
plt.xticks(x,labels)
plt.ylabel('Smoothness')
plt.xlabel('Frequency bands in the cortical envelope signal')
plt.title('Smoothness Response')
fig.savefig('/homes/v20subra/S4B2/Graph-related_analysis/Functional_graph_setup/smoothness_violin.png', dpi=300, bbox_inches='tight')
# %%

plt.figure(figsize=(25,25))

a = 5  # number of rows
b = 5  # number of columns
c = 1  # initialize plot counter
import seaborn as sns
sns.set_theme()
for i in range(25):
    plt.subplot(a, b, c)
    plt.scatter(boxcox(np.array(smoothness_roughness_time_series_averaged_temporally)[:,i],0),isc_result[0])
    c+=1
    plt.xlabel('Smoothness')
    plt.ylabel('ISC')
    r, p =scipy.stats.pearsonr(boxcox(np.array(smoothness_roughness_time_series_averaged_temporally)[:,i],0),isc_result[0])
    print(r, p)
    plt.annotate(f'r = {np.round(r,2)}, p = {np.round(p,2)}', xy=(0, 1), xytext=(155, -20), va='top',
             xycoords='axes fraction', textcoords='offset points')
    # print(scipy.stats.ttest_rel(boxcox(np.array(smoothness_roughness_time_series_averaged_temporally)[:,i],0),isc_result[0]))
plt.suptitle('The relationship between smoothness quotient & ISC subject-wise')
plt.tight_layout()


#%%

print(np.where(isc_result[0]<=0.02))
# %%

significance = np.array(np.where(np.max(np.array(noise_floor_source)[:,1,:],axis=0)<isc_result[1]))


# %%
set(significance[0])
# %%





# %%

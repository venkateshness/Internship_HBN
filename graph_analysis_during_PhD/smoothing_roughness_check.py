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


import pandas as pd
from scipy.stats import boxcox
import scipy
from scipy.stats import boxcox
sns.set_theme()
############################################################
##########Getting the Graph ready###########################
############################################################ 
def graph_setup(thresholding, percentage):
    path_Glasser='/homes/v20subra/S4B2/GSP/Glasser_masker.nii.gz'
    res_path=''

    # Load structural connectivity matrix
    connectivity = sio.loadmat('/homes/v20subra/S4B2/GSP/SC_avg56.mat')['SC_avg56']
    connectivity.shape
    coordinates = sio.loadmat('/homes/v20subra/S4B2/GSP/Glasser360_2mm_codebook.mat')['codeBook'] # coordinates in brain space

    G=graphs.Graph(connectivity,gtype='HCP subject',lap_type='combinatorial',coords=coordinates) 
    G.set_coordinates('spring')
    print('{} nodes, {} edges'.format(G.N, G.Ne))

    if thresholding:
        weights = csr_matrix(G.W).toarray()
        weights[weights<np.percentile(weights,percentage)] =0

        G = graphs.Graph(weights)
        print(G.is_connected())
        print('{} nodes, {} edges'.format(G.N, G.Ne))

    return G


G = graph_setup(False,90)
###############################
####Decomposing into eigenmodes
###############################
G.compute_fourier_basis()

# %%

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


isc_result = np.load('/users2/local/Venkatesh/Generated_Data/25_subjects/sourceCCA_ISC.npz')['sourceISC']
noise_floor_source = np.load('/users2/local/Venkatesh/Generated_Data/25_subjects/noise_floor_source.npz')['sourceCCA']
isc_sliced = isc_result[0,:][np.hstack([np.arange(0,5),np.arange(7,9),np.arange(13,17), np.arange(26,30),np.arange(33,38),np.arange(40,44),np.arange(58,59),np.arange(62,66),
                        np.arange(70,74),np.arange(85,86),np.arange(88,95),np.arange(99,100),np.arange(104,110),np.arange(118,124),np.arange(127,128),np.arange(129,133),np.arange(144,145),np.arange(148,153),np.arange(155,158)])]




envelope_signal_bandpassed = np.load('/users2/local/Venkatesh/Generated_Data/25_subjects/envelope_signal_bandpassed.npz', mmap_mode='r')

alpha = envelope_signal_bandpassed['alpha']
beta = envelope_signal_bandpassed['beta']
theta = envelope_signal_bandpassed['theta']


#%%
from statannot import add_stat_annotation
import pandas as pd

def barplot(data_for_strong,data_for_weak,title,pvalues,band):
    labels = ['Theta','Beta','Alpha']
    x = np.arange(len(labels))  # the label locations
    width = 1.5# the width of the bars

    sns.set_theme()
    def stats_SEM(freqs):
        print(np.shape(freqs))
        return scipy.stats.sem(freqs,axis=1)#/np.sqrt(25)

    std_err_weak = np.hstack([stats_SEM(data_for_weak)])
    std_err_strong = np.hstack([stats_SEM(data_for_strong)])

#     print(std_err_strong)
#     print(std_err_weak)
#     fig, ax = plt.subplots()
#     rects1 = ax.bar(x - width, np.average(data_for_strong), width, label="Strong ISC", yerr = std_err_strong, align='center', alpha=0.5, ecolor='black', capsize=10)
#     rects2 = ax.bar(x + width, np.average(data_for_weak), width, label="Weak ISC", yerr = std_err_weak, align='center', alpha=0.5, ecolor='black', capsize=10)
    
  
    data = [data_for_strong,data_for_weak]
    error = [std_err_strong,std_err_weak]
    print(np.shape(np.array(error)))
    fig, ax = plt.subplots()
    ax.bar(x,np.average(np.squeeze(data).T,axis=0),yerr=np.squeeze(np.array(error)),color=('C0','C1'))
    
    y = "Smoothness coefficients"
    order = ["Strong ISC", "Weak ISC"]
    print(np.shape(data_for_strong))
    df = pd.DataFrame([np.average(data_for_strong),np.average(data_for_weak)],columns=['Smoothness coefficients'])
    df['conditions'] = ["Strong ISC", "Weak ISC"]
#     add_stat_annotation(ax, data=df,y='Smoothness coefficients',x='conditions',order=order,
# box_pairs=[("Strong ISC", "Weak ISC")],perform_stat_test=False, pvalues=[(pvalues)],
# line_offset_to_box=0.20, line_offset=0.1, line_height=0.05, text_format='star', loc='inside', verbose=2)
    
    plt.ylabel(y)
    plt.title(title)
    plt.xticks(x,labels)
    plt.show()
    fig.savefig(f'/homes/v20subra/S4B2/graph_analysis_during_PhD/Gretsi/{band}.png')
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
    print("np.shape(smoothness_roughness_time_series):",np.shape(smoothness_roughness_time_series))

    items_weak = np.hstack([np.arange(0*125,5*125),np.arange(33*125,38*125),np.arange(49*125,54*125),np.arange(62*125,66*125),
                        np.arange(70*125,74*125),np.arange(88*125,95*125),np.arange(129*125,133*125),np.arange(148*125,153*125)])


    items_strong = np.hstack([np.arange(7*125,9*125),np.arange(13*125,17*125), np.arange(27*125-62,30*125+63),np.arange(40*125,44*125),np.arange(58*125-62,58*125+63),
    np.arange(85*125-62,85*125+63),np.arange(99*125-62,99*125+63),np.arange(104*125,110*125),np.arange(118*125,124*125),np.arange(155*125,158*125),np.arange(127*125-62,127*125+63),np.arange(144*125-62,144*125+63)])

    print(len(items_strong)/125)
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
    pvalues.append(ttest1)
    # for i in range(25):
    #     r, p = scipy.stats.pearsonr(boxcox(np.array(to_append)[:,i],lmbda=0),isc_result[0])
    #     print(f'r ={r}, p ={p}')

    # barplot(squeezed_strong_averaged,squeezed_weak_averaged,f'Smoothness for the envelope {band}-band',pvalues=ttest1[1],band=band)
    return squeezed_strong_averaged, squeezed_weak_averaged

smoothness_roughness_time_series_dict = dict()

smoothness_roughness_time_series_dict['theta'] = master(signal_to_calculate_smoothness=theta,band='theta')
smoothness_roughness_time_series_dict['alpha'] = master(signal_to_calculate_smoothness=alpha,band='alpha')
smoothness_roughness_time_series_dict['beta'] = master(signal_to_calculate_smoothness=beta,band='beta')

# %%
import seaborn as sns
sns.set_theme()
def plot(comp,band):
    significance = np.array(np.where(np.max(np.array(noise_floor_source)[:,comp,:],axis=0)<isc_result[comp]))

    def drawingline(axes):
        
        axes.axvspan(20,25,alpha=0.1, color='C1')
        axes.axvspan(33,38,alpha=0.1, color='C1')
        axes.axvspan(62,66,alpha=0.1, color='C1')
        axes.axvspan(70,82,alpha=0.1, color='C1')
        axes.axvspan(88,95,alpha=0.1, color='C1')
        axes.axvspan(129,141,alpha=0.1, color='C1')
        axes.axvspan(148,153,alpha=0.1, color='C1')
        axes.axvspan(49,54,alpha=0.1, color='C1')
        axes.axvspan(49,54,alpha=0.1, color='C1')

        axes.axvspan(7,8,alpha=0.1, color='C0')
        axes.axvspan(13,16,alpha=0.1, color='C0')
        axes.axvspan(27,29,alpha=0.1, color='C0')
        axes.axvspan(40,44,alpha=0.1, color='C0')
        axes.axvspan(118,124,alpha=0.1, color='C0')
        axes.axvspan(104,110,alpha=0.1, color='C0')
        axes.axvspan(155,158,alpha=0.1, color='C0')
        items_strong = np.hstack([np.arange(7*125,9*125),np.arange(13*125,17*125), np.arange(27*125-62,30*125+63),np.arange(40*125,44*125),np.arange(58*125-62,58*125+63),
        np.arange(85*125-62,85*125+63),np.arange(99*125-62,99*125+63),np.arange(104*125,110*125),np.arange(118*125,124*125),np.arange(155*125,158*125),np.arange(127*125-62,127*125+63),np.arange(144*125-62,144*125+63)])

  

    fig = plt.figure(figsize = (10,10))
    ax1 = plt.subplot(211, frameon=False)
    ax2 = plt.subplot(212, frameon=False)

    ax1.plot(range(170),isc_result[comp])
    ax1.fill_between(range(170),np.max(np.array(noise_floor_source)[:,comp,:],axis=0).T,np.min(np.array(noise_floor_source)[:,comp,:],axis=0).T,color ='grey',alpha=0.8)
    ax1.plot(significance,isc_result[comp][significance],
                marker='o', ls="",color='red',markersize=4)

    drawingline(ax1)

    print(np.shape(smoothness_roughness_time_series_averaged_temporally))

    mean_t = np.average(np.squeeze(smoothness_roughness_time_series_averaged_temporally),axis=1)
    std_t = scipy.stats.sem(np.squeeze(smoothness_roughness_time_series_averaged_temporally),axis=1)
    ax2.plot(mean_t)
    ax2.fill_between(range(170),mean_t-std_t,mean_t+std_t, color='b', alpha=.3)

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

fig,ax = plt.subplots(figsize=(8,8))

labels = ['Theta','Beta','Alpha']
x = np.arange(len(labels))  # the label locations
width=0.35
error_theta = stats_SEM(np.squeeze(smoothness_roughness_time_series_dict['theta']))
error_beta = stats_SEM(np.squeeze(smoothness_roughness_time_series_dict['beta']))
error_alpha = stats_SEM(np.squeeze(smoothness_roughness_time_series_dict['alpha']))

strong_isc = [np.average(np.squeeze(smoothness_roughness_time_series_dict['theta']),axis=1)[0],
            np.average(np.squeeze(smoothness_roughness_time_series_dict['beta']),axis=1)[0],
            np.average(np.squeeze(smoothness_roughness_time_series_dict['alpha']),axis=1)[0]]

weak_isc = [np.average(np.squeeze(smoothness_roughness_time_series_dict['theta']),axis=1)[1],
            np.average(np.squeeze(smoothness_roughness_time_series_dict['beta']),axis=1)[1],
            np.average(np.squeeze(smoothness_roughness_time_series_dict['alpha']),axis=1)[1]]


ax.bar(x - width/2, strong_isc, width, label='Strong ISC',yerr=[error_theta[0],error_beta[0],error_alpha[0]], align='center',color='C0')
ax.bar(x + width/2, weak_isc, width, label='Weak ISC',yerr=[error_theta[1],error_beta[1],error_alpha[1]], align='center', color='C1')

labels = ['Theta', 'Beta', 'Alpha']

data = pd.DataFrame({'labels':labels,'Smoothness':weak_isc})

data2 = pd.DataFrame({'labels':labels,'Smoothness':strong_isc})
data_fin = data.append(data2,ignore_index=True)
data_fin['cond'] = ['Weak','Weak','Weak','Strong','Strong','Strong']
pvalues_slicing =[pvalues[i][1] for i in range(3)]
add_stat_annotation(ax,data=data_fin, y='Smoothness', x ='labels', hue='cond',
                    box_pairs=[(("Theta", "Strong"), ("Theta", "Weak")),
                    (("Beta", "Strong"), ("Beta", "Weak")),
                    (("Alpha", "Strong"), ("Alpha", "Weak"))],
                                 perform_stat_test=False, pvalues=pvalues_slicing,
line_offset_to_box=0.25, line_offset=0.1, line_height=0.05, text_format='simple', loc='inside', verbose=2)

plt.tight_layout()
plt.legend()
plt.xticks(x,labels)
plt.ylabel('Smoothness coefficients')
plt.xlabel('Hilbert transform Bands')
plt.show()
fig.savefig('/homes/v20subra/S4B2/graph_analysis_during_PhD/Gretsi/smoothness.png', dpi=300, bbox_inches='tight')
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
plt.suptitle('The relationship between smoothness quotient & ISC subject-wise')
plt.tight_layout()
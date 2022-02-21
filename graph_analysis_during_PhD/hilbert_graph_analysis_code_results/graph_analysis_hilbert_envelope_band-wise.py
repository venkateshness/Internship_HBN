#%%
from cProfile import label
from fileinput import filename
from logging import error
from turtle import color
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import size
from pygsp import graphs, filters
from pygsp import plotting as gsp_plt
from nilearn import image, plotting, datasets

from pathlib import Path
from scipy import io as sio
from pygsp import graphs
from seaborn.utils import axis_ticklabels_overlap
from scipy.sparse import csr_matrix
import scipy
import matplotlib
import seaborn as sns
from scipy import stats


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

#%%

envelope_signal_bandpassed = np.load('/users/local/Venkatesh/Generated_Data/25_subjects/envelope_signal_bandpassed.npz', mmap_mode='r')

alpha = envelope_signal_bandpassed['alpha']
beta = envelope_signal_bandpassed['beta']
theta = envelope_signal_bandpassed['theta']

envelope_signal_bandpassed_gft_total = dict()

def GFT(band,which_band):
    envelope_signal_bandpassed_gft_total[which_band] = [G.gft(np.array(band[0])),G.gft(np.array(band[1])), 
        G.gft(np.array(band[2])), G.gft(np.array(band[3])), 
        G.gft(np.array(band[4])), G.gft(np.array(band[5])),
        G.gft(np.array(band[6])), G.gft(np.array(band[7])), 
        G.gft(np.array(band[8])),  G.gft(np.array(band[9]))]
    
GFT(theta,'theta')
GFT(alpha,'alpha')
GFT(beta,'beta')

#%%
def split(which_gft):
    which_gft = np.array(which_gft)

    which_gft_low_freq = which_gft[:,1:51,:]
    which_gft_medium_freq = which_gft[:,51:200,:]
    which_gft_high_freq = which_gft[:,200:,:]

    return which_gft_low_freq, which_gft_medium_freq, which_gft_high_freq

# %%
#########################################
######Utility function###################
#########################################
def mean_std(freq,ax):
    if ax>2:
        d = np.average(np.array(np.abs(freq)),axis=2)[:,1:]
    else: d = np.abs(freq[1:,:])
    mean_t = np.mean(d,axis=0)
    std_t = 2 * np.std(d,axis=0)
    top = mean_t + std_t
    bottom = mean_t - std_t
    
    return mean_t,std_t,top,bottom



isc_result = np.load('/users/local/Venkatesh/Generated_Data/sourceCCA_ISC.npz')['sourceCCA']
noise_floor_source = np.load('/users/local/Venkatesh/Generated_Data/noise_floor_1000_on_SI_full.npz')['a']

significance = np.array(np.where(np.max(np.array(noise_floor_source)[:,0,:],axis=0)<isc_result[0]))

def axvspan(axes,fs,c,alpha):
    # axes.axvspan(154*fs, 162*fs, alpha=alpha, color=c)
    # axes.axvspan(84*fs, 87*fs, alpha=alpha, color=c)
    # axes.axvspan(102*fs, 107*fs, alpha=alpha, color=c)
    # axes.axvspan(113*fs, 115*fs, alpha=alpha, color=c)
    # axes.axvspan(119*fs, 123*fs, alpha=alpha, color=c)
    # axes.axvspan(48*fs, 49*fs, alpha=alpha, color=c)

    # axes.axvspan(53*fs, 58*fs, alpha=alpha, color=c)
    # axes.axvspan(131*fs, 135*fs, alpha=alpha, color=c)
    # axes.axvspan(16*fs, 17*fs, alpha=alpha, color=c)
    # axes.axvspan(74*fs, 75*fs, alpha=alpha, color=c)
    axes.axvspan(5*fs, 13*fs, alpha=alpha, color='r')
    axes.axvspan(40*fs, 46*fs, alpha=alpha, color='r')
    axes.axvspan(164*fs, 170*fs, alpha=alpha, color='r')
    axes.axvspan(53*fs, 59*fs, alpha=alpha, color='b')
    axes.axvspan(84*fs, 87*fs, alpha=alpha, color='b')
    axes.axvspan(102*fs, 105*fs, alpha=alpha, color='b')
    axes.axvspan(154*fs, 162*fs, alpha=alpha, color='b')

    # items_weak = np.hstack([np.arange(5*125,13*125),np.arange(40*125,46*125),np.arange(164*125,170*125)])
    # items_strong = np.hstack([np.arange(53*125,59*125),np.arange(84*125,87*125),np.arange(102*125,105*125),np.arange(154*125,162*125)])




def plot(kwargs,title,filename):
    axes =[411,412,413,414]
    fig = plt.figure(figsize = (35,25))
    ax1 = plt.subplot(axes[0], frameon=False)
    ax2 = plt.subplot(axes[1], frameon=False)
    ax3 = plt.subplot(axes[2], frameon=False)
    ax4 = plt.subplot(axes[3], frameon=False)

    ax1.plot(np.sum(np.average(np.abs(kwargs[0]),axis=0),axis=0), marker = 'x', markeredgecolor = 'black', markevery=significance[0]*125,color='g',label='Low (2 to 50 freqs)')[0]
    axvspan(ax1,125, c = 'lavender', alpha = 0.8)
    ax1.legend(loc="upper left")

    ax2.plot(np.sum(np.average(np.abs(kwargs[1]),axis=0),axis=0), marker = 'x', markeredgecolor = 'black', markevery=significance[0]*125,color='b',label='Med (51 to 200)')[0]
    axvspan(ax2,125, c = 'lavender', alpha = 0.8)
    ax2.legend(loc="upper left")

    ax3.plot(np.sum(np.average(np.abs(kwargs[2]),axis=0),axis=0), marker = 'x', markeredgecolor = 'black', markevery=significance[0]*125,color='r',label='high (200 & above)')[0]
    axvspan(ax3,125, c = 'lavender', alpha = 0.8)
    ax3.legend(loc="upper left")

    # xticks= np.arange(1,21250,125)
    #ax1.legend((a1), ('Low', 'Med','High'), loc='lower left', shadow=True)

    xticks= np.arange(1,21250,2125)


    ax4.plot(range(1,171),isc_result[0], label='Source-level estimation')
    ax4.fill_between(range(1,171),np.max(np.array(noise_floor_source)[:,0,:],axis=0).T,np.min(np.array(noise_floor_source)[:,0,:],axis=0).T,color ='grey',alpha=0.8)
    ax4.plot(significance,isc_result[0][significance],
                marker='o', ls="",color='red',markersize=4)
    ax4.legend(loc="upper left")
    def set_ticks(ax):
        ax.set_xticks([])
        ax.set_xticks(xticks)
        ax.set_xticklabels(labels=np.arange(1,170,17),rotation='vertical')
    set_ticks(ax1)
    set_ticks(ax2)
    set_ticks(ax3)
    ax4.set_xticks([])
    ax4.set_xticks(np.arange(1,170,17))
    axvspan(ax4,1, c = 'lavender', alpha = 0.2)
    ax4.set_xticklabels(labels=np.arange(1,170,17),rotation='vertical')

    # ax1.get_shared_x_axes().join(ax1, ax2,ax3,ax4)
    ax1.set_xticklabels([])
    ax2.set_xticklabels([])
    ax3.set_xticklabels([])
    
    fig.text(0.5, 0.04, 'time (s)', ha='center')
    fig.text(0.08, 0.15, 'ISC coefficients', rotation='vertical')
    fig.text(0.08, 0.65, 'graph PSD',  rotation='vertical')
    plt.suptitle(title)
    plt.show()
    # fig.savefig(f'/homes/v20subra/S4B2/graph_analysis_during_PhD/hilbert_graph_analysis_code_results/{filename}')

plot(split(envelope_signal_bandpassed_gft_total['alpha']),title='The GFT of the Alpha band (8 - 13 Hz) envelope signal',filename='AlphaBand')


def slicing(what_to_slice,where_to_slice):
    array_to_append = list()
    array_to_append.append ( what_to_slice[:,where_to_slice] )
    return array_to_append


# %%
print(significance)
items_weak = np.hstack([np.arange(5*125,13*125),np.arange(40*125,46*125),np.arange(164*125,170*125)])
items_strong = np.hstack([np.arange(53*125,59*125),np.arange(84*125,87*125),np.arange(102*125,105*125),np.arange(154*125,162*125)])

low,med,high = split(np.abs(envelope_signal_bandpassed_gft_total['alpha']))

def slice_and_sum(freqs,items_strong,items_weak):

    summed_strong_time = np.average(slicing(freqs,items_strong),axis=2)
    summed_weak_time =  np.average(slicing(freqs,items_weak),axis=2)

    summed_strong_subjs =  np.sum(summed_strong_time,axis=1)
    summed_weak_subjs =  np.sum(summed_weak_time,axis=1)

    return summed_strong_subjs, summed_weak_subjs,scipy.stats.ttest_rel(summed_strong_time,summed_weak_time,axis=1)


summed_low_freqs = np.sum(low,axis=1)
summed_med_freqs = np.sum(med,axis=1)
summed_high_freqs = np.sum(high,axis=1)


labels = ['Low', 'Med', 'High']
to_plot_strong_ISC_low_freqs, to_plot_weak_ISC_low_freqs, ttest_low_freqs = slice_and_sum(summed_low_freqs,items_strong,items_weak)
to_plot_strong_ISC_med_freqs, to_plot_weak_ISC_med_freqs, ttest_med_freqs = slice_and_sum(summed_med_freqs,items_strong,items_weak)
to_plot_strong_ISC_high_freqs, to_plot_weak_ISC_high_freqs, ttest_high_freqs = slice_and_sum(summed_high_freqs,items_strong,items_weak)

to_plot_strong_ISC = [to_plot_strong_ISC_low_freqs, to_plot_strong_ISC_med_freqs, to_plot_strong_ISC_high_freqs]
to_plot_weak_ISC = [to_plot_weak_ISC_low_freqs, to_plot_weak_ISC_med_freqs, to_plot_weak_ISC_high_freqs]

print(ttest_low_freqs)
print(ttest_med_freqs)
print(ttest_high_freqs)


x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars


def stats_SEM(freqs,items):
    return  scipy.stats.sem(np.average(np.array(slicing(freqs,items)),axis=2).T)

std_err_weak = np.hstack([stats_SEM(summed_low_freqs,items_weak),
                stats_SEM(summed_med_freqs,items_weak),
                stats_SEM(summed_high_freqs,items_weak)])

std_err_strong = np.hstack([stats_SEM(summed_low_freqs,items_strong),
                stats_SEM(summed_med_freqs,items_strong),
                stats_SEM(summed_high_freqs,items_strong)])


rects1 = plt.bar(x - width/2, to_plot_strong_ISC[0], width, label='Strong ISC', yerr = std_err_strong, align='center', alpha=0.5, ecolor='black', capsize=10)
rects2 = plt.bar(x + width/2, to_plot_weak_ISC[0], width, label='Weak ISC', yerr = std_err_weak, align='center', alpha=0.5, ecolor='black', capsize=10)

plt.ylabel("gPSD")
plt.xticks(x,labels)


# %%


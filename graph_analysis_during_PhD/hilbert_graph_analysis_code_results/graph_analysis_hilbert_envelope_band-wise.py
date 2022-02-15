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

envelope_signal_bandpassed = np.load('/users/local/Venkatesh/Generated_Data/eLORETA_extensive_validation/envelope_signal_bandpassed.npz', mmap_mode='r')

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



isc_result = np.load('/users/local/Venkatesh/Generated_Data/sourceCCA_ISC.npz')['sourceCCA']
noise_floor_source = np.load('/users/local/Venkatesh/Generated_Data/noise_floor_1000_on_SI_full.npz')['a']

significance = np.array(np.where(np.max(np.array(noise_floor_source)[:,0,:],axis=0)<isc_result[0]))

def plot(kwargs,title,filename):
    axes =[411,412,413,414]
    fig = plt.figure(figsize = (20,20))
    ax1 = plt.subplot(axes[0], frameon=False)
    ax2 = plt.subplot(axes[1], frameon=False)
    ax3 = plt.subplot(axes[2], frameon=False)
    ax4 = plt.subplot(axes[3], frameon=False)

    ax1.plot(np.average(np.average(np.abs(kwargs[0]),axis=0),axis=0),color='g',label='Low')[0]
    ax1.legend(loc="upper left")

    ax2.plot(np.average(np.average(np.abs(kwargs[1]),axis=0),axis=0),color='b',label='Med')[0]
    ax2.legend(loc="upper left")

    ax3.plot(np.average(np.average(np.abs(kwargs[2]),axis=0),axis=0),color='r',label='high')[0]
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
    ax4.set_xticklabels(labels=np.arange(1,170,17),rotation='vertical')


    ax1.get_shared_x_axes().join(ax1, ax2,ax3,ax4)
    ax1.set_xticklabels([])
    ax2.set_xticklabels([])
    ax3.set_xticklabels([])


    fig.text(0.5, 0.04, 'time (s)', ha='center')
    fig.text(0.08, 0.15, 'ISC coefficients', rotation='vertical')
    fig.text(0.08, 0.65, 'graph PSD',  rotation='vertical')
    plt.suptitle(title)
    plt.show()
    fig.savefig(f'/homes/v20subra/S4B2/graph_PhD/hilbert_graph_analysis_code_results/{filename}')

plot(split(envelope_signal_bandpassed_gft_total['beta']),title='The GFT of the Beta band (13-30Hz) envelope signal',filename='BetaBand')
# %%
